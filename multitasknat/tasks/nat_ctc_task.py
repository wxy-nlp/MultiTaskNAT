# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
from copy import deepcopy
from dataclasses import dataclass, field

import torch

from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask, load_langpair_dataset
from fairseq.tasks.translation_lev import TranslationLevenshteinTask, TranslationLevenshteinConfig
from fairseq.utils import new_arange
from fairseq.data.dictionary import Dictionary, ExpendDictionary
from fairseq import utils
from fairseq.data import data_utils

EVAL_BLEU_ORDER = 4
logger = logging.getLogger(__name__)

@dataclass
class NATCTCConfig(TranslationLevenshteinConfig):
    upsample_scale: int = field(
        default=3, metadata={"help": "the amount of src_tokens upsample scale."}
    )
    ctc_loss: bool = field(
        default=True, metadata={"help": "use ctc loss."}
    )
    upsample_strict: bool = field(
        default=False, metadata={"help": "if upsample with strict in a strict way."}
    )
    ctc_decode_with_beam: int = field(
        default=1, metadata={"help": "the ctc beam decode size."}
    )
    use_ctc_bs: bool = field(
        default=False, metadata={"help": "if upsample with strict in a strict way."}
    )
    expand_dataset: bool = field(
        default=False, metadata={"help": "if set, expand special token with <blk> and <mask>."}
    )


@register_task('nat_ctc_task', dataclass=NATCTCConfig)
class NATCTC_Task(TranslationLevenshteinTask):
    def __init__(self, cfg, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        self.ctc_decode_with_beam = getattr(cfg, "ctc_decode_with_beam", 1)

    def get_upsample_src_tokens(self, model, sample):
        """
        src_tokens:
        0 x x x 2 1 1 1
        0 x x x x 2 1 1
        normal:
        0 0 0 x x x x x x 2 2 2 1 1 1 1 1 1
        0 0 0 x x x x x x x x 2 2 2 1 1 1 1
        strict:
        0 x x x x x x 2 1 1 1 1 1 1
        0 x x x x x x x x 2 1 1 1 1
        """
        def strict(src_tokens, src_lengths, src_dict):
            bsz = src_tokens.size(0)
            src_len_with_pad = src_tokens.size(1)
            upsample_src_len_with_pad = (src_len_with_pad - 2) * model.scale + 2
            upsample_src_tokens = torch.zeros([bsz, upsample_src_len_with_pad]).fill_(-1).type_as(src_tokens)
            # fix bos and eos
            upsample_src_tokens[:, 0] = src_dict.bos()
            upsample_src_tokens[torch.arange(0, bsz, 1), (src_lengths - 2) * model.scale + 1] = src_dict.eos()
            # fix pad
            upsample_src_lengths = (src_lengths - 2) * model.scale + 2
            upsample_src_lengths = upsample_src_lengths.expand(upsample_src_len_with_pad, -1).transpose(0, 1)
            arange = torch.arange(1, upsample_src_len_with_pad + 1, 1).expand(bsz, -1).type_as(src_tokens)
            upsample_src_tokens = upsample_src_tokens.masked_fill_(
                arange > upsample_src_lengths, src_dict.pad()
            )
            return upsample_src_tokens

        def normal(src_tokens, scale, src_dict):
            pad = src_dict.pad()
            bos = src_dict.bos()
            eos = src_dict.eos()
            unk = src_dict.unk()

            _mask = (
                    src_tokens.eq(bos) | src_tokens.eq(eos) | src_tokens.eq(pad)
            )
            src_tokens = src_tokens.masked_fill(~_mask, unk)
            bsz = src_tokens.size(0)
            upsample_src_tokens = src_tokens.unsqueeze(-1).expand(bsz, -1, scale).reshape(bsz, -1)
            return upsample_src_tokens

        src_tokens = sample["net_input"]["src_tokens"]
        src_lengths = sample["net_input"]["src_lengths"]
        if self.cfg.upsample_strict: return strict(src_tokens, src_lengths)
        else: return normal(src_tokens, self.cfg.upsample_scale, self.src_dict)

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False, **kwargs
    ):
        model.train()
        upsample_src_tokens = self.get_upsample_src_tokens(model, sample)
        sample["prev_target"] = upsample_src_tokens
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion, **kwargs):
        model.eval()
        with torch.no_grad():
            upsample_src_tokens = self.get_upsample_src_tokens(model, sample)
            sample["prev_target"] = self.inject_noise(upsample_src_tokens)
            loss, sample_size, logging_output = criterion(model, sample)
            # 以下为后来添加
            if self.cfg.eval_bleu:
                bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
                logging_output["_bleu_sys_len"] = bleu.sys_len
                logging_output["_bleu_ref_len"] = bleu.ref_len
                # we split counts into separate entries so that they can be
                # summed efficiently across workers using fast-stat-sync
                assert len(bleu.counts) == EVAL_BLEU_ORDER
                for i in range(EVAL_BLEU_ORDER):
                    logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                    logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def build_generator(self, models, args, **unused):
        # add models input to match the API for SequenceGenerator
        from fairseq.iterative_refinement_generator import IterativeRefinementGenerator

        return IterativeRefinementGenerator(
            self.target_dictionary,
            eos_penalty=getattr(args, "iter_decode_eos_penalty", 0.0),
            max_iter=getattr(args, "iter_decode_max_iter", 0),
            beam_size=self.ctc_decode_with_beam,
            reranking=getattr(args, "iter_decode_with_external_reranker", False),
            decoding_format=getattr(args, "decoding_format", None),
            adaptive=not getattr(args, "iter_decode_force_max_iter", True),
            retain_history=getattr(args, "retain_iter_history", False),
            ctc_model=True
        )

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        model_args = kwargs['model']
        args.left_pad_source = utils.eval_bool(args.left_pad_source)
        args.left_pad_target = utils.eval_bool(args.left_pad_target)

        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(paths[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = cls.load_dictionary(args, model_args, os.path.join(paths[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(args, model_args, os.path.join(paths[0], 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info('[{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        logger.info('[{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

    @classmethod
    def load_dictionary(cls, args, model_args, filename):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        if getattr(args, "expand_dataset", False):
            d = ExpendDictionary.load(filename)
        else:
            d = Dictionary.load(filename)
        if getattr(args, "ctc_loss", False):
            d.blank_index = d.add_symbol("<blank>")
            d.nspecial += 1
            d.blank_word = "<blank>"
        return d

    def filter_indices_by_size(
            self,
            indices,
            dataset,
            max_positions=None,
            ignore_invalid_inputs=False,
    ):
        """
        Filter examples that are too large

        Args:
            indices (np.array): original array of sample indices
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
        Returns:
            np.array: array of filtered sample indices
        """
        original_size = len(indices)
        if ignore_invalid_inputs and hasattr(self.cfg, "upsample_scale"):
            max_positions = (
                (dataset.tgt_sizes[indices] * self.cfg.upsample_scale).tolist(),
                (dataset.src_sizes[indices] * self.cfg.upsample_scale).tolist(),
            )
        indices, ignored = dataset.filter_indices_by_size(indices, max_positions)
        if len(ignored) > 0:
            if not ignore_invalid_inputs:
                raise Exception((
                                    'Size of sample #{} is invalid (={}) since max_positions={}, '
                                    'skip this example with --skip-invalid-size-inputs-valid-test'
                                ).format(ignored[0], dataset.size(ignored[0]), max_positions))
            if hasattr(self.cfg, "upsample_scale"):
                logger.warning((
                                   'when ctc loss enabled, {} samples have invalid sizes and will be skipped, '
                                   'where the src_len * {} < tgt_len'
                               ).format(len(ignored), self.cfg.upsample_scale))
            else:
                logger.warning((
                                   '{} samples have invalid sizes and will be skipped, '
                                   'max_positions={}, first few sample ids={}'
                               ).format(len(ignored), max_positions, ignored[:10]))

            logger.info(f"Dataset original size: {original_size}, filtered size: {len(indices)}")

        return indices



