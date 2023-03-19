# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from copy import deepcopy
import logging

import torch

from dataclasses import dataclass, field
from fairseq.tasks import register_task
from fairseq import utils
from multitasknat.tasks.nat_ctc_task import NATCTCConfig, NATCTC_Task

EVAL_BLEU_ORDER = 4
logger = logging.getLogger(__name__)


def drop_sentences_(sample, rate, indexes=None):
    def gen_randperm(bsz, droprate):
        nbsz = max(1, int((1.0 - droprate) * bsz))
        return torch.randperm(bsz)[:nbsz]

    bsz = sample['nsentences']
    if indexes is None:
        indexes = gen_randperm(bsz, rate)
    nbsz = indexes.size(0)
    for k, v in sample['net_input'].items():
        if isinstance(v, torch.Tensor):
            sample['net_input'][k] = v[indexes]
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            sample[k] = v[indexes]
    sample['ntokens'] = sample['ntokens'] * nbsz // bsz
    sample['nsentences'] = nbsz
    return sample


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


@dataclass
class MTCTCConfig(NATCTCConfig):
    if_deepcopy_at_sample: bool = field(
        default=False, metadata={"help": "if set, shuffle at sample."}
    )
    start_p: float = field(
        default=0.5, metadata={"help": "minus prob"}
    )
    minus_p: float = field(
        default=0.2, metadata={"help": "minus prob"}
    )
    total_up: int = field(
        default=300000, metadata={"help": "total updates"}
    )
    glat: bool = field(
        default=False,
    )


@register_task('mt_ctc_task', dataclass=MTCTCConfig)
class MT_CTC_Task(NATCTC_Task):
    def train_step(self,
                   sample,
                   model,
                   criterion,
                   optimizer,
                   update_num,
                   ignore_grad=False, **kwargs):
        model.train()
        glat = None
        if getattr(self.cfg, "glat", False):
            train_ratio = max(0, min(1, update_num / self.cfg.total_up))
            glat = {"context_p": self.cfg.start_p - self.cfg.minus_p * train_ratio}
        at_sample = sample
        if getattr(self.cfg, "if_deepcopy_at_sample", False):
            at_sample = deepcopy(sample)
            at_sample = drop_sentences_(at_sample, rate=0.0)
        nat_sample = sample
        src_tokens = sample["net_input"]["src_tokens"].clone()
        upsample_src_tokens = normal(src_tokens, self.cfg.upsample_scale, self.src_dict)
        nat_sample['prev_target'] = upsample_src_tokens
        loss, sample_size, logging_output = criterion(model, at_sample, nat_sample, None, glat=glat, **kwargs)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion, **kwargs):
        model.eval()
        with torch.no_grad():
            at_sample = deepcopy(sample)
            nat_sample = sample
            src_tokens = sample["net_input"]["src_tokens"].clone()
            upsample_src_tokens = normal(src_tokens, self.cfg.upsample_scale, self.src_dict)
            nat_sample['prev_target'] = upsample_src_tokens
            loss, sample_size, logging_output = criterion(model, at_sample, nat_sample)
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

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            extra_symbols_to_ignore = []
            if hasattr(self.tgt_dict, "blank_index"): extra_symbols_to_ignore.append(self.tgt_dict.blank_index)
            if hasattr(self.tgt_dict, "mask_index"): extra_symbols_to_ignore.append(self.tgt_dict.mask_index)
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.cfg.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
                extra_symbols_to_ignore=extra_symbols_to_ignore or None
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyp = gen_out[i][0]['tokens']
            if not self.cfg.use_ctc_bs:
                _toks = hyp.int().tolist()
                hyp = hyp.new_tensor([v for i, v in enumerate(_toks) if i == 0 or v != _toks[i - 1]])
            hyps.append(decode(hyp))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.cfg.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.cfg.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])
