import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import FairseqNATDecoder, FairseqNATModel, ensemble_decoder
from fairseq.models.transformer import Embedding, TransformerDecoder, TransformerModel
from fairseq.models.nat.nonautoregressive_transformer import NATransformerModel, NATransformerDecoder
from multitasknat.models.nat_ctc import NAT_ctc_model, NAT_ctc_encoder, NAT_ctc_decoder
from fairseq.models.fairseq_incremental_decoder import FairseqIncrementalDecoder
from fairseq.modules.transformer_layer import TransformerDecoderLayer
import copy
import random
from typing import Optional, List, Dict
from torch import Tensor
from omegaconf import DictConfig


class ShallowTranformerDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=False)

        self.dictionary = dictionary
        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TransformerDecoderLayer(args, no_encoder_attn)
                for _ in range(args.shallow_at_decoder_layers)
            ]
        )


class hybrid_decoder(NAT_ctc_decoder):
    def __init__(self, args, dictionary, embed_tokens, at_dec_nat_enc,
                 at_dec_nat_dec_1,
                 at_dec_nat_dec_2,
                 at_dec_nat_dec_3,
                 at_dec_nat_dec_4,
                 at_dec_nat_dec_5,
                 at_dec_nat_dec_6):
        super().__init__(args, dictionary, embed_tokens)
        self.at_dec_nat_enc = at_dec_nat_enc
        self.at_dec_nat_dec_1 = at_dec_nat_dec_1
        self.at_dec_nat_dec_2 = at_dec_nat_dec_2
        self.at_dec_nat_dec_3 = at_dec_nat_dec_3
        self.at_dec_nat_dec_4 = at_dec_nat_dec_4
        self.at_dec_nat_dec_5 = at_dec_nat_dec_5
        self.at_dec_nat_dec_6 = at_dec_nat_dec_6

    def forward(self, encoder_out, prev_output_tokens, normalize: bool = False, features_only: bool = False):
        features, _ = self.extract_features(
            encoder_out=encoder_out,
            prev_output_tokens=prev_output_tokens
        )
        if features_only:  # used for mt_ctc
            return features, _
        decoder_out = self.output_layer(features)
        return F.log_softmax(decoder_out, -1) if normalize else decoder_out


@register_model("mt_ctc_multi")
class mt_ctc_multi_model(NAT_ctc_model):
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        # nat_decoder = NATransformerModel.build_decoder(args, tgt_dict, embed_tokens)
        if getattr(args, "share_at_decoder", False):
            at_dec = ShallowTranformerDecoder(args, tgt_dict, embed_tokens)
            return hybrid_decoder(args, tgt_dict, embed_tokens,
                                  at_dec, at_dec, at_dec, at_dec, at_dec, at_dec, at_dec)
        at_dec_nat_enc = ShallowTranformerDecoder(args, tgt_dict, embed_tokens)
        at_dec_nat_dec_1 = ShallowTranformerDecoder(args, tgt_dict, embed_tokens)
        at_dec_nat_dec_2 = ShallowTranformerDecoder(args, tgt_dict, embed_tokens)
        at_dec_nat_dec_3 = ShallowTranformerDecoder(args, tgt_dict, embed_tokens)
        at_dec_nat_dec_4 = ShallowTranformerDecoder(args, tgt_dict, embed_tokens)
        at_dec_nat_dec_5 = ShallowTranformerDecoder(args, tgt_dict, embed_tokens)
        at_dec_nat_dec_6 = ShallowTranformerDecoder(args, tgt_dict, embed_tokens)
        return hybrid_decoder(args, tgt_dict, embed_tokens, at_dec_nat_enc,
                              at_dec_nat_dec_1,
                              at_dec_nat_dec_2,
                              at_dec_nat_dec_3,
                              at_dec_nat_dec_4,
                              at_dec_nat_dec_5,
                              at_dec_nat_dec_6
                              )

    def add_args(parser):
        NAT_ctc_model.add_args(parser)
        parser.add_argument("--shallow-at-decoder-layers", type=int, metavar='N',
                            help="the number of at decoder.")
        parser.add_argument("--share-at-decoder", default=False, action='store_true',
                            help='if set, share all at decoder\'s param.')
        parser.add_argument("--is-random", default=False, action='store_true',
                            help='if set, randomly select at decoder layer.')
        parser.add_argument("--without-enc", default=False, action='store_true',
                            help='do not use nat encoder output.')

    def forward(self, at_src_tokens, nat_src_tokens, src_lengths, prev_nat, prev_at, tgt_tokens, **kwargs):
        # 执行两次模型。at_src_tokens和nat_src_tokens的区别就在于顺序不同
        nat_encoder_out = self.encoder(nat_src_tokens, src_lengths=src_lengths, **kwargs)
        at_encoder_out = nat_encoder_out
        if getattr(self.args, "if_deepcopy_at_sample", False):
            at_encoder_out = self.encoder(at_src_tokens, src_lengths=src_lengths, **kwargs)
            nat_decode_output = self.decoder(nat_encoder_out,
                                             prev_nat,
                                             normalize=False,
                                             features_only=False)
            _, dec_each_layer_output_and_attn = self.decoder(at_encoder_out,
                                                             prev_nat,
                                                             normalize=False,
                                                             features_only=True)
        else:
            nat_decode_features, dec_each_layer_output_and_attn = self.decoder(nat_encoder_out,
                                                                               prev_nat,
                                                                               normalize=False,
                                                                               features_only=True)

            nat_decode_output = self.decoder.output_layer(nat_decode_features)

        # AT shallow decoding
        at_dec_nat_output = []
        if not getattr(self.args, "without_enc", False):
            # on nat encoder
            at_dec_nat_enc_output, _ = self.decoder.at_dec_nat_enc(prev_at,
                                                                   encoder_out=at_encoder_out,
                                                                   features_only=False,
                                                                   return_all_hiddens=False)
            at_dec_nat_output.append(at_dec_nat_enc_output)
        # on each nat decoder layer
        dec_each_layer_output = dec_each_layer_output_and_attn['inner_states']
        dec_dict = {
            1: self.decoder.at_dec_nat_dec_1,
            2: self.decoder.at_dec_nat_dec_2,
            3: self.decoder.at_dec_nat_dec_3,
            4: self.decoder.at_dec_nat_dec_4,
            5: self.decoder.at_dec_nat_dec_5,
            6: self.decoder.at_dec_nat_dec_6
        }
        dec_list = [1, 2, 3, 4, 5, 6]
        if getattr(self.args, "is_random", False):
            random.shuffle(dec_list)
            dec_list = dec_list[:3]
        for idx, dec_layer_output in enumerate(dec_each_layer_output):
            # initial x
            if idx not in dec_list:
                continue
            shallow_at_encode_output = {
                "encoder_out": [dec_layer_output],
                "encoder_padding_mask": [at_encoder_out["upsample_mask"]]
            }
            at_dec_nat_dec_layer_output, _ = dec_dict[idx](prev_at,
                                                           encoder_out=shallow_at_encode_output,
                                                           features_only=False,
                                                           return_all_hiddens=False)
            at_dec_nat_output.append(at_dec_nat_dec_layer_output)

        return ({
                    "out": nat_decode_output,  # T x B x C
                    "name": "NAT"
                },
                {
                    "out": at_dec_nat_output,  # B x T x C
                    "name": "AT"
                }
        )


@register_model_architecture("mt_ctc_multi", "mt_ctc_multi")
def base_architecture(args):
    # This is actually nat_ctc_decoder.
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.upsample_scale = getattr(args, "upsample_scale", 3)
    args.shallow_at_decoder_layers = getattr(args, "shallow_at_decoder_layers", 1)

    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)
