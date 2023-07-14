#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

savedir=/path/to/savedir
dataset=/path/to/dataset
userdir=multitasknat
task=mt_ctc_task
criterion=mt_ctc_loss
arch=mt_ctc_multi
max_token=8192
max_update=300000
update_freq=2

echo "=============Training============="

python ../train.py \
    --save-dir ${savedir} \
    --user-dir ${userdir} \
    ${dataset} \
    --arch ${arch} \
    --task ${task} \
    --criterion ${criterion} \
    --fp16 \
    --ddp-backend=no_c10d \
    --shallow-at-decoder-layers 1 \
    --lambda-nat-at 0.5 \
    --is-random \
    --share-at-decoder \
    --noise full_mask \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9,0.999)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --stop-min-lr '1e-09' --warmup-updates 10000 \
    --warmup-init-lr '1e-07' --label-smoothing 0.1 \
    --dropout 0.1 --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --apply-bert-init \
    --log-format 'simple' --log-interval 100 \
    --fixed-validation-seed 7 \
    --max-tokens ${max_token} \
    --update-freq ${update_freq} \
    --max-update ${max_update} \
    --eval-bleu \
    --eval-bleu-args '{"beam": 1, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --keep-last-epochs 3 \
    --keep-best-checkpoints 5

echo "=============Averaging checkpoints============="

python ../scripts/average_checkpoints.py \
    --inputs ${savedir} \
    --num-best-checkpoints 5 \
    --output ${savedir}/checkpoint.best_average_5.pt

python ../scripts/average_checkpoints.py \
    --inputs ${savedir} \
    --num-top-checkpoints 5 \
    --output ${savedir}/checkpoint.top5_average_5.pt

echo "=============Generating by average============="

python ../generate.py \
    --path ${savedir}/checkpoint.best_average_5.pt \
    ${dataset} \
    --gen-subset test \
    --task mt_ctc_task \
    --iter-decode-max-iter 0 \
    --iter-decode-eos-penalty 0 \
    --beam 1 \
    --remove-bpe \
    --print-step \
    --batch-size 256

python ../generate.py \
    --path ${savedir}/checkpoint.top5_average_5.pt \
    ${dataset} \
    --gen-subset test \
    --task mt_ctc_task \
    --iter-decode-max-iter 0 \
    --iter-decode-eos-penalty 0 \
    --beam 1 \
    --remove-bpe \
    --print-step \
    --batch-size 256

echo "=============Finish============="