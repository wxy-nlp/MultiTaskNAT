import numpy as np
import torch
from argparse import Namespace
import random
import math
import sacrebleu
from sacrebleu.metrics.bleu import BLEU
from sacrebleu.utils import my_log


# a = (1, 2)
# b = 1
# print(isinstance(a, tuple))
# print(isinstance(b, tuple))
# # x = torch.tensor([[2.0, 0, 4, 5, 1, 1], [2, 0, 4, 5, 6, 7]])
# # y = x.ne(1).sum(dim=1)
# # x = x.unsqueeze(-1)
# #
# # reverse_x = x.clone()
# # for i, length in enumerate(y):
# #     reverse_x[i, 1] = x[i, 0].clone()
# #     reverse_x[i, 0] = x[i, 1].clone()
# #     _body = x[i, 2:length].clone()
# #     _body_reverse = torch.flip(_body, [-2])
# #     reverse_x[i, 2:length] = _body_reverse
# # print(reverse_x)
# # a = torch.tensor([1.0, 2.20])
# # b = torch.tensor(2.0, requires_grad=TrueE)
# # _a = a.clone()
# # _a = _a + b
# # y = _a**2
# #
# # Z = a**2+a*3
# # y.backward()
# # print(a.grad)
# # a = torch.flip(a, [0])
# # print(a)
# # b = 1
def check_ckpt():
    local_path = '/home/wangxy/mt_ctc_multi-wmt14-without-enc/checkpoint.best_average_5.pt'
    with open(local_path, "rb") as f:
        state = torch.load(f, map_location=torch.device("cpu"))
    wxy = 1


# str = "a"
# print(ord(str) - 97)

# str1 = ["12ap"]
# print(len(str1)
# x = torch.tensor([1,2,3,4])
# y = 2
# z = x > y
# a = torch.tensor([5,5,5,5])
# a[z] = 2
# wxy = 1
# check_ckpt()

# a = torch.zeros(5, 1)
# b = torch.ones(5, 4).fill_(2)
# c = torch.cat((a, b), 1)
# print(c)
# input = [
#     [2, 3, 4, 5, 0, 0],
#     [1, 4, 3, 0, 0, 0],
#     [4, 2, 2, 5, 7, 0],
#     [1, 0, 0, 0, 0, 0]
# ]
# input = torch.tensor(input)
# #注意index的类型
# length = torch.LongTensor([[0,1,2]])
# #index之所以减1,是因为序列维度是从0开始计算的
# out = torch.gather(input, 0, length)
# ssh = "pabu"

# a = torch.tensor([1, 2, 3.], requires_grad=True)
# print(a.grad)
# out = a.sigmoid()
#
# out.sum().backward()
# print(a.grad)

# a = torch.tensor([[[1,2,3,4],[1,2,3,4],[1,2,3,4]], [[3,4,3,4],[3,4,3,4],[3,4,3,4]]])

# c = b.expand(-1,-1,-1,3)
# d = c.reshape(6,3,4)
# print(a)
# print(b)
# print(c)
# print(d)
## 并行操作reverse
# a = torch.tensor([[1,2,3,4,-1,-1],[10,20,30,40,50,-1],[100,200,300,-1,-1,-1]])
# a = torch.flip(a, [-1])
# b = (a == -1)
# c = b.int()
# _, idx = torch.sort((a == -1).type_as(a))
# b = a.gather(1, idx)
# d = torch.arange(5)
# aa1 = torch.roll(a, shifts=())
# for aaa in a:
#     wx = 2
# c = a.scatter_(1, [b[:, None],b[:,None]-1], -1)
# wxy =1

all_decode = ['and our algorithms -- of course , not one person , but our algorithms , our algorithms , the related '
              'search is &quot; i &apos;m boring . &quot;', 'and our algorithms -- of course , not one person , '
                                                            'but our algorithms , our algorithms , the related search '
                                                            'is &quot; i &apos;m bored . &quot;', 'and our algorithms '
                                                                                                  '-- of course , '
                                                                                                  'not one person , '
                                                                                                  'but our algorithms '
                                                                                                  ', our algorithms '
                                                                                                  '-- that the '
                                                                                                  'related search is '
                                                                                                  '&quot; i &apos;m '
                                                                                                  'bored . &quot;',
              'and our algorithms -- of course , not one person , but our algorithms , our algorithms -- that the '
              'related search is &quot; i am bored . &quot;', 'and our algorithms -- of course , not one person , '
                                                              'but our algorithms , our algorithms , that the related'
                                                              ' search is &quot; i &apos;m bored . &quot;']


def my_compute_bleu(all_decode, beam_size):
    def compute(match_dict, total_dict, hyp, ref_list):
        score_list = []
        for ref in ref_list:
            if ref == hyp:
                continue
            precision = [100. * match_array[hyp, ref] / total_dict[order][hyp]
                         for order, match_array in match_dict.items()]
            sys_len, ref_len = len_array[hyp], len_array[ref]
            if sys_len < ref_len:
                bp = math.exp(1 - ref_len / sys_len)
            else:
                bp = 1.0
            score = bp * math.exp(sum(map(my_log, precision)) / 4)
            score_list.append(score)
        return score_list

    # all_decode: candidate
    ngrams_list = [BLEU.extract_ngrams(sentence) for sentence in all_decode]
    match_array_1 = np.zeros([beam_size, beam_size])
    match_array_2 = np.zeros([beam_size, beam_size])
    match_array_3 = np.zeros([beam_size, beam_size])
    match_array_4 = np.zeros([beam_size, beam_size])
    match_dict = {
        1: match_array_1,
        2: match_array_2,
        3: match_array_3,
        4: match_array_4
    }
    total_array_1 = np.zeros(beam_size)
    total_array_2 = np.zeros(beam_size)
    total_array_3 = np.zeros(beam_size)
    total_array_4 = np.zeros(beam_size)
    total_dict = {
        1: total_array_1,
        2: total_array_2,
        3: total_array_3,
        4: total_array_4
    }
    len_array = np.zeros(beam_size)
    for i in range(beam_size):
        len_array[i] += len(all_decode[i].split())
        sys_ngrams = ngrams_list[i]
        if i == beam_size - 1:
            for ngram in sys_ngrams.keys():
                n = len(ngram.split())
                total_dict[n][i] += sys_ngrams[ngram]
        for j in range(i + 1, beam_size):
            ref_ngrams = ngrams_list[j]
            for ngram in sys_ngrams.keys():
                n = len(ngram.split())
                if j == beam_size - 1:
                    total_dict[n][i] += sys_ngrams[ngram]
                match_dict[n][i, j] += min(sys_ngrams[ngram], ref_ngrams.get(ngram, 0))
            for _, match_array in match_dict.items():
                match_array[j, i] = match_array[i, j]
    arr = np.arange(beam_size)
    for hyp in arr:
        if hyp == 2:
            wxy = compute(match_dict, total_dict, hyp, arr)
    score_list = [compute(match_dict, total_dict, hyp, arr) for hyp in arr]
    tmp1 = [all_decode[0], all_decode[1], all_decode[3], all_decode[4]]
    tmp = ([sacrebleu.corpus_bleu(all_decode[2], [[tmp1[i]]], tokenize='none').score
            for i in range(4)])
    return score_list


my_compute_bleu(all_decode, 5)
