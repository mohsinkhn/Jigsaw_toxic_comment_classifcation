#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 18:49:28 2018

@author: mohsin
"""
import numpy as np
from collections import Counter
import itertools


class Tokenizer:
    def __init__(self, max_features=20000, max_len=10, tokenizer=str.split):
        self.max_features = max_features
        self.tokenizer = tokenizer
        self.doc_freq = None
        self.vocab = None
        self.vocab_idx = None
        self.max_len = max_len

    def fit_transform(self, texts):
        tokenized = []
        n = len(texts)

        tokenized = [self.tokenizer(text) for text in texts]
        self.doc_freq = Counter(itertools.chain.from_iterable(tokenized))

        vocab = [t[0] for t in self.doc_freq.most_common(self.max_features)]
        vocab_idx = {w: (i + 1) for (i, w) in enumerate(vocab)}
        # doc_freq = [doc_freq[t] for t in vocab]

        # self.doc_freq = doc_freq
        self.vocab = vocab
        self.vocab_idx = vocab_idx

        result_list = []
        #tokenized = [self.tokenizer(text) for text in texts]
        for text in tokenized:
            text = self.text_to_idx(text, self.max_len)
            result_list.append(text)

        result = np.zeros(shape=(n, self.max_len), dtype=np.int32)
        for i in range(n):
            text = result_list[i]
            result[i, :len(text)] = text

        return result

    def text_to_idx(self, tokenized, max_len):
        return [self.vocab_idx[t] for i, t in enumerate(tokenized) if (t in self.vocab_idx) and (i < max_len)]

    def transform(self, texts):
        n = len(texts)
        result = np.zeros(shape=(n, self.max_len), dtype=np.int32)

        for i in range(n):
            text = self.tokenizer(texts[i])
            text = self.text_to_idx(text, self.max_len)
            result[i, :len(text)] = text

        return result

    def vocabulary_size(self):
        return len(self.vocab) + 1