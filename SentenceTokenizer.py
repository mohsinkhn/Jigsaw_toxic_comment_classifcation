#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 10:51:19 2018

@author: mohsin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 18:49:28 2018

@author: mohsin
"""
import numpy as np
from collections import Counter
import itertools


class SentenceTokenizer:
    def __init__(self, max_features=20000, max_sentence_len=20, max_sentences=10, tokenizer=lambda x: [str.split(s) for s in str(x).split('.')]):
        self.max_features = max_features
        self.tokenizer = tokenizer
        self.doc_freq = None
        self.vocab = None
        self.vocab_idx = None
        self.max_sentence_len = max_sentence_len
        self.max_sentences = max_sentences

    def fit_transform(self, texts):
        tokenized = []
        n = len(texts)

        tokenized = [self.tokenizer(text) for text in texts]
        self.doc_freq = Counter(itertools.chain.from_iterable(itertools.chain.from_iterable(tokenized)))

        vocab = [t[0] for t in self.doc_freq.most_common(self.max_features)]
        vocab_idx = {w: (i + 1) for (i, w) in enumerate(vocab)}
        # doc_freq = [doc_freq[t] for t in vocab]

        # self.doc_freq = doc_freq
        self.vocab = vocab
        self.vocab_idx = vocab_idx

        #result_list = []
        #tokenized = [self.tokenizer(text) for text in texts]
        #for text in tokenized:
        #    text = self.text_to_idx(text)
        #    result_list.append(text)

        result = np.zeros(shape=(n, self.max_sentences, self.max_sentence_len), dtype=np.int32)
        for i, sentences in enumerate(tokenized):
            for j, sent in enumerate(sentences):
                if j< self.max_sentences:
                    for k, word in enumerate(sent):
                        if (k < self.max_sentence_len) and (word in self.vocab_idx):
                            result[i,j,k] = self.vocab_idx[word]
                              

        return result

    def text_to_idx(self, ):
        return [[self.vocab_idx[t] for i, t in enumerate(sentence) if (t in self.vocab_idx) and (i < self.max_sentence_len)]
                 for j, sentence in enumerate(tokenized) if j < self.max_sentences]

    def transform(self, texts):
        n = len(texts)
        tokenized = [self.tokenizer(text) for text in texts]
        
        result = np.zeros(shape=(n, self.max_sentences, self.max_sentence_len), dtype=np.int32)
        for i, sentences in enumerate(tokenized):
            for j, sent in enumerate(sentences):
                if j< self.max_sentences:
                    for k, word in enumerate(sent):
                        if (k < self.max_sentence_len) and (word in self.vocab_idx):
                            result[i,j,k] = self.vocab_idx[word]

        return result

    def vocabulary_size(self):
        return len(self.vocab) + 1