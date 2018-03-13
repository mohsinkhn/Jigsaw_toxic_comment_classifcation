#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 20:10:59 2018

@author: mohsin
"""
import pandas as pd
from sklearn.base  import BaseEstimator, TransformerMixin

class ExtraFeats(BaseEstimator, TransformerMixin):
    
    def  __init__(self, regex_feats=True):
        self.regex_feats = regex_feats

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        if isinstance(X, pd.Series):
            num_bullets = X.str.count("\n\s{0,2}\d{1}\.")
            num_lines = X.str.count("\n\n")
            num_sent = X.str.count("[a-z]\. ")
            indented = X.str.contains(":{2,}").astype(int)
            num_chars = X.str.count("[a-z]")
            num_nums = X.str.count("[0-9]")
            num_words = X.str.count("\s+")
            big_cap_words = X.str.count("\s[A-Z]{3,}\s")
            grammar_aware = X.str.count("[a-z]. [A-Z]")
            exclaims = X.str.count("!{2,}")
            other_syms = X.str.count("@*$")
            
            X = pd.concat([num_bullets, num_lines, num_sent, indented, num_chars, num_nums, num_words, big_cap_words,
                      grammar_aware, exclaims, other_syms], axis=1)
            return X
        else:
            raise ValueError("Need pandas series as input")