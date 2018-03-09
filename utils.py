#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 19:16:41 2018

@author: mohsin
"""

import numpy as np
import keras.backend as K
import gc
from RocAucEvaluation import RocAucEvaluation
from sklearn.metrics import roc_auc_score

def shuffle_crossvalidator(model, X, y, cvlist, callbacks, X_test=None, predict_test=False, 
                           scorer = roc_auc_score):
    y_trues = []
    y_preds = []
    scores = []
    y_test_preds = []
    for tr_index, val_index in cvlist:
        X_tr, y_tr = X[tr_index, :], y[tr_index, :]
        X_val, y_val = X[val_index, :], y[val_index, :]

        RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
        callbacks.append(RocAuc) 
        
        model.set_params(**{'callbacks':callbacks})
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_val)
        
        if predict_test:
            y_test_preds = model.predict(X_test)
        score = scorer(y_val, y_pred)
        scores.append(score)
        print("Score for this fold is ", score)
        y_trues.append(y_val)
        y_preds.append(y_pred)
        K.clear_session()
        gc.collect()
        #break
    y_trues = np.concatenate(y_trues)
    y_preds = np.concatenate(y_preds)
    if predict_test:
        y_test_preds = np.mean(y_test_preds, axis=0)
    score = scorer(y_trues, y_preds)
    print("Overall score on 10 fold CV is {}".format(score))
    
    return y_preds, y_trues, y_test_preds


def outoffold_crossvalidator(model, X, y, cvlist, callbacks, X_test=None, predict_test=False, 
                           scorer = roc_auc_score):
    y_preds = np.zeros(y.shape)

    for tr_index, val_index in cvlist:
        X_tr, y_tr = X[tr_index, :], y[tr_index, :]
        X_val, y_val = X[val_index, :], y[val_index, :]
        
        RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
        callbacks.append(RocAuc)        
        model.set_params(**{'callbacks':callbacks})
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_val)
        if predict_test:
            y_test_preds = model.predict(X_test)
        print("Score for this fold is ", scorer(y_val, y_pred))
        y_preds[val_index] = y_pred
        K.clear_session()
        gc.collect()
        
    if predict_test:
        y_test_preds = np.mean(y_test_preds, axis=0)
    score = scorer(y, y_preds)
    print("Overall score on 10 fold CV is {}".format(score))
    
    return y_preds, y, y_test_preds


def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')

def initialize_embeddings(filename, tokenizer):
    oov_list = []
    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(filename))
    
    word_index = tokenizer.vocab_idx
    nb_words = len(word_index) + 1
    embed_size = len(embeddings_index.get("the"))
    embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            oov_list.append(word)
    return  embedding_matrix, oov_list
