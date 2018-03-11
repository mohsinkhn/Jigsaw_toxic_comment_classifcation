#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 07:56:09 2018

@author: mohsin
"""

import sys, os, re, csv, codecs, gc, numpy as np, pandas as pd
import tensorflow as tf
#from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Permute, GRU, Conv1D, LSTM, Embedding, Dropout, Activation, CuDNNLSTM, CuDNNGRU, concatenate, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalAveragePooling1D, BatchNormalization, SpatialDropout1D, Dot
from keras.optimizers import Adam, RMSprop
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras_tqdm import TQDMNotebookCallback
import keras.backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from functools import reduce
from keras.layers import Layer, PReLU, SpatialDropout1D, TimeDistributed, Subtract
from keras import initializers
from sklearn.model_selection import cross_val_predict

from nltk.tokenize import word_tokenize, wordpunct_tokenize, TweetTokenizer, MWETokenizer, ToktokTokenizer
from nltk.corpus import stopwords

import unicodedata
from collections import Counter
import itertools

np.random.seed(786)

from Tokenizer import Tokenizer
from GRU_classifier import GRUClassifier
from utils import shuffle_crossvalidator, outoffold_crossvalidator, initialize_embeddings, get_coefs

#%%

def unicodeToAscii(series):
    return series.apply(lambda s: unicodedata.normalize('NFKC', str(s)))


def multiple_replace(text, adict):
    rx = re.compile('|'.join(map(re.escape, adict)))

    def one_xlat(match):
        return adict[match.group(0)]

    return rx.sub(one_xlat, text)

STOP_WORDS = set(stopwords.words( 'english' ))
# Lowercase, trim, and remove non-letter characters
def normalizeString(series):
    series = unicodeToAscii(series)
    series = series.str.lower()
    series = series.str.replace(r"(\n){1,}", " ")
    series = series.str.replace(r"\'", "")
    series = series.str.replace(r"\-", "")
    series = series.str.replace(r"[^0-9a-z]+", " ")
    series = series.str.replace("([a-z0-9]{2,}\.){2,}[a-z]{2,}", " ") 
    series = series.str.replace(" \d ", "")
    return series


#%%
if __name__=="__main__":
    
    path = '../input/'
    utility_path = '../utility/'
    comp = 'jigsaw-toxic-comment-classification-challenge/'
    EMBEDDING_FILE=f'{utility_path}glove.42B.300d.txt'
    TRAIN_DATA_FILE=f'{path}train_preprocessed_v1.csv'
    TEST_DATA_FILE=f'{path}test_preprocessed_v1.csv'
    
    MAX_FEATURES= 150000
    MAX_LEN = 200
    MODEL_IDENTIFIER = "GLOVE_PREPROCSSED_2"
    
    train = pd.read_csv(TRAIN_DATA_FILE)
    test = pd.read_csv(TEST_DATA_FILE)

    print(train.shape, test.shape)

    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y = train[list_classes].values

    #Get validation folds
    train['target_str'] = reduce(lambda x,y: x+y, [train[col].astype(str) for col in list_classes])
    train['target_str'] = train['target_str'].replace('110101', '000000').replace('110110','000000')
    cvlist1 = list(StratifiedKFold(n_splits=10, random_state=786).split(train, train['target_str'].astype('category')))
    cvlist2 = list(StratifiedShuffleSplit(n_splits=5, test_size=0.05, random_state=786).split(train, train['target_str'].astype('category')))
    
    #NOrmalize text
    #for df in train, test:
    #    df["comment_text"] = normalizeString(df["comment_text"])
        
    #Tokenize comments    S
    tok = Tokenizer(max_features=MAX_FEATURES, max_len=MAX_LEN, tokenizer=wordpunct_tokenize)
    X = tok.fit_transform(pd.concat([train["comment_text"].astype(str), test["comment_text"].astype(str)]))
    X_train = X[:len(train), :]
    X_test = X[len(train):, :]
    
    print(X_train.shape, X_test.shape)
    print("<+++++++>")
    print("Total words found by tokenizer in train and test are {}".format(len(tok.doc_freq)))
    print("Top 10 words in vocab are {}".format(tok.doc_freq.most_common(10)))
    print("Last 10 words to be used vocab with their freq are {}".format(tok.doc_freq.most_common(MAX_FEATURES)[-10:]))
    
    #Initialize embeddings
    embedding_matrix, oov_list = initialize_embeddings(EMBEDDING_FILE, tok)
    print("<+++++++>")
    print("Size of initialized matrix is {}".format(embedding_matrix.shape))
    print("No. of words in that were not found in embedding are ".format(len(oov_list)))
    random_indices = np.random.randint(0, len(oov_list), 10)
    print("Some out of vocab words are".format(np.array(oov_list)[random_indices]))
    
    #checkPoint = ModelCheckpoint("Model_"+str(MODEL_IDENTIFIER)+".check", save_best_only=True, verbose=1)
    #callbacks = [checkPoint]
    #Model params to be used
    MODEL_PARAMS = {
            "max_seq_len": MAX_LEN,
            "embed_vocab_size":MAX_FEATURES+1,
            "embed_dim": 300,
            "spatial_dropout": 0.5,
            "gru_dim" : 150,
            "cudnn" : True,
            "bidirectional" : True,
            "gru_layers": 2,
            "pooling": 'max_attention',
            "fc_dim": 256,
            "fc_dropout": 0.2,
            "fc_layers": 0,
            "optimizer": 'adam',
            "out_dim": 6,
            "batch_size": 256,
            "epochs": 10,
            "callbacks": [],
            "model_id": MODEL_IDENTIFIER,  
            "embed_kwargs": {"weights": [embedding_matrix]}
            }
    
    #Initialize model
    model = GRUClassifier(**MODEL_PARAMS)
    check_filename="Model_"+str(MODEL_IDENTIFIER)+".check"
    y_preds, y_trues, y_test = outoffold_crossvalidator(model, X_train, y, cvlist1, check_filename=check_filename,
                                                      predict_test=True, X_test=X_test)
    
    #write out train oof and test
    oof_preds: pd.DataFrame = train[['id']]
    for i, col in enumerate(list_classes):
        oof_preds.loc[:, col] = y_preds[:, i]
    
    test_preds: pd.DataFrame = test[['id']]
    for i, col in enumerate(list_classes):
        test_preds.loc[:, col] = y_test[:, i]   
    
    oof_preds.to_csv("../utility/oof_{}.csv".format(MODEL_IDENTIFIER), index=False)
    test_preds.to_csv("../utility/test_{}.csv".format(MODEL_IDENTIFIER), index=False)
    