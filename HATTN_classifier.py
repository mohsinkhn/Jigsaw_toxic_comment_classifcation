#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 17:51:50 2018

GRU CLassifier for text classification
@author: mohsin
"""

import inspect
import numpy as np
#from keras.preprocessing.text import Tokenizer
from keras.optimizers import Adam, RMSprop
from keras.models import Model, load_model
from keras.layers import Input, Embedding, SpatialDropout1D, CuDNNGRU, GRU, Bidirectional, Dropout, Dense, PReLU, Flatten
from keras.layers import concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D, TimeDistributed
from keras.callbacks import ModelCheckpoint
from ZeroMaskedLayer import ZeroMaskedLayer
from AttentionLayer import AttentionLayer, AttentionWithContext

from sklearn.base import BaseEstimator, ClassifierMixin

#%%
class HATTNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 sent_max_len = 80,
                 doc_max_len = 20,
                 word_vocab_size= 200000,
                 word_embed_dim = 300,
                 word_spatial_dropout=0.2,
                 word_embed_trainable_flag=False,
                 mask_zero=False,
                 sent_rnn_dim=300,
                 sent_rnn_type = 'gru',
                 sent_bidirectional_flag=True,
                 sent_rnn_layers = 1,
                 sent_pooling = "max_attention",
                 doc_rnn_dim=100,
                 doc_rnn_type = 'gru',
                 doc_bidirectional_flag=False,
                 doc_rnn_layers = 1,
                 doc_pooling = "max_attention",
                 cudnn=True,
                 fc_dim = 300,
                 fc_dropout = 0.2,
                 fc_layers = 1,
                 fc_prelu = True,
                 optimizer = 'adam',
                 out_dim = 6,
                 batch_size=256,  
                 epochs=1,
                 verbose=1,
                 callbacks=None,
                 word_embed_kwargs = {},
                 sent_rnn_kwargs = {},
                 doc_rnn_kwargs = {},
                 opt_kwargs = {}
                ):
        
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)
            
    def _rnn_block(self, x, rnn_dim=64, rnn_type='gru', bidirectional_flag=False, cudnn=True, rnn_kwargs={}):
        #Learn document encoding using CuDNNGRU and return hidden sequences
        if cudnn:
            if rnn_type == "gru":
                rnn_layer =  CuDNNGRU(rnn_dim, return_sequences=True, **rnn_kwargs)
            elif rnn_type == "lstm":
                rnn_layer =  CuDNNLSTM(rnn_dim, return_sequences=True, **rnn_kwargs)
        else:
            if rnn_type == "gru":
                rnn_layer =  GRU(rnn_dim, return_sequences=True, **rnn_kwargs)
            elif rnn_type == "lstm":
                rnn_layer =  LSTM(rnn_dim, return_sequences=True, **rnn_kwargs)
        
        #Apply bidirectional wrapper if flag is True
        if bidirectional_flag:
            x = Bidirectional(rnn_layer)(x)
        else:
            x = rnn_layer(x)  
        return x
    
    def _pool_block(self, x, pooling='max_attention', seq_len=1):
        #Pool layers
        if pooling == 'mean':
            x = GlobalAveragePooling1D()(x)
            #x = concatenate([x, state])
            
        if pooling == 'max':
            x = GlobalMaxPooling1D()(x)
            #x = concatenate([x, state])
            
        if pooling == 'attention':
            x = AttentionLayer(seq_len)(x)
            #x = concatenate([x, state])
        
        if pooling == "mean_max":
            x1 = GlobalAveragePooling1D()(x)
            x2 = GlobalMaxPooling1D()(x)
            #x3 = AttentionLayer(self.max_seq_len)(x)
            x = concatenate([x1, x2])

        if pooling == 'max_attention':
            #x1 = GlobalAveragePooling1D()(emb)
            x2 = GlobalMaxPooling1D()(x)
            x3 = AttentionLayer(seq_len)(x)
            x = concatenate([x2, x3])
        return x
    
    
    def _fc_block(self, x, fc_dim=256, fc_dropout=0.2, fc_kwargs={}, prelu=True):
        #Fully connected layer
        x = Dropout(fc_dropout)(x)
        x = Dense(fc_dim, **fc_kwargs)(x)
        if prelu:
            x = PReLU()(x)
        return x

    def _build_model(self):
        #Set sentences and words inputs with max sequence lengths 
        sent_input = Input(shape=(self.sent_max_len,))
        doc_inp = Input(shape=(self.doc_max_len, self.sent_max_len))
        
        #Word embedding
        word_emb = Embedding(self.word_vocab_size, 
                        self.word_embed_dim,
                        trainable=self.word_embed_trainable_flag,
                        mask_zero = self.mask_zero,
                        **self.word_embed_kwargs)(sent_input)
        
        #Add zero mask optionally
        if self.mask_zero:
            word_emb = ZeroMaskedLayer()(word_emb)
            
        #Spatial Dropout (randomnly drop words from sequence)
        x = SpatialDropout1D(self.word_spatial_dropout)(word_emb)
        print(x.shape)
        #Apply rnn on sentences
        for _ in range(self.sent_rnn_layers):
            x = self._rnn_block(x, self.sent_rnn_dim, self.sent_rnn_type,
                                   self.sent_bidirectional_flag, cudnn=True, rnn_kwargs=self.sent_rnn_kwargs)
        print(x.shape)
        #Apply Pooling on rnn sequence output
        sent_enc = self._pool_block(x, self.sent_pooling, self.sent_max_len)
        
        #Wrap sentence encoder in a model
        sentEncoder = Model(sent_input, sent_enc)
            
        #Apply sentence encoder on all sentences
        x = TimeDistributed(sentEncoder)(doc_inp)
        
        #Apply rnn on sentence outputs
        for _ in range(self.doc_rnn_layers):
            x = self._rnn_block(x, self.doc_rnn_dim, self.doc_rnn_type,
                                   self.doc_bidirectional_flag, cudnn=self.cudnn, rnn_kwargs=self.doc_rnn_kwargs)
        
        #Apply pooling on rnn output on sentences
        x = self._pool_block(x, self.doc_pooling, self.doc_max_len)
        
        #Apply fully connected layer
        for _ in range(self.fc_layers):
            x = self._fc_block(x, self.fc_dim, self.fc_dropout, self.fc_kwargs, self.fc_prelu)
        
        out = Dense(self.out_dim, activation="sigmoid")(x)
        
        #Select Optimizer #TODO clean this up
        if self.optimizer == 'adam':
            opt = Adam(**self.opt_kwargs)
        elif self.optimizer == 'rmsprop':
            opt = RMSprop(**self.opt_kwargs)
        elif self.optimizer == 'nadam':
            opt = RMSprop(**self.opt_kwargs)
            
        #Set Model and compile
        model = Model(inputs=doc_inp, outputs=out)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model
    
    def fit(self, X, y):
        self.model = self._build_model()
        
        if self.callbacks:
            self.model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs,
                       verbose=self.verbose,
                       callbacks=self.callbacks,
                       shuffle=True)
        else:
            self.model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs,
                       verbose=self.verbose,
                       shuffle=True)
        return self
    
    def predict(self, X, y=None):
        if self.model:
            y_hat = self.model.predict(X, batch_size=1024)
        else:
            raise ValueError("Model not fit yet")
        return y_hat
