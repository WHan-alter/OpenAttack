#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 15:56:05 2021

@author: janie
"""
import torch
import torch.nn as nn
import sys
import os
import esm
from transformers.file_utils import ModelOutput
from typing import Optional

class output(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None

class ValuePredictionHead(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 out_dim: int,
                 droupout: float=0.1) -> None:
        super().__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(in_dim,hid_dim,bias=True),
            nn.ReLU(),
            nn.Dropout(droupout,inplace=True),
            nn.Linear(hid_dim,out_dim)
            )
   
    def forward(self,pooled_output):
        value_pred = self.fc_layer(pooled_output)
        outputs = value_pred
        return(outputs)


class ESM_value_prediction_model(nn.Module):
    def __init__(self,
                 freeze_bert: bool=False,
                 esm_path: str="./model",
                 embedding_size: int=1280,
                 dropout_prob: float=0.1,
                 pooling: str="first_token") -> None:
        super().__init__()
        self.esm_model = esm.pretrained.load_model_and_alphabet_local(esm_path)[0]
        self.model_name = os.path.basename(esm_path)
        self.model_layers = self.model_name.split("_")[1][1:]
        self.repr_layers = int(self.model_layers)
        self.predict = ValuePredictionHead(embedding_size,512,1,dropout_prob)
        self.pooling = pooling
        
        if freeze_bert:
            for p in self.esm_model.parameters():
                p.requires_gard = False
        
    def forward(self,input_ids,labels):
        outputs = self.esm_model(input_ids, repr_layers=[self.repr_layers])["representations"][self.repr_layers]
        if self.pooling == "first_token":
            outputs = outputs[:,0,:]
        elif self.pooling == "average" :
            outputs = outputs.mean(1)
        else:
            ValueError("Not implemented! Please choose the pooling methods from first_token/average")
        logits = self.predict(outputs)
        loss = None
        if labels is not None:  
            criterion = nn.MSELoss()
            loss = criterion(logits.view(-1), labels)
        return(output(
        loss = loss,
        logits = logits.view(-1)))
    
    
