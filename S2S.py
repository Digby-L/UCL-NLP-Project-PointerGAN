import os
import sys
import time
import numpy as np 
import pandas as pd
import gc
import matplotlib.pyplot as plt
import re
import unicodedata
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import json
import pickle
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from nltk import sent_tokenize
import torch.nn.functional as F
import matplotlib.pyplot as plt
from rouge import Rouge

import math
import os
import random
import string

# Pytorch library for training
import torch
from torch import optim

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder1(nn.Module):
    def __init__(self, input_size, emb_size, hid_size, n_layers, dropout, embeddings):
        super().__init__()
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.input_size = input_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = embeddings
        self.lstm = nn.LSTM(emb_size, hid_size, n_layers, dropout=dropout)
        self.linear = nn.Linear
        
#         self.lstm = nn.LSTM(emb_size, hid_size, n_layers, dropout=dropout, bidirectional = True)
        

    def forward(self, x, x_length):
        embedded = torch.tensor([[self.embedding[i] for i in x[:, seq]]for seq in range(x.shape[1])]).permute(1, 0, 2)
        embedded = nn.utils.rnn.pack_padded_sequence(embedded, x_length.numpy(),batch_first=False)
        outputs, (hidden, cell) = self.lstm(embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        return hidden, cell

class Decoder1(nn.Module):
    def __init__(self, output_size, emb_size, hid_size, n_layers, dropout, embeddings):
        super().__init__()
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = embeddings
        self.lstm = nn.LSTM(emb_size, hid_size, n_layers, dropout=dropout)
        self.out = nn.Linear(hid_size, output_size, bias = True)

    def forward(self, output, hidden, cell):
        embedded = torch.tensor([self.embedding[x] for x in output]).float().unsqueeze(0)

        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        
        prediction = self.out(outputs.squeeze(0))
        return prediction, hidden, cell

class Seq2Seq1(nn.Module):
    def __init__(self, encoder: Encoder1, decoder: Decoder1, device: torch.device, embeddings):
        super().__init__()
        self.embedding = embeddings
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, text_batch, text_batch_len, headline_batch, teacher_forcing_ratio: float=0.5):
        max_len, batch_size = headline_batch.shape
        headline_vocab_size = self.decoder.output_size

        # tensor to store decoder's output
        outputs = torch.zeros(max_len, batch_size, headline_vocab_size).to(self.device)

        # last hidden & cell state of the encoder is used as the decoder's initial hidden state
        hidden, cell = self.encoder(text_batch, text_batch_len)
        
        hl_batch_i = headline_batch[0]
        
        for i in range(1, max_len):
            prediction, hidden, cell = self.decoder(hl_batch_i, hidden, cell)
            outputs[i] = prediction

            if random.random() < teacher_forcing_ratio:
                hl_batch_i = headline_batch[i]
            else:
                hl_batch_i = prediction.argmax(1)

        return outputs

