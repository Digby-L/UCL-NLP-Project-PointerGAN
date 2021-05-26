# pointer generator
# reference: https://arxiv.org/pdf/1704.04368.pdf
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import public as pb


# auxiliary function for parameter initialisation
# reference: https://github.com/atulkum/pointer_summarizer
def init_lstm_wt(lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-pb.unif_init_bound, pb.unif_init_bound)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)


def init_linear_wt(linear):
    linear.weight.data.normal_(std=pb.norm_init_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=pb.norm_init_std)


def init_wt_normal(wt):
    wt.data.normal_(std=pb.norm_init_std)


def init_wt_unif(wt):
    wt.data.uniform_(-pb.unif_init_bound, pb.unif_init_bound)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(pb.vocab_size, pb.gen_emb_size)
        init_wt_normal(self.embedding.weight)
        self.lstm = nn.LSTM(pb.gen_emb_size, pb.gen_hid_size, n_layers=1,  batch_first=True, bidirectional = True)
        init_lstm_wt(self.lstm)

        # reduce the bidirectional
        self.reduce_h = nn.Linear(pb.gen_hid_size * 2, pb.gen_hid_size)
        init_linear_wt(self.reduce_h)
        self.reduce_c = nn.Linear(pb.gen_hid_size * 2, pb.gen_hid_size)
        init_linear_wt(self.reduce_c)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, state = self.lstm(embedded)
        hidden, cell = state

        # Concatenate bidirectional lstm states
        hidden = torch.cat((hidden[0],hidden[1]),dim=-1)
        cell = torch.cat((cell[0],cell[1]),dim=-1)
        reduced_hidden = F.relu(self.reduce_h(hidden))
        reduced_cell = F.relu(self.reduce_c(cell))

        return outputs, (reduced_hidden, reduced_cell)  # dim=1*batch_size*hid_size


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

        self.v = nn.Linear(pb.gen_hid_size * 2, 1, bias=False)
        self.W_h = nn.Linear(pb.gen_hid_size * 2, pb.gen_hid_size * 2, bias=False)
        self.W_s = nn.Linear(pb.gen_hid_size, pb.gen_hid_size * 2, bias=True)
        self.w_c = nn.Linear(1, pb.gen_hid_size * 2, bias=False)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(pb.vocab_size, pb.gen_emb_size)
        init_wt_normal(self.embedding.weight)
        self.lstm = nn.LSTM(pb.gen_emb_size, pb.gen_hid_size, n_layers=1, batch_first=True)
        init_lstm_wt(self.lstm)
        self.out = nn.Linear(pb.gen_hid_size, pb.vocab_size, bias=True)

    def forward(self, output, hidden, cell):
        embedded = self.embedding(output.unsqueeze(0))

        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))

        prediction = self.out(outputs.squeeze(0))
        return prediction, hidden, cell


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    # initial hidden state of LSTM
    def init_hidden(self, batch_size=pb.batch_size, hid_size=pb.gen_hid_size):
        h = torch.zeros(1, batch_size, hid_size)
        c = torch.zeros(1, batch_size, hid_size)

        if pb.gpu:
            return h.cuda(), c.cuda()
        else:
            return h, c

    def forward(self,x, hidden_state):
        y, h = nn.LSTM(x, hidden_state)
        return y,h
    # y dim=(batch_size * seq_len) * vocab_size

    def policy_gradient_loss(self, x, labels, rewards):  # rewards via mc-search, dim=batch_size
        one_hot = F.one_hot(labels, pb.vocab_size).float()
        y, _ = self.forward(x, self.init_hidden())
        policy = torch.sum(one_hot * y.view(pb.batch_size, pb.max_seq_len, pb.vocab_size), dim=-1)  # batch_size*seq_len

        return -torch.sum(policy * rewards)    # policy  loss
