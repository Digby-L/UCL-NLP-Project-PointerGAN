# pointer generator
from numpy import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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
    def __init__(self, embeddings):
        super().__init__()
        self.hid_size = pb.gen_hid_size
        self.input_size = pb.vocab_size
        self.emb_size = pb.gen_emb_size
        self.dropout = pb.dropout

        self.embedding = embeddings
        self.lstm = nn.LSTM(self.emb_size, self.hid_size, num_layers=2, dropout=self.dropout)

    def forward(self, x, x_length):
        embedded = torch.tensor([[self.embedding[i] for i in x[:, seq]] for seq in range(x.shape[1])]).permute(1, 0, 2)
        embedded = pack_padded_sequence(embedded, list(x_length), batch_first=False, enforce_sorted=False)
        outputs, (hidden, cell) = self.lstm(embedded)
        outputs, _ = pad_packed_sequence(outputs)

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        self.hid_size = pb.gen_hid_size
        self.output_size = pb.output_vocab_size
        self.emb_size= pb.gen_emb_size
        self.dropout = pb.dropout

        self.embedding = embeddings
        self.lstm = nn.LSTM(self.emb_size, self.hid_size, num_layers=2, dropout=self.dropout)
        self.out = nn.Linear(self.hid_size, self.output_size, bias=True)

    def forward(self, output, hidden, cell):
        embedded = torch.tensor([self.embedding[x] for x in output]).float().unsqueeze(0)

        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))

        prediction = self.out(outputs.squeeze(0))

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, text_embeddings, headline_embeddings):
        super().__init__()
        self.text_embedding = text_embeddings
        self.headline_embedding = headline_embeddings
        self.encoder = Encoder(self.text_embedding)
        self.decoder = Decoder(self.headline_embedding)

    def forward(self, text_batch, text_batch_len, headline_batch,
                teacher_forcing_ratio: float = 0.5, need_hidden=False):
        max_len, batch_size = headline_batch.shape

        # tensor to store decoder's output
        outputs = torch.zeros(max_len, batch_size, pb.output_vocab_size)

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

        if not need_hidden:
            return outputs
        else:
            return outputs, (hidden, cell)

    def predict(self, text_batch, text_batch_len, headline_batch):
        outputs = self.forward(text_batch, text_batch_len, headline_batch, teacher_forcing_ratio=0)

        return torch.argmax(torch.tensor(outputs), dim=2)

    # debugging...
    # policy gradient loss for GAN use
    def policy_gradient_loss(self, text_batch, text_batch_len, headline_batch, rewards):  # rewards via mc-search, dim=batch_size
        one_hot = F.one_hot(headline_batch, pb.output_vocab_size).float()
        y, _ = self.forward(text_batch, text_batch_len, headline_batch)
        policy = torch.sum(one_hot * y.view(pb.batch_size, pb.output_seq_len, pb.output_vocab_size), dim=-1)  # batch_size*seq_len

        return -torch.sum(policy * rewards)    # policy  loss

    # initial hidden state of LSTM
    def init_hidden(self, batch_size=pb.batch_size, hid_size=pb.gen_hid_size):
        h = torch.zeros(1, batch_size, hid_size)
        c = torch.zeros(1, batch_size, hid_size)

        if pb.gpu:
            return h.cuda(), c.cuda()
        else:
            return h, c
