# pointer generator
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
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(pb.input_seq_len, pb.gen_emb_size)
        init_wt_normal(self.embedding.weight)
        self.lstm = nn.LSTM(pb.gen_emb_size, pb.gen_hid_size, batch_first=True)
        init_lstm_wt(self.lstm)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, state = self.lstm(embedded)

        return state


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(pb.output_seq_len, pb.gen_emb_size)
        init_wt_normal(self.embedding.weight)
        self.lstm = nn.LSTM(pb.gen_emb_size, pb.gen_hid_size, batch_first=True)
        init_lstm_wt(self.lstm)
        self.out = nn.Linear(pb.gen_hid_size, pb.output_seq_len, bias=True)

    def forward(self, output, state):
        embedded = self.embedding(output.unsqueeze(0))

        outputs, state = self.lstm(embedded, state)

        prediction = self.out(outputs.squeeze(0))
        return prediction, state


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.encoder.embedding.weight = self.decoder.embedding.weight

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


class LSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(pb.vocab_size, pb.gen_emb_size, padding_idx=0)
        init_wt_normal(self.embedding.weight)
        self.lstm = nn.LSTM(pb.gen_emb_size, pb.gen_hid_size, batch_first=True)
        init_lstm_wt(self.lstm)

        # reduce the bidirectional
        # self.reduce_h = nn.Linear(pb.gen_hid_size * 2, pb.gen_hid_size)
        # init_linear_wt(self.reduce_h)
        # self.reduce_c = nn.Linear(pb.gen_hid_size * 2, pb.gen_hid_size)
        # init_linear_wt(self.reduce_c)

        self.out1 = nn.Linear(pb.gen_hid_size, pb.max_seq_len)
        init_linear_wt(self.out1)
        self.out2 = nn.LogSoftmax(dim=-1)

    def forward(self, x, h):
        emb = self.embedding(x)
        if len(x.size()) == 1:
            emb = emb.unsqueeze(1)
        pred, h = self.lstm(emb, h)

        # hidden, cell = h
        # hidden = torch.cat((hidden[0],hidden[1]),dim=-1)
        # cell = torch.cat((cell[0],cell[1]),dim=-1)
        # reduced_hidden = F.relu(self.reduce_h(hidden))
        # reduced_cell = F.relu(self.reduce_c(cell))
        # pred = pred.contiguous().view(-1, pb.gen_hid_size)
        y = self.out2(self.out1(pred))
        return y, h

    def sample(self, num_samples):
        num_batch = num_samples // pb.batch_size+1 if num_samples!=pb.batch_size else 1
        samples = torch.zeros(num_batch*pb.batch_size, pb.max_seq_len).long()

        for b in range(num_batch):
            h = self.init_hidden(pb.batch_size)
            x = torch.LongTensor([pb.start_letter_idx] * pb.batch_size)
            if self.gpu:
                x = x.cuda()

            for i in range(pb.max_seq_len):
                y, h = self.forward(x, h)
                next_token = torch.multinomial(torch.exp(y), 1)
                samples[b*pb.batch_size:(b+1)*pb.batch_size, i] = next_token.view(-1)
                x = next_token.view(-1)
        samples = samples[:num_samples]

        return samples

    # initial hidden state of LSTM
    def init_hidden(self, batch_size=pb.batch_size, hid_size=pb.gen_hid_size, num_direction=1):
        h = torch.zeros(num_direction, batch_size, hid_size)
        c = torch.zeros(num_direction, batch_size, hid_size)

        if pb.gpu:
            return h.cuda(), c.cuda()
        else:
            return h, c

    def policy_gradient_loss(self, x, labels, rewards):  # rewards via mc-search, dim=batch_size
        one_hot = F.one_hot(labels, pb.vocab_size).float()
        y, _ = self.forward(x, self.init_hidden())
        policy = torch.sum(one_hot * y.view(pb.batch_size, pb.max_seq_len, pb.vocab_size), dim=-1)  # batch_size*seq_len

        return -torch.sum(policy * rewards)    # policy  loss
