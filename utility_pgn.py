# utility for pgn only
import numpy as np

# Pytorch library for training
import torch
from torch import optim
import torch.nn as nn
from torch.optim import Adagrad
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from numpy import random

import public as pb
from discriminator import Discriminator
from data_loader import DisDataLoader


class Encoder(nn.Module):
    def __init__(self, input_size, hid_size, emb_size, embedding):
        super(Encoder, self).__init__()
        self.hid_size = hid_size
        self.embedding = embedding

        self.lstm = nn.LSTM(emb_size, hid_size, num_layers=1, batch_first=True, bidirectional=True)
        self.W_h = nn.Linear(hid_size * 2, hid_size * 2, bias=False)

    # seq_lens should be in descending order
    def forward(self, enc_input, enc_input_len, hidden_state):
        embedded = torch.tensor([[self.embedding[i] for i in enc_input[:, seq]]
                                 for seq in range(enc_input.shape[1])]).permute(1, 0, 2)

        packed = pack_padded_sequence(embedded, enc_input_len, batch_first=True)

        output, hidden = self.lstm(packed, hidden_state)

        enc_outputs, _ = pad_packed_sequence(output, batch_first=True)
        enc_outputs = enc_outputs.contiguous()

        enc_feature = enc_outputs.view(-1, 2 * self.hid_size)  # B * L x 2*hidden_dim
        enc_feature = self.W_h(enc_feature)

        return enc_outputs, enc_feature, hidden


class ReduceState(nn.Module):
    def __init__(self, hid_size):
        super().__init__()
        self.hid_size = hid_size

        self.reduce_h = nn.Linear(self.hid_size * 2, self.hid_size)
        self.reduce_c = nn.Linear(self.hid_size * 2, self.hid_size)

    def forward(self, hidden):
        h, c = hidden

        h_in = h.transpose(0, 1).contiguous().view(-1, self.hid_size * 2)
        hid_reduced_h = F.relu(self.reduce_h(h_in))

        c_in = c.transpose(0, 1).contiguous().view(-1, self.hid_size * 2)
        hid_reduced_c = F.relu(self.reduce_c(c_in))

        return hid_reduced_h.unsqueeze(0), hid_reduced_c.unsqueeze(0)


class Attention(nn.Module):
    def __init__(self, hid_size, use_coverage):
        super().__init__()
        self.hid_size = hid_size
        self.use_coverage = use_coverage

        # Coverage layer
        if use_coverage:
            self.w_c = nn.Linear(1, hid_size * 2, bias=False)

        self.decode_proj = nn.Linear(hid_size * 2, hid_size * 2)
        self.v = nn.Linear(hid_size * 2, 1, bias=False)

    def forward(self, h_c_hat, enc_outputs, enc_feature, enc_padding_mask, coverage):
        """""
        h_c_hat: hidden, cell from decoder
        enc_outputs: first output of encoder
        enc_feature: second output of encoder
        enc_padding_mask: text_padmask
        coverage: initialize: Variable(torch.zeros((batch_size, 2 * hid_size)))

        Return:
        context_vec: sum(attention weights)*encoder hidden states
        attn_dist: attention distribution
        coverage: updated coverage
        """""
        b, m, n = list(enc_outputs.size())

        dec_feature = self.decode_proj(h_c_hat)  # B x 2*hid_size

        dec_feature_expanded = dec_feature.unsqueeze(1).expand(b, m, n).contiguous()  # B x m x 2*hid_size

        dec_feature_expanded = dec_feature_expanded.view(-1, n)  # (B * m )x 2*hid_size

        attn_feature = enc_feature + dec_feature_expanded  # (B * m) x 2*hid_size

        if self.use_coverage:
            coverage_input = coverage.view(-1, 1)  # (B * m) x 1
            coverage_feature = self.w_c(coverage_input)  # (B * m) x 2*hid_size
            att_feature = attn_feature + coverage_feature

        scores = torch.tanh(att_feature)  # (B * m) x 2*hidden_dim
        scores = self.v(scores)  # (B * m) x 1
        scores = scores.view(-1, m)  # B x m

        attn_dist = F.softmax(scores, dim=1) * (1 - enc_padding_mask)  # B x m
        normalization_factor = attn_dist.sum(1, keepdim=True)

        attn_dist = attn_dist / normalization_factor

        attn_dist = attn_dist.unsqueeze(1)  # B x 1 x m
        context_vec = torch.bmm(attn_dist, enc_outputs)  # B x 1 x n
        context_vec = context_vec.view(-1, self.hid_size * 2)  # B x 2*hidden_dim

        attn_dist = attn_dist.view(-1, m)  # B x m

        if self.use_coverage:
            coverage = coverage.view(-1, m)
            coverage = coverage + attn_dist

        return context_vec, attn_dist, coverage


class Decoder(nn.Module):
    def __init__(self, input_size, hid_size, vocab_size, emb_size, embedding_headline, use_coverage, use_p_gen):
        super().__init__()
        self.hid_size = hid_size

        self.attn_net = Attention(hid_size, use_coverage)

        self.use_coverage = use_coverage  # True/False

        self.use_p_gen = use_p_gen  # True/False

        self.embedding = embedding_headline

        self.x_context = nn.Linear(hid_size * 2 + emb_size, emb_size)

        self.lstm = nn.LSTM(emb_size, hid_size, num_layers=1, batch_first=True, bidirectional=False)

        if use_p_gen:
            self.p_gen_linear = nn.Linear(hid_size * 4 + emb_size, 1)

        # p_vocab
        self.out1 = nn.Linear(hid_size * 3, hid_size)
        self.out2 = nn.Linear(hid_size, vocab_size)

    def forward(self, target, h_c_1, enc_outputs, enc_feature, enc_padding_mask,
                cont_v, enc_oov_len, enc_batch, coverage, step):
        """
        target: headline batch
        h_c_1: reduced_state(enc_hidden)
        h_c_hat: updated hidden for attn_net
        enc_outputs: first output of encoder
        enc_feature: second output of encoder
        enc_padding_mask:
        cont_v: context vector input to decoder (initialization: Variable(torch.zeros((batch_size, 2 * hid_size))))
        enc_oov_len: text_batch_oov
        enc_batch: text_train_pad (OOV has index)
        coverage: initialization: Variable(torch.zeros(text_batch.size()))

        extro_zeros: initialization: Variable(torch.zeros((batch_size, max_oov_len)))

        """
        if not self.training and step == 0:
            h_decoder, c_decoder = h_c_1
            h_c_hat = torch.cat((h_decoder.view(-1, self.hid_size),
                                 c_decoder.view(-1, self.hid_size)), 1)  # B x 2*hid_size
            context_vec, _, coverage_new = self.attn_net(h_c_hat, enc_outputs, enc_feature,
                                                         enc_padding_mask, coverage)
            coverage = coverage_new

        max_oov_len = torch.max(enc_oov_len)

        target_embbed = torch.tensor([self.embedding[i] for i in target]).float()

        x = self.x_context(torch.cat((cont_v, target_embbed), 1))
        lstm_out, h_c = self.lstm(x.unsqueeze(1), h_c_1)

        h_decoder, c_decoder = h_c
        h_c_hat = torch.cat((h_decoder.view(-1, self.hid_size),
                             c_decoder.view(-1, self.hid_size)), 1)  # B x 2*hid_size
        context_vec, attn_dist, coverage_new = self.attn_net(h_c_hat, enc_outputs, enc_feature,
                                                             enc_padding_mask, coverage)

        if self.training or step > 0:
            coverage = coverage_new

        p_gen = None

        if self.use_p_gen:
            p_gen_input = torch.cat((cont_v, h_c_hat, x), 1)  # B x (2*2*hid_size + emb_size)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = torch.sigmoid(p_gen)

        output = torch.cat((lstm_out.view(-1, self.hid_size), cont_v), 1)  # B x hid_size * 3
        output = self.out1(output)  # B x hid_size

        # output = F.relu(output)

        output = self.out2(output)  # B x vocab_size
        vocab_dist = F.softmax(output, dim=1)

        dist_size = vocab_dist.size()[0]
        extra_zeros = torch.zeros([dist_size, max_oov_len])

        if self.use_p_gen:
            vocab_dist_p = p_gen * vocab_dist
            attn_dist_p = (1 - p_gen) * attn_dist

            if extra_zeros is not None:
                vocab_dist_p = torch.cat([vocab_dist_p, extra_zeros], 1)

            final_dist = vocab_dist_p.scatter_add(1, enc_batch, attn_dist_p)
        else:
            final_dist = vocab_dist

        return final_dist, h_c, cont_v, attn_dist, p_gen, coverage


class Parameters():
    ## Set parameter
    input_size = pb.vocab_size
    output_size = pb.output_vocab_size

    vocab_size = input_size - 1
    unk_idx = 3
    use_coverage = True
    use_p_gen = True

    # input_size = int(7778 + 1)
    # output_size = int(1410 + 1)

    enc_emb_size = 200
    dec_emb_size = 200
    hid_size = 128

    n_layers = 1
    enc_dropout = 0.5
    dec_dropout = 0.5

    beam_size = 50
    max_dec_steps = 10
    min_dec_steps = 1

    cov_weight = 1.0

    # training and optimizer
    lr = 0.1
    opt_acc = 0.1


class Pointer_Generator(nn.Module):
    def __init__(self, para, encoder: Encoder, reduced_net: ReduceState, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.reduced_net = reduced_net
        self.decoder = decoder
        self.para = para

    def forward(self, text_batch, text_batch_len, text_batch_padmask, text_batch_oov,
                headline_batch, headline_batch_len, headline_batch_no, headline_batch_padmask, hidden_state=None):
        """""
        text_batch: text batch with oov index (eg. text_train_pad)
        text_batch_len: len of sentence in the batch before padding (eg. text_train_len)

        text_batch_padmask: padding mask of each sentence in text batch (padded:1, no pad: 0), (eg. text_train_padmask)
        text_batch_oov: number of oov in each sentence in text batch (eg. text_train_oov)

        headline_batch: headline batch with oov index (eg. headline_train_pad)
        headline_batch_len: len of sentence in the batch before padding (eg. headline_train_len)

        headline_batch_no: headline batch with oov index == unk_idx == 3 (eg. headline_train_no)
        headline_batch_padmask: padding mask of each sentence in hl batch (padded:1, no pad: 0), (eg. headline_train_padmask)

        hidden_state: hidden state for GAN (hidden_state = None, if not specified)
        """""

        batch_size, max_len = headline_batch.shape
        headline_vocab_size = self.para.output_size - 1

        # tensor to store decoder's output
        outputs = torch.zeros(max_len, batch_size, headline_vocab_size)

        # last hidden & cell state of the encoder is used as the decoder's initial hidden state
        enc_outputs, enc_feature, enc_hidden = self.encoder(text_batch, text_batch_len, hidden_state)

        h_c_1 = self.reduced_net(enc_hidden)

        c_t_1 = Variable(torch.zeros((100, 2 * self.para.hid_size)))
        coverage = Variable(torch.zeros(text_batch.size()))

        dist = torch.zeros((batch_size, max_len, self.para.vocab_size))

        step_loss_rec = []

        for i in range(max_len):
            final_dist, h_c, c_t, attn_dist, p_gen, coverage_new = self.decoder(headline_batch_no[:, i], h_c_1,
                                                                                enc_outputs, enc_feature,
                                                                                text_batch_padmask, c_t_1,
                                                                                text_batch_oov,
                                                                                text_batch, coverage, i)
            # Record distribution for GAN
            dist[:, i, :] = final_dist

            # Calculate loss for this batch
            headline = headline_batch[:, i]
            total_dist = torch.gather(final_dist, 1, headline.unsqueeze(1)).squeeze()
            step_loss = - torch.log(total_dist + 1e-10)

            if self.para.use_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)

                step_loss = step_loss + step_coverage_loss
                coverage = coverage_new

            step_mask = headline_batch_padmask[:, i]
            step_loss = step_loss + 1.0 * step_mask
            step_loss_rec.append(step_loss)

        total_loss = torch.sum(torch.stack(step_loss_rec, 1), 1)

        batch_loss_average = total_loss / headline_batch_len

        # Final loss is the average of batch_loss_average
        loss = torch.mean(batch_loss_average)

        return dist, loss


class GAN:
    def __init__(self, PGN):
        self.generator = PGN
        self.discriminator = Discriminator()
        if pb.gpu:
            self.generator.cuda(), self.discriminator.cuda()

        # Criterion
        # self.gen_criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.dis_criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        self.parameter = list(self.generator.encoder.parameters()) + list(self.generator.decoder.parameters())\
                         + list(self.generator.reduced_net.parameters())
        self.gen_optimizer = optim.Adam(self.parameter, lr=pb.gen_lr)
        self.dis_optimizer = optim.Adam(self.discriminator.parameters(), lr=pb.dis_lr)

    def train_gen(self, GenDataIter):
        self.generator.train()
        loss = 0
        for text_batch, hl_batch, text_batch_padmask, headline_batch_padmask, text_batch_len, \
            headline_batch_len, text_batch_oov, headline_batch_oov, text_batch_no, headline_batch_no in GenDataIter:

            # h = self.generator.init_hidden()
            _, l = self.generator(text_batch, text_batch_len, text_batch_padmask, text_batch_oov,
                hl_batch, headline_batch_len, headline_batch_no, headline_batch_padmask)

            self.gen_optimizer.zero_grad()
            l.backward()
            clip_grad_norm_(self.generator.parameters(), max_norm=pb.max_grad_norm)
            self.gen_optimizer.step()
            loss += l.item()

        return loss

    def train_dis(self, DisDataIter):
        self.discriminator.train()
        accuracy = 0
        count = 0
        for i, data in enumerate(DisDataIter.data_loader):
            x, y = data['input'], data['label']
            if pb.gpu:
                x, y = x.cuda(), y.cuda()

            pred_y = self.discriminator.forward(x).view(-1)
            loss = self.dis_criterion(pred_y, y.float())
            self.dis_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            clip_grad_norm_(self.discriminator.parameters(), max_norm=pb.max_grad_norm)
            self.dis_optimizer.step()

            accuracy += torch.sum(torch.tensor(pred_y.argmax(dim=-1) == y)).item()
            count += x.size(0)

        return accuracy/count

    # compromised version of adversarial training w/o policy gradient
    def train_gan(self, data_loader):
        ##### pre-train #####
        print('generator pre-training via maximum likelihood begins')
        for i in range(pb.gen_pretrain_epochs):
            _ = self.train_gen(data_loader)

        print('discriminator pre-training begins')
        for step in range(pb.d_steps):
            # random sample true headline batch_size
            idx = torch.tensor(random.randint(0, pb.num_samples, size=pb.batch_size)).long()

            positive = headline_pad[idx, :]
            samples = text_pad[idx, :]
            samples = torch.transpose(samples, 0, 1)   # due to lstm dimension reverse issue
            samples_len = text_len[idx]
            negative = self.generator.predict(samples, samples_len, torch.transpose(positive, 0, 1))
            negative = torch.transpose(negative, 0, 1)   # due to lstm dimension reverse issue
            mixed = DisDataLoader(positive, negative)

            for epoch in range(pb.k):
                _ = self.train_dis(mixed)

        print('==============================================')
        print('adversarial training begins')
        for j in range(pb.gan_epochs):
            for text_train_pad, headline_train_pad, text_train_lengths, headline_train_lengths in data_loader:
                text_train_pad = torch.transpose(text_train_pad, 0, 1)
                headline_train_pad = torch.transpose(headline_train_pad, 0, 1)
                text_train_pad = text_train_pad[:text_train_lengths.max()]
                headline_train_pad = headline_train_pad[:headline_train_lengths.max()]
                if pb.gpu:
                    text_train_pad.cuda(), headline_train_pad.cuda(), text_train_lengths.cuda()

                pred_y = self.generator.forward(text_train_pad, text_train_lengths, headline_train_pad)
                outputs_flatten = pred_y[1:].view(-1, pred_y.shape[-1])
                hl_flatten = headline_train_pad[1:].reshape(-1)

                ce_loss = self.gen_criterion(outputs_flatten, hl_flatten)

                positive = headline_train_pad
                negative = self.generator.predict(text_train_pad, text_train_lengths, positive)
                negative = torch.transpose(negative, 0, 1)  # due to lstm dimension reverse issue
                dis_pred_y = self.discriminator.forward(negative).view(-1)

                dis_loss = self.dis_criterion(dis_pred_y, torch.ones_like(dis_pred_y).view(-1))

                # combine two losses
                adv_l = dis_loss * ce_loss

                self.gen_optimizer.zero_grad()
                adv_l.backward(retain_graph=True)
                self.gen_optimizer.step()

                print('generator loss: {}'.format(adv_l))

                for step in range(pb.d_steps):
                    # random sample true headline batch_size
                    idx = torch.tensor(random.randint(0, pb.num_samples, size=pb.batch_size*5)).long()

                    positive = headline_pad[idx, :]
                    samples = text_pad[idx, :]
                    samples = torch.transpose(samples, 0, 1)  # due to lstm dimension reverse issue
                    samples_len = text_len[idx]
                    negative = self.generator.predict(samples, samples_len, torch.transpose(positive, 0, 1))
                    negative = torch.transpose(negative, 0, 1)  # due to lstm dimension reverse issue
                    mixed = DisDataLoader(positive, negative)

                    for epoch in range(pb.k):
                        accuracy = self.train_dis(mixed)
        torch.save(self.generator.state_dict(), 'model.pt')
