# training class for discriminator
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from numpy import random

import public as pb
from discriminator import Discriminator
from generator import Seq2Seq
from data_loader import DisDataLoader


class GAN:
    def __init__(self, embedding):
        self.generator = Seq2Seq(embedding)
        self.discriminator = Discriminator()
        if pb.gpu:
            self.generator.cuda(), self.discriminator.cuda()

        # Criterion
        self.gen_criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.dis_criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=pb.gen_lr)
        self.reinforce_optimizer = optim.Adam(self.generator.parameters(), lr=pb.gen_lr)
        self.dis_optimizer = optim.Adam(self.discriminator.parameters(), lr=pb.dis_lr)

    # debugging...
    def mc_search(self, x):
        with torch.no_grad():
            rewards = torch.zeros([pb.rollout_num * pb.output_seq_len, pb.batch_size]).float()
            if pb.gpu:
                rewards = rewards.cuda()

            count = 0
            for i in range(pb.rollout_num):
                for given_num in range(1, pb.output_seq_len+1):
                    hidden_state = self.generator.init_hidden()
                    inp = x[:, :given_num]
                    out, hidden_state = self.generator.forward(inp, hidden_state)
                    out = out.view(pb.batch_size, -1, pb.output_vocab_size)[:, -1]

                    samples = torch.zeros(pb.batch_size, pb.output_seq_len).long()
                    samples[:, :given_num] = x[:, :given_num]

                    if pb.gpu:
                        samples = samples.cuda()

                    # monte-carlo search
                    for j in range(given_num, pb.output_seq_len):
                        out = torch.multinomial(torch.exp(out), 1)
                        samples[:,j] = out.view(-1).data
                        inp = out.view(-1)

                        out, hidden_state = self.generator.forward(inp, hidden_state)

                    y = F.softmax(self.discriminator.forward(samples), dim=-1)
                    rewards[count] = y[:,1]
                    count += 1

        rewards = torch.mean(rewards.view(pb.batch_size, pb.output_seq_len, pb.rollout_num), dim=-1)

        return rewards

    def train_gen(self, GenDataIter):
        self.generator.train()
        loss = 0
        for text_train_pad, headline_train_pad, text_train_lengths, headline_train_lengths in GenDataIter:
            text_train_pad = torch.transpose(text_train_pad, 0, 1)
            headline_train_pad = torch.transpose(headline_train_pad, 0, 1)
            text_train_pad = text_train_pad[:text_train_lengths.max()]
            headline_train_pad = headline_train_pad[:headline_train_lengths.max()]
            if pb.gpu:
                text_train_pad.cuda(), headline_train_pad.cuda(), text_train_lengths.cuda()

            # h = self.generator.init_hidden()
            pred_y = self.generator.forward(text_train_pad, text_train_lengths, headline_train_pad)
            outputs_flatten = pred_y[1:].view(-1, pred_y.shape[-1])
            hl_flatten = headline_train_pad[1:].reshape(-1)

            l = self.gen_criterion(outputs_flatten, hl_flatten)
            self.gen_optimizer.zero_grad()
            l.backward(retain_graph=True)
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
    def train_gan(self, data_loader, text_pad, headline_pad, text_len):
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

                dis_scale = torch.sum(1/dis_pred_y)

                # combine two losses
                adv_l = dis_scale * ce_loss

                self.gen_optimizer.zero_grad()
                adv_l.backward(retain_graph=True)
                self.gen_optimizer.step()

                print('generator loss: {}'.format(adv_l))

                for step in range(pb.d_steps):
                    # random sample true headline batch_size
                    idx = torch.tensor(random.randint(0, pb.num_samples, size=pb.batch_size)).long()

                    positive = headline_pad[idx, :]
                    samples = text_pad[idx, :]
                    samples = torch.transpose(samples, 0, 1)  # due to lstm dimension reverse issue
                    samples_len = text_len[idx]
                    negative = self.generator.predict(samples, samples_len, torch.transpose(positive, 0, 1))
                    negative = torch.transpose(negative, 0, 1)  # due to lstm dimension reverse issue
                    mixed = DisDataLoader(positive, negative)

                    dis_best = []
                    for epoch in range(pb.k):
                        accuracy = self.train_dis(mixed)
                        dis_best.append(accuracy)
                    print(max(dis_best))

    # debugging...
    # adversarial training with policy gradient
    def train_gan_rl(self, data_loader, text_pad, headline_pad, text_len):
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

                rewards = self.mc_search(y)
                pg_loss = self.generator.policy_gradient_loss(x, x_batch_len, y, rewards)

                pred_y = self.generator.forward(text_train_pad, text_train_lengths, headline_train_pad)
                outputs_flatten = pred_y[1:].view(-1, pred_y.shape[-1])
                hl_flatten = headline_train_pad[1:].reshape(-1)

                ce_loss = self.gen_criterion(outputs_flatten, hl_flatten)

                # combine two loss
                l = pg_loss * pb.adv_lambda + ce_loss * (1 - pb.adv_lambda)

                self.reinforce_optimizer.zero_grad()
                l.backward(retain_graph=True)
                self.reinforce_optimizer.step()

                print('generator loss: {}'.format(l))

            for step in range(pb.d_steps):
                idx = random.randint(0, pb.num_samples, size=pb.batch_size)
                positive = headline[torch.tensor(idx).long(), :]
                negative = self.generator.sample(pb.batch_size, text)
                mixed = DisDataLoader(positive, negative)

                for epoch in range(pb.k):
                    accuracy = self.train_dis(mixed)
                    print(accuracy)
