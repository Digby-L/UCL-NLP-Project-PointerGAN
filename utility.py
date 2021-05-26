# training class for discriminator
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

import public as pb
from discriminator import Discriminator
from generator import Generator, LSTM
from data_loader import GenDataLoader, DisDataLoader


class GAN:
    def __init__(self):
        self.generator = LSTM()
        self.discriminator = Discriminator()
        if pb.gpu:
            self.generator.cuda(), self.discriminator.cuda()

        # Criterion
        self.pretrain_criterion = nn.NLLLoss()
        self.gen_criterion = nn.CrossEntropyLoss()
        self.dis_criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=pb.gen_lr)
        self.reinforce_optimizer = optim.Adam(self.generator.parameters(), lr=pb.gen_lr)
        self.dis_optimizer = optim.Adam(self.discriminator.parameters(), lr=pb.dis_lr)

    def mc_search(self, x):
        with torch.no_grad():
            rewards = torch.zeros([pb.rollout_num * pb.max_seq_len, pb.batch_size]).float()
            if pb.gpu:
                rewards = rewards.cuda()

            count = 0
            for i in range(pb.rollout_num):
                for given_num in range(1, pb.max_seq_len+1):
                    hidden_state = self.generator.init_hidden()
                    inp = x[:, :given_num]
                    out, hidden_state = self.generator.forward(inp, hidden_state)
                    out = out.view(pb.batch_size, -1, pb.vocab_size)[:, -1]

                    samples = torch.zeros(pb.batch_size, pb.max_seq_len).long()
                    samples[:, :given_num] = x[:, :given_num]

                    if pb.gpu:
                        samples = samples.cuda()

                    # monte-carlo search
                    for j in range(given_num, pb.max_seq_len):
                        out = torch.multinomial(torch.exp(out), 1)
                        samples[:,j] = out.view(-1).data
                        inp = out.view(-1)

                        out, hidden_state = self.generator.forward(inp, hidden_state)

                    y = F.softmax(self.discriminator.forward(samples), dim=-1)
                    rewards[count] = y[:,1]
                    count += 1

        rewards = torch.mean(rewards.view(pb.batch_size, pb.max_seq_len, pb.rollout_num), dim=-1)

        return rewards

    def train_gen(self, GenDataIter):
        self.generator.train()
        loss = 0
        for x,y in GenDataIter:
            if pb.gpu:
                x.cuda(), y.cuda()

            h = self.generator.init_hidden()
            pred_y, _ = self.generator.forward(x,h)
            l = self.gen_criterion(pred_y, y.view(-1))
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
        for i, data in enumerate(DisDataIter):
            x, y = data['input'], data['label']
            if pb.gpu:
                x, y = x.cuda(), y.cuda()

            pred_y = self.discriminator.forward(x)
            loss = self.dis_criterion(pred_y, y)
            self.dis_optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.discriminator.parameters(), max_norm=pb.max_grad_norm)
            self.dis_optimizer.step()

            accuracy += torch.sum(torch.tensor(pred_y.argmax(dim=-1) == y)).item()
            count += x.size(0)

        return accuracy/count

    def train_gan(self, data_loader):
        ##### pre-train #####
        print('generator pre-training via maximum likelihood begins')
        for i in range(pb.gen_pretrain_epochs):
            _ = self.train_gen(data_loader)

        print('discriminator pre-training begins')
        for step in range(pb.d_steps):
            positive = data_loader.headline
            negative = self.generator.sample(pb.num_samples)
            mixed = DisDataLoader(positive, negative)

            for epoch in range(pb.k):
                _ = self.train_dis(mixed)

        print('==============================================')
        print('adversarial training begins')
        for j in range(pb.gan_epochs):
            y = self.generator.sample(pb.batch_size)
            x = torch.zeros(y.size()).long()
            x[:, 0] = pb.start_letter_idx
            x[:,1:] = y[:, :pb.max_seq_len-1]
            if pb.gpu:
                y.cuda(), x.cuda()

            rewards = self.mc_search(y)
            pg_loss = self.generator.policy_gradient_loss(x, y, rewards)
            self.reinforce_optimizer.zero_grad()
            pg_loss.backward()
            self.reinforce_optimizer.step()

            print('generator loss: {}'.format(pg_loss))

            for step in range(pb.d_steps):
                positive = data_loader.headline
                negative = self.generator.sample(pb.num_samples)
                mixed = DisDataLoader(positive, negative)

                for epoch in range(pb.k):
                    accuracy = self.train_dis(mixed)
                    print(accuracy)
