# CNN discriminator
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import public as pb


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # network
        self.embeddings = nn.Embedding(pb.vocab_size, pb.dis_embed_dim, padding_idx=pb.padding_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, pb.dis_embed_dim)) for (n, f) in zip(pb.dis_num_filters, pb.dis_filter_sizes)
        ])
        self.highway_unit = nn.Linear(pb.dis_feature_dim, pb.dis_feature_dim)
        self.dropout = nn.Dropout(pb.dis_dropout)
        self.fc = nn.Linear(pb.dis_feature_dim, 1)

        # initialisation
        self.init_dist = pb.dis_init_dist
        self.init_params()

    def forward(self, sentence):
        """
        Get predictions of discriminator
        :param sentence: batch_size * seq_len
        :return: pred: batch_size
        """
        embedded = self.embeddings(sentence).unsqueeze(1)  # batch_size * 1 * max_seq_len * embed_dim
        convs = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]  # batch_size * num_filter * length
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]  # batch_size * num_filter
        pred = torch.cat(pools, 1)  # batch_size * feature_dim

        # highway layer
        highway_unit = self.highway_unit(pred)
        pred = torch.sigmoid(highway_unit) * F.relu(highway_unit) + (1. - torch.sigmoid(highway_unit)) * pred

        pred = self.fc(self.dropout(pred))  # batch_size

        return pred

    def predict(self, sentence):
        return [1 if y >= 0 else 0 for y in self.forward(sentence)]

    def init_params(self):
        for param in self.parameters():
            if param.requires_grad and len(param.shape) > 0:
                stddev = 1 / np.sqrt(param.shape[0])
                if self.init_dist == 'uniform':
                    torch.nn.init.uniform_(param, a=-pb.unif_init_bound, b=pb.unif_init_bound)
                elif self.init_dist == 'normal':
                    torch.nn.init.normal_(param, std=pb.norm_init_std)
