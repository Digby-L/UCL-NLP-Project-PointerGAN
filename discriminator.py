# CNN discriminator
# reference: https://github.com/williamSYSU/TextGAN-PyTorch/tree/891635af6845edfee382de147faa4fc00c7e90eb/models

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, embed_dim, vocab_size, filter_sizes, num_filters, padding_idx, dropout, init_dist, gpu):
        super(Discriminator, self).__init__()
        self.embedding_dim = embed_dim
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.feature_dim = sum(num_filters)
        self.gpu = gpu

        # network
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, embed_dim)) for (n, f) in zip(num_filters, filter_sizes)
        ])
        self.highway_unit = nn.Linear(self.feature_dim, self.feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.feature_dim, 1)

        # initialisation
        self.init_dist = init_dist
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

    def init_params(self):
        for param in self.parameters():
            if param.requires_grad and len(param.shape) > 0:
                stddev = 1 / np.sqrt(param.shape[0])
                if self.init_dist == 'uniform':
                    torch.nn.init.uniform_(param, a=-0.05, b=0.05)
                elif self.init_dist == 'normal':
                    torch.nn.init.normal_(param, std=stddev)
