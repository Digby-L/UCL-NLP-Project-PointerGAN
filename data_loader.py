# data loader for discriminator using

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import public as pb


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class MyDataLoader:
    def __init__(self, pos_samples, neg_samples):
        '''
        torch.utils.data.DataLoader object
        :param pos_samples: real (summarization) sentence
        :param neg_samples: summarization by generator/synthetic (summarization) sentence
        '''
        self.pos_samples = pos_samples
        self.neg_samples = neg_samples

        self.data_loader = DataLoader(dataset=MyDataset(self.load()),
                                      batch_size=pb.batch_size,
                                      shuffle=pb.shuffle,
                                      drop_last=True)

    def load(self):
        '''
        load labelled data, and mix the samples with different labels
        :return: list of dictionaries for each sample point
        '''
        X = torch.cat((self.pos_samples, self.neg_samples), dim=0).long().detach()
        y = torch.ones(X.size(0)).long()
        y[self.pos_samples.size(0):] = 0

        mix = torch.randperm(X.size(0))
        X,y = X[mix], y[mix]
        if pb.gpu:
            X.cuda(), y.cuda()
        dataset = [{'input':i, 'label':j} for (i, j) in zip(X, y)]
        return dataset

    def random_batch(self):
        '''
        choose one batch randomly
        '''
        idx = np.random.randint(0, len(self.data_loader) - 1)
        return list(self.data_loader)[idx]
