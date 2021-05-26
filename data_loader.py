# data loader for GAN training
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader

import public as pb


class GenDataLoader:
    def __init__(self, text, headline):
        self.data_loader = DataLoader(dataset=TensorDataset(text, headline),
                                      batch_size=pb.batch_size,
                                      shuffle=pb.shuffle,
                                      drop_last=True)


class DisDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class DisDataLoader:
    def __init__(self, pos_samples, neg_samples):
        '''
        torch.utils.data.DataLoader object
        :param pos_samples: real (summarization) sentence
        :param neg_samples: summarization by generator/synthetic (summarization) sentence
        '''
        self.pos_samples = pos_samples
        self.neg_samples = neg_samples

        self.data_loader = DataLoader(dataset=DisDataset(self.load()),
                                      batch_size=pb.batch_size,
                                      shuffle=pb.shuffle,
                                      drop_last=True)

    def load(self):
        '''
        load labelled data, and mix the samples with different labels
        :return: list of dictionaries for each sample point
        '''
        x = torch.cat((self.pos_samples, self.neg_samples), dim=0).long().detach()
        y = torch.ones(x.size(0)).long()
        y[self.pos_samples.size(0):] = 0
        # mix +/- labels
        mix = torch.randperm(x.size(0))
        x, y = x[mix], y[mix]
        if pb.gpu:
            x.cuda(), y.cuda()
        dataset = [{'input':i, 'label':j} for (i, j) in zip(x, y)]
        return dataset
