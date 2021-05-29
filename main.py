import torch
import numpy as np
from utility import GAN
from torch.utils.data import TensorDataset, DataLoader
import public as pb


if __name__ == '__main__':
    embedding = np.load('./Dataset3/embedding.npy')
    model = GAN(embedding)

    headline_train_lengths = np.load('./Dataset3/headline_train_lengths.npy')
    headline_train_pad = np.load('./Dataset3/headline_train_pad.npy')
    text_train_lengths = np.load('./Dataset3/text_train_lengths.npy')
    text_train_pad = np.load('./Dataset3/text_train_pad.npy')
    data = TensorDataset(torch.tensor(text_train_pad), torch.tensor(headline_train_pad),
                         torch.tensor(text_train_lengths), torch.tensor(headline_train_lengths))
    data = DataLoader(dataset=data, batch_size=pb.batch_size, shuffle=pb.shuffle, drop_last=True)

    model.train_gan(data, torch.tensor(text_train_pad), torch.tensor(headline_train_pad), torch.tensor(text_train_lengths))
