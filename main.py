import torch
import numpy as np
from utility import GAN
from torch.utils.data import TensorDataset, DataLoader
import public as pb

method = 'Seq2Seq'

if __name__ == '__main__':
    if method == 'Seq2Seq':
        embedding = np.load('./data/embedding(1).npy')
        embedding_headline = np.load('./data/embedding_headline(1).npy')
        model = GAN(embedding, embedding_headline)

        headline_train_lengths = np.load('./data/headline_train_lengths(1).npy')
        headline_train_pad = np.load('./data/headline_train_pad(1).npy')
        text_train_lengths = np.load('./data/text_train_lengths(1).npy')
        text_train_pad = np.load('./data/text_train_pad(1).npy')
        data = TensorDataset(torch.tensor(text_train_pad), torch.tensor(headline_train_pad),
                             torch.tensor(text_train_lengths), torch.tensor(headline_train_lengths))
        data = DataLoader(dataset=data, batch_size=pb.batch_size, shuffle=pb.shuffle, drop_last=True)

        model.train_gan(data, torch.tensor(text_train_pad), torch.tensor(headline_train_pad),
                        torch.tensor(text_train_lengths))

    else:
        pass
