import torch

from utility import GAN
from torch.utils.data import DataLoader
import public as pb


if __name__ == '__main__':
    model = GAN()
    data  = torch.load('./Dataset3/yyz_train.pt')
    data = DataLoader(dataset=data, batch_size=pb.batch_size, shuffle=pb.shuffle, drop_last=True)
    model.train_gan(data)
