import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import public as pb

method = 'Seq2Seq'

if __name__ == '__main__':
    if method == 'Seq2Seq':
        from utility import GAN
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
        traindata_zip = torch.load('./data/Dataset3/traindata_zip.pt')
        from utility_pgn import Parameters, Encoder, ReduceState, Decoder, Pointer_Generator, GAN
        trainloader = DataLoader(traindata_zip, batch_size=pb.batch_size, shuffle=False, num_workers=0)

        embedding = np.load('./data/embedding.npy')
        embedding_headline = np.load('./data/embedding_headline.npy')

        para = Parameters()
        encoder = Encoder(para.input_size, para.hid_size, para.enc_emb_size, embedding)
        reduce = ReduceState(para.hid_size)
        decoder = Decoder(para.input_size, para.hid_size, para.vocab_size, para.dec_emb_size, embedding_headline,
                          para.use_coverage, para.use_p_gen)

        PG = Pointer_Generator(para, encoder, reduce, decoder)
        model = GAN(PG)
        model.train_gan(trainloader)
