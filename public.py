# configuration file for GAN
import torch


# basic hyper-parameters
shuffle = False
batch_size = 48
vocab_size = 0
padding_idx = 0

if_gpu = False
gpu = if_gpu and torch.cuda.is_available()

# for parameter initialisation
unif_init_bound = 0.05
norm_init_std = 1e-3

# max norm of the gradients, for gradient clipping
max_norm = 5.0


# for generator
gen_emb_size = 32
gen_hid_size = 128


# for discriminator
dis_embed_dim = 64
dis_filter_sizes = [2, 3, 4, 5, 8, 10]
dis_num_filters = [100, 200, 200, 200, 100]
dis_dropout = 0.25
dis_init_dist = 'uniform'
dis_lr = 1e-3
dis_update_steps = 5
dis_update_epoch = 5
