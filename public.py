# configuration file for GAN
import torch


# basic hyper-parameters
shuffle = False
batch_size = 64  # 48
padding_idx = 0

# TODO: set from data
vocab_size = 0
max_seq_len = 0

if_gpu = False
gpu = if_gpu and torch.cuda.is_available()

# for parameter initialisation
unif_init_bound = 0.05
norm_init_std = 1e-3

max_grad_norm = 5.0


# for generator
gen_emb_size = 32
gen_hid_size = 128
gen_lr = 0.01
rollout_num = 4  # 16


# for discriminator
dis_embed_dim = 64
dis_filter_sizes = [2, 3, 5, 8, 12]
dis_num_filters = [100, 200, 200, 100, 160]
dis_feature_dim = sum(dis_num_filters)
dis_dropout = 0.25
dis_init_dist = 'uniform'
dis_lr = 1e-3
dis_update_steps = 5
dis_update_epoch = 5
