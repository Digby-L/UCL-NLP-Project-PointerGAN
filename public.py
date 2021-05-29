# configuration file for GAN
import torch
if_gpu = False
gpu = if_gpu and torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# basic hyper-parameters
shuffle = False
batch_size = 64  # 48
padding_idx = 0
start_letter_idx = 1

# data information
num_samples = 611
vocab_size = 7560
output_vocab_size = 1411
input_seq_len = 207
output_seq_len = 52

# dropout for seq2seq
dropout = 0.1

# for parameter initialisation
unif_init_bound = 0.05
norm_init_std = 1e-3

# for gradient clipping
max_grad_norm = 5.0


# for generator
gen_pretrain_epochs = 1  # small number for test, 50
gan_epochs = 1000
gen_emb_size = 200
gen_hid_size = 128
gen_lr = 0.01
# for monte-carlo search
rollout_num = 4  # 16
# while gan training
adv_lambda = 0.5


# for discriminator
dis_embed_dim = 64
dis_filter_sizes = [2, 3, 5, 8, 12]
dis_num_filters = [100, 200, 200, 100, 160]
dis_feature_dim = sum(dis_num_filters)

dis_dropout = 0.25
dis_init_dist = 'uniform'
dis_lr = 1e-3
# while adversarial training
d_steps = 5
k = 10  # discriminator update epochs
