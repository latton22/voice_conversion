"""
src_train_dir = '/localtmp/for_pretrain/out/train/mspec'
tgt_train_dir = '/localtmp/for_pretrain/out/train/label'
src_valid_dir = '/localtmp/for_pretrain/out/valid/mspec'
tgt_valid_dir = '/localtmp/for_pretrain/out/valid/label'
src_test_dir  = '/localtmp/for_pretrain/out/test/mspec'
"""
src_train_dir = '../../../01_extract_features/for_pretrain/out/train/mspec'
tgt_train_dir = '../../../01_extract_features/for_pretrain/out/train/label'
src_valid_dir  = '../../../01_extract_features/for_pretrain/out/valid/mspec'
tgt_valid_dir  = '../../../01_extract_features/for_pretrain/out/valid/label'
src_test_dir  = '../../../01_extract_features/for_pretrain/out/test/mspec'

warmup_steps = 15000
n_epoch    = 25
batchsize = 4
lr         = 0.0025
maxpool_kernel = 3
conv_kernel = 7

hidden_dim = [128, 256, 512, 1024]
piramid_scale = 4

expand_input_dim = 48
mspec_dim = 80
ppg_dim  = 35

out_path   = '../out'
train_path = out_path + '/train_loss.csv'
valid_path = out_path + '/valid_loss.csv'
model_path = out_path + '/model_epoch.pth'
out_ppg_path   = out_path + '/ppg'
lr_log_path = out_path + '/lr_log.csv'

test_model_path = out_path + '/model_best.pth'
