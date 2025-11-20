import math
import random
import subprocess
from datetime import datetime


def run_cmd(cmd, arguments_dict):
    for k, v in arguments_dict.items():
        cmd += f" --{k}={v}"

    print("\nrunning cmd: ", cmd)
    start = datetime.now()
    p = subprocess.Popen(cmd, shell=True)
    p.wait()

    end = datetime.now()
    print("running used time:{}\n".format(end - start))


val_num_beams = 1
test_num_beams = 50

train_batch_size = 1024
eval_batch_size = math.floor(train_batch_size // val_num_beams)
test_batch_size = math.floor(train_batch_size // test_num_beams)

arguments_dict = {
    'dataset': 'AmazonReviews2023',
    'run_time': None,
    'train_batch_size': train_batch_size,
    'eval_batch_size': eval_batch_size,
    'test_batch_size': test_batch_size,
    'lr': 3e-3,
    'weight_decay': 0.0,
    'warmup_ratio': 0.1,
    'epochs': 150,
    'max_users': 10000000,
    'eval_interval': 1,
    'patience': 50,
    'val_topk': '[1]',
    'val_metric': 'ndcg@1',
    'topk': '[1,5,10]',
}


def run_tokenizer(category):
    CUDA_VISIBLE_DEVICES = "0"

    token_paras = {
        'category': category,
        'tokenizer_name': 'MultiHeadVQVAE',
        'sent_emb_pca': 128,
        'vq_n_codebooks': 4,
        'vq_codebook_size': 256,
        'vqvae_hidden_sizes': '[512,256]',
        'vqvae_e_dim': 32,
        'vqvae_dropout': 0.0,
        'vqvae_lr': 0.001,
        'vqvae_l2': 1e-4,
        'vqvae_batch_size': 2048,
        'vqvae_epoch': 10000,
        'vqvae_verbose': 50,
        'vqvae_beta': 0.25,
        'vqvae_sk_epsilon': 0.003,
        'vqvae_sk_iters': 10,
        'vqvae_quant_weight': 1,
        'kmeans_init': True,
        'kmeans_iters': 100,
        'vqvae_save_method': 'collision',
    }
    arguments_dict.update(token_paras)

    arguments_dict['run_time'] = datetime.now().strftime(r"%Y%m%d-%H%M%S")

    cmd = f'CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} python -u main_token.py'

    run_cmd(cmd, arguments_dict)

    return arguments_dict['run_time']


def run_model(category, sem_id_time):
    CUDA_VISIBLE_DEVICES = "0,1"

    if category == 'Industrial_and_Scientific':
        model_paras = {
            'category': category,
            'model': 'LLaDARec',
            'tokenizer_name': 'MultiHeadVQVAE',
            'sem_ids_path': f"{sem_id_time}.sem_ids",
            'lr': 0.003,
            'weight_decay': 0.05,
            'max_item_seq_len': 20,
            'dropout_rate': 0.1,
            'd_model': 256,
            'mlp_ratio': 4,
            'n_layers': 4,
            'n_heads': 4,
            'n_kv_heads': 4,
            'his_mask_w': 3.5,
            'gen_steps': 4,
            'val_num_beams': val_num_beams,
            'num_beams': test_num_beams,
            'temperature': 0.3,
        }
    elif category == 'Musical_Instruments':
        model_paras = {
            'category': category,
            'model': 'LLaDARec',
            'tokenizer_name': 'MultiHeadVQVAE',
            'sem_ids_path': f"{sem_id_time}.sem_ids",
            'lr': 0.003,
            'weight_decay': 0.05,
            'max_item_seq_len': 20,
            'dropout_rate': 0.1,
            'd_model': 256,
            'mlp_ratio': 4,
            'n_layers': 4,
            'n_heads': 8,
            'n_kv_heads': 8,
            'his_mask_w': 4,
            'gen_steps': 4,
            'val_num_beams': val_num_beams,
            'num_beams': test_num_beams,
            'temperature': 0.7,
        }
    elif category == 'Video_Games':
        model_paras = {
            'category': category,
            'model': 'LLaDARec',
            'tokenizer_name': 'MultiHeadVQVAE',
            'sem_ids_path': f"{sem_id_time}.sem_ids",
            'lr': 0.003,
            'weight_decay': 0.001,
            'max_item_seq_len': 20,
            'dropout_rate': 0.1,
            'd_model': 256,
            'mlp_ratio': 4,
            'n_layers': 6,
            'n_heads': 8,
            'n_kv_heads': 8,
            'his_mask_w': 5,
            'gen_steps': 4,
            'val_num_beams': val_num_beams,
            'num_beams': test_num_beams,
            'temperature': 1.0,
        }
    else:
        raise NotImplementedError
    arguments_dict.update(model_paras)

    arguments_dict['run_time'] = datetime.now().strftime(r"%Y%m%d-%H%M%S")

    if len(CUDA_VISIBLE_DEVICES.split(',')) > 1:
        cmd = f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} \
        accelerate launch --num_processes={len(CUDA_VISIBLE_DEVICES.split(','))} --main_process_port {random.randint(49152,65535)} main.py"

    else:
        cmd = f'CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} python -u main.py'

    run_cmd(cmd, arguments_dict)


# Amazon2023
# Industrial_and_Scientific / Musical_Instruments / Video_Games
for category in ['Industrial_and_Scientific']:
    sem_id_time = run_tokenizer(category=category)
    run_model(category=category, sem_id_time=sem_id_time)
