import os
import sys
import json
import math
from time import time
import torch
import argparse
from tqdm import tqdm
import torch.optim as optim
from model import TransRec
from utils import *
from recbole.utils import ensure_dir, get_local_time, early_stopping, calculate_valid_score, dict2str
import logging
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--beta', default=0.1, type=float)
parser.add_argument('--gamma', default=0.1, type=float)
parser.add_argument('--ratio', default=0.1, type=float)
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--train_batch_size', default=256, type=int)
parser.add_argument('--test_batch_size', default=1024, type=int)
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--maxlen', default=20, type=int)
parser.add_argument('--hidden_size', default=64, type=int)
parser.add_argument('--inner_size', default=256, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--n_heads', default=2, type=int)
parser.add_argument('--n_layers', default=2, type=int)
parser.add_argument('--hidden_dropout_prob', default=0.5, type=float)
parser.add_argument('--attn_dropout_prob', default=0.5, type=float)
parser.add_argument('--layer_norm_eps', default=1e-12, type=float)
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--initializer_range', default=0.02, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--hidden_act', default='gelu', type=str)
parser.add_argument('--loss_type', default='CE', type=str)
parser.add_argument('--valid_metric', default='NDCG@10', type=str)
parser.add_argument('--stopping_step', default=10, type=int)
parser.add_argument('--eval_step', default=1, type=int)
parser.add_argument('--valid_metric_bigger', default=True, type=bool)
parser.add_argument('--log_dir', default='./log/', type=str)
parser.add_argument('--seed', default=2024, type=int)
parser.add_argument('--dataset_file', default='./data/Tools/Tools.txt', type=str) # Beauty Sports Toys Tools
parser.add_argument('--temperature', default=1.0, type=float)
parser.add_argument('--temp_inter', default=1.0, type=float)
parser.add_argument('--temp_intra', default=1.0, type=float)
parser.add_argument('--cl_pool_file', default=r'data/Tools/semantic_only_top5.json', type=str)

args = parser.parse_args()
set_seed(args.seed)

output_dir = os.path.join(args.log_dir, args.dataset + '_' + args.train_dir)
ensure_dir(output_dir)

with open(os.path.join(output_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
args.saved_model_file = os.path.join(output_dir, f"model_{current_time}.pth")
args.saved_result_file = os.path.join(output_dir, f"results_beta_{args.beta:.3f}.txt")

log_file = os.path.join(output_dir, f"train_{current_time}.log")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)])
logger = logging.getLogger()

def train_epoch(args, model, train_data, optimizer, epoch_idx, cur_beta, cur_gamma, loss_func=None, show_progress=True, device=None):
    model.train()
    total_loss = 0.0
    total_rec_loss = 0.0
    total_cl_base_loss = 0.0
    total_intra_loss = 0.0
    
    iter_data = (tqdm(enumerate(train_data), total=len(train_data), desc=f"Train {epoch_idx:>5}") if show_progress else enumerate(train_data))
    
    for batch_idx, batch_data in iter_data:
        
        uid, item_seq, pos_items, item_seq_len, pos_item_id, cand_seq_1, cand_seq_2 = batch_data
        
        item_seq = item_seq.to(args.device)
        pos_items = pos_items.to(args.device)
        item_seq_len = item_seq_len.to(args.device)
        pos_item_id = pos_item_id.to(args.device)
        cand_seq_1 = cand_seq_1.to(args.device)
        cand_seq_2 = cand_seq_2.to(args.device)
        
        optimizer.zero_grad()
        
        rec_loss = model.calculate_loss(item_seq, pos_items)
        cl_loss_combined, _ = model.calculate_align_loss(item_seq, pos_item_id, item_seq_len)
        
        weighted_cl = cur_beta * cl_loss_combined
        
        if cur_gamma > 0:
            intra_loss = model.calculate_intra_loss(item_seq, cand_seq_1, cand_seq_2, item_seq_len, args.ratio)
            weighted_intra = cur_gamma * intra_loss
        else:
            intra_loss = torch.tensor(0.0, device=args.device)
            weighted_intra = torch.tensor(0.0, device=args.device)
            
        losses = rec_loss + weighted_cl + weighted_intra
       
        check_nan(losses)
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        total_rec_loss += rec_loss.item()
        total_cl_base_loss += weighted_cl.item()
        total_intra_loss += weighted_intra.item()

    return (total_loss, total_rec_loss, total_cl_base_loss, total_intra_loss)

def valid_epoch(args, model, valid_data, show_progress=False):
    valid_result = evaluate(args, model, valid_data, load_best_model=False)
    valid_score = calculate_valid_score(valid_result, args.valid_metric)
    return valid_score, valid_result

def save_checkpoint(epoch, model, saved_model_file):
    state = {'state_dict': model.state_dict()}
    torch.save(state, saved_model_file)
    
def generate_train_loss_output(epoch_idx, s_time, e_time, losses):
    total_loss, rec_loss, cl_base, intra_loss = losses
    train_loss_output = (
        f"epoch {epoch_idx} training [time: {e_time - s_time:.2f}s, "
        f"Total Loss: {total_loss:.4f}, Rec Loss: {rec_loss:.4f}, "
        f"Inter CL: {cl_base:.4f}, Intra CL: {intra_loss:.4f}]"
    )
    return train_loss_output

def get_emb_lr_factor(epoch, start_anneal=25, end_anneal=50, min_factor=0.1):
    if epoch < start_anneal:
        return 1.0
    elif epoch >= end_anneal:
        return min_factor
    else:
        progress = (epoch - start_anneal) / (end_anneal - start_anneal)
        factor = min_factor + 0.5 * (1.0 - min_factor) * (1.0 + math.cos(math.pi * progress))
        return factor

if __name__ == '__main__':
    with open(args.cl_pool_file, 'r', encoding='utf-8') as f:
        item_neighbors_dict = json.load(f)
    
    dataset = data_partition(args.dataset_file)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    args.item_num = itemnum
    
    uid_list, item_list, target_list, item_list_length = data_augmentation(user_train, args.maxlen)
    
    TrainData = SequenceDataset(args, uid_list, item_list, target_list, item_list_length, args.maxlen, item_neighbors_dict, args.ratio)
    TrainDataLoader = DataLoader(TrainData, batch_size=args.train_batch_size, shuffle=True)
    
    ValData = TestDataset(user_valid, args.maxlen)
    ValDataLoader = DataLoader(ValData, batch_size=args.test_batch_size, shuffle=False)
    TestData = TestDataset(user_test, args.maxlen)
    TestDataLoader = DataLoader(TestData, batch_size=args.test_batch_size, shuffle=False)
    
    model = TransRec(args).to(args.device) 

    if args.inference_only:
        if args.state_dict_path is None:
            sys.exit()
        checkpoint = torch.load(args.state_dict_path, map_location=args.device)
        model.load_state_dict(checkpoint['state_dict'])
        test_result = evaluate(args, model, TestDataLoader, load_best_model=False)
        df = pd.DataFrame([test_result])
        result_file = args.state_dict_path.replace(".pth", "_inference_test.txt")
        df.to_csv(result_file, index=False, sep='\t')
        sys.exit() 

    emb_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'item_embedding' in name:
            emb_params.append(param)
        else:
            other_params.append(param)

    optimizer = optim.Adam([
        {'params': other_params, 'lr': args.learning_rate, 'name': 'other'},
        {'params': emb_params, 'lr': args.learning_rate, 'name': 'item_embedding'}
    ], weight_decay=args.weight_decay)
    
    start_epoch = 0
    verbose = True
    saved = True
    best_valid_score = -np.inf if args.valid_metric_bigger else np.inf
    cur_step = 0
    
    for epoch_idx in range(start_epoch, args.epochs):
        if epoch_idx < 30:
            cur_beta = args.beta
            cur_gamma = 0
        elif 30 <= epoch_idx < 40:
            progress = (epoch_idx - 30) / 10
            cur_beta = args.beta * 0.5 * (1 + math.cos(math.pi * progress))
            cur_gamma = args.gamma * 0.5 * (1 - math.cos(math.pi * progress))
        else:
            cur_beta = 0.0
            cur_gamma = args.gamma

        current_emb_lr = args.learning_rate
        
        for param_group in optimizer.param_groups:
            if param_group.get('name') == 'item_embedding':
                param_group['lr'] = current_emb_lr

        if verbose:
            curriculum_log = f"[SACA] Epoch {epoch_idx}: beta={cur_beta:.4f}, gamma={cur_gamma:.4f}, temp_inter={args.temp_inter:.2f}, temp_intra={args.temp_intra:.2f}"
            logger.info(curriculum_log)

        training_start_time = time()
        train_loss = train_epoch(args, model, TrainDataLoader, optimizer, epoch_idx, cur_beta, cur_gamma)
        training_end_time = time()
        
        train_loss_output = generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
        if verbose:
            logger.info(train_loss_output)
        
        if (epoch_idx) % args.eval_step == 0:
            valid_start_time = time()
            valid_score, valid_result = valid_epoch(args, model, ValDataLoader, show_progress=False)
            best_valid_score, cur_step, stop_flag, update_flag = early_stopping(
                valid_score, best_valid_score, cur_step, max_step=args.stopping_step, bigger=args.valid_metric_bigger)
            valid_end_time = time()
            
            valid_result_output = ('valid result') + ': \n' + dict2str(valid_result)
            logger.info(valid_result_output)
            
            if update_flag and saved:
                save_checkpoint(epoch_idx, model, args.saved_model_file)
                best_valid_result = valid_result

            if stop_flag:
                break
    
    test_result = evaluate(args, model, TestDataLoader, load_best_model=True)
    df = pd.DataFrame([test_result])
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    df.to_csv(args.saved_result_file, index=False, sep='\t')
    logger.info(('best valid ') + f': {best_valid_result}')
    logger.info(('test result') + f': {test_result}')