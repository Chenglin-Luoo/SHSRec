import sys
import copy
import torch
import random
import argparse
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
import math 
import torch.nn as nn
import torch.nn.functional as F

def calculate_valid_score(valid_result, valid_metric=None):
    if valid_metric:
        return valid_result[valid_metric]
    else:
        return valid_result["Recall@10"]

def check_nan(loss):
    if torch.isnan(loss):
        raise ValueError('Training loss is nan')

def get_intra_gamma_factor(epoch, start_anneal=25, end_anneal=50):
    if epoch < start_anneal:
        return 0.0
    elif epoch >= end_anneal:
        return 1.0
    else:
        progress = (epoch - start_anneal) / (end_anneal - start_anneal)
        factor = 0.5 * (1.0 - math.cos(math.pi * progress))
        return factor

class SequenceDataset(Dataset):
    def __init__(self, args, uid_list, item_list, target_list, item_list_length, maxlen, item_neighbors, ratio):
        self.item_list = item_list
        self.target_list = target_list
        self.item_list_length = item_list_length
        self.uid_list = uid_list
        self.maxlen = maxlen
        self.item_neighbors = item_neighbors
        self.ratio = ratio
        self.pos_rng = random.Random(args.seed)
        self.aug_rng = random.Random(args.seed)

    def __len__(self):
        return len(self.item_list)

    def padding_and_truncation(self, seq):
        length = len(seq)
        if length < self.maxlen:
            padded_seq = np.zeros(self.maxlen, dtype=np.int32)
            padded_seq[-length:] = seq[:]
        else:
            padded_seq = seq
        return padded_seq[-self.maxlen:]

    def item_substitution(self, seq, item_neighbors):
        seq_len = len(seq)
        if seq_len == 0:
            return seq
        aug_seq = list(seq)
        for idx in range(seq_len):
            item_id_str = str(aug_seq[idx])
            if item_id_str in item_neighbors and item_neighbors[item_id_str]:
                aug_seq[idx] = int(self.aug_rng.choice(item_neighbors[item_id_str]))
        return aug_seq
        
    def __getitem__(self, idx):
        uid = self.uid_list[idx]
        seq = self.item_list[idx]
        target = self.target_list[idx]
        length = self.item_list_length[idx]
       
        padded_seq = self.padding_and_truncation(seq)
        
        target_str = str(target)
        if target_str in self.item_neighbors and self.item_neighbors[target_str]:
            pos_item_id = int(self.pos_rng.choice(self.item_neighbors[target_str]))
        else:
            pos_item_id = target 

        aug_seq_1 = self.item_substitution(seq, self.item_neighbors)
        aug_seq_2 = self.item_substitution(seq, self.item_neighbors)
        
        padded_aug_seq_1 = self.padding_and_truncation(aug_seq_1)
        padded_aug_seq_2 = self.padding_and_truncation(aug_seq_2)

        return (
            torch.tensor(uid, dtype=torch.long), 
            torch.tensor(padded_seq, dtype=torch.long), 
            torch.tensor(target, dtype=torch.long), 
            torch.tensor(length, dtype=torch.long), 
            torch.tensor(pos_item_id, dtype=torch.long),
            torch.tensor(padded_aug_seq_1, dtype=torch.long),
            torch.tensor(padded_aug_seq_2, dtype=torch.long)
        )

class TestDataset(Dataset):
    def __init__(self, item_list, max_seq_len):
        self.uid_list, self.item_list, self.target_list, self.item_list_length = self.data_process(item_list)
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, idx):
        uid = self.uid_list[idx]
        seq = self.item_list[idx]
        target = self.target_list[idx]
        length = self.item_list_length[idx]

        if length < self.max_seq_len:
            padded_seq = np.zeros(self.max_seq_len, dtype=np.int32)
            padded_seq[-length:] = seq[:]
        else:
            padded_seq = seq
        padded_seq = padded_seq[-self.max_seq_len:]
        return torch.tensor(uid, dtype=torch.long), torch.tensor(padded_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long), torch.tensor(length, dtype=torch.long)

    def data_process(self, data_dict):
        uid_list, item_list, target_list, item_list_length = [], [], [], []
        for uid, item_id_seq in data_dict.items():
            if len(item_id_seq)>1:
                uid_list.append(uid)
                item_list.append(item_id_seq[:-1])  
                target_list.append(item_id_seq[-1]) 
                item_list_length.append(len(item_id_seq[:-1]))
        return uid_list, item_list, target_list, item_list_length

def data_augmentation(data_dict, max_seq_len):
    uid_list, item_list, target_list, item_list_length = [], [], [], []

    for uid, item_id_seq in data_dict.items():
        seq_start = 0
        for i in range(1, len(item_id_seq)):
            if i - seq_start > max_seq_len:
                seq_start += 1
            
            current_seq = item_id_seq[seq_start:i]
            target_item = item_id_seq[i] 
            
            uid_list.append(uid)
            item_list.append(current_seq)  
            target_list.append(target_item) 
            item_list_length.append(len(current_seq))
            
    return uid_list, item_list, target_list, item_list_length
    
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    f = open(fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = (User[user][:-1])
            user_test[user] = (User[user][:])
    return [user_train, user_valid, user_test, usernum, itemnum]

def evaluate(args, model, eval_data, load_best_model=True, model_file=None, show_progress=False):
    if not eval_data:
        return
    if load_best_model:
        if model_file:
            checkpoint_file = model_file
        else:
            checkpoint_file = args.saved_model_file
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['state_dict'])
        
    model.eval()
    prog_iter = tqdm(eval_data, leave=False)
    scores = []
    labels = []
    
    for batch in prog_iter:
        uid, item_seq, pos_item, item_seq_len = \
            batch[0].to(args.device), batch[1].to(args.device), batch[2].to(args.device), batch[3].to(args.device)                                             
        bs_scores = model.full_sort_predict(item_seq).detach().cpu()
        bs_labels = (pos_item-1).reshape(-1,1).cpu()
        scores.append(bs_scores)
        labels.append(bs_labels)
        
    scores = torch.cat(scores, axis=0).numpy()
    partitioned_indices = np.argpartition(-scores, 20, axis=1)[:, :20]
    pred_list = partitioned_indices[np.arange(scores.shape[0])[:, None], np.argsort(-scores[np.arange(scores.shape[0])[:, None], partitioned_indices], axis=1)].tolist()
    labels = torch.cat(labels, axis=0).numpy().tolist()

    result = get_full_sort_score(labels, pred_list)
    return result

def full_sort_batch_eval(args, model, batched_data):
    uid, item_seq, target, item_seq_len = batched_data
    scores = model.full_sort_predict(item_seq.to(args.device))
    scores = scores.view(-1, args.item_num)
    scores[:, 0] = -np.inf
    return item_seq, scores

def get_full_sort_score(answers, pred_list):
    recall, ndcg, mrr = [], [], []
    for k in [5, 10, 20]:
        recall.append(recall_at_k(answers, pred_list, k))
        ndcg.append(ndcg_k(answers, pred_list, k))
        mrr.append(mrr_at_k(answers, pred_list, k))
    result_dic = {
        "HIT@10": round(recall[1], 4), "NDCG@10": round(ndcg[1], 4),
        "HIT@20": round(recall[2], 4), "NDCG@20": round(ndcg[2], 4),
    }
    return result_dic
        
def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users

def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in set(actual[user_id])) / math.log(j+2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))

def idcg_k(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res
    
def mrr_at_k(actual, predicted, topk):
    sum_mrr = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        for rank, item in enumerate(predicted[i][:topk], start=1):
            if item in act_set:
                sum_mrr += 1.0 / rank
                break
    return sum_mrr / num_users

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  

def info_nce_positive(z_anchor, z_pos, temp_cl, batch_size, sim='dot'):
    if sim == 'cos':
        z_anchor = F.normalize(z_anchor, dim=-1)
        z_pos = F.normalize(z_pos, dim=-1)

    pos_logits = (z_anchor * z_pos).sum(dim=-1, keepdim=True) / temp_cl

    sim_matrix_1 = torch.mm(z_anchor, z_pos.T) / temp_cl
    sim_matrix_2 = torch.mm(z_anchor, z_anchor.T) / temp_cl
    
    mask_self = torch.eye(batch_size, dtype=torch.bool, device=z_anchor.device)
    
    sim_matrix_1 = sim_matrix_1.masked_fill(mask_self, -1e9)
    sim_matrix_2 = sim_matrix_2.masked_fill(mask_self, -1e9)

    logits = torch.cat((pos_logits, sim_matrix_1, sim_matrix_2), dim=1)
    labels = torch.zeros(batch_size, dtype=torch.long, device=z_anchor.device)

    return logits, labels

def info_nce(z_i, z_j, temp, batch_size, sim='dot'):
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)
        if sim == 'cos':
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temp
    
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
    
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    
        mask = mask_correlated_samples(batch_size)
        negative_samples = sim[mask].reshape(N, -1)
    
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

def mask_correlated_samples(batch_size):
    N = 2 * batch_size
    mask = torch.ones((N, N), dtype=bool)
    mask = mask.fill_diagonal_(0)
    for i in range(batch_size):
        mask[i, batch_size + i] = 0
        mask[batch_size + i, i] = 0
    return mask

def info_nce_single(seq, seq_similar, temp, batch_size, sim='dot'):
    if sim == 'cos':
        sim = nn.functional.cosine_similarity(seq.unsqueeze(1), seq_similar.unsqueeze(0), dim=2) / temp
    elif sim == 'dot':
        sim = torch.mm(seq, seq_similar.T) / temp
    
    positive_samples = torch.diag(sim).reshape(batch_size, 1)
    
    mask = mask_correlated_samples_single(batch_size)
    negative_samples = sim[mask].reshape(batch_size, -1)
    
    labels = torch.zeros(batch_size).to(positive_samples.device).long()
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    return logits, labels

def mask_correlated_samples_single(batch_size):
    mask = torch.ones((batch_size, batch_size), dtype=bool)
    mask = mask.fill_diagonal_(0)  
    return mask

def compute_cosine_similarity_batch(user_semantic_emb, top_k=100, batch_size=1024):
    num_users = user_semantic_emb.size(0)
    user_sorted_indices = np.zeros((num_users, top_k), dtype=np.int32)

    user_semantic_emb = user_semantic_emb / user_semantic_emb.norm(dim=1, keepdim=True)

    dataset = TensorDataset(user_semantic_emb)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for i, batch in enumerate(dataloader):
        batch_emb = batch[0].cuda()  
        batch_emb = batch_emb / batch_emb.norm(dim=1, keepdim=True)  
        batch_cosine_sim = torch.mm(batch_emb, user_semantic_emb.t().cuda())  
        _, batch_topk_indices = torch.topk(batch_cosine_sim, top_k, dim=1, largest=True, sorted=True)  
        batch_topk_indices = batch_topk_indices.cpu().numpy()  

        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, num_users)
        user_sorted_indices[start_idx:end_idx, :] = batch_topk_indices

    return user_sorted_indices