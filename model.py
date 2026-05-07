import numpy as np
import torch
import torch.nn as nn
from recbole.model.layers import TransformerEncoder
from utils import info_nce, info_nce_single, info_nce_positive
import torch.nn.functional as F

class TransRec(torch.nn.Module):
    def __init__(self, config):
        super(TransRec, self).__init__()
        self.n_layers = config.n_layers
        self.n_heads = config.n_heads
        self.hidden_size = config.hidden_size  
        self.inner_size = config.inner_size  
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.attn_dropout_prob = config.attn_dropout_prob
        self.hidden_act = config.hidden_act
        self.layer_norm_eps = config.layer_norm_eps
        self.batch_size = config.train_batch_size
        self.initializer_range = config.initializer_range
        self.loss_type = config.loss_type
        self.n_items = config.item_num
        self.max_seq_length = config.maxlen
        self.temperature = config.temperature
        self.temp_inter = config.temp_inter if hasattr(config, 'temp_inter') else self.temperature
        self.temp_intra = config.temp_intra if hasattr(config, 'temp_intra') else self.temperature
        
        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.loss_fct = nn.CrossEntropyLoss()
        self.nce_fct = nn.CrossEntropyLoss()
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) 
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, item_seq):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = output[:, -1, :]
        return output  

    def calculate_loss(self, item_seq, pos_items):
        seq_output = self.forward(item_seq)
        test_item_emb = self.item_embedding.weight[1:self.n_items+1]  
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        pos_items = pos_items - 1
        loss = self.loss_fct(logits, pos_items)
        return loss
    
    def calculate_align_loss(self, anchor_seq, pos_item_id, seq_len):
        z_anchor = self.forward(anchor_seq)
        
        valid_indices = seq_len > 0
        if valid_indices.sum() == 0:
            zero_tensor = torch.tensor(0.0, requires_grad=True).to(z_anchor.device)
            return zero_tensor, zero_tensor

        z_anchor_v = z_anchor[valid_indices]
        pos_item_id_v = pos_item_id[valid_indices]
        
        all_item_emb = self.item_embedding.weight[1:self.n_items+1]
        logits = torch.matmul(z_anchor_v, all_item_emb.transpose(0, 1)) / self.temp_inter
        labels = pos_item_id_v - 1
        
        cl_loss = self.nce_fct(logits, labels)
        zero_loss = torch.tensor(0.0, device=z_anchor.device)
        return cl_loss, zero_loss

    def dynamic_augmentation(self, item_seq, cand_seq, ratio):
        valid_mask = (item_seq > 0)
        if valid_mask.sum() == 0:
            return item_seq
        
        prob = torch.full(item_seq.shape, ratio, dtype=torch.float, device=item_seq.device)
        replace_mask = torch.bernoulli(prob).bool() & valid_mask
        aug_seq = torch.where(replace_mask, cand_seq, item_seq)
        return aug_seq

    def calculate_intra_loss(self, item_seq, cand_seq_1, cand_seq_2, seq_len, ratio):
        aug_seq_1 = self.dynamic_augmentation(item_seq, cand_seq_1, ratio)
        aug_seq_2 = self.dynamic_augmentation(item_seq, cand_seq_2, ratio)
        
        z_1 = self.forward(aug_seq_1)
        z_2 = self.forward(aug_seq_2)
        
        valid_indices = seq_len > 0
        if valid_indices.sum() == 0:
            zero_tensor = torch.tensor(0.0, requires_grad=True).to(z_1.device)
            return zero_tensor

        z_1_v = z_1[valid_indices]
        z_2_v = z_2[valid_indices]
        
        batch_size_v = z_1_v.shape[0]
        
        logits, labels = info_nce(
            z_1_v, z_2_v, 
            temp=self.temp_intra, 
            batch_size=batch_size_v, 
            sim="dot"
        )
        
        intra_cl_loss = self.nce_fct(logits, labels)
        
        return intra_cl_loss

    def predict(self, item_seq, test_item):
        seq_output = self.forward(item_seq)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1) 
        return scores

    def full_sort_predict(self, item_seq):
        seq_output = self.forward(item_seq)
        test_items_emb = self.item_embedding.weight[1:self.n_items+1]  
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  
        return scores
    
    def gather_indexes(self, output, gather_index):
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)