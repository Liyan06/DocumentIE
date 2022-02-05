import torch
import torch.nn as nn
import torch.optim as optim

import json
from nltk.tokenize import sent_tokenize

from utils import get_att_regu_mask
import numpy as np

from load_data import load_data, load_document, load_relation, sort_relations, MOST_COMMON_R
from transformers import get_linear_schedule_with_warmup, AdamW, RobertaModel, RobertaConfig, RobertaTokenizer


# MOST_COMMON_R= ['P17', 'P131', 'P27', 'P150', 'P577', 'P175', 'P569', 'P570', 'P527', 'P161']
MOST_COMMON_R = sort_relations()


class RERobertaDatasetBase(torch.utils.data.Dataset):
    def __init__(self, args, df, data: json, cat, weakly=False):
        
        self.df = df
        self.classes = [load_relation()[r] for r in MOST_COMMON_R]
        self.max_len = args.max_input_len
        self.cat = cat
        self.weakly = weakly
        self.tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_class)
        self.dataset = data

    def __len__(self):
        return len(self.df)


class RERobertaDataset(RERobertaDatasetBase):

    def __getitem__(self, index):
        data = {}
        row = self.df.iloc[index]  

        head_token_ids = self.tokenizer.encode(row.h, add_special_tokens=False)
        tail_token_ids = self.tokenizer.encode(row.t, add_special_tokens=False)
        text_token_ids = self.tokenizer.encode(load_document(self.dataset, row.doc_idx, cat=self.cat), 
                                               add_special_tokens=False,
                                               max_length=self.max_len,
                                               truncation=True)
        avail_len = self.max_len - (5 + len(head_token_ids) + len(tail_token_ids))
        input_ids = [0] + head_token_ids + [2, 2] + tail_token_ids + [2] + text_token_ids[:avail_len] + [2]
        
        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            input_ids += [1] * pad_len
        
        data['text'] = load_document(self.dataset, row.doc_idx, cat=self.cat)
        data['input_ids'] = torch.tensor(input_ids)
        data['attention_mask'] = torch.where(torch.tensor(input_ids) != 1, torch.tensor(1), torch.tensor(0))
        data['evidence'] = str(row.evidence)
        data['head'] = row.h
        data['tail'] = row.t
        data['relation'] = torch.tensor(self.classes.index(row.r))
        data['doc_idx'] = row.doc_idx
        
        ################### attention regularization ######################
        att = get_att_regu_mask(data['text'], data['input_ids'], data["evidence"], data['relation'])
        if self.weakly:
            if index % 9 == 0: data['regu_mask'] = att
            else: data['regu_mask'] = torch.ones_like(data['input_ids'])
        else: data['regu_mask'] = att
            
        ################### maximize entropy when input is not sufficient ############
        if self.weakly:
             data['reduced_flag'] = 0 if len(eval(data['evidence'])) == 0 or index % 9 != 0 else 1
        else:
            data['reduced_flag'] = 0 if len(eval(data['evidence'])) == 0 else 1
        data['reduced_text'], data['reduced_input_ids'], data['reduced_attention_mask'] = self.get_reduced(data['evidence'],
                                                                                                           data['text'],
                                                                                                           head_token_ids,
                                                                                                           tail_token_ids)
        return data
    
    
    def get_reduced(self, evidence, text, head_token_ids, tail_token_ids):
        to_delete = eval(evidence)
        original = sent_tokenize(text)
        reduced_text = " ".join([sent for i, sent in enumerate(original) if i not in to_delete])
        
        reduced_text_token_ids = self.tokenizer.encode(reduced_text,
                                                       add_special_tokens=False,
                                                       max_length=self.max_len,
                                                       truncation=True)
        avail_len = self.max_len - (5 + len(head_token_ids) + len(tail_token_ids))
        reduced_input_ids = [0] + head_token_ids + [2, 2] + tail_token_ids + [2] + reduced_text_token_ids[:avail_len] + [2]
        
        pad_len = self.max_len - len(reduced_input_ids)
        if pad_len > 0:
            reduced_input_ids += [1] * pad_len
            
        reduced_input_ids = torch.tensor(reduced_input_ids)
        reduced_attention_mask = torch.where(reduced_input_ids.clone().detach() != 1, torch.tensor(1), torch.tensor(0))
        
        return reduced_text, reduced_input_ids, reduced_attention_mask
            
        
class RERobertaDatasetMask(RERobertaDataset):

    def __getitem__(self, index):
        data = {}
        row = self.df.iloc[index]  

        head_token_ids = self.tokenizer.encode("entity1", add_special_tokens=False)
        tail_token_ids = self.tokenizer.encode("entity2", add_special_tokens=False)
        
        doc = load_document(self.dataset, row.doc_idx, cat=self.cat)
        doc = doc.replace(row.h, 'entity1')
        doc = doc.replace(row.t, 'entity2')
        text_token_ids = self.tokenizer.encode(doc, 
                                               add_special_tokens=False,
                                               max_length=self.max_len,
                                               truncation=True)
        avail_len = self.max_len - (5 + len(head_token_ids) + len(tail_token_ids))
        input_ids = [0] + head_token_ids + [2, 2] + tail_token_ids + [2] + text_token_ids[:avail_len] + [2]
        
        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            input_ids += [1] * pad_len
        
        data['text'] = doc
        data['input_ids'] = torch.tensor(input_ids)
        data['attention_mask'] = torch.where(torch.tensor(input_ids) != 1, torch.tensor(1), torch.tensor(0))
        data['evidence'] = str(row.evidence)
        data['head'] = "entity1"
        data['tail'] = "entity2"
        data['relation'] = torch.tensor(self.classes.index(row.r))
        data['doc_idx'] = row.doc_idx
        
        return data
    
    
# len = 148 
class RERobertaDatasetShort(RERobertaDatasetBase):

    def __getitem__(self, index):
        data = {}
        row = self.df.iloc[index]  

        head_token_ids = self.tokenizer.encode(row.h, add_special_tokens=False)
        tail_token_ids = self.tokenizer.encode(row.t, add_special_tokens=False)
        
        input_ids = [0] + head_token_ids + [2, 2] + tail_token_ids + [2]
        
        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            input_ids += [1] * pad_len
        
        data['text'] = load_document(self.dataset, row.doc_idx, cat=self.cat)
        data['input_ids'] = torch.tensor(input_ids)
        data['attention_mask'] = torch.where(torch.tensor(input_ids) != 1, torch.tensor(1), torch.tensor(0))
        data['evidence'] = str(row.evidence)
        data['head'] = row.h
        data['tail'] = row.t
        data['relation'] = torch.tensor(self.classes.index(row.r))
        data['doc_idx'] = row.doc_idx
        
        return data
    

class RERobertaDatasetRelMask(RERobertaDatasetBase):

    def __getitem__(self, index):
        data = {}
        row = self.df.iloc[index]  

        head_token_ids = self.tokenizer.encode("entity1", add_special_tokens=False)
        tail_token_ids = self.tokenizer.encode("entity2", add_special_tokens=False)
        
        doc, pos_indices = self.get_rel_text(row, self.cat)
        doc = doc.replace(row.h, 'entity1')
        doc = doc.replace(row.t, 'entity2')
        text_token_ids = self.tokenizer.encode(doc, 
                                               add_special_tokens=False,
                                               max_length=self.max_len,
                                               truncation=True)
        avail_len = self.max_len - (5 + len(head_token_ids) + len(tail_token_ids))
        input_ids = [0] + head_token_ids + [2, 2] + tail_token_ids + [2] + text_token_ids[:avail_len] + [2]
        
        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            input_ids += [1] * pad_len
        
        data['text'] = doc
        data['input_ids'] = torch.tensor(input_ids)
        data['attention_mask'] = torch.where(torch.tensor(input_ids) != 1, torch.tensor(1), torch.tensor(0))
        data['evidence'] = str(row.evidence)
        data['head'] = "entity1"
        data['tail'] = "entity2"
        data['relation'] = torch.tensor(self.classes.index(row.r))
        data['doc_idx'] = row.doc_idx
        data['pos_indices'] = str(pos_indices)
        
        return data

    
    def get_rel_text(self, row, cat):
        h, t = row.h, row.t
        sentences = np.array(sent_tokenize(load_document(self.dataset, row.doc_idx, cat)))
        valid_sentences = []
        pos_indices = []
        for i, sentence in enumerate(sentences):
            if h in sentence or t in sentence:
                valid_sentences.append(sentence)
                pos_indices.append(i)
        extracted_sentences = " ".join(valid_sentences)
        return extracted_sentences, pos_indices
    

class RERobertaDatasetRel(RERobertaDatasetBase):

    def __getitem__(self, index):
        data = {}
        row = self.df.iloc[index]  
        
        head_token_ids = self.tokenizer.encode(row.h, add_special_tokens=False)
        tail_token_ids = self.tokenizer.encode(row.t, add_special_tokens=False)

        doc, pos_indices = self.get_rel_text(row, self.cat)
        text_token_ids = self.tokenizer.encode(doc, 
                                               add_special_tokens=False,
                                               max_length=self.max_len,
                                               truncation=True)
        
        avail_len = self.max_len - (5 + len(head_token_ids) + len(tail_token_ids))
        input_ids = [0] + head_token_ids + [2, 2] + tail_token_ids + [2] + text_token_ids[:avail_len] + [2]
        
        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            input_ids += [1] * pad_len
            
        
        data['text'] = doc
        data['input_ids'] = torch.tensor(input_ids)
        data['attention_mask'] = torch.where(torch.tensor(input_ids) != 1, torch.tensor(1), torch.tensor(0))
        data['evidence'] = str(row.evidence)
        data['head'] = row.h
        data['tail'] = row.t
        data['relation'] = torch.tensor(self.classes.index(row.r))
        data['doc_idx'] = row.doc_idx
        data['pos_indices'] = str(pos_indices)
        
        return data
    

    def get_rel_text(self, row, cat):
        h, t = row.h, row.t
        words = h.split() + t.split()
        sentences = np.array(sent_tokenize(load_document(self.dataset, row.doc_idx, cat)))
        valid_sentences = []
        pos_indices = []
        for i, sentence in enumerate(sentences):
            if any([word in sentence for word in words]):
                valid_sentences.append(sentence)
                pos_indices.append(i)
        extracted_sentences = " ".join(valid_sentences)
        return extracted_sentences, pos_indices        
    
    
class RERobertaDatasetTopk(RERobertaDatasetBase):

    def __getitem__(self, index):
        data = {}
        row = self.df.iloc[index]  

        head_token_ids = self.tokenizer.encode(row.h, add_special_tokens=False)
        tail_token_ids = self.tokenizer.encode(row.t, add_special_tokens=False)
        
        text = load_document(self.dataset, row.doc_idx, cat=self.cat)
        text = self.get_topk_text(text, row)
        text_token_ids = self.tokenizer.encode(text, 
                                               add_special_tokens=False,
                                               max_length=self.max_len,
                                               truncation=True)
        
        avail_len = self.max_len - (5 + len(head_token_ids) + len(tail_token_ids))
        input_ids = [0] + head_token_ids + [2, 2] + tail_token_ids + [2] + text_token_ids[:avail_len] + [2]
        
        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            input_ids += [1] * pad_len
        
        data['text'] = load_document(self.dataset, row.doc_idx, cat=self.cat)
        data['topk_text'] = text
        data['input_ids'] = torch.tensor(input_ids)
        data['attention_mask'] = torch.where(torch.tensor(input_ids) != 1, torch.tensor(1), torch.tensor(0))
        data['evidence'] = str(row.evidence)
        data['head'] = row.h
        data['tail'] = row.t
        data['relation'] = torch.tensor(self.classes.index(row.r))
        data['doc_idx'] = row.doc_idx
        
        return data

    
    def get_topk_text(self, text, row):
        sentences = np.array(sent_tokenize(text))
        length = len(sentences)
        valid_sentences = []
        for i in range(len(row.evidence)):
            if i < length:
                valid_sentences.append(sentences[i])
        extracted_sentences = " ".join(valid_sentences)
        return extracted_sentences
    
  
class RERobertaDatasetTop2(RERobertaDatasetBase):

    def __getitem__(self, index):
        data = {}
        row = self.df.iloc[index]  

        head_token_ids = self.tokenizer.encode(row.h, add_special_tokens=False)
        tail_token_ids = self.tokenizer.encode(row.t, add_special_tokens=False)
        
        text = load_document(self.dataset, row.doc_idx, cat=self.cat)
        text = self.get_top2_text(text)
        text_token_ids = self.tokenizer.encode(text, 
                                               add_special_tokens=False,
                                               max_length=self.max_len,
                                               truncation=True)
        avail_len = self.max_len - (5 + len(head_token_ids) + len(tail_token_ids))
        input_ids = [0] + head_token_ids + [2, 2] + tail_token_ids + [2] + text_token_ids[:avail_len] + [2]
        
        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            input_ids += [1] * pad_len
        
        data['text'] = load_document(self.dataset, row.doc_idx, cat=self.cat)
        data['top2_text'] = text
        data['input_ids'] = torch.tensor(input_ids)
        data['attention_mask'] = torch.where(torch.tensor(input_ids) != 1, torch.tensor(1), torch.tensor(0))
        data['evidence'] = str(row.evidence)
        data['head'] = row.h
        data['tail'] = row.t
        data['relation'] = torch.tensor(self.classes.index(row.r))
        data['doc_idx'] = row.doc_idx
        
        return data
    
    def get_top2_text(self, text):
        
        sentences = np.array(sent_tokenize(text))
        length = len(sentences)
        valid_sentences = []
        if length >= 2:
            valid_sentences.extend([sentences[0], sentences[1]])
        elif length == 1:
            valid_sentences.append(sentences[0])
        extracted_sentences = " ".join(valid_sentences)
        
        return extracted_sentences


class RERobertaDatasetTop3(RERobertaDatasetBase):

    def __getitem__(self, index):
        data = {}
        row = self.df.iloc[index]  

        head_token_ids = self.tokenizer.encode(row.h, add_special_tokens=False)
        tail_token_ids = self.tokenizer.encode(row.t, add_special_tokens=False)
        
        text = load_document(self.dataset, row.doc_idx, cat=self.cat)
        text = self.get_top3_text(text)
        text_token_ids = self.tokenizer.encode(text, 
                                               add_special_tokens=False,
                                               max_length=self.max_len,
                                               truncation=True)
        
        avail_len = self.max_len - (5 + len(head_token_ids) + len(tail_token_ids))
        input_ids = [0] + head_token_ids + [2, 2] + tail_token_ids + [2] + text_token_ids[:avail_len] + [2]
        
        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            input_ids += [1] * pad_len
        
        data['text'] = load_document(self.dataset, row.doc_idx, cat=self.cat)
        data['top3_text'] = text
        data['input_ids'] = torch.tensor(input_ids)
        data['attention_mask'] = torch.where(torch.tensor(input_ids) != 1, torch.tensor(1), torch.tensor(0))
        data['evidence'] = str(row.evidence)
        data['head'] = row.h
        data['tail'] = row.t
        data['relation'] = torch.tensor(self.classes.index(row.r))
        data['doc_idx'] = row.doc_idx
        
        return data
    
    def get_top3_text(self, text):
        sentences = np.array(sent_tokenize(text))
        length = len(sentences)
        valid_sentences = []
        if length >= 3:
            valid_sentences.extend([sentences[0], sentences[1], sentences[2]])
        elif length == 2:
            valid_sentences.extend([sentences[0], sentences[1]])
        elif length == 1:
            valid_sentences.append(sentences[0])
        extracted_sentences = " ".join(valid_sentences)
        return extracted_sentences


def get_train_loader(train_df, train_data: json, Dataset: torch.utils.data.Dataset, args):

    train_loader = torch.utils.data.DataLoader(
        Dataset(args, train_df, train_data, cat='train', weakly=args.weakly),
        batch_size=args.per_gpu_train_batch_size, 
        shuffle=True, 
        num_workers=2
    )
    return train_loader


def get_val_loader(val_df, val_data: json, Dataset: torch.utils.data.Dataset, args):

    val_loader = torch.utils.data.DataLoader(
        Dataset(args, val_df, val_data, cat='val', weakly=False),
        batch_size=args.per_gpu_eval_batch_size, 
        shuffle=False, 
        num_workers=2
    )
    return val_loader


class RERoberta(nn.Module):
    def __init__(self, args, n_class, regularization=False):
        super(RERoberta, self).__init__()
        
        self.roberta = RobertaModel.from_pretrained(
            args.model_name_or_path, 
            output_attentions=True
        )
        self.dropout = nn.Dropout(0.1)
        self.regularization = regularization
        
        self.logits = nn.Linear(self.roberta.config.hidden_size, n_class)
        

    def forward(self, input_ids, attention_mask):
        _, cls, attention = self.roberta(input_ids, attention_mask, return_dict=False)
        logit = self.logits(cls)  
        
        if self.regularization:
            return logit, attention[-1][:, :, 0, :].mean(1)
        else:
            return logit


def loss_fn(logit, relation):
    ce_loss = nn.CrossEntropyLoss(reduction='mean')
    return ce_loss(logit, relation)

def load_standard_data(data, device):
    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)  
    relation = data['relation'].to(device)
    return input_ids, attention_mask, relation

def load_regu_data(data, device):
    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)   
    relation = data['relation'].to(device)
    regu_mask = data['regu_mask'].to(device)
    return input_ids, attention_mask, relation, regu_mask

def load_entro_data(data, device):
    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)   
    relation = data['relation'].to(device)

    reduced_input_ids = data['reduced_input_ids'].to(device)
    reduced_attention_mask = data['reduced_attention_mask'].to(device)
    flag = data['reduced_flag'].to(device)
    return input_ids, attention_mask, relation, reduced_input_ids, reduced_attention_mask, flag

def load_regu_entro_data(data, device):
    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)   
    relation = data['relation'].to(device)
    regu_mask = data['regu_mask'].to(device)
    
    reduced_input_ids = data['reduced_input_ids'].to(device)
    reduced_attention_mask = data['reduced_attention_mask'].to(device)
    flag = data['reduced_flag'].to(device)
    return input_ids, attention_mask, relation, regu_mask, reduced_input_ids, reduced_attention_mask, flag
