import json
import numpy as np
import pandas as pd
import random
from collections import Counter
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import sent_tokenize

from typing import Dict, List, Tuple

def load_relation() -> List:
    with open('./data/DocRed/rel_info.json', 'r') as f:
        relation_dict = json.load(f)
        relation_dict['NA'] = 'NA'
    return relation_dict

RELATION_DICT = load_relation()

def load_data(data_path) -> json:
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data
    

def sort_relations() -> List:
    with open('./data/DocRed/train_annotated.json', 'r') as f:
        train_data = json.load(f)

    all_relations = []
    for i in range(len(train_data)):
        for item in train_data[i]['labels']:
            all_relations.append(item['r'])
    return [item[0] for item in Counter(all_relations).most_common(96)] + ["NA"]


# MOST_COMMON_R= ['P17', 'P131', 'P27', 'P150', 'P577', 'P175', 'P569', 'P570', 'P527', 'P161']
MOST_COMMON_R = sort_relations()


def load_tuples(data, idx, most_common_r=MOST_COMMON_R, relation_dict=RELATION_DICT):
    '''Load tuples from one document. Longest mention is selected for each entity.'''
    tuples = []

    for i in range(len(data[idx]['labels'])):
        head_idx = data[idx]['labels'][i]['h']
        tail_idx = data[idx]['labels'][i]['t']
        relation = data[idx]['labels'][i]['r']
        evidence = data[idx]['labels'][i]['evidence']
        head = max(list({item['name'] for item in data[idx]['vertexSet'][head_idx]}), key=len)
        tail = max(list({item['name'] for item in data[idx]['vertexSet'][tail_idx]}), key=len)
        if relation in most_common_r:
            tuples.append((idx, evidence, head, relation_dict[relation], tail))
    return tuples


def load_document(data, idx, cat):
    '''Untokenization'''
    if cat == 'train':
        return " ".join([TreebankWordDetokenizer().detokenize(i) for i in data[idx]['sents']])
    elif cat == 'val':
        return " ".join([TreebankWordDetokenizer().detokenize(i) for i in data[idx]['sents']])
    else:
        return " ".join([TreebankWordDetokenizer().detokenize(i) for i in data[idx]['sents']])


def load_all_tuples(data) -> Tuple : 
    '''
    Load tuples from all documents.
    format: (doc_idx, evidence, (h, r, t))
    
    '''
    all_tuples = []
        
    for idx in range(len(data)):
        all_tuples.extend(load_tuples(data, idx))

    return all_tuples


def substr_idx(source, alist):
    for idx, string in enumerate(alist):
        if string in source:
            return idx
        

def get_entity_list(data, idx):
    mentions = []
    for item in load_tuples(data, idx):
        mentions.append(item[2])
        mentions.append(item[4])
    mentions = list(set(mentions))
    
    entity_list = []
    for mention in mentions:
        if all(mention not in entity for entity in entity_list):
            if any(entity in mention for entity in entity_list):
                entity_list.remove(entity_list[substr_idx(mention, entity_list)])
                entity_list.append(mention)
            else:
                entity_list.append(mention)
    return entity_list


def find_entity(mention, ent_list):
    for ent in ent_list:
        if mention in ent:
            return ent


def generate_na_tuples(data, idx):
    
    entity_list = get_entity_list(data, idx)
    valid_pairs = [(find_entity(item[2], entity_list), find_entity(item[4], entity_list)) 
                   for item in load_tuples(data, idx)]
    
    head_list = []
    tail_list = []
    for item in load_tuples(data, idx):
        if find_entity(item[2], entity_list) not in head_list:
            head_list.append(find_entity(item[2], entity_list))
        if find_entity(item[4], entity_list) not in tail_list:
            tail_list.append(find_entity(item[4], entity_list))
    
    na_pairs = []
    for h in head_list:
        for t in tail_list:
            if (h, t) not in valid_pairs and (t, h) not in valid_pairs:
                if h not in t and t not in h:
                    na_pairs.append((idx, [], h, "NA", t))
    
    return na_pairs


def generate_all_na_tuples(data):
    all_na_tuples = []
        
    for idx in range(len(data)):
        all_na_tuples.extend(generate_na_tuples(data, idx))
        
    return all_na_tuples


def load_df(data_dir):
    data = load_data(data_dir)
    return pd.DataFrame(load_all_tuples(data), columns=['doc_idx', 'evidence', 'h', 'r', 't'])


def get_processed_df(data_dir):
    entity_relation_dict = {}
    for idx, row in load_df(data_dir).iterrows():
        if f"({row.h}, {row.t})" not in entity_relation_dict.keys():
            entity_relation_dict[f"({row.h}, {row.t})"] = [[row.r], [idx]]
        else:
            if row.r not in entity_relation_dict[f"({row.h}, {row.t})"][0]:
                entity_relation_dict[f"({row.h}, {row.t})"][0].append(row.r)
                entity_relation_dict[f"({row.h}, {row.t})"][1].append(idx)

    std_df_merge = pd.DataFrame({"(h,t)": list(entity_relation_dict.keys()),
                                 "r": [r for r, idx in list(entity_relation_dict.values())],
                                 "indices": [idx for r, idx in list(entity_relation_dict.values())]
                                })
    std_df_merge["num_r"] = std_df_merge['indices'].apply(lambda x: len(x))
    std_multi_r_df = std_df_merge[std_df_merge['num_r'] > 1]
    return std_df_merge, std_multi_r_df


def load_na_df(data, loader_size, proportion):
    random.seed(42)
    return pd.DataFrame(random.sample(generate_all_na_tuples(data), int(loader_size*proportion)), 
                        columns=['doc_idx', 'evidence', 'h', 'r', 't'])


def load_df_with_na(data_dir, cat, loader_size, proportion):
    data = load_data(data_dir)
    df1 = load_na_df(data, loader_size=loader_size, proportion=proportion)
    if cat == 'train':
        df2 = load_df(data_dir).sample(loader_size, random_state=42)
    else:
        df2 = load_df(data_dir)
    return pd.concat([df1, df2])


def load_document_rel(idx, cat):
    """
    For each instance, get sentences with two entities
    """
    if cat == 'train':
        with open("./data/document_rel_train.json", "r") as f:
            data = json.load(f)
    else:
        with open("./data/document_rel_val.json", "r") as f:
            data = json.load(f)
    return data[idx]


def load_document_topk(idx, cat):
    """
    For each instance: get top k sentences (k is the number of evidence sentences)
    """
    if cat == 'train':
        with open("../data/document_topk_train.json", "r") as f:
            data = json.load(f)
    else:
        with open("../data/document_topk_val.json", "r") as f:
            data = json.load(f)
    return data[idx]


def load_document_top(idx, cat, k):
    """
    For each instance: get top k sentences
    """
    if cat == 'train':
        with open(f"../data/document_top{k}_train.json", "r") as f:
            data = json.load(f)
    else:
        with open(f"../data/document_top{k}_val.json", "r") as f:
            data = json.load(f)
    return data[idx]


def get_partial_document(data_dir, cat):
    """
    For each instance, get sentences with two entities
    """
    df = load_df(data_dir)
    documents = []
    evidences = []

    data = load_data(data_dir)

    for idx, row in df.iterrows():
        h, t = row.h, row.t
        sentences = np.array(sent_tokenize(load_document(data, row.doc_idx, cat)))
        valid_sentences = []
        valid_evd = []
        for i, sentence in enumerate(sentences):
            if h in sentence or t in sentence:
                valid_sentences.append(sentence)
                valid_evd.append(i)
        extracted_sentences = " ".join(valid_sentences)
        documents.append(extracted_sentences)
        evidences.append(valid_evd)
    return documents, evidences    


def get_document_top_k(data_dir, cat):
    """
    For each instance: get top k sentences (k is the number of evidence sentences)
    """
    df = load_df(data_dir)
    documents = []
    evidences = []

    data = load_data(data_dir)

    for idx, row in df.iterrows():
        sentences = np.array(sent_tokenize(load_document(data, row.doc_idx, cat)))
        length = len(sentences)
        valid_sentences = []
        for i in range(len(row.evidence)):
            if i < length:
                valid_sentences.append(sentences[i])
        extracted_sentences = " ".join(valid_sentences)
        documents.append(extracted_sentences)
    return documents


def get_document_top_2(data_dir, cat):
    """
    For each instance: get top k sentences (k is the number of evidence sentences)
    """
    df = load_df(data_dir)
    documents = []
    evidences = []

    data = load_data(data_dir)

    for idx, row in df.iterrows():
        sentences = np.array(sent_tokenize(load_document(data, row.doc_idx, cat)))
        length = len(sentences)
        valid_sentences = []
        if length >= 2:
            valid_sentences.extend([sentences[0], sentences[1]])
        elif length == 1:
            valid_sentences.append(sentences[0])
        extracted_sentences = " ".join(valid_sentences)
        documents.append(extracted_sentences)
    return documents


def get_document_top_3(data_dir, cat):
    """
    For each instance: get top k sentences (k is the number of evidence sentences)
    """
    df = load_df(data_dir)
    documents = []
    evidences = []

    data = load_data(data_dir)

    for idx, row in df.iterrows():
        sentences = np.array(sent_tokenize(load_document(data, row.doc_idx, cat)))
        length = len(sentences)
        valid_sentences = []
        if length >= 3:
            valid_sentences.extend([sentences[0], sentences[1], sentences[2]])
        elif length == 2:
            valid_sentences.extend([sentences[0], sentences[1]])
        elif length == 1:
            valid_sentences.append(sentences[0])
        extracted_sentences = " ".join(valid_sentences)
        documents.append(extracted_sentences)
    return documents