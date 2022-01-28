import torch
import numpy as np
import random, os
from transformers import RobertaTokenizer
from nltk.tokenize import sent_tokenize

import argparse


tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        
        
def find_start_end_idx(tensor):
    '''
    Start finding top attribution tokens after two entities and before the last </s>.
    '''
    
    def find_start_idx():
        count = 0
        for i, num in enumerate(tensor):
            if num == 2:
                count += 1
            if count == 3:
                return i + 1
    def find_end_idx():
        count = 0
        for i, num in enumerate(tensor):
            if num == 2:
                count += 1
            if count == 4:
                return i

    return find_start_idx(), find_end_idx()
        
    
def get_token_offset(encoding):
    offsets = []; idx = 0; t = ''
    for en in encoding:
        w = tokenizer.convert_ids_to_tokens(en)
        if w == "<unk>":
            offsets.append((idx, idx + 1))
            idx += 1
        else:
            offsets.append((idx, idx + len(w)))
            idx += len(tokenizer.decode(en))   #len(w)

    return offsets


def get_sent_offset(text, sent):
    start = text.find(sent)
    end = start + len(sent)
    return start, end


def get_sents_offsets(text):
    sents = sent_tokenize(text)
    l = []
    for sent in sents:
        start, end = get_sent_offset(text, sent)
        l.append((start, end))
    return l


def locate_token_to_sent(token_offset, sents_offset):
    for i in range(len(sents_offset)):
        if token_offset[1] <= sents_offset[0][1]:
            return 0
        elif sents_offset[i-1][1] < token_offset[1] <= sents_offset[i][1]:
            return i
        
        
def get_att_regu_mask(text, input_ids, evidence, relation):

    start, end = find_start_end_idx(input_ids)
    evidence = eval(evidence)
    relation = int(relation)
    isNA = int(relation) == 96
    
    token_offsets = get_token_offset(input_ids.tolist()[start:end])
    sents_offset = get_sents_offsets(text)

    if isNA or (len(evidence) == 0):
        att = torch.ones_like(input_ids)
    else:
        att = torch.zeros_like(input_ids)
        for i, idx in enumerate(range(start, end)):
            sent_idx = locate_token_to_sent(token_offsets[i], sents_offset)
            if sent_idx in evidence:
                att[idx] = 1
            
    return att
        

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience, verbose=True, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, acc, model, model_path, model_name):

        score = acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(acc, model, model_path, model_name)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(acc, model, model_path, model_name)
            self.counter = 0

    def save_checkpoint(self, acc, model, model_path, model_name):
        '''Saves model when validation score improve.'''
        torch.save(model.state_dict(), os.path.join(model_path, model_name))
        self.val_loss_min = acc
        
    def show_best_acc(self):
        return self.best_score


def common_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dir", default="data/DocRed/train_annotated.json")
    parser.add_argument("--val_dir", default="data/DocRed/dev.json")
    parser.add_argument("--test_dir", default="data/DocRed/test.json")

    parser.add_argument(
        "--train_type",
        type=str,
        help="Train type selected in the list: ['std', 'regu', 'entro', 'regu_entro']",
    )
    parser.add_argument(
        "--dataset_name",
        default="RERobertaDataset",
        type=str,
        help="choose from ['RERobertaDataset', 'RERobertaDatasetMask', 'RERobertaDatasetShort', \
                           'RERobertaDatasetRelMask', 'RERobertaDatasetRel', 'RERobertaDatasetTopk', \
                           'RERobertaDatasetTop2', 'RERobertaDatasetTop3']",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="roberta-base",
        type=str,
        help="Check path to pre-trained model or shortcut name",
    )
    parser.add_argument(
        "--tokenizer_class",
        default="roberta-base",
        type=str,
        help="Check path to pre-trained model or shortcut name",
    )
    parser.add_argument(
        "--trained_model_name_or_path",
        type=str,
        help="model fine-tuned on the data",
    )

    parser.add_argument(
        "--input_dir",
        default=None,
        type=str,
        help="Check path to pre-trained model or shortcut name",
    )
    parser.add_argument(
        "--params_output_dir",
        default='./params/',
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--params_output_name", default='./params/', type=str, help="The output name.")
    parser.add_argument(
        "--eval_output_dir",
        default='./predictions/relation_pred/',
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--eval_output_name", default='./predictions/relation_pred/', type=str, help="The output name.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run eval on the test set.")

    parser.add_argument("--n_gpu", default=1, type=int, help="Number of Gpu to use.", )
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size training.", )
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size evaluation.", )
    parser.add_argument("--gpu_device", default=0, type=int, help="gpu device")
    parser.add_argument("--gpu_device_ids", nargs="+", type=int, default=[0, 1, 2, 3], help="gpu device")

    parser.add_argument("--max_input_len", default=296, type=int, help="Max gradient norm.")
    parser.add_argument("--weakly", action="store_true", help="Whether to run training.")

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="No. steps before backward pass.",)
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--beta1", default=0.9, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--beta2", default=0.999, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=25, type=int, help="Total number of training epochs", )
    parser.add_argument("--num_train_steps", default=40000, type=int, help="Number of training steps.",)
    parser.add_argument("--num_warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X updates steps.")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the output directory", )
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached data sets", )
    parser.add_argument("--predict", action="store_true", help="Generate summaries for dev set", )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--patience", type=int, default=42, help="set patience")
    parser.add_argument("--log_path", type=str, help="set up log path to save training stats.")
   
    return parser