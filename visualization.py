import captum
from captum.attr._utils.input_layer_wrapper import ModelInputWrapper
from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization, LayerDeepLift, InputXGradient, LimeBase
from models import RERoberta
from transformers import (RobertaModel, RobertaTokenizer)
import torch
import torch.nn as nn

from loadData import MOST_COMMON_R, load_data, load_df
from utils import common_args
from models import get_val_loader, RERobertaDataset, RERobertaDatasetMask


# parser = common_args()
# args = parser.parse_args()
# model = RERoberta(args, n_class=len(MOST_COMMON_R), regularization=False)
# model.cuda()

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

ref_token_id = tokenizer.mask_token_id
sep_token_id = tokenizer.sep_token_id 
cls_token_id = tokenizer.cls_token_id
pad_token_id = tokenizer.pad_token_id


def view_rows(args, val_df, val_data, start, end, mask=False):

    if mask:
        val_loader = get_val_loader(val_df, val_data, RERobertaDatasetMask, args)
    else:
        val_loader = get_val_loader(val_df, val_data, RERobertaDataset, args)
    
    count = 0
    request_list = []
    for data in val_loader:
        if count in range(start, end + 1): 
            text = data['text']
            input_ids = data['input_ids'].to(args.device)
            attention_mask = data['attention_mask'].to(args.device)
            head = data['head']
            relation = data['relation']
            tail = data['tail']
            evidence = eval(data['evidence'][0])
            doc_idx = data['doc_idx']
            request_list.append([args, input_ids, attention_mask, text, evidence, head, relation, tail, doc_idx])
            count += 1
        else:
            count += 1
    return request_list


def view_rows_list(args, val_df, val_data, alist, mask=False):

    if mask:
        val_loader = get_val_loader(val_df, val_data, RERobertaDatasetMask, args)
    else:
        val_loader = get_val_loader(val_df, val_data, RERobertaDataset, args)
    
    count = 0
    request_list = []
    for data in val_loader:
        if count in alist: 
            text = data['text']
            input_ids = data['input_ids'].to(args.device)
            attention_mask = data['attention_mask'].to(args.device)
            head = data['head']
            relation = data['relation']
            tail = data['tail']
            evidence = eval(data['evidence'][0])
            doc_idx = data['doc_idx']
            request_list.append([args, input_ids, attention_mask, text, evidence, head, relation, tail, doc_idx])
            count += 1
        else:
            count += 1
        break
    return request_list


def squad_pos_forward_func(input_ids, attention_mask):
    logit = model(input_ids, attention_mask)
    return logit.max(1).values


def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


class ModelWrapper(nn.Module):
    
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def set_inputs(self, input_ids, attention_mask):
        self.input_ids = input_ids

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.model.roberta.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        self.extended_attention_mask = extended_attention_mask
        
    def forward(self, embeddings):   

        encoding_output = self.model.roberta.encoder(embeddings, self.extended_attention_mask)['last_hidden_state']
        logit = self.model.logits(self.model.roberta.pooler(encoding_output))

        return logit


class InterpretationTool:
    """
    Interpretation tool wrapper.
    Valid interpretation methods include:
    IntegratedGradients, DeepLift, InputXGradient, LIME
    """

    def __init__(self, tool_name, model):
        self.tool_names = ['IG', 'DL', 'IXG', 'LIME']
        self.tool_name = tool_name
        assert self.tool_name in self.tool_names, 'Invalid tool name.'
           
        self.model = model        
        self.set_tool()

    def _squad_pos_forward_func(self, input_ids, attention_mask):
        logit = self.model(input_ids, attention_mask)
        return logit.max(1).values

    def set_tool(self):
        if self.tool_name == 'IG':
            self.tool = LayerIntegratedGradients(self.model, self.model.roberta.embeddings)
        elif self.tool_name == 'DL':
            self.tool = LayerDeepLift(self.model, self.model.roberta.embeddings)
        elif self.tool_name == 'IXG':
            self.tool = InputXGradient(self.model)
        elif self.tool_name == 'LIME':
            pass
    
    def attribute(self, **params):
        return self.tool.attribute(**params)


def add_visualization_helper(args, input_ids, attention_mask, text, head, tail, tool):

    head_token_ids = tokenizer.encode(head, add_special_tokens=False)
    tail_token_ids = tokenizer.encode(tail, add_special_tokens=False)
    text_token_ids = tokenizer.encode(text, 
                                      add_special_tokens=False,
                                      max_length=args.max_input_len,
                                      truncation=True)
    avail_len = args.max_input_len - (5 + len(head_token_ids) + len(tail_token_ids))
    ref_input_ids = [cls_token_id] + [ref_token_id] * len(head_token_ids) + [sep_token_id] + \
                    [sep_token_id] + [ref_token_id] * len(tail_token_ids) + [sep_token_id] + \
                    [ref_token_id] * len(text_token_ids[:avail_len]) + [sep_token_id]
    pad_len = args.max_input_len - len(ref_input_ids)
    if pad_len > 0:
        ref_input_ids += [pad_token_id] * pad_len
    ref_input_ids = torch.tensor(ref_input_ids).to(args.device).unsqueeze(dim=0)
    
    if args.interpret_tool == 'IXG':
        logit = tool.model.model(input_ids, attention_mask)
    else:
        logit = tool.model(input_ids, attention_mask)

    pred_idx = int(torch.softmax(logit[0], dim=0).argmax().cpu().detach())

    return logit, pred_idx, ref_input_ids


def add_visualization(args, input_ids, attention_mask, text, evidence, head, relation, tail, doc_idx, tool):
    
    
    logit, pred_idx, ref_input_ids = add_visualization_helper(args, input_ids, attention_mask, text, head, tail, tool)
    
    if args.interpret_tool == 'IG':
        attributions, delta = tool.attribute(
            inputs=input_ids,
            target=pred_idx,
            baselines=ref_input_ids,
            additional_forward_args=(attention_mask),
            return_convergence_delta=True,
            n_steps=30
        )
    elif args.interpret_tool == 'DL':
        attributions, delta = tool.attribute(
            inputs=input_ids,
            target=pred_idx,
            baselines=ref_input_ids,
            additional_forward_args=(attention_mask),
            return_convergence_delta=True,
        )
    elif args.interpret_tool == 'IXG':
        tool.model.set_inputs(input_ids, attention_mask)
        attributions = tool.attribute(
            inputs=tool.model.model.roberta.embeddings(input_ids),
            target=pred_idx, 
        )
        delta = None

    attributions_sum = summarize_attributions(attributions)
    
    vis_data_record = visualization.VisualizationDataRecord(
        word_attributions=attributions_sum,
        pred_prob=torch.max(torch.softmax(logit[0], dim=0)),
        pred_class=MOST_COMMON_R[pred_idx],
        true_class=MOST_COMMON_R[relation],                         
        attr_class=evidence,   
        attr_score=attributions_sum.sum(),       
        raw_input=[tokenizer.decode([i]) for i in input_ids[0]],
        convergence_score=delta
    )
    
    return vis_data_record


if __name__ == "__main__":

    parser = common_args()
    args = parser.parse_args()

    args.per_gpu_eval_batch_size = 1
    device = torch.device("cuda", args.gpu_device)
    args.device = device
    args.interpret_tool = "IXG"

    val_data = load_data(args.val_dir)
    val_df = load_df(args.val_dir)

    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_class)
    ref_token_id = tokenizer.mask_token_id
    sep_token_id = tokenizer.sep_token_id 
    cls_token_id = tokenizer.cls_token_id
    pad_token_id = tokenizer.pad_token_id

    model = RERoberta(args, n_class=len(MOST_COMMON_R), regularization=False)
    model.to(args.device)

    model.load_state_dict(torch.load(f"params/docred_97_NA_40000/docred_97_NA_40000.pth"))
    model.eval();

    if args.interpret_tool == 'IXG':
        tool = InterpretationTool(args.interpret_tool, ModelWrapper(model))
    else:
        tool = InterpretationTool(args.interpret_tool, model)
        
    vis_data_records_ig=[]
    for instance in view_rows_list(args, val_df, val_data, [0], mask=False):
        instance.append(tool)
        vis_data_records_ig.append(add_visualization(*instance))