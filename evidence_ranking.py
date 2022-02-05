from visualization import *
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from collections import Counter
import json


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


def get_ranked_sentence(text, tensor, attributions_sum):
    start, end = find_start_end_idx(tensor)
    token_offsets = get_token_offset(tensor.tolist()[start:end])
    sents_offset = get_sents_offsets(text)
        
    sent_sum_score = Counter()
    token_per_sent = Counter()
        
    for i, idx in enumerate(range(start, end)):
        sent_idx = locate_token_to_sent(token_offsets[i], sents_offset)        
        token_per_sent[sent_idx] += 1
        sent_sum_score[sent_idx] += abs(attributions_sum[idx].item())
                
    sent_mean_score = Counter()
    for sent_idx in sent_sum_score:
        sent_mean_score[sent_idx] = round(sent_sum_score[sent_idx] / token_per_sent[sent_idx], 5)
                
    sent_rank = [item[0] for item in sent_mean_score.most_common()]
    
    if None in sent_rank:
        # sometimes there are some wierd tokens in the text, which makes the offsets wrong. ~600
        sent_rank.remove(None)
    
    return sent_rank


def get_all_ranks(args, val_df, val_data, tool, mask=False):

    if mask:
        val_loader = get_val_loader(val_df, val_data, RERobertaDatasetMask, args)
    else:
        val_loader = get_val_loader(val_df, val_data, RERobertaDataset, args)
    
    rank_list = []
    for data in tqdm(val_loader):
        text = data['text']
        input_ids = data['input_ids'].to(args.device)
        attention_mask = data['attention_mask'].to(args.device)
        head = data['head']
        tail = data['tail']
    
        _, pred_idx, ref_input_ids = add_visualization_helper(args, input_ids, attention_mask, text, head, tail, tool)
    
        if args.interpret_tool == 'IG':
            attributions = tool.attribute(
                inputs=input_ids,
                target=pred_idx,
                baselines=ref_input_ids,
                additional_forward_args=(attention_mask),
                n_steps=30
            )
        elif args.interpret_tool == 'DL':
            attributions = tool.attribute(
                inputs=input_ids,
                target=pred_idx,
                baselines=ref_input_ids,
                additional_forward_args=(attention_mask),
            )
        elif args.interpret_tool == 'IXG':
            tool.model.set_inputs(input_ids, attention_mask)
            attributions = tool.attribute(
                inputs=tool.model.model.roberta.embeddings(input_ids),
                target=pred_idx, 
            )
        elif args.interpret_tool == 'LIME':
            prefix_end, postfix_idx = get_prefix_postfix_idx(input_ids)
            attributions = tool.attribute(
                inputs=input_ids,
                target=pred_idx,
                additional_forward_args=(attention_mask,),
                n_samples=100,
                show_progress=False,
                device=args.gpu_device,
                prefix_end=prefix_end,
                postfix_idx=postfix_idx,
                classifer=tool.model
            )
            
        attributions_sum = summarize_attributions(attributions)
        sent_rank = get_ranked_sentence(text[0], input_ids[0], attributions_sum)
        rank_list.append(sent_rank)


    return rank_list



if __name__ == "__main__":

    parser = common_args()
    parser.add_argument("--evidence_ranking_path", type=str, help="dir to save ranked sentences.")
    args = parser.parse_args()

    device = torch.device("cuda", args.gpu_device)
    args.device = device

    val_data = load_data(args.val_dir)
    val_df = load_df(args.val_dir)

    torch.cuda.empty_cache()
    model = RERoberta(args, n_class=len(MOST_COMMON_R), regularization=False)
    model.to(args.device)
    model.load_state_dict(torch.load(args.trained_model_name_or_path))
    model.eval();

    if args.interpret_tool == 'IXG':
        tool = InterpretationTool(args.interpret_tool, ModelWrapper(model))
    else:
        tool = InterpretationTool(args.interpret_tool, model)

    print(f"Start evidence sentences ranking with: {args.interpret_tool}.")
    result = get_all_ranks(args, val_df, val_data, tool, mask=False)
    with open(args.evidence_ranking_path, "w") as f:
        json.dump(result, f) 

    