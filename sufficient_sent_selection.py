import numpy as np
import torch
from load_data import load_data
from models import get_val_loader, RERoberta

from load_data import load_df, load_relation, MOST_COMMON_R
import json
from evidence_ranking import find_start_end_idx, get_token_offset, get_sents_offsets, locate_token_to_sent
from tqdm import tqdm
from utils import common_args
from train import DATASETDICT


def extract_enough_sentence_org(args, val_idx, input_ids, k, evidence, text):

    start, end = find_start_end_idx(input_ids)
    evi = evidence[val_idx]
    evi = evi[:k]
    
    token_offsets = get_token_offset(input_ids.tolist()[start:end])
    sents_offset = get_sents_offsets(text)
    
    new_input_ids = input_ids[:start].tolist()
    for i, idx in enumerate(range(start, end)):
        sent_idx = locate_token_to_sent(token_offsets[i], sents_offset)
        if sent_idx in evi:
            new_input_ids.append(input_ids[idx].item())

    new_input_ids += [2] 
    pad_len = args.max_input_len - len(new_input_ids)
    if pad_len > 0:
        new_input_ids += [1] * pad_len
        
    new_input_ids = torch.tensor(new_input_ids).unsqueeze(0)
    new_attention_mask = torch.where(new_input_ids.clone().detach() != 1, torch.tensor(1), torch.tensor(0)).unsqueeze(0)
        
    return new_input_ids.to(args.device), new_attention_mask.to(args.device)


if __name__ == '__main__':
    
    parser = common_args()
    parser.add_argument("--evidence_ranking_path", type=str, help="dir to save ranked sentences.")
    parser.add_argument("--sufficient_sent_path", type=str, help="dir to save sufficient sentences.")
    args = parser.parse_args()

    device = torch.device("cuda", args.gpu_device)
    args.device = device

    Dataset = DATASETDICT[args.dataset_name]
    relation_dict = load_relation()

    val_data = load_data(args.val_dir)
    val_df = load_df(args.val_dir)
    val_loader = get_val_loader(val_df, val_data, Dataset, args)
    
    model = RERoberta(args, n_class=len(MOST_COMMON_R), regularization=False)
    model.to(args.device)
    model.load_state_dict(torch.load(args.trained_model_name_or_path))
    model.eval();

    with open(args.evidence_ranking_path, "r") as f:
        evidence = json.load(f)

    predictions = []
    refined_evi = []
    probs = []

    for val_idx, data in enumerate(tqdm(val_loader)):

        input_ids = data['input_ids']
        attention_mask = data['attention_mask']
        text = data['text']

        with torch.no_grad():
            logit = model(input_ids.to(args.device), attention_mask.to(args.device))
            pred_softmax = torch.nn.Softmax(1)(logit).cpu().detach().numpy()
            pred = torch.from_numpy(np.argmax(pred_softmax, axis=1))

            model_prediction = pred.item()
            model_prob = float(pred_softmax[0][np.argmax(pred_softmax, axis=1)])

            k = 1; pred_label = -999; selected_evi = []; flag = True
            while flag:
                new_input_ids, new_attention_mask = extract_enough_sentence_org(args, val_idx, input_ids[0], k, evidence, text[0])
                logit = model(new_input_ids, new_attention_mask)
                pred_softmax = torch.nn.Softmax(1)(logit).cpu().detach().numpy()
                pred = torch.from_numpy(np.argmax(pred_softmax, axis=1))
                pred_label = pred.item()
                prob = float(pred_softmax[0][np.argmax(pred_softmax, axis=1)])
                selected_evi = evidence[val_idx][:k]
                k += 1
                flag = pred_label != model_prediction or prob < 0.8 * model_prob
                if k > len(evidence[val_idx]) or (len(selected_evi) > 5 and pred_label == model_prediction):
                    flag = False
            refined_evi.append(selected_evi)
            probs.append(prob)
            predictions.append(relation_dict[MOST_COMMON_R[pred_label]])
        assert(len(predictions) == val_idx + 1 == len(refined_evi))

    with open(args.sufficient_sent_path, 'w') as f:
        json.dump(refined_evi, f)