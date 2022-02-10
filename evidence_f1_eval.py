from utils import common_args
from load_data import load_df
import json
from sklearn.utils import resample
from tqdm import tqdm


def get_macro_f1_R_P_score(truth_dict, pred_dict, verbose=True):
    recall = []
    precision = []
    f1 = []
    for key in pred_dict.keys():
        if len(truth_dict[key]) != 0 and len(pred_dict[key]) != 0:
            recall.append(len(set(truth_dict[key]).intersection(set(pred_dict[key]))) / len(truth_dict[key]))
            precision.append(len(set(truth_dict[key]).intersection(set(pred_dict[key]))) / len(pred_dict[key]))
    
    recall = [num for num in recall] 
    precision = [num for num in precision]
    
    for i in range(len(recall)):
        if precision[i] + recall[i] == 0:
            f1.append(0)
        else:
            f1.append(2 * precision[i] * recall[i] / (precision[i] + recall[i]))
    macro_f1 = round(sum(f1) / len(f1) * 100, 1)
    macro_pre = round(sum(precision) / len(precision) * 100, 1)
    macro_recall = round(sum(recall) / len(recall)* 100, 1)
    
    if verbose:
        print(f"Macro:   F1: {macro_f1}, Precision: {macro_pre}, Recall: {macro_recall}")
    return macro_f1, macro_pre, macro_recall


def get_micro_f1_R_P_score(truth_dict, pred_dict, verbose=True):
    
    num_correct_preds = 0
    num_truth = 0
    num_pred = 0
    
    for key in pred_dict.keys():
        if len(truth_dict[key]) != 0 and len(pred_dict[key]) != 0:
            num_correct_preds += len(set(truth_dict[key]).intersection(set(pred_dict[key])))
            num_truth += len(truth_dict[key])
            num_pred += len(pred_dict[key])
    
    recall = round(num_correct_preds / num_truth * 100, 1)
    precision = round(num_correct_preds / num_pred * 100, 1)
    f1 = round(2 * recall * precision / (recall + precision), 1)
    
    if verbose:
        print(f"Micro:   F1: {f1}, Precision: {precision}, Recall: {recall}")
    return f1, precision, recall


def f1_eval(args, evi_pred, evi_truth=None, all_evi=True, verbose=True):

    if evi_truth:
        evi_truth_dict = evi_truth
    else:
        evi_truth_dict = dict(enumerate(load_df(args.val_dir).evidence.values.tolist()))
    
    evi_pred_dict = {}

    if all_evi:
        for i, evi in enumerate(evi_pred):
            evi_pred_dict[i] = evi
    else:
        for i, evi in enumerate(evi_pred):
            if None in evi:
                evi.remove(None)
            evi_pred_dict[i] = evi[:len(evi_truth_dict[i])]

    macro_f1, macro_pre, macro_recall = get_macro_f1_R_P_score(evi_truth_dict, evi_pred_dict, verbose=verbose)

    evi_len = [len(evi) for evi in evi_pred]

    if verbose:
        print(f"Average Evidence Length: {round(sum(evi_len) / len(evi_len), 2)}.")

    return macro_f1, macro_pre, macro_recall


def DocRED_boostrap_evidence_p_value(baseline_evis, better_evis, truth, times):
    count = 0
    
    for t in tqdm(range(times)):
        sampled_indices = resample(range(len(truth)), replace=True, n_samples=len(truth), random_state=t)
        sampled_base_evis = [baseline_evis[idx] for idx in sampled_indices]
        sampled_both_evis = [better_evis[idx] for idx in sampled_indices]
        sampled_truth = [truth[idx] for idx in sampled_indices]

        base_f1, _ , _ = f1_eval(args, sampled_base_evis, sampled_truth, all_evi=True, verbose=False)
        better_f1, _ , _ = f1_eval(args, sampled_both_evis, sampled_truth, all_evi=True, verbose=False)
        
        if better_f1 > base_f1:
            count += 1

    print(count)
    return 1 - count / times


if __name__  == "__main__":

    parser = common_args()
    parser.add_argument("--sufficient_sent_path", type=str, help="dir to save sufficient sentences.")
    args = parser.parse_args()

    sufficient_sent_path = args.sufficient_sent_path
    with open(sufficient_sent_path, "r") as f:
        evi_pred = json.load(f)

    f1_eval(args, evi_pred, all_evi=True, verbose=False)
    