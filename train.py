import os
import time
import json
from utils import seed_everything, common_args, EarlyStopping
from tqdm import tqdm

from models import *
from loadData import load_df, load_df_with_na

import pandas as pd
import warnings
from sklearn.metrics import f1_score
warnings.filterwarnings('ignore')


DATASETDICT = {
    "RERobertaDataset": RERobertaDataset,
    "RERobertaDatasetMask": RERobertaDatasetMask,
    "RERobertaDatasetShort": RERobertaDatasetShort,
    "RERobertaDatasetRelMask": RERobertaDatasetRelMask,
    "RERobertaDatasetRel": RERobertaDatasetRel,
    "RERobertaDatasetTopk": RERobertaDatasetTopk,
    "RERobertaDatasetTop2": RERobertaDatasetTop2,
    "RERobertaDatasetTop3": RERobertaDatasetTop3
}


def evaluate(args, model, val_loader):

    eval_output_dir = args.eval_output_dir

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    model.load_state_dict(torch.load(args.trained_model_name_or_path))
    model.eval()

    predictions = []
    probability = []

    for data in tqdm(val_loader):
        input_ids = data['input_ids'].to(args.device)
        attention_mask = data['attention_mask'].to(args.device)

        with torch.no_grad():
            logit = model(input_ids, attention_mask)
            pred_softmax = torch.nn.Softmax(1)(logit).cpu().detach().numpy()
            pred = torch.from_numpy(np.argmax(pred_softmax, axis=1))
            predictions.extend(pred.tolist())
            probability.extend(pred_softmax.tolist())

    predictions = [load_relation()[MOST_COMMON_R[p]] for p in predictions]

    truth = load_df(args.val_dir)['r'].tolist()
    acc = sum(np.array(truth) == np.array(predictions)) / len(truth)

    print(f"Eval Accuracy: {100*acc:.2f}%")
    print(f"Eval macro-F1: {100*f1_score(truth, predictions, average='macro'):.1f}%")
    print(f"Eval micro-F1: {100*f1_score(truth, predictions, average='micro'):.1f}%")

    if args.do_eval:
        with open(os.path.join(args.eval_output_dir, args.eval_output_name), 'w') as f:
            json.dump(predictions, f)

    return acc


def train(args, model, dataloaders_dict, train_type):
    
    es = EarlyStopping(patience=args.patience)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.num_warmup_steps, 
        num_training_steps=args.num_train_steps
    )
    
    model.zero_grad()
    torch.cuda.empty_cache()

    for epoch in tqdm(range(args.num_train_epochs)):

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            count = 0
            epoch_loss = 0.0

            for data in tqdm(dataloaders_dict[phase]):
                
                if train_type == 'std': input_ids, attention_mask, relation = load_standard_data(data, args.device)
                elif train_type == 'regu': input_ids, attention_mask, relation, regu_mask = load_regu_data(data, args.device)
                elif train_type == 'entro': input_ids, attention_mask, relation, reduced_input_ids, reduced_attention_mask, flag = load_entro_data(data, args.device)
                else: input_ids, attention_mask, relation, regu_mask, reduced_input_ids, reduced_attention_mask, flag = load_regu_entro_data(data, args.device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    if train_type == 'std': 
                        logit = model(input_ids, attention_mask)
                        loss = loss_fn(logit, relation)
                    elif train_type == 'regu': 
                        logit, att = model(input_ids, attention_mask) 
                        loss = loss_fn(logit, relation) - torch.log(att.mul(regu_mask).sum(1)+1e-30).mean()  
                    elif train_type == 'entro': 
                        logit = model(input_ids, attention_mask)
                        reduced_logit = model(reduced_input_ids, reduced_attention_mask)
                        reduced_pred_softmax = torch.nn.Softmax(1)(reduced_logit)
                        reduced_pred_log_softmax = torch.nn.LogSoftmax(1)(reduced_logit)
                        loss = loss_fn(logit, relation) + reduced_pred_softmax.mul(reduced_pred_log_softmax).sum(1).mul(flag).mean() * 0.1
                    else:
                        logit, att = model(input_ids, attention_mask) 
                        reduced_logit, _ = model(reduced_input_ids, reduced_attention_mask)
                        reduced_pred_softmax = torch.nn.Softmax(1)(reduced_logit)
                        reduced_pred_log_softmax = torch.nn.LogSoftmax(1)(reduced_logit)
                        loss = loss_fn(logit, relation) + \
                            reduced_pred_softmax.mul(reduced_pred_log_softmax).sum(1).mul(flag).mean() * 0.1 - \
                            torch.log(att.mul(regu_mask).sum(1)+1e-30).mean() 
                            
                            
                    pred_softmax = torch.nn.Softmax(1)(logit).cpu().detach().numpy()
                    pred = torch.from_numpy(np.argmax(pred_softmax, axis=1))
                    count += int(torch.sum(pred == relation.cpu()))

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                    epoch_loss += loss.item() * len(relation)    

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = count / len(dataloaders_dict[phase].dataset)
            
            print("Epoch {}/{} | {:^5} | Loss: {:.4f} | Acc: {:.4f}".format(
                epoch + 1, args.num_train_epochs, phase, epoch_loss, epoch_acc))

            with open(args.log_path, 'a') as f:
                f.write("Epoch {}/{} | {:^5} | Loss: {:.4f} | Acc: {:.4f}".format(
                    epoch + 1, args.num_train_epochs, phase, epoch_loss, epoch_acc))
                f.write(f"  {time.asctime()}\n")
                
        es(epoch_acc, model, args.params_output_dir, args.params_output_name)

        if es.early_stop:
            print(f"Best Accuracy: {es.show_best_acc():.4f}")
            break
        
        if epoch == args.num_train_epochs - 1:
            print(f"Best Accuray: {es.show_best_acc():.4f}")
            
    return es.show_best_acc()



if __name__ == '__main__':

    parser = common_args()
    args = parser.parse_args()

    device = torch.device("cuda", args.gpu_device)
    args.device = device

    seed_everything(args.seed)   

    if (os.path.exists(args.params_output_dir)
        and os.listdir(args.params_output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overwrite.".format(
                args.params_output_dir
            )
        )

    if (os.path.exists(args.params_output_dir)
        and os.listdir(args.params_output_dir)
        and args.do_train
        and args.overwrite_output_dir
    ):
        print("Overwrite previous parameters.")

    if not os.path.exists(args.params_output_dir):
        os.makedirs(args.params_output_dir)

    Dataset = DATASETDICT[args.dataset_name]
    train_type = args.train_type
    if train_type not in ('std', 'regu', 'entro', 'regu_entro'):
        raise AssertionError ("Invalid type. Choose from: std, regu, entro, regu_entro") 

    regularization = True if train_type in ('regu', 'regu_entro') else False
    model = RERoberta(args, n_class=len(MOST_COMMON_R), regularization=regularization)
    model.to(args.device)

    
    val_data = load_data(args.val_dir)
    val_df = load_df_with_na(args.val_dir, cat='val', loader_size=len(load_df(args.val_dir)), proportion=0)

    if args.do_train:
        train_data = load_data(args.train_dir)
        train_df = load_df_with_na(args.train_dir, 'train', len(load_df(args.train_dir)), 0.5).sample(len(load_df(args.train_dir)), random_state=42)   

        dataloaders_dict = {}
        dataloaders_dict['train'] = get_train_loader(train_df, train_data, Dataset, args)
        dataloaders_dict['val'] = get_val_loader(val_df, val_data, Dataset, args)

        train(args, model, dataloaders_dict, train_type)

    if args.do_eval:
        evaluate(args, model, get_val_loader(val_df, val_data, Dataset, args))