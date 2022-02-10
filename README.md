# Making Document-Level Information Extraction Right for the Right Reasons

## Abstract

Document-level information extraction is a flexible framework compatible with applications where information is not necessarily localized in a single sentence. For example, key features of a diagnosis in radiology a report may not be explicitly stated, but nevertheless can be inferred from the report’s text. However, document-level neural models can easily learn spurious correlations from irrelevant information. This work studies how to ensure that these models make correct inferences from complex text and make those inferences in an auditable way: beyond just being right, are these models "right for the right reasons?" We experiment with post-hoc evidence extraction in a predict-select-verify framework using feature attribution techniques. While this basic approach can extract reasonable evidence, it can be regularized with small amounts of evidence supervision during training, which substantially improves the quality of extracted evidence. We evaluate on two domains: a small-scale labeled dataset of brain MRI reports and a large-scale modified version of DocRED (Yao et al., 2019) and show that models’ plausibility can be improved with no loss in accuracy.

Please see our full paper [here](https://arxiv.org/abs/2110.07686).

## Setup
This project runs on Python 3.6.\
numpy==1.19.5 \
torch==1.10.1 \
transformers==2.11.0 \
tqdm==4.62.3 \
nltk==3.6.7 \
captum==0.4.1 

DocRED dataset (Yao et al., 2019) can be downloaded from [thunlp/DocRED](https://github.com/thunlp/DocRED). The adapted DocRED we use in the paper is processed in `load_data.py`, and can be directly loaded for training and evaluation. Details can be found in `train.py`. We will provide our model checkpoint after the anonymous period for future fair comparisons.

## Training and Evaluation

Run `run_pipeline.sh` to train/evaluate models. Unless stated otherwise, commands in this repo show the mimimum rquired arguments. See `utils.py` for the complete set of arguments.

### Train
```
python train.py \
    --do_train \
    --train_type TRAIN_TYPE \
    --train_dir TRAIN_FILENAME_PATH \
    --val_dir VAL_FILENAME_PATH \
    --params_output_dir PARAM_OUTPUT_PATH \
    --params_output_name PARAM_OUTPUT_FILENAME \
    --log_path LOG_PATH 
```

`--train_type` can be `std, regu, entro, regu_entro`, where `regu_entro` has highest performance across both datasets in our experiments.

### Evaluation
```
python train.py \
    --do_eval \
    --val_dir VAL_FILENAME_PATH \
    --trained_model_name_or_path PATH_TO_SAVED_MODEL_PARAMS \
```

## Evidence Extraction and Evaluation

Evidence extraction consists of two steps. (1) We first use any attribution method to obtain a ranking of sentences through `evidence_ranking.py` for all inputs; (2) then select evidence sentences with our SUFFICIENT method via `sufficient_sent_selection.py`.

### Sentence Ranking
```
python evidence_ranking.py \
    --trained_model_name_or_path PATH_TO_SAVED_MODEL_PARAMS \
    --evidence_ranking_path PATH_TO_SAVE_RANKED_SENTENCES \
    --interpret_tool TOOL \
    --per_gpu_eval_batch_size 1 \
```

`--interpret_tool` can be any of the interpretation methods `IG, DL, IXG, LIME` mentioned in the paper. `LIME` and `IG` runs slower compared to the rest of interpretation methods.

### Evidence Selection
```
python sufficient_sent_selection.py \
    --trained_model_name_or_path PATH_TO_SAVED_MODEL_PARAMS \
    --evidence_ranking_path PATH_THAT_SAVED_RANKED_SENTENCES \
    --sufficient_sent_path PATH_TO_SAVE_SUFFICIENT_EVIDENCE \
    --per_gpu_eval_batch_size 1 \
```

`--per_gpu_eval_batch_size` is set to 1 for easier computations. 

### Evidence Evaluation
Once evidence sentences are collected by our SUFFICIENT method, use the following command to evaluate.

```
python evidence_f1_eval.py --sufficient_sent_path PATH_THAT_SAVED_SUFFICIENT_EVIDENCE
```
