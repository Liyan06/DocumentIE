python evidence_ranking.py \
    --trained_model_name_or_path params/roberta_97_NA_40000.pth \
    --evidence_ranking_path evidence_ranking/docred_97_NA_40000_sent_rank_IG.json \
    --interpret_tool IG \
    --tokenizer_class roberta-base \
    --val_dir data/DocRed/dev.json \
    --dataset_name RERobertaDataset \
    --max_input_len 296 \
    --per_gpu_eval_batch_size 1 \
    --gpu_device 2 \

