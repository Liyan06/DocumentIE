python train.py \
    --do_eval \
    --val_dir data/DocRed/dev.json \
    --trained_model_name_or_path params/docred/roberta_97_NA_40000.pth \
    --max_input_len 296 \
    --dataset_name RERobertaDataset \
    --per_gpu_train_batch_size 8 \
    --gpu_device 0 \
    --eval_output_dir ./predictions/relation_pred/ \
    # --eval_output_name docred_97_NA_40000_regu_entro_weak_relation_pred.txt \


# python train.py \
#     --do_train \
#     --train_dir data/DocRed/train_annotated.json \
#     --val_dir data/DocRed/dev.json \
#     --weakly \
#     --train_type regu_entro \
#     --model_name_or_path roberta-base \
#     --tokenizer_class roberta-base \
#     --max_input_len 296 \
#     --dataset_name RERobertaDataset \
#     --params_output_dir ./params/docred_97_NA_40000_regu_entro_week/ \
#     --params_output_name docred_97_NA_40000_regu_entro_week.pth \
#     --num_train_epochs 25 \
#     --per_gpu_train_batch_size 8 \
#     --patience 3 \
#     --gpu_device 0 \
#     --log_path ./log/docred_97_NA_40000_regu_entro_week.txt \
    


    