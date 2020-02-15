CUDA_VISIBLE_DEVICES=2 \
python pretrain_padmask_dialogue_bert.py  \
    --train_data_file ./kakou_data/corpusForMLM_train.txt \
    --output_dir ./kei/params/pretrain_bert_e20 \
    --eval_data_file ./kakou_data/corpusForMLM_eval.txt \
    --model_name_or_path ./kei/params/pretrain_padmask_bert_dialogue/ \
    --mlm \
    --mlm_probability 0.15 \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --per_gpu_train_batch_size 128 \
    --per_gpu_eval_batch_size 16 \
    --num_train_epochs 10 \
    --save_steps 5400 \
    --logging_steps 1800 \
    --overwrite_output_dir \
    --block_size 128
