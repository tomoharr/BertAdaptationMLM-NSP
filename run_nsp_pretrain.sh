# CUDA_VISIBLE_DEVICES= \
python BERT_NSP_pretrain.py  \
    --train_data_file ./kakou_data/CorpusForNSP_train.txt \
    --output_dir ./kei/params/nsp/pretrain_method1 \
    --eval_data_file ./kakou_data/CorpusForNSP_eval.txt \
    --model_name_or_path ./resorce/bert/Japanese_L-12_H-768_A-12_E-30_BPE_transformers \
    --nsp_swap_ratio 0.5 \
    --do_train \
    --do_eval \
    --per_gpu_train_batch_size 64 \
    --per_gpu_eval_batch_size 64 \
    --num_train_epochs 10 \
    --save_steps 2486 \
    --logging_steps 2486 \
    --overwrite_output_dir \
    --block_size 512
    --method method1
