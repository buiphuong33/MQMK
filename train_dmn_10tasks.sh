
python main.py \
        mqmk \
        --model vit_base_patch16_224 \
        --batch-size 64 \
        --data-path /nas/xxx/dataset/data/ \
        --dataset Split-Domainnet\
        --output_dir ./output \
        --multi_query True \
        --multi_key True \
        --KEY_replace False \
        --NCM False \
        --Match_NCM False \
        --gpu_devices 0 \
        --length 90 \
        --epochs 430 \
        --use_g_prompt True \
        --g_prompt_layer_idx 0 1 \
        --g_prompt_length 5 \
        --e_prompt_layer_idx 2 3 4 5 6 7 8 9 10 \
        --k_key 1 \
        --class_group 1 \
        --perfect_match False \
        --lr 0.005 \
        --num_tasks 10 \
        --size 10 \
        --name mq_mk_dmn_10tasks