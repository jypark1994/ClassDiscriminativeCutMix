python3 kfold_main.py --gpus 0 --batch_size 64 --pretrained --learning_rate 5e-3 --weight_decay 1e-3 --flag_vervose --expr_name R50_Vanilla_MosquitoDL_L5e-3_W1e-3 \
|
python3 kfold_main.py --gpus 1 --batch_size 32 --k 1 --cut_mode A --cut_prob 0.5 --pretrained --learning_rate 1e-2 --flag_vervose --train_mode MACM --expr_name R50_MACM_A_k1_Prob05_MosquitoDL_L5e-3 \
|
python3 kfold_main.py --gpus 2 --batch_size 32 --k 3 --cut_mode A --cut_prob 0.5 --pretrained --learning_rate 1e-2 --flag_vervose --train_mode MACM --expr_name R50_MACM_A_k3_Prob05_MosquitoDL_L5e-3 \
|
python3 kfold_main.py --gpus 3 --batch_size 32 --k 5 --cut_mode A --cut_prob 0.5 --pretrained --learning_rate 1e-2 --flag_vervose --train_mode MACM --expr_name R50_MACM_A_k5_Prob05_MosquitoDL_L5e-3;

# python3 kfold_main.py --gpus 0 --batch_size 32 --cut_mode A --cam_mode 'label' --cut_prob 0.9 --pretrained --learning_rate 1e-2 --flag_vervose --train_mode MCACM --expr_name R50_L_MCACM_Label_A_Prob09_MosquitoDL_L1e-2 \
# |
# python3 kfold_main.py --gpus 1 --batch_size 32 --cut_mode B --cam_mode 'label' --cut_prob 0.9 --pretrained --learning_rate 1e-2 --flag_vervose --train_mode MCACM  --expr_name R50_MCACM_Label_B_Prob09_MosquitoDL_L1e-2 \
# |
# python3 kfold_main.py --gpus 2 --batch_size 32 --cut_mode A --cam_mode 'label' --cut_prob 0.1 --pretrained --learning_rate 1e-2 --flag_vervose --train_mode MCACM --expr_name R50_MCACM_Label_A_Prob01_MosquitoDL_L1e-2 \
# |
# python3 kfold_main.py --gpus 3 --batch_size 32 --cut_mode B --cam_mode 'label' --cut_prob 0.1 --pretrained --learning_rate 1e-2 --flag_vervose --train_mode MCACM  --expr_name R50_MCACM_Label_B_Prob01_MosquitoDL_L1e-2;