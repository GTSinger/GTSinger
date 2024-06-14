# binarizer
CONFIG_NAME=singing/svs/config/base_rms.yaml
CUDA_VISIBLE_DEVICES=2 python data_gen/tts/runs/binarize.py --config $CONFIG_NAME

# 48khz singer
CONFIG_NAME=singing/svs/config/base_rms.yaml
HPRAMS="val_check_interval=8000,max_updates=300000"
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config $CONFIG_NAME --hparam $HPRAMS --exp_name svs/diff --reset

CONFIG_NAME=singing/svs/config/base_rms.yaml
HPRAMS="val_check_interval=8000,max_updates=300000,f0_gen=gmdiff"
CUDA_VISIBLE_DEVICES=1 python tasks/run.py --config $CONFIG_NAME --hparam $HPRAMS --exp_name svs/diff_gm --reset

# gm larger model
CONFIG_NAME=singing/svs/config/base_rms.yaml
HPRAMS="val_check_interval=8000,max_updates=300000,f0_gen=gmdiff,f0_residual_layers=15,f0_max_beta=0.04,f0_timesteps=200,f0_K_step=200"
CUDA_VISIBLE_DEVICES=2 python tasks/run.py --config $CONFIG_NAME --hparam $HPRAMS --exp_name svs/diff_gm_l --reset

# crepe f0
CONFIG_NAME=singing/svs/config/base_rms.yaml
HPRAMS="val_check_interval=8000,max_updates=300000"
CUDA_VISIBLE_DEVICES=1 python tasks/run.py --config $CONFIG_NAME --hparam $HPRAMS --exp_name svs/diff_cp --reset

CONFIG_NAME=singing/svs/config/base_rms.yaml
HPRAMS="val_check_interval=8000,max_updates=300000,param_=x0"
CUDA_VISIBLE_DEVICES=1 python tasks/run.py --config $CONFIG_NAME --hparam $HPRAMS --exp_name svs/diffx0 --reset

# diff postnet
CONFIG_NAME=singing/svs/config/diffpostnet.yaml
HPRAMS="val_check_interval=8000"
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config $CONFIG_NAME --hparam $HPRAMS --exp_name svs/diff_dpost --reset

CONFIG_NAME=singing/svs/config/diffpostnet.yaml
HPRAMS="val_check_interval=8000,f0_gen=gmdiff,fs2_ckpt_dir=checkpoints/svs/diff_gm_l,f0_residual_layers=15,f0_max_beta=0.04,f0_timesteps=200,f0_K_step=200"
CUDA_VISIBLE_DEVICES=1 python tasks/run.py --config $CONFIG_NAME --hparam $HPRAMS --exp_name svs/diff_dpost_gm_l --reset