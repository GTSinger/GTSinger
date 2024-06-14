# aishell3 pitch shift
CONFIG_NAME=singing/multi_svs/config/base_aishell3.yaml
CUDA_VISIBLE_DEVICES=1 python tasks/run.py --config $CONFIG_NAME --exp_name multi_svs/aishell3_ps --reset

# xiaoma binarizer 
CONFIG_NAME=singing/multi_svs/config/base_xiaoma.yaml
CUDA_VISIBLE_DEVICES=0 python data_gen/tts/runs/binarize.py --config $CONFIG_NAME

# m4singer binarizer
CONFIG_NAME=singing/multi_svs/config/base_m4.yaml
CUDA_VISIBLE_DEVICES=0 python data_gen/tts/runs/binarize.py --config $CONFIG_NAME

# opensvi binarizer
CONFIG_NAME=singing/multi_svs/config/base_opensvi.yaml
CUDA_VISIBLE_DEVICES=0 python data_gen/tts/runs/binarize.py --config $CONFIG_NAME

CONFIG_NAME=singing/multi_svs/config/base_xiaoma.yaml
HPRAMS="val_check_interval=8000,max_updates=300000"
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config $CONFIG_NAME --hparam $HPRAMS --exp_name multi_svs/xiaoma_m4 --reset

# 只训练声学模型部分，其他部分不考虑
CONFIG_NAME=singing/multi_svs/config/base_xiaoma.yaml
HPRAMS="val_check_interval=8000,max_updates=120000,binary_data_dir=data/binary/xiaoma_m4_opensvi"
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config $CONFIG_NAME --hparam $HPRAMS --exp_name multi_svs/base --reset
# 只训练pitch 部分
CONFIG_NAME=singing/multi_svs/config/diff_pitch.yaml
HPRAMS="pretrain_singer=/home/renyi/hjz/NATSpeech/checkpoints/multi_svs/base/model_ckpt_steps_120000.ckpt"
CUDA_VISIBLE_DEVICES=1 python tasks/run.py --config $CONFIG_NAME --hparam $HPRAMS --exp_name multi_svs/diffpitch --reset

# xiaoma_m4 but pred x_start not noise
CONFIG_NAME=singing/multi_svs/config/base_xiaoma.yaml
HPRAMS="val_check_interval=8000,max_updates=304000"
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config $CONFIG_NAME --hparam $HPRAMS --exp_name multi_svs/xiaoma_m4_x0 --reset