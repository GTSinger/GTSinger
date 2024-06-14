################################### 最开始尝试 ###################################

# 启动尝试。有个问题就是，全0的target会导致分数直接是0
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config research/singtech/config/te.yaml --exp_name 240221-te-01 --reset


CUDA_VISIBLE_DEVICES=2,3 python tasks/run.py --config research/singtech/config/te.yaml --exp_name 240221-te-02 --reset -hp "lr=0.00001"

CUDA_VISIBLE_DEVICES=0,1 python tasks/run.py --config research/singtech/config/te.yaml --exp_name 240221-te-03 --reset -hp "lr=0.000005"

CUDA_VISIBLE_DEVICES=2,3 python tasks/run.py --config research/singtech/config/te.yaml --exp_name 240221-te-04 --reset -hp "lr=0.000005,updown_rates=2-2-2,channel_multiples=1-1-1,frames_multiple=8"

# 基于 240221-te-04，但是加入了三个 variance 信息
CUDA_VISIBLE_DEVICES=0,1 python tasks/run.py --config research/singtech/config/te1.yaml --exp_name 240222-te-01 --reset -hp "lr=0.000005,updown_rates=2-2-2,channel_multiples=1-1-1,frames_multiple=8"
