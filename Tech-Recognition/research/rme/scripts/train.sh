####################### 第一部分 ########################
# 第一次尝试，note_bd的loss总是负数，然后最后就变成nan了，不知道为什么
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tasks/run.py --config research/rme/config/base_me.yaml --exp_name 231113-01-me --reset

# 试一试不要ph_bd预测了，直接预测note_bd会不会还是有负数的问题
# 结果是，一开始没问题，一旦note_pitch参与梯度就变成nan了
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tasks/run.py --config research/rme/config/base_me.yaml --exp_name 231114-01-me --hparams "note_bd_start=0,note_pitch_start=5000" --reset

# 试一试全部一起开始。结果是，一开始就是nan
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tasks/run.py --config research/rme/config/base_me.yaml --exp_name 231114-02-me --hparams "note_bd_start=0,note_pitch_start=0" --reset

# 重写了结构 me2，不让多任务都基于同一个feature了。试一试全部一起开始，毕竟反正已经是并行了
# 这次好像不会 nan！
# 但是会出现 note_bd loss < 0 的情况...
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tasks/run.py --config research/rme/config/base_me.yaml --exp_name 231114-03-me2 --hparams "note_bd_start=0,note_pitch_start=0" --reset

# 进行了对 nan 值的预防，且调整了学习率、validate cycle等参数
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tasks/run.py --config research/rme/config/base_me.yaml --exp_name 231114-me2-04 --hparams "note_bd_start=0,note_pitch_start=0,lr=0.0001" --reset

# 容易过拟合
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tasks/run.py --config research/rme/config/base_me.yaml --exp_name 231114-me2-05 --hparams "note_bd_start=20000,note_pitch_start=10000,lr=0.0001,max_updates=80000,scheduler=warmup" --reset

# 试一试 dot atten，把别的改回去
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tasks/run.py --config research/rme/config/base_me.yaml --exp_name 231115-me2-02 --hparams "note_bd_start=0,note_pitch_start=0,lr=0.0001,max_updates=40000,scheduler=step_lr,dropout=0.2,max_sentences=4" --reset

# 基于231115-me2-02，给 mel 加不同程度的噪音，给目标target也加高斯噪音，对 pitch 的预测使用 label smoothing
# 反而更加难train了，loss下降非常慢
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tasks/run.py --config research/rme/config/base_me.yaml --exp_name 231115-me2-03 --hparams "note_bd_start=0,note_pitch_start=0,lr=0.0001,max_updates=40000,scheduler=step_lr,dropout=0.2,max_sentences=4,note_bd_add_noise=gaussian:0.05,ph_bd_add_noise=gaussian:0.05,mel_add_noise=gaussian:0.3,note_pitch_label_smoothing=0.1" --reset

# 采用参数初始化，梯度累计.
# 结果: ph_bd预测更好但容易过拟合，note_bd和note_pitch仍然比原来低
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tasks/run.py --config research/rme/config/base_me.yaml --exp_name 231116-me2-01 --hparams "lr=0.0001,max_updates=60000,scheduler=step_lr,dropout=0.0,max_sentences=4,note_bd_add_noise=gaussian:0.02,ph_bd_add_noise=gaussian:0.02,mel_add_noise=gaussian:0.1,note_pitch_label_smoothing=0.05,accumulate_grad_batches=2,label_pos_weight_decay=0.9" --reset

# 换一个scheduler，加入note_pitch温度，调小label_smoothing，调大soft_label的宽度，调小boundary的噪声
# 这好像对ph_bd的loss非常有效，但是分数却更低了。temperature对pitch确实有提升
# 总体更差了
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tasks/run.py --config research/rme/config/base_me.yaml --exp_name 231116-me2-02 --hparams "lr=0.06,max_updates=60000,scheduler=rsqrt,warmup_updates=4000,dropout=0.1,max_sentences=4,note_bd_add_noise=gaussian:0.01,ph_bd_add_noise=gaussian:0.01,mel_add_noise=gaussian:0.05,note_pitch_label_smoothing=0.01,note_pitch_temperature=0.1,accumulate_grad_batches=2,label_pos_weight_decay=0.9,soft_label_func=gaussian:200" --reset

# 缩小 soft_label宽度，加大hidden_size
# 居然没oom，loss确实是小了，也确实严重地过拟合了
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tasks/run.py --config research/rme/config/base_me.yaml --exp_name 231116-me2-03 --hparams "lr=0.0001,max_updates=40000,scheduler=step_lr,warmup_updates=4000,hidden_size=256,dropout=0.2,max_sentences=4,note_bd_add_noise=gaussian:0.005,mel_add_noise=gaussian:0.04,note_pitch_label_smoothing=0.005,note_pitch_temperature=0.05,accumulate_grad_batches=2,label_pos_weight_decay=0.9,soft_label_func=gaussian:100" --reset

# 想办法解决过拟合
# 更差了
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tasks/run.py --config research/rme/config/base_me.yaml --exp_name 231116-me2-04 --hparams "lr=0.00005,max_updates=40000,scheduler=step_lr,warmup_updates=4000,hidden_size=256,dropout=0.3,max_sentences=4,note_bd_add_noise=gaussian:0.005,mel_add_noise=gaussian:0.1,note_pitch_label_smoothing=0.005,note_pitch_temperature=0.03,accumulate_grad_batches=2,label_pos_weight_decay=0.9" --reset

# 想办法解决过拟合
# 稍微好些，ph_bd更差了
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tasks/run.py --config research/rme/config/base_me.yaml --exp_name 231116-me2-05 --hparams "lr=0.00001,max_updates=40000,scheduler=step_lr,warmup_updates=4000,hidden_size=256,dropout=0.2,max_sentences=4,note_bd_add_noise=gaussian:0.005,mel_add_noise=gaussian:0.05,note_pitch_label_smoothing=0.005,note_pitch_temperature=0.03,accumulate_grad_batches=2,label_pos_weight_decay=0.95" --reset

# note bd 还行，pitch变差了
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tasks/run.py --config research/rme/config/base_me.yaml --exp_name 231116-me2-06 --hparams "lr=0.00001,max_updates=60000,scheduler=step_lr,warmup_updates=4000,hidden_size=256,dropout=0.2,max_sentences=4,note_bd_add_noise=gaussian:0.005,ph_bd_add_noise=gaussian:0.02,mel_add_noise=gaussian:0.05,note_pitch_label_smoothing=0.005,note_pitch_temperature=0.03,accumulate_grad_batches=4,label_pos_weight_decay=0.95" --reset

# 这个在ph_bd上非常不错，在note onset precision 上很一般。
# 这个综合来看确实是最好的
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tasks/run.py --config research/rme/config/base_me.yaml --exp_name 231116-me2-07 --hparams "lr=0.00002,max_updates=60000,scheduler=step_lr,warmup_updates=4000,hidden_size=256,dropout=0.2,max_sentences=4,note_bd_add_noise=gaussian:0.005,ph_bd_add_noise=gaussian:0.02,mel_add_noise=gaussian:0.05,lambda_note_bd=0.5,note_pitch_label_smoothing=0.005,note_pitch_temperature=0.03,accumulate_grad_batches=4,label_pos_weight_decay=0.95" --reset

# 想办法改进 note_bd
# ？把 soft_filter 提出来，然后分别给ph_bd和note_bd来做的时候，爆炸了？loss非常大
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tasks/run.py --config research/rme/config/base_me.yaml --exp_name 231116-me2-08 --hparams "lr=0.00002,max_updates=60000,scheduler=step_lr,hidden_size=256,dropout=0.2,max_sentences=4,note_bd_add_noise=gaussian:0.002,ph_bd_add_noise=gaussian:0.02,mel_add_noise=gaussian:0.05,note_pitch_label_smoothing=0.005,note_pitch_temperature=0.03,accumulate_grad_batches=4,label_pos_weight_decay=0.95,num_valid_stats=50" --reset

# 重新做231116-me2-08，但是把soft_filter稍微往前，把get soft filter部分放到函数里，同时把 note_bd 和 ph_bd 的soft部分分开，且分开储存filter
# 这个的效果好像还挺好
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tasks/run.py --config research/rme/config/base_me.yaml --exp_name 231116-me2-12 --hparams "lr=0.00002,max_updates=60000,scheduler=step_lr,hidden_size=256,dropout=0.2,max_sentences=4,note_bd_add_noise=gaussian:0.002,ph_bd_add_noise=gaussian:0.02,mel_add_noise=gaussian:0.05,note_pitch_label_smoothing=0.005,note_pitch_temperature=0.03,accumulate_grad_batches=4,label_pos_weight_decay=0.95,num_valid_stats=50" --reset

# 重新做231116-me2-08，但是把soft_filter稍微往前，把get soft filter部分放到函数里，同时把 note_bd 和 ph_bd 的soft部分分开，且分开储存filter，且用两个不同的hparams来控制
# 还是没问题？？？？ 无语了，那为什么231116-me2-08的loss这么爆炸？不管了...
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tasks/run.py --config research/rme/config/base_me.yaml --exp_name 231116-me2-13 --hparams "lr=0.00002,max_updates=60000,scheduler=step_lr,hidden_size=256,dropout=0.2,max_sentences=4,note_bd_add_noise=gaussian:0.002,ph_bd_add_noise=gaussian:0.02,mel_add_noise=gaussian:0.05,note_pitch_label_smoothing=0.005,note_pitch_temperature=0.03,accumulate_grad_batches=4,label_pos_weight_decay=0.95,num_valid_stats=50" --reset

# 想办法改进 note_bd
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tasks/run.py --config research/rme/config/base_me.yaml --exp_name 231116-me2-14 --hparams "lr=0.00002,max_updates=60000,scheduler=step_lr,hidden_size=256,dropout=0.2,max_sentences=4,note_bd_add_noise=gaussian:0.01,ph_bd_add_noise=gaussian:0.02,mel_add_noise=gaussian:0.05,note_pitch_label_smoothing=0.005,note_pitch_temperature=0.03,accumulate_grad_batches=4,label_pos_weight_decay=0.95,num_valid_stats=50,soft_note_bd_func=gaussian:150" --reset

# 抛弃小模型识别ph_bd，直接用ph_gt看看效果天花板是什么
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tasks/run.py --config research/rme/config/base_me.yaml --exp_name 231120-me2-01 --hparams "lr=0.00002,max_updates=60000,scheduler=step_lr,hidden_size=256,dropout=0.1,max_sentences=4,note_bd_add_noise=gaussian:0.002,use_soft_ph_bd=False,mel_add_noise=gaussian:0.05,note_pitch_label_smoothing=0.005,note_pitch_temperature=0.03,accumulate_grad_batches=4,label_pos_weight_decay=0.95,num_valid_stats=50,task_cls=research.rme.me_task2.MidiExtractionTask2" --reset

# 略微加大 lr
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tasks/run.py --config research/rme/config/base_me.yaml --exp_name 231120-me2-02 --hparams "lr=0.00005,max_updates=60000,scheduler=step_lr,hidden_size=256,dropout=0.1,max_sentences=4,note_bd_add_noise=gaussian:0.002,use_soft_ph_bd=False,mel_add_noise=gaussian:0.05,note_pitch_label_smoothing=0.005,note_pitch_temperature=0.03,accumulate_grad_batches=4,label_pos_weight_decay=0.95,num_valid_stats=50,task_cls=research.rme.me_task2.MidiExtractionTask2" --reset

# 把 lr 减回，增大 max_updates，增加 focal_loss
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tasks/run.py --config research/rme/config/base_me.yaml --exp_name 231120-me2-03 --hparams "lr=0.00002,max_updates=100000,scheduler=step_lr,hidden_size=256,dropout=0.1,max_sentences=4,note_bd_add_noise=gaussian:0.002,use_soft_ph_bd=False,mel_add_noise=gaussian:0.05,note_pitch_label_smoothing=0.005,note_pitch_temperature=0.03,accumulate_grad_batches=4,label_pos_weight_decay=0.95,num_valid_stats=50,task_cls=research.rme.me_task2.MidiExtractionTask2,note_bd_focal_loss=2.0,lambda_note_bd_focal=0.5,lambda_note_bd=0.5" --reset

# 基于231120-me2-03，给 focal loss 调参
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tasks/run.py --config research/rme/config/base_me.yaml --exp_name 231120-me2-04 --hparams "lr=0.00002,max_updates=100000,scheduler=step_lr,hidden_size=256,dropout=0.1,max_sentences=4,note_bd_add_noise=gaussian:0.002,use_soft_ph_bd=False,mel_add_noise=gaussian:0.05,note_pitch_label_smoothing=0.005,note_pitch_temperature=0.03,accumulate_grad_batches=4,label_pos_weight_decay=0.95,num_valid_stats=50,task_cls=research.rme.me_task2.MidiExtractionTask2,note_bd_focal_loss=5.0,lambda_note_bd_focal=1.0,lambda_note_bd=1.0" --reset

####################### 第二部分 ########################
# 抛弃 ph_bd，使用 word_bd。且先不关注 word_bd 的预测，而是直接使用 gt word_bd。同时，加入 mty 数据。
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tasks/run.py --config research/rme/config/base_me2.yaml --exp_name 231122-me3-01 --hparams "max_sentences=4,lambda_note_bd_focal=2.0" --reset

# 尝试重新用融合模块来多任务预测note bd和pitch。奇怪，这次好像也没有nan或者负loss。ps：模型名应该换成me5
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tasks/run.py --config research/rme/config/base_me2.yaml --exp_name 231122-me3-02 --hparams "lr=0.00002,max_updates=100000,max_sentences=4,lambda_note_bd_focal=4.0,model=me5" --reset

# 同231122-me3-02。调参。尝试只用40bin的mel
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_me2.yaml --exp_name 231123-me5-01 --hparams "lr=0.00002,max_updates=100000,max_sentences=4,lambda_note_bd_focal=5.0,model=me5,use_mel_bins=40,note_pitch_temperature=0.02" --reset

# 同231123-me5-01。调参
# 这个好像还没收敛完全啊，可以继续跑几个epoch
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_me2.yaml --exp_name 231123-me5-02 --hparams "lr=0.00002,max_updates=120000,max_sentences=4,lambda_note_bd_focal=5.0,model=me5,use_mel_bins=30,note_pitch_temperature=0.01,soft_note_bd_func=gaussian:80" --reset

# 同231123-me5-02。调参。加入note_bd_temperature，加入word_bd的regulate
# 150k steps 甚至还不够，甚至还能继续收敛
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_me2.yaml --exp_name 231123-me5-03 --hparams "lr=0.00002,max_updates=150000,max_sentences=4,lambda_note_bd_focal=5.0,model=me5,use_mel_bins=30,note_pitch_temperature=0.01,soft_note_bd_func=gaussian:80,note_bd_temperature=0.1,note_bd_ref_min_gap=40" --reset

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_me2.yaml --exp_name 231123-me5-04 --hparams "lr=0.00004,max_updates=120000,max_sentences=4,lambda_note_bd_focal=5.0,model=me5,use_mel_bins=30,note_pitch_temperature=0.01,soft_note_bd_func=gaussian:80,note_bd_temperature=0.2,note_bd_ref_min_gap=40" --reset

# 使用 hopsize256的数据，mel大小会小一半
# 会出现大量的 tgt pitch 和 pred pitch 数量不一致的问题。
# 问题解决，因为dataset2里计算bd的时候会有舍入问题。新问题：24k和hopsize256会产生分辨率不足的问题，即0.01秒的间隔在这里会变成只有一个frame。可以先试试，但不抱希望
# 一个解决办法是，减小note_bd_min_gap
# 精度损失了很多...
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tasks/run.py --config research/rme/config/base_me3.yaml --exp_name 231124-me5-01 --hparams "lr=0.00003,max_updates=120000,max_sentences=8,lambda_note_bd_focal=3.0,model=me5,use_mel_bins=30,note_pitch_temperature=0.01,soft_note_bd_func=gaussian:20,note_bd_temperature=0.4,note_bd_min_gap=20,note_bd_ref_min_gap=10" --reset

# 尝试失败。增大hopsize没用
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tasks/run.py --config research/rme/config/base_me3.yaml --exp_name 231124-me5-02 --hparams "lr=0.00002,max_updates=150000,max_sentences=8,lambda_note_bd_focal=5.0,model=me5,use_mel_bins=30,note_pitch_temperature=0.01,soft_note_bd_func=gaussian:40,note_bd_temperature=0.4,note_bd_min_gap=40,note_bd_ref_min_gap=20" --reset

# 纯粹是为了调试新的cuda
# 1. 老的fairseq2环境可以在cuda11.8上运行
# 2. 准备安装 torch2.0
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_me2.yaml --exp_name 231212-me5-01 --hparams "lr=0.00002,max_updates=150000,max_sentences=4,lambda_note_bd_focal=5.0,model=me5,use_mel_bins=30,note_pitch_temperature=0.01,soft_note_bd_func=gaussian:80,note_bd_temperature=0.1,note_bd_ref_min_gap=40" --reset

# 测试 flash attn
# 实验证明使用 flash attn 的普通 transformer 效果不行
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_me2.yaml --exp_name 231212-me5-02 --hparams "lr=0.00002,max_updates=150000,max_sentences=4,lambda_note_bd_focal=5.0,model=me5,use_mel_bins=30,note_pitch_temperature=0.01,soft_note_bd_func=gaussian:80,note_bd_temperature=0.1,note_bd_ref_min_gap=40" --reset

# 重新跑 231123-me5-03，再看看到底是不是 dataset 的原因
# 这个是对的，看来一切都走上正轨，torch2也没问题。除了经常oom....
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tasks/run.py --config research/rme/config/base_me2.yaml --exp_name 231213-me5-01 --hparams "lr=0.00002,max_updates=150000,max_sentences=4,lambda_note_bd_focal=5.0,model=me5,use_mel_bins=30,note_pitch_temperature=0.01,soft_note_bd_func=gaussian:80,note_bd_temperature=0.1,note_bd_ref_min_gap=40" --reset

# 重新跑 231213-me5-01，把dataset改成并行的格式，再看看有没有地方可以提速的
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tasks/run.py --config research/rme/config/base_me2.yaml --exp_name 231213-me5-02 --hparams "lr=0.00002,max_updates=150000,max_sentences=4,lambda_note_bd_focal=5.0,model=me5,use_mel_bins=30,note_pitch_temperature=0.01,soft_note_bd_func=gaussian:80,note_bd_temperature=0.1,note_bd_ref_min_gap=40,pin_memory=True" --reset

# 更换模型！！！看看纯cnn+wavenet会有啥结果。先试试多点参数，看看过拟合效果
# （果然，主要是conformer的部分非常慢，gpu跑不满啊）
# 确实过拟合，可能lr还是太大了。
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tasks/run.py --config research/rme/config/base_me2.yaml --exp_name 231214-me6-01 --hparams "lr=0.0001,max_updates=100000,max_sentences=16,accumulate_grad_batches=1,scheduler_lr_step_size=1000,lambda_note_bd_focal=5.0,model=me6,use_mel_bins=30,note_pitch_temperature=0.01,soft_note_bd_func=gaussian:80,note_bd_temperature=0.1,note_bd_ref_min_gap=40" --reset

# for rsqrt scheduler, lr=1 and warmup=4000 and hidden=256 gives approximate 0.001 maximum lr
# 依旧过拟合，early stop
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tasks/run.py --config research/rme/config/base_me2.yaml --exp_name 231214-me6-02 --hparams "lr=1.0,max_updates=100000,max_sentences=16,accumulate_grad_batches=1,scheduler=rsqrt,warmup_updates=4000,lambda_note_bd_focal=5.0,model=me6,use_mel_bins=40,note_pitch_temperature=0.01,soft_note_bd_func=gaussian:80,note_bd_temperature=0.1,note_bd_ref_min_gap=40,wn_kernel=3" --reset

# 换成原来的step_lr，减小学习率，把网络变浅，防止过拟合
# 仍然过拟合
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tasks/run.py --config research/rme/config/base_me2.yaml --exp_name 231214-me6-03 --hparams "lr=0.00002,max_updates=150000,max_sentences=16,accumulate_grad_batches=1,lambda_note_bd_focal=5.0,model=me6,use_mel_bins=40,note_pitch_temperature=0.01,soft_note_bd_func=gaussian:80,note_bd_temperature=0.1,note_bd_ref_min_gap=40,wn_layers=12" --reset

####################### 第三部分 ########################
# 使用 conformer + wav2vec2
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tasks/run.py --config research/rme/config/base_me4.yaml --exp_name 231217-me7-01 --hparams "lr=0.00002,max_updates=150000,max_sentences=3,accumulate_grad_batches=5" --reset

# 对 slur 进行惩罚 (先试试只加个bce).
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tasks/run.py --config research/rme/config/base_me4.yaml --exp_name 231217-me7-03 --hparams "lr=0.00002,max_updates=150000,max_sentences=3,accumulate_grad_batches=4,lambda_note_bd_focal=4,label_pos_weight_decay=0.8,lambda_note_bd_slur_punish=1.0" --reset

# 使用 bce 对 slur 进行惩罚不太行，试试直接加起来
# 这些在 note_bd 的 loss 部分全都过拟合了
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tasks/run.py --config research/rme/config/base_me4.yaml --exp_name 231217-me7-04 --hparams "lr=0.00002,max_updates=150000,max_sentences=3,accumulate_grad_batches=4,lambda_note_bd_focal=4,label_pos_weight_decay=0.8,lambda_note_bd_slur_punish=1.0" --reset

# 减小 lambda，减小 feat 的 noise
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tasks/run.py --config research/rme/config/base_me4.yaml --exp_name 231217-me7-05 --hparams "lr=0.00002,max_updates=150000,max_sentences=3,accumulate_grad_batches=4,lambda_note_bd_focal=1,label_pos_weight_decay=0.8,lambda_note_bd_slur_punish=1.0,feat_add_noise=gaussian:0.02" --reset

# 继续调参
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python tasks/run.py --config research/rme/config/base_me4.yaml --exp_name 231217-me7-06 --hparams "lr=0.00005,max_updates=150000,max_sentences=3,accumulate_grad_batches=4,lambda_note_bd_focal=1,label_pos_weight_decay=0.8,lambda_note_bd_slur_punish=0.5,feat_add_noise=gaussian:0.02" --reset

############ me8 ###########
# 基于 231123-me5-03，模型换成me8，具体是把backbone换成了unet
# 先试试 conv
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_me2.yaml --exp_name 231219-me8-01 --hparams "lr=0.0001,max_updates=80000,max_sentences=32,max_tokens=80000,accumulate_grad_batches=1,val_check_interval=1000,scheduler_lr_step_size=1000,lambda_note_bd_focal=5.0,model=me8,use_mel_bins=30,note_pitch_temperature=0.01,soft_note_bd_func=gaussian:80,note_bd_temperature=0.2,note_bd_ref_min_gap=40,bkb_net=conv,frames_multiple=8,bkb_layers=12" --reset

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_me2.yaml --exp_name 231219-me8-02 --hparams "lr=0.00004,max_updates=120000,max_sentences=32,max_tokens=80000,accumulate_grad_batches=1,val_check_interval=1000,scheduler_lr_step_size=1000,lambda_note_bd_focal=5.0,model=me8,use_mel_bins=30,note_pitch_temperature=0.01,soft_note_bd_func=gaussian:80,note_bd_temperature=0.2,note_bd_ref_min_gap=40,bkb_net=conv,frames_multiple=8,bkb_layers=12" --reset

# 换成 conformer，把下采样层数减少一层
CUDA_VISIBLE_DEVICES=2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_me2.yaml --exp_name 231219-me8-04 --hparams "lr=0.00004,max_updates=150000,max_sentences=32,max_tokens=60000,accumulate_grad_batches=1,lambda_note_bd_focal=5.0,model=me8,use_mel_bins=30,note_pitch_temperature=0.01,soft_note_bd_func=gaussian:80,note_bd_temperature=0.2,note_bd_ref_min_gap=40,conformer_kernel=9,bkb_net=conformer,bkb_layers=2,frames_multiple=4,updown_rates=2-2,channel_multiples=1.125-1.125" --reset

# conformer，使用 channel constant
# 容易过拟合
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_me2.yaml --exp_name 231224-me8-01 --hparams "lr=0.00002,max_updates=150000,max_sentences=32,max_tokens=60000,accumulate_grad_batches=1,lambda_note_bd_focal=3.0,model=me8,use_mel_bins=30,note_pitch_temperature=0.01,soft_note_bd_func=gaussian:80,note_bd_temperature=0.2,note_bd_ref_min_gap=40,conformer_kernel=9,bkb_net=conformer,bkb_layers=2,frames_multiple=8,channel_multiples=1-1-1,ds_workers=4" --reset

# conformer，使用 channel constant，减小学习率，减小 hidden
# 可能是学习率太低，精度不太行，但至少减小 hidden 之后不过拟合了。精度的问题主要是p太小但是r太大
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_me2.yaml --exp_name 231224-me8-02 --hparams "lr=0.000005,max_updates=150000,max_sentences=80,max_tokens=160000,accumulate_grad_batches=1,val_check_interval=1000,lambda_note_bd_focal=1.0,dropout=0.0,model=me8,hidden_size=128,use_mel_bins=30,note_pitch_temperature=0.01,soft_note_bd_func=gaussian:80,note_bd_temperature=0.2,note_bd_ref_min_gap=40,conformer_kernel=9,bkb_net=conformer,bkb_layers=2,frames_multiple=8,channel_multiples=1.125-1.125-1.125,ds_workers=4" --reset

# conformer，使用 channel constant，减小学习率，加入 skip layer
# 可能是学习率太低，精度不太行，还会过拟合。精度的问题主要是p太小但是r太大
CUDA_VISIBLE_DEVICES=2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_me2.yaml --exp_name 231224-me8-03 --hparams "lr=0.000005,max_updates=150000,max_sentences=32,max_tokens=60000,accumulate_grad_batches=1,val_check_interval=1000,lambda_note_bd_focal=1.0,dropout=0.0,model=me8,use_mel_bins=30,note_pitch_temperature=0.01,soft_note_bd_func=gaussian:80,note_bd_temperature=0.2,note_bd_ref_min_gap=40,conformer_kernel=9,bkb_net=conformer,bkb_layers=2,frames_multiple=8,channel_multiples=1-1-1,ds_workers=4,unet_skip_layer=True" --reset

# 基于 231224-me8-02，加大学习率，加入 slur 的惩罚，略微还原 focal 的 lambda。减小 pos_weight，对 f0 进行加噪
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_me2.yaml --exp_name 231224-me8-04 --hparams "lr=0.00001,max_updates=60000,max_sentences=80,max_tokens=160000,accumulate_grad_batches=1,scheduler_lr_step_size=500,val_check_interval=500,lambda_note_bd_focal=3.0,dropout=0.0,model=me8,hidden_size=128,use_mel_bins=30,note_pitch_temperature=0.01,soft_note_bd_func=gaussian:80,note_bd_temperature=0.2,note_bd_ref_min_gap=40,conformer_kernel=9,bkb_net=conformer,bkb_layers=2,frames_multiple=8,channel_multiples=1.125-1.125-1.125,ds_workers=4,lambda_note_bd_slur_punish=1.0,label_pos_weight_decay=0.85,f0_add_noise=gaussian:0.04" --reset

# 基于 231224-me8-01，对 f0 进行加噪
# 效果非常好！
CUDA_VISIBLE_DEVICES=2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_me2.yaml --exp_name 231224-me8-05 --hparams "lr=0.00001,max_updates=60000,max_sentences=32,max_tokens=60000,accumulate_grad_batches=1,scheduler_lr_step_size=500,val_check_interval=500,lambda_note_bd_focal=3.0,model=me8,use_mel_bins=30,note_pitch_temperature=0.01,soft_note_bd_func=gaussian:80,note_bd_temperature=0.2,note_bd_ref_min_gap=40,conformer_kernel=9,bkb_net=conformer,bkb_layers=2,frames_multiple=8,channel_multiples=1-1-1,ds_workers=4,f0_add_noise=gaussian:0.04" --reset

####################### 第四部分 ########################
# 基于231123-me5-03，先搞个只有conformer作为bkbone的，先不搞unet啥的。多加一个f0加噪
# 每跑两次val就会oom，无奈换成bsz=3
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_rme.yaml --exp_name 231226-me5-01 --hparams "max_sentences=3,accumulate_grad_batches=5,scheduler_lr_step_size=2500,val_check_interval=2500,use_mel_bins=30" --reset

# 基于 231224-me8-05，试试unet在噪声数据上的表现
# 非常好！
CUDA_VISIBLE_DEVICES=2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_rme.yaml --exp_name 231227-me8-01 --hparams "lr=0.00001,max_updates=80000,max_sentences=32,max_tokens=80000,accumulate_grad_batches=1,scheduler_lr_step_size=500,val_check_interval=500,lambda_note_bd_focal=3.0,model=me8,use_mel_bins=30,conformer_kernel=9,bkb_net=conformer,bkb_layers=2,frames_multiple=8,channel_multiples=1-1-1,ds_workers=4" --reset

# 基于231227-me8-01，随便跑跑，加大channel multiple，加大 f0 的噪声
CUDA_VISIBLE_DEVICES=2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_rme.yaml --exp_name 231227-me8-02 --hparams "lr=0.00001,max_updates=80000,max_sentences=32,max_tokens=80000,accumulate_grad_batches=1,scheduler_lr_step_size=500,val_check_interval=500,lambda_note_bd_focal=3.0,model=me8,use_mel_bins=30,conformer_kernel=9,bkb_net=conformer,bkb_layers=2,frames_multiple=8,channel_multiples=1.0625-1.0625-1.0625,ds_workers=4,lambda_note_bd_slur_punish=1,f0_add_noise=gaussian:0.08" --reset

####################### 第五部分：减少数据 ########################
# 基于 231224-me8-05 (无加噪)，但是减少一半的数据
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_me2.yaml --exp_name 231228-me8-01 --hparams "lr=0.00001,max_updates=60000,max_sentences=32,max_tokens=60000,accumulate_grad_batches=1,scheduler_lr_step_size=500,val_check_interval=500,lambda_note_bd_focal=3.0,model=me8,use_mel_bins=30,note_pitch_temperature=0.01,soft_note_bd_func=gaussian:80,note_bd_temperature=0.2,note_bd_ref_min_gap=40,conformer_kernel=9,bkb_net=conformer,bkb_layers=2,frames_multiple=8,channel_multiples=1-1-1,ds_workers=4,f0_add_noise=gaussian:0.04,dataset_downsample_rate=0.5" --reset

# 基于 231224-me8-05 (无加噪)，但是只有0.2的数据
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_me2.yaml --exp_name 231228-me8-02 --hparams "lr=0.00001,max_updates=60000,max_sentences=32,max_tokens=60000,accumulate_grad_batches=1,scheduler_lr_step_size=500,val_check_interval=500,lambda_note_bd_focal=3.0,model=me8,use_mel_bins=30,note_pitch_temperature=0.01,soft_note_bd_func=gaussian:80,note_bd_temperature=0.2,note_bd_ref_min_gap=40,conformer_kernel=9,bkb_net=conformer,bkb_layers=2,frames_multiple=8,channel_multiples=1-1-1,ds_workers=4,f0_add_noise=gaussian:0.04,dataset_downsample_rate=0.2" --reset

# 基于 231224-me8-05 (无加噪)，但是只有0.1的数据
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_me2.yaml --exp_name 231228-me8-03 --hparams "lr=0.00001,max_updates=60000,max_sentences=32,max_tokens=60000,accumulate_grad_batches=1,scheduler_lr_step_size=500,val_check_interval=500,lambda_note_bd_focal=3.0,model=me8,use_mel_bins=30,note_pitch_temperature=0.01,soft_note_bd_func=gaussian:80,note_bd_temperature=0.2,note_bd_ref_min_gap=40,conformer_kernel=9,bkb_net=conformer,bkb_layers=2,frames_multiple=8,channel_multiples=1-1-1,ds_workers=4,f0_add_noise=gaussian:0.04,dataset_downsample_rate=0.1" --reset

# 基于 231224-me8-05 (无加噪)，但是只有0.02的数据
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_me2.yaml --exp_name 231228-me8-04 --hparams "lr=0.00001,max_updates=60000,max_sentences=32,max_tokens=60000,accumulate_grad_batches=1,scheduler_lr_step_size=500,val_check_interval=500,lambda_note_bd_focal=3.0,model=me8,use_mel_bins=30,note_pitch_temperature=0.01,soft_note_bd_func=gaussian:80,note_bd_temperature=0.2,note_bd_ref_min_gap=40,conformer_kernel=9,bkb_net=conformer,bkb_layers=2,frames_multiple=8,channel_multiples=1-1-1,ds_workers=4,f0_add_noise=gaussian:0.04,dataset_downsample_rate=0.02" --reset

# 基于 231227-me8-01 (加噪)，但是只有0.5的数据
CUDA_VISIBLE_DEVICES=2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_rme.yaml --exp_name 231229-me8-01 --hparams "lr=0.00001,max_updates=80000,max_sentences=32,max_tokens=80000,accumulate_grad_batches=1,scheduler_lr_step_size=500,val_check_interval=500,lambda_note_bd_focal=3.0,model=me8,use_mel_bins=30,conformer_kernel=9,bkb_net=conformer,bkb_layers=2,frames_multiple=8,channel_multiples=1-1-1,ds_workers=4,dataset_downsample_rate=0.5" --reset

# 基于 231227-me8-01 (加噪)，但是只有0.2的数据
CUDA_VISIBLE_DEVICES=2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_rme.yaml --exp_name 231229-me8-02 --hparams "lr=0.00001,max_updates=80000,max_sentences=32,max_tokens=80000,accumulate_grad_batches=1,scheduler_lr_step_size=500,val_check_interval=500,lambda_note_bd_focal=3.0,model=me8,use_mel_bins=30,conformer_kernel=9,bkb_net=conformer,bkb_layers=2,frames_multiple=8,channel_multiples=1-1-1,ds_workers=4,dataset_downsample_rate=0.2" --reset

# 基于 231227-me8-01 (加噪)，但是只有0.1的数据
CUDA_VISIBLE_DEVICES=2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_rme.yaml --exp_name 231229-me8-03 --hparams "lr=0.00001,max_updates=80000,max_sentences=32,max_tokens=80000,accumulate_grad_batches=1,scheduler_lr_step_size=500,val_check_interval=500,lambda_note_bd_focal=3.0,model=me8,use_mel_bins=30,conformer_kernel=9,bkb_net=conformer,bkb_layers=2,frames_multiple=8,channel_multiples=1-1-1,ds_workers=4,dataset_downsample_rate=0.1" --reset

# 基于 231227-me8-01 (加噪)，但是只有0.02的数据
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_rme.yaml --exp_name 231229-me8-04 --hparams "lr=0.00001,max_updates=80000,max_sentences=32,max_tokens=80000,accumulate_grad_batches=1,scheduler_lr_step_size=500,val_check_interval=500,lambda_note_bd_focal=3.0,model=me8,use_mel_bins=30,conformer_kernel=9,bkb_net=conformer,bkb_layers=2,frames_multiple=8,channel_multiples=1-1-1,ds_workers=4,dataset_downsample_rate=0.02" --reset

# 基于 231227-me8-01 (加噪)，但是只使用 m4singer
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_rme.yaml --exp_name 231230-me8-01 --hparams "lr=0.00001,max_updates=80000,max_sentences=32,max_tokens=80000,accumulate_grad_batches=1,scheduler_lr_step_size=500,val_check_interval=500,lambda_note_bd_focal=3.0,model=me8,use_mel_bins=30,conformer_kernel=9,bkb_net=conformer,bkb_layers=2,frames_multiple=8,channel_multiples=1-1-1,ds_workers=4,ds_names_in_training=m4" --reset

# 基于 231227-me8-01 (加噪)，但是只使用 m4singer 的 0.5
CUDA_VISIBLE_DEVICES=2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_rme.yaml --exp_name 231230-me8-02 --hparams "lr=0.00001,max_updates=80000,max_sentences=32,max_tokens=80000,accumulate_grad_batches=1,scheduler_lr_step_size=500,val_check_interval=500,lambda_note_bd_focal=3.0,model=me8,use_mel_bins=30,conformer_kernel=9,bkb_net=conformer,bkb_layers=2,frames_multiple=8,channel_multiples=1-1-1,ds_workers=4,ds_names_in_training=m4,dataset_downsample_rate=0.5" --reset

# 基于 231227-me8-01 (加噪)，但是只使用 m4singer 的 0.1
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_rme.yaml --exp_name 231230-me8-03 --hparams "lr=0.00001,max_updates=80000,max_sentences=32,max_tokens=80000,accumulate_grad_batches=1,scheduler_lr_step_size=500,val_check_interval=500,lambda_note_bd_focal=3.0,model=me8,use_mel_bins=30,conformer_kernel=9,bkb_net=conformer,bkb_layers=2,frames_multiple=8,channel_multiples=1-1-1,ds_workers=4,ds_names_in_training=m4,dataset_downsample_rate=0.1" --reset

# 基于 231227-me8-01 (加噪)，但是只使用 m4singer 的 0.02
CUDA_VISIBLE_DEVICES=2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_rme.yaml --exp_name 231230-me8-04 --hparams "lr=0.00001,max_updates=80000,max_sentences=32,max_tokens=80000,accumulate_grad_batches=1,scheduler_lr_step_size=500,val_check_interval=500,lambda_note_bd_focal=3.0,model=me8,use_mel_bins=30,conformer_kernel=9,bkb_net=conformer,bkb_layers=2,frames_multiple=8,channel_multiples=1-1-1,ds_workers=4,ds_names_in_training=m4,dataset_downsample_rate=0.02" --reset

# 基于 231227-me8-01 (加噪)，但是只使用 rms
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_rme.yaml --exp_name 231231-me8-01 --hparams "lr=0.00001,max_updates=80000,max_sentences=32,max_tokens=80000,accumulate_grad_batches=1,scheduler_lr_step_size=500,val_check_interval=500,lambda_note_bd_focal=3.0,model=me8,use_mel_bins=30,conformer_kernel=9,bkb_net=conformer,bkb_layers=2,frames_multiple=8,channel_multiples=1-1-1,ds_workers=4,ds_names_in_training=rms" --reset

# 基于 231227-me8-01 (加噪)，但是只使用 rms 的 0.5
CUDA_VISIBLE_DEVICES=2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_rme.yaml --exp_name 231231-me8-02 --hparams "lr=0.00001,max_updates=80000,max_sentences=32,max_tokens=80000,accumulate_grad_batches=1,scheduler_lr_step_size=500,val_check_interval=500,lambda_note_bd_focal=3.0,model=me8,use_mel_bins=30,conformer_kernel=9,bkb_net=conformer,bkb_layers=2,frames_multiple=8,channel_multiples=1-1-1,ds_workers=4,ds_names_in_training=rms,dataset_downsample_rate=0.5" --reset

# 基于 231227-me8-01 (加噪)，但是只使用 rms 的 0.1
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_rme.yaml --exp_name 231231-me8-03 --hparams "lr=0.00001,max_updates=80000,max_sentences=32,max_tokens=80000,accumulate_grad_batches=1,scheduler_lr_step_size=500,val_check_interval=500,lambda_note_bd_focal=3.0,model=me8,use_mel_bins=30,conformer_kernel=9,bkb_net=conformer,bkb_layers=2,frames_multiple=8,channel_multiples=1-1-1,ds_workers=4,ds_names_in_training=rms,dataset_downsample_rate=0.1" --reset

# 基于 231227-me8-01 (加噪)，但是只使用 rms 的 0.02
CUDA_VISIBLE_DEVICES=2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_rme.yaml --exp_name 231231-me8-04 --hparams "lr=0.00001,max_updates=80000,max_sentences=32,max_tokens=80000,accumulate_grad_batches=1,scheduler_lr_step_size=500,val_check_interval=500,lambda_note_bd_focal=3.0,model=me8,use_mel_bins=30,conformer_kernel=9,bkb_net=conformer,bkb_layers=2,frames_multiple=8,channel_multiples=1-1-1,ds_workers=4,ds_names_in_training=rms,dataset_downsample_rate=0.02" --reset

# 重新在rme框架下跑 231224-me8-05，主要为了对比加噪的效果.此为不加噪
# 对比还是很明显的！
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_rme.yaml --exp_name 240101-me8-01 --hparams "lr=0.00001,max_updates=60000,max_sentences=32,max_tokens=60000,accumulate_grad_batches=1,scheduler_lr_step_size=500,val_check_interval=500,lambda_note_bd_focal=3.0,model=me8,use_mel_bins=30,note_pitch_temperature=0.01,soft_note_bd_func=gaussian:80,note_bd_temperature=0.2,note_bd_ref_min_gap=40,conformer_kernel=9,bkb_net=conformer,bkb_layers=2,frames_multiple=8,channel_multiples=1-1-1,ds_workers=4,f0_add_noise=gaussian:0.04,noise_prob=-1" --reset
# 重新在rme框架下跑 231224-me8-05，主要为了对比加噪的效果。此为加噪
CUDA_VISIBLE_DEVICES=2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_rme.yaml --exp_name 240101-me8-02 --hparams "lr=0.00001,max_updates=60000,max_sentences=32,max_tokens=60000,accumulate_grad_batches=1,scheduler_lr_step_size=500,val_check_interval=500,lambda_note_bd_focal=3.0,model=me8,use_mel_bins=30,note_pitch_temperature=0.01,soft_note_bd_func=gaussian:80,note_bd_temperature=0.2,note_bd_ref_min_gap=40,conformer_kernel=9,bkb_net=conformer,bkb_layers=2,frames_multiple=8,channel_multiples=1-1-1,ds_workers=4,f0_add_noise=gaussian:0.04,noise_prob=0.8" --reset

# 重试231227-me8-01，试试如果不使用 f0，只使用 mel，会怎么样
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_rme.yaml --exp_name 240102-me8-01 --hparams "lr=0.00001,max_updates=80000,max_sentences=32,max_tokens=80000,accumulate_grad_batches=1,scheduler_lr_step_size=500,val_check_interval=500,lambda_note_bd_focal=3.0,model=me8,use_mel_bins=30,use_f0=False,conformer_kernel=9,bkb_net=conformer,bkb_layers=2,frames_multiple=8,channel_multiples=1-1-1,ds_workers=4" --reset

# 在240101-me8-02基础上，试试如果加深下采样会怎么样
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_rme.yaml --exp_name 240208-me8-01 --hparams "lr=0.00001,max_updates=60000,max_sentences=32,max_tokens=60000,accumulate_grad_batches=1,scheduler_lr_step_size=500,val_check_interval=500,lambda_note_bd_focal=3.0,model=me8,use_mel_bins=30,note_pitch_temperature=0.01,conformer_kernel=9,bkb_net=conformer,bkb_layers=2,frames_multiple=16,updown_rates=2-2-2-2,channel_multiples=1-1-1-1,ds_workers=4,f0_add_noise=gaussian:0.04,noise_prob=0.8" --reset

# 基于 240208-me8-01，继续加深
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_rme.yaml --exp_name 240208-me8-02 --hparams "lr=0.00001,max_updates=60000,max_sentences=32,max_tokens=60000,accumulate_grad_batches=1,scheduler_lr_step_size=500,val_check_interval=500,lambda_note_bd_focal=3.0,model=me8,use_mel_bins=30,note_pitch_temperature=0.01,conformer_kernel=9,bkb_net=conformer,bkb_layers=2,frames_multiple=32,updown_rates=2-2-2-2-2,channel_multiples=1-1-1-1-1,ds_workers=4,f0_add_noise=gaussian:0.04,noise_prob=0.8" --reset

# 在240101-me8-02基础上，试试如果把 conformer 换成 conv 会怎么样
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_rme.yaml --exp_name 240209-me8-01 --hparams "lr=0.00001,max_updates=60000,max_sentences=32,max_tokens=60000,accumulate_grad_batches=1,scheduler_lr_step_size=500,val_check_interval=500,lambda_note_bd_focal=3.0,model=me8,use_mel_bins=30,note_pitch_temperature=0.01,conformer_kernel=9,bkb_net=conv,bkb_layers=2,frames_multiple=8,updown_rates=2-2-2,channel_multiples=1-1-1,ds_workers=4,f0_add_noise=gaussian:0.04,noise_prob=0.8" --reset

# 调参，把mid的激活换成了swish，并加深
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_rme.yaml --exp_name 240209-me8-02 --hparams "lr=0.00001,max_updates=60000,max_sentences=32,max_tokens=60000,accumulate_grad_batches=1,scheduler_lr_step_size=500,val_check_interval=500,lambda_note_bd_focal=3.0,model=me8,use_mel_bins=30,note_pitch_temperature=0.01,bkb_net=conv,bkb_layers=8,frames_multiple=8,updown_rates=2-2-2,channel_multiples=1-1-1,ds_workers=4,f0_add_noise=gaussian:0.04,noise_prob=0.95" --reset

# 基于231227-me8-01，但是在unet的mid layer中加入w2v2特征，且不使用 f0
# 很容易过拟合！
CUDA_VISIBLE_DEVICES=2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_rme2.yaml --exp_name 240104-me9-01 --hparams "lr=0.00001,max_sentences=32,use_mel_bins=30,use_f0=False,conformer_kernel=9,bkb_net=conformer,bkb_layers=2,frames_multiple=8,ds_workers=8" --reset

####################### 第六部分：加入 w2v2 特征 ########################
# 基于231227-me8-01，但是在unet的mid layer中加入w2v2特征，使用 f0
# 估计这个也很容易过拟合。减少数据的尝试之后再做吧
CUDA_VISIBLE_DEVICES=2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_rme2.yaml --exp_name 240104-me9-02 --hparams "lr=0.00001,max_sentences=32,use_mel_bins=30,conformer_kernel=9,bkb_net=conformer,bkb_layers=2,frames_multiple=8,ds_workers=8" --reset

# 基于240104-me9-02，调参，减少过拟合。ssl_feat_encoder 改成了新的，加入了conv_blocks
# 这个是最好的，但也不比之前的好
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_rme2.yaml --exp_name 240104-me9-06 --hparams "lr=0.000003,scheduler_lr_step_size=1000,max_sentences=32,use_mel_bins=30,ssl_feat_add_noise=gaussian:0.3,conformer_kernel=5,bkb_net=conformer,bkb_layers=2,frames_multiple=8,ds_workers=8" --reset

# 终于有个和 240104-me9-06 差不多的了
# 放弃了，就拿这个当 w2v2 的 baseline 吧
CUDA_VISIBLE_DEVICES=2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_rme2.yaml --exp_name 240104-me9-11 --hparams "lr=0.000003,scheduler_lr_step_size=1500,max_sentences=32,use_mel_bins=40,ssl_feat_add_noise=gaussian:0.4,conformer_kernel=7,bkb_net=conformer,bkb_layers=2,frames_multiple=8,ds_workers=8" --reset

# 基于 240104-me9-11，在unet的mid layer中加入w2v2特征，使用 f0，但是只使用 m4singer
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_rme2.yaml --exp_name 240106-me9-1 --hparams "lr=0.000003,scheduler_lr_step_size=1500,max_sentences=32,use_mel_bins=40,ssl_feat_add_noise=gaussian:0.4,conformer_kernel=7,bkb_net=conformer,bkb_layers=2,frames_multiple=8,ds_workers=8,ds_names_in_training=m4" --reset

# 基于 240104-me9-11，在unet的mid layer中加入w2v2特征，使用 f0，但是只使用 m4singer 的 0.5
CUDA_VISIBLE_DEVICES=2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_rme2.yaml --exp_name 240106-me9-2 --hparams "lr=0.000003,scheduler_lr_step_size=1500,max_sentences=32,use_mel_bins=40,ssl_feat_add_noise=gaussian:0.4,conformer_kernel=7,bkb_net=conformer,bkb_layers=2,frames_multiple=8,ds_workers=8,ds_names_in_training=m4,dataset_downsample_rate=0.5" --reset

# 基于 240104-me9-11，在unet的mid layer中加入w2v2特征，使用 f0，但是只使用 m4singer 的 0.1
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_rme2.yaml --exp_name 240106-me9-3 --hparams "lr=0.000003,scheduler_lr_step_size=1500,max_sentences=32,use_mel_bins=40,ssl_feat_add_noise=gaussian:0.4,conformer_kernel=7,bkb_net=conformer,bkb_layers=2,frames_multiple=8,ds_workers=8,ds_names_in_training=m4,dataset_downsample_rate=0.1" --reset

# 基于 240104-me9-11，在unet的mid layer中加入w2v2特征，使用 f0，但是只使用 m4singer 的 0.02
CUDA_VISIBLE_DEVICES=2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_rme2.yaml --exp_name 240106-me9-4 --hparams "lr=0.000003,scheduler_lr_step_size=1500,max_sentences=32,use_mel_bins=40,ssl_feat_add_noise=gaussian:0.4,conformer_kernel=7,bkb_net=conformer,bkb_layers=2,frames_multiple=8,ds_workers=8,ds_names_in_training=m4,dataset_downsample_rate=0.02" --reset

####################### 第六部分：训练word bd ########################
# 基于 231230-me8-01
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_rwbd.yaml --exp_name 240205-rwbd-01 --reset

# 基于 240205-rwbd-01
CUDA_VISIBLE_DEVICES=2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_rwbd.yaml --exp_name 240205-rwbd-02 -hp "use_mel_bins=30,bkb_layers=1" --reset

# 基于 240104-me9-11
#CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_rwbd2.yaml --exp_name 240206-rwbd2-01 --reset
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_rwbd2.yaml --exp_name 240206-rwbd2-01 -hp "word_bd_threshold=0.6" --reset

