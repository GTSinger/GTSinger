# auto restart: counter-oom
while true
do
	ps -ef | grep tasks/run.py | grep -v "grep"
	if [ "$?" -eq 1 ]
	then

		CUDA_VISIBLE_DEVICES=0,1,2,3 PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 python tasks/run.py --config research/rme/config/base_rme.yaml --exp_name 231226-me5-01 --hparams "use_mel_bins=30" --reset

		echo "process has been restarted!"
	fi
	sleep 10
done

