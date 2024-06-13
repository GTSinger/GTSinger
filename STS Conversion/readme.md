# STS Conversion

Download GTSinger and try STS conversion.

### Data Preparation 

1. Downlowad GTSinger and process all json to `metadata.json`.
2. Put `metadata.json` , `spker_set.json` (including all singers and their id), and `phone_set.json` (all phonemes of your dictionary) in `data/processed/style`
3. Set `processed_data_dir`, `binary_data_dir`,`valid_prefixes`, `test_prefixes` in the [config](./configs/singing/speech2singing/alignsts.yaml).

### Training

Run
```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name alignsts --reset --hparams "gen_dir_name=test" --config configs/singing/speech2singing/alignsts.yaml --reset
```

### Inference

`Run
```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name alignsts --infer --hparams "gen_dir_name=test" --config configs/singing/speech2singing/alignsts.yaml --reset
```

## Acknowledgements
This implementation uses parts of the code from the following Github repos:
[AlignSTS](https://github.com/RickyL-2000/AlignSTS)