# Technique Recognition

## Data Preparation

1. Downlowad GTSinger and process all json to `metadata.json`.
2. Put `metadata.json` (including ph, word, item_name, ph_durs, wav_fn, singer, ep_pitches, ep_notedurs, ep_types for each singing voice) into `./processed_data_dir`.
3. Set `valid_prefixes`, `test_prefixes`, `processed_data_dir`, `binary_data_dir` in the [config](./research/singtech/config/te.yaml).
4. Preprocess Dataset 

```bash
export PYTHONPATH=.
python data_gen/tts/runs/binarize.py --config research/singtech/config/te.yaml
```

## Training Technique Recognition

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config research/singtech/config/te2.yaml --exp_name <your exp_name> --reset
```

## Inference

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name <your exp_name> --infer --hparams "load_ckpt={path-to-the-best-ckpt}"
```

## Acknowledgements

This implementation uses parts of the code from the following Github repos: [ROSVOT](https://github.com/RickyL-2000/ROSVOT) as described in our code.