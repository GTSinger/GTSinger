# Technique Recognition

## Quick Start

Our model for technique recognition is tested in **python3.9, torch 2.1.1, CUDA 11.8**. We provide a series of install commands([install.sh](./research/singtech/scripts/install.sh)) for easier installation.

```bash
conda create -n tech-recog python==3.9
conda activate tech-recog
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install tensorflow==2.9.0 tensorflow-estimator==2.9.0 tensorboardX==2.5
pip install pyyaml matplotlib==3.5 pandas pyworld==0.2.12 librosa torchmetrics
pip install mir_eval pretty_midi pyloudnorm scikit-image textgrid g2p_en npy_append_array einops webrtcvad
export PYTHONPATH=.
```

We also provide the [despedencies](./requirements.txt) of the environment, you can also use the command below for installation after you install **torch 2.1.1**.

```bash
cd Tech-Recognition
pip install -r requirements.txt
```

## Data Preparation

1. Downlowad GTSinger and process all json to `metadata.json`.
2. Put `metadata.json` (including ph, word, item_name, ph_durs, wav_fn, singer, ep_pitches, ep_notedurs, ep_types for each singing voice) into `./processed_data_dir`.
3. Set `valid_prefixes`, `test_prefixes`, `processed_data_dir`, `binary_data_dir` in the [config](./research/singtech/config/te.yaml).
4. Preprocess Dataset 

```bash
cd Tech-Recognition
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