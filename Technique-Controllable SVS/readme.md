# Technique-Controllable SVS

### Data Preparation 

1. Downlowad GTSinger and process all json to `metadata.json`.
2. Put `metadata.json` (including ph, word, item_name, ph_durs, tech,mix_tech, falsetto_tech, breathy_tech, pharyngeal_tech, vibrato_tech, glissando_tech, wav_fn, singer, ep_pitches, ep_notedurs, ep_types for each singing voice), `spker_set.json` (including all singers and their id), and `phone_set.json` (all phonemes of your dictionary) in `data/processed/style`
3. Set `processed_data_dir`, `binary_data_dir`,`valid_prefixes`, `test_prefixes` in the [config](./egs/stylesinger.yaml).
4. Preprocess Dataset 

```bash
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=$GPU python data_gen/tts/bin/binarize.py --config egs/stylesinger.yaml
```

### Training StyleSinger

```bash
CUDA_VISIBLE_DEVICES=$GPU python tasks/run.py --config egs/stylesinger.yaml  --exp_name StyleSinger --reset
```

### Inference using StyleSinger

```bash
CUDA_VISIBLE_DEVICES=$GPU python tasks/run.py --config egs/stylesinger.yaml  --exp_name StyleSinger --infer
```

## Acknowledgements

This implementation uses parts of the code from the following Github repos:
[StyleSinger](https://github.com/AaronZ345/StyleSinger)
as described in our code.
