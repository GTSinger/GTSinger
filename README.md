#  GTSinger: A Global Multi-Technique Singing Corpus with Realistic Music Scores for All Singing Tasks

We introduce GTSinger, a large Global, multi-Technique, free-to-use, high-quality singing corpus with realistic music scores, designed for all singing tasks, along with its benchmarks.

We provide the processing codes for our dataset and benchmarks' implementation in this repository. 

Also, you can visit our [Demo Page](https://gtsinger.github.io/) for the audio samples of our dataset as well as the results of our benchmarks.

## Dataset

### Where to download

Click [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?label=Dataset)](https://huggingface.co/datasets/GTSinger/GTSinger) to access our dataset (audio along with TextGrid, json, musicxml) on Hugging Face **for free**! Hope our data is helpful for your research.

Besides, we also provide our raw data on [Google Drive](https://drive.google.com/drive/folders/1vYtq_b7-rdBB4nIkzpfiTb6zFJJhgOP_?usp=drive_link).

**Please note that, if you are using GTSinger, it means that you have accepted the terms of [license](https://github.com/GTSinger/GTSinger/blob/master/dataset_license.md).**

### Data Architecture

Our dataset is organized hierarchically. 

It presents nine top-level folders, each corresponding to a distinct language. 

Within each language folder, there are five sub-folders, each representing a specific singing technique.

These technique folders contain numerous song entries, with each song further divided into several controlled comparison groups: a control group (natural singing without the specific technique), and a technique group (densely employing the specific technique).

Our singing voices and speech are recorded at a 48kHz sampling rate with 24-bit resolution in WAV format. 

Alignments and annotations are provided in TextGrid files, including word boundaries, phoneme boundaries, phoneme-level annotations for six techniques, and global style labels (singing method, emotion, pace, and range). 

We also provide realistic music scores in musicxml format.

Notably, we provide an additional JSON file for each singing voice, facilitating data parsing and processing for singing models.

Here is the data structure of our dataset:

```
.
├── Chinese
│   ├── ZH-Alto-1
│   └── ZH-Tenor-1
├── English
│   ├── EN-Alto-1
│   │   ├── Breathy
│   │   ├── Glissando
│   │   │   └── my love
│   │   │       ├── Control_Group
│   │   │       ├── Glissando_Group
│   │   │       └── Paired_Speech_Group
│   │   ├── Mixed_Voice_and_Falsetto
│   │   ├── Pharyngeal
│   │   └── Vibrato
│   ├── EN-Alto-2
│   │   ├── Breathy
│   │   ├── Glissando
│   │   ├── Mixed_Voice_and_Falsetto
│   │   ├── Pharyngeal
│   │   └── Vibrato
│   └── EN-Tenor-1
│       ├── Breathy
│       ├── Glissando
│       ├── Mixed_Voice_and_Falsetto
│       ├── Pharyngeal
│       └── Vibrato
├── French
│   ├── FR-Soprano-1
│   └── FR-Tenor-1
├── German
│   ├── DE-Soprano-1
│   └── DE-Tenor-1
├── Italian
│   ├── IT-Bass-1
│   ├── IT-Bass-2
│   └── IT-Soprano-1
├── Japanese
│   ├── JA-Soprano-1
│   └── JA-Tenor-1
├── Korean
│   ├── KO-Soprano-1
│   ├── KO-Soprano-2
│   └── KO-Tenor-1
├── Russian
│   └── RU-Alto-1
└── Spanish
    ├── ES-Bass-1
    └── ES-Soprano-1
```

### Code for preprocessing

The code for processing the dataset is provided in the `./Data-Process`.

#### Dependencies

A suitable [conda](https://docs.conda.io/en/latest/) environment named `gt_dataprocess` can be created and activated with:

```bash
conda create -n gt_dataprocess python=3.8 -y
conda activate gt_dataprocess
pip install -r requirements.txt
```

#### Data Check

The code for checking the dataset is provided in `./Data-Process/data_check/`, including the following files:

- `check_file_and_folder.py`: Check the file and folder structure of the dataset.

- `check_valid_bandwidth.py`: Check the sample rate and valid bandwidth of the dataset.

- `count_time.py`: Count the time of the singing voice and speech in the dataset.

- `plot_f0.py`: Plot the pitch(f0) of the singing voice audio.

- `plot_mel.py`: Plot the mel-spectrogram of audio.

#### Data Preprocessing

The code for preprocessing the dataset is provided in `./Data-Process/data_preprocess/`, including the following files:

- `gen_final_json.py`: Generate the final JSON file for each singing voice based on the TextGrid file and musicxml file that have been annotated.

- `global2tgjson.py`: Add global style labels to the JSON file and TextGrid file.

- `seg_singing.py` & `seg_speech.py`: Segment the singing voice and speech based on the TextGrid file.

## Technique Controllable Singing Voice Synthesis

The code for our benchmarks for [Technique Controllable Singing Voice Synthesis](./Technique-Controllable%20SVS/readme.md).

## Technique Recognition

The code for our benchmarks for [Technique Recognition](./Tech-Recognition/readme.md).

## Style Transfer

The code for our benchmarks for [Style Transfer](./Style%20Transfer/readme.md).

## Speech-to-Singing Conversion

The code for our benchmarks for [Speech-to-Singing-Conversion](./STS%20Conversion/readme.md).

 ![visitors](https://visitor-badge.laobi.icu/badge?page_id=GTSinger/GTSinger)
