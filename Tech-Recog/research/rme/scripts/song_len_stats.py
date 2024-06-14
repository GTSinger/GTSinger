# %%
import os
import glob
from pathlib import Path
import wave
import librosa
import soundfile as sf
from tqdm import tqdm

# %%
# root_dir = r'D:\MyDrive\09_Research\YiWiseLab\projects\华为歌声项目TTSing二期2023\文件\男声小批量样例1206\男声小批量样例1206'
# root_dir = r'D:\MyDrive\09_Research\YiWiseLab\projects\华为歌声项目TTSing二期2023\文件\男声小批量样例1206\男声小批量样例1206'
# root_dir = r'/mnt/sdb/liruiqi/SingingDictation/data/raw/temp/女声第1批'
root_dir = r'/mnt/sdb/liruiqi/SingingDictation/data/raw/temp/男声第1批231220'

def get_wav_num_frames(path, sr=None):
    try:
        with wave.open(path, 'rb') as f:
            sr_ = f.getframerate()
            if sr is None:
                sr = sr_
            return int(f.getnframes() / (sr_ / sr))
    except wave.Error:
        wav_file, sr_ = sf.read(path, dtype='float32')
        if sr is None:
                sr = sr_
        return int(len(wav_file) / (sr_ / sr))
    except:
        if sr is None:
            wav_file, sr_ = librosa.core.load(path, sr=None)
        else:
            wav_file, _ = librosa.load(path, sr)
        return len(wav_file)

wav_paths = []
for root, dirs, files in os.walk(root_dir):
    if len(files) > 0:
        for f_name in files:
            if Path(f_name).suffix in ['.mp3', '.wav']:
                wav_paths.append(os.path.join(root, f_name))

total_sec = 0
sr = 48000
for wav_path in tqdm(wav_paths, total=len(wav_paths)):
    total_sec += get_wav_num_frames(wav_path, sr) / sr

print("seconds", total_sec)
print("hours", total_sec / 3600)