import os
from pathlib import Path
import argparse
from tqdm import tqdm
import librosa

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--read', action='store_true', help='Enable overwrite')
    
    args = parser.parse_args()
    wav_dir = args.data_dir
    read_length_flag = args.read

    # get all wav files
    wav_paths = []
    for root, dirs, files in os.walk(wav_dir):
        if len(files) > 0:
            for f_name in files:
                if Path(f_name).suffix in ['.wav']:
                    wav_paths.append(os.path.join(root, f_name))
    wav_paths = sorted(wav_paths)

    # the statistics of time
    time = 0
    for wav_path in tqdm(wav_paths, total=len(wav_paths)):
        wav_duration = librosa.get_duration(filename=wav_path)
        if read_length_flag == False:
            if 'Speech' not in wav_path:
                time += wav_duration
        else:
            if 'Speech' in wav_path:
                time += wav_duration

    time=time/3600.0
    if read_length_flag == True:
        print("The reading time is", time)
    else:
        print("The singing time is", time)