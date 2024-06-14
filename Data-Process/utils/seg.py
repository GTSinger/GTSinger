import os
import audio
import json

# process the segementation of wav
def seg_wav(item_name, sen_id, sr, start_time, end_time, wav_data, target_dir):
    seg_wav_path = item_name + '#' + str(sen_id)
    min_frame = int(sr * start_time)
    max_frame = int(sr * end_time)
    wav_out = wav_data[min_frame:max_frame]
    target_dir = f'{target_dir}/{item_name}'
    os.makedirs(target_dir, exist_ok=True)
    audio.save_wav(wav_out, f'{target_dir}/{seg_wav_path}.wav', sr)
    
def check_empty(segments):
    for seg in segments:
        if seg["word"] not in [ "_NONE",'<SP>','<AP>']:
            return False
    return True
