import os
import glob
import argparse
import json
import textgrid
from tqdm import tqdm

def global2tg(tg_path,global_json_path):
    tg = textgrid.TextGrid.fromFile(tg_path)
    word_tier = tg[0]
    with open(global_json_path, 'r') as f:
        global_json = json.load(f)
    
    pace = global_json['pace']
    range = global_json['range']
    singing_method = global_json['singing_method']
    emotion = global_json['emotion']
    # add new tier named global
    global_tier = textgrid.IntervalTier(name='global', maxTime=tg.maxTime)
    global_tier.addInterval(textgrid.Interval(minTime=0, maxTime=tg.maxTime, mark=f'{pace},{range},{singing_method},{emotion}'))
    
    assert global_tier[-1].maxTime == word_tier[-1].maxTime, f"global_tier and word_tier have different time length for {tg_path}\n word_tier:{word_tier[-1]},\n global_tier:{global_tier[-1]},\n word_tier.maxTime:{word_tier.maxTime},\n global_tier.maxTime:{global_tier.maxTime}"
    
    tg.append(global_tier)
    # save new tg
    tg.write(tg_path)
    
    
def global2json(json_path,global_json_path):
    with open(global_json_path, 'r') as f:
        global_json = json.load(f)
    with open(json_path, 'r') as f:
        word_dict_list = json.load(f)
    for i in range(len(word_dict_list)):
        word_dict_list[i]['pace'] = global_json['pace']
        word_dict_list[i]['range'] = global_json['range']
        word_dict_list[i]['singing_method'] = global_json['singing_method']
        word_dict_list[i]['emotion'] = global_json['emotion']
    with open(json_path, 'w') as f:
        json.dump(word_dict_list, f, ensure_ascii=False, indent=4)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data')
    args = parser.parse_args()
    
    data_dir = args.data_dir
    wav_dir = sorted(glob.glob(f'{data_dir}/*/*/*/*/*/*.wav'))
    
    wav_dir = [i for i in wav_dir if 'Speech' not in i]
    
    for wav_path in tqdm(wav_dir, desc='global2tg',total=len(wav_dir)):
        tg_path = wav_path.replace('.wav', '.TextGrid')
        global_json_path = wav_path.replace('.wav', '_global.json')
        global2tg(tg_path, global_json_path)
    
    for wav_path in tqdm(wav_dir, desc='global2json',total=len(wav_dir)):
        json_path = wav_path.replace('.wav', '.json')
        global_json_path = wav_path.replace('.wav', '_global.json')
        global2json(json_path, global_json_path)