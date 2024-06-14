import glob
import os
import textgrid
import json
import argparse
import argparse
import tqdm
import librosa

from data_process.utils.phonemes import ALL_PHONES_LIST, enum

def get_wav_time(wav_path):
    """
    Args:
        wav_path: the absolute path of the wav file
    """
    # print(wav_path)
    if not os.path.exists(wav_path):
        print(f'{wav_path} not exists')
        return None
    try:
        # song = AudioSegment.from_wav(wav_path)
        # print(song.channels)    
        # return (song.duration_seconds)
        wav_duration = librosa.get_duration(filename=wav_path)
        return wav_duration
    except Exception as e:
        print(f'[ERROR] the wav file in \033[93m{wav_path}\033[0m cannot be opend!\n')
        # return None
        return 0

def merge_small_silence(tg_path):
    try:
        tg = textgrid.TextGrid.fromFile(tg_path)
    except:
        print(f'[ERROR] Cannot open {tg_path}')
        return
    
    word_intervals = tg[0]
    ph_intervals = tg[1]
    
    new_word_intervals = textgrid.IntervalTier(name='word')
    new_ph_intervals = textgrid.IntervalTier(name='phone')
    
    for i, word in enumerate(word_intervals):
        mark = word.mark
        # if the interval is empty and the duration is less than 0.1s, merge it with the next interval
        if mark in ['',None,' '] and float(word.maxTime) - float(word.minTime) < 0.1:
            # print(f'[WARNING] {tg_path} word {word} is empty and will be merged')
            if i < len(word_intervals) - 1:
                word_intervals[i+1].minTime = word.minTime
                continue
            else:
                new_word_intervals[-1].maxTime = word.maxTime
                continue
        # remove the leading and trailing spaces
        mark = mark.strip()
        new_word_intervals.addInterval(word)
        
    for i in range(len(new_word_intervals)):
        new_word_intervals[i].minTime = round(float(new_word_intervals[i].minTime),3)
        new_word_intervals[i].maxTime = round(float(new_word_intervals[i].maxTime),3)
    
    # process the phone intervals
    for i,ph in enumerate(ph_intervals):
        mark = ph.mark
        if mark in ['',None,' '] and float(ph.maxTime) - float(ph.minTime) < 0.1:
            print(f'[WARNING] {tg_path} phone {ph} is empty and will be merged')
            if i < len(ph_intervals) - 1:
                ph_intervals[i+1].minTime = ph.minTime
                continue
            else:
                new_ph_intervals[-1].maxTime = ph.maxTime
                continue
        mark = mark.strip()
        # if float(ph.maxTime) - float(ph.minTime) < 0.019:
        #     print(f'phone duration is too short!\n{relative_name}\n\033[93{new_ph_intervals[i]}\033[0m\n')
        ph.mark = ph.mark.strip()
        new_ph_intervals.addInterval(ph)
    
    for i,ph in enumerate(new_ph_intervals):
        new_ph_intervals[i].minTime = round(float(new_ph_intervals[i].minTime),3)
        new_ph_intervals[i].maxTime = round(float(new_ph_intervals[i].maxTime),3)
        
    # write the new textgrid
    new_tg = textgrid.TextGrid()
    new_tg.append(new_word_intervals)
    new_tg.append(new_ph_intervals)
    
    new_tg.write(tg_path) 

def tg2text(tg_path):
    """generate text file from textgrid file

    Args:
        tg_path (str): the path of the textgrid file
    """
    # print(tg_path)
    tg = textgrid.TextGrid.fromFile(tg_path)
    word_intervals = tg[0]
    ph_intervals = tg[1]
    
    assert word_intervals[-1].maxTime==ph_intervals[-1].maxTime
    
    base_name = os.path.basename(tg_path)
    word_name = base_name.replace('.TextGrid','_word.txt')
    ph_name = base_name.replace('.TextGrid','_ph.txt')
    dir_path = os.path.dirname(tg_path)
    with open(os.path.join(dir_path,word_name),'w') as f:
        for word in word_intervals:
            if mark is None or mark in ['',' ','_None']:
                mark = "_NONE"
            elif mark in ['breath','breathe','bre']:
                mark='breathe'
            mark=mark.strip()
            write_line = f'{mark} || {str(word.minTime)} || {str(word.maxTime)}\n'
            f.write(write_line)
    with open(os.path.join(dir_path,ph_name),'w') as f:
        for ph in ph_intervals:
            mark = ph.mark.strip()
            if mark is None or mark in ['',' ','_None']:
                mark = "_NONE"
            elif mark in ['breath','breathe','bre']:
                mark='breathe'
            mark=mark.strip()
            write_line = f'{mark} || {str(ph.minTime)} || {str(ph.maxTime)}\n'
            f.write(write_line)
            
def tg2json(tg_path,language='english'):
    name_list = tg_path.split('/')[-4:]
    base_name = '/'.join(name_list)
    # print(base_name)
    word_dict_list = []
    # get the word and phone information from the txt file
    word_path = tg_path.replace('.TextGrid','_word.txt')
    ph_path = tg_path.replace('.TextGrid','_ph.txt')
    with open(word_path,'r') as f:
        word_lines = f.readlines()
    with open(ph_path,'r') as f:
        ph_lines = f.readlines()
    
     # first we process the word information
    word_id = -1
    for word in word_lines:
        word = word.strip()
        word_dict = {}
        mark, min, max = word.split(' || ')
        mark = mark.strip() 
        mark = mark.replace('9','')
        if mark is None or mark in ['',' ','_NONE']:
            mark = '<SP>'
       
        word_dict["word"] = mark
        word_dict["start_time"] = round(float(min),3)
        word_dict["end_time"] = round(float(max),3)
        if mark != '<SP>':
            word_id = word_id + 1
        word_dict['word_id'] = word_id
        word_dict["ph"] = []
        word_dict["ph_start"] = []
        word_dict["ph_end"] = []
        word_dict_list.append(word_dict)
        
    # then we process the phone information
    idx = 0
    for i,ph in enumerate(ph_lines):
        if idx >= len(word_dict_list):
            print(f'{ph}{idx}, word_dict_list length is {len(word_dict_list)}')
            for element in word_dict_list:
                print(f'{element["word"]},{element["ph"]}')
            assert False,f'{base_name}'
        ph = ph.strip()
        next_ph = ph_lines[i+1].strip() if i+1<len(ph_lines) else None
        mark, min, max = ph.split(' || ')
        next_mark = next_min = next_max = None
        if next_ph is not None:
            next_mark, next_min, next_max = next_ph.split(' || ')
        # for SP
        if mark is None or mark in ['',' ','_NONE']:
            word_dict_list[idx]["ph"].append('<SP>')
            word_dict_list[idx]["ph_start"].append(round(float(word_dict_list[idx]["start_time"]),3))
            word_dict_list[idx]["ph_end"].append(round(float(word_dict_list[idx]["end_time"]),3))
            idx = idx + 1
        # normal phone
        else:
            if next_max is not None and float(next_max) < float(word_dict_list[idx]["end_time"]) + 0.005:
                word_dict_list[idx]["ph"].append(mark)
                # add phone start
                if len(word_dict_list[idx]["ph_start"]) == 0:
                    word_dict_list[idx]["ph_start"].append(round(float(word_dict_list[idx]["start_time"]),3))
                else:
                    word_dict_list[idx]["ph_start"].append(round(float(min),3))
                # add phone end
                if next_min is not None and float(next_min) - float(max) < 0.005:
                    word_dict_list[idx]["ph_end"].append(round(float(next_min),3))
                elif next_min is None:
                    word_dict_list[idx]["ph_end"].append(round(float(word_dict_list[idx]["end_time"]),3))
                else:
                    print(f'{base_name} {i} {ph} {next_ph}, the phone is not align')
            else:
                # if the phone is the last phone of the word
                word_dict_list[idx]["ph"].append(mark)
                # add phone start
                if len(word_dict_list[idx]["ph_start"]) == 0:
                    word_dict_list[idx]["ph_start"].append(round(float(word_dict_list[idx]["start_time"]),3))
                else:
                    word_dict_list[idx]["ph_start"].append(round(float(min),3))
                # add and adjust phone end
                word_dict_list[idx]["ph_end"].append(round(float(word_dict_list[idx]["end_time"]),3))
                idx = idx + 1
    assert idx == len(word_dict_list), f"{tg_path}ph 未对齐, {idx} != {len(word_dict_list)}"
    
    # check for spn
    for i in range(len(word_dict_list)):
        if 'spn' in word_dict_list[i]["ph"]:
            if word_dict_list[i]['word'] == 'ohhh':
                print(f'[Invalid Phone] {base_name} {word_dict_list[i]}')
    # check the alignment of word and phone
    assert word_dict_list[0]['start_time'] == 0, f'{base_name} start_time is not 0'
    assert word_dict_list[-1]['end_time'] == word_dict_list[-1]['ph_end'][-1], f'{base_name} end_time is not equal to ph_end'
    for i in range(len(word_dict_list)-1):
        assert word_dict_list[i]['end_time'] == word_dict_list[i+1]['start_time'], f'{base_name} {i} {word_dict_list[i]}word and word not align'
        assert round(word_dict_list[i]['ph_end'][-1],3) == round(word_dict_list[i+1]['ph_start'][0],3), f'{base_name} {i} {word_dict_list[i]}ph and ph not align'
        # adjust the start and end time of the word
        word_dict_list[i]['ph_start'][0] == word_dict_list[i]['start_time']
        word_dict_list[i]['ph_end'][-1] = word_dict_list[i]['end_time']
    # adjust the start and end time of the phone
        for j in range(len(word_dict_list[i]['ph'])-1):
            word_dict_list[i]['ph_end'][j] = word_dict_list[i]['ph_start'][j+1]
    
    # check for breathe
    for i in range(len(word_dict_list)):
        if word_dict_list[i]['word'] == 'breathe' and word_dict_list[i]['ph'] == ['breathe']:
            word_dict_list[i]['word'] = '<SP>'
            word_dict_list[i]['ph'] = ['<SP>']
            word_dict_list[i]['ph_start'] = [word_dict_list[i]['start_time']]
            word_dict_list[i]['ph_end'] = [word_dict_list[i]['end_time']]
            
    
    # check the validity of the phones
    for i in range(len(word_dict_list)):
        for ph in word_dict_list[i]['ph']:
            if ph in ALL_PHONES_LIST[enum[language]] or ph == '<SP>':
                continue
            else:
                print(f'[ph error]\n{base_name},{word_dict_list[i]}{ph}')
    # round the time to 3 decimal places
    for i in range(len(word_dict_list)):
        word_dict_list[i]['start_time'] = round(word_dict_list[i]['start_time'],3)
        word_dict_list[i]['end_time'] = round(word_dict_list[i]['end_time'],3)
        word_dict_list[i]['ph_start'] = [round(element,3) for element in word_dict_list[i]['ph_start']]
        word_dict_list[i]['ph_end'] = [round(element,3) for element in word_dict_list[i]['ph_end']]
    # check the time alignment
    for i in range(len(word_dict_list)):
        assert word_dict_list[i]['end_time'] > word_dict_list[i]['start_time'], f'{base_name} {i} {word_dict_list[i]} word end_time < start_time'
        for j in range(len(word_dict_list[i]['ph'])):
            assert word_dict_list[i]['ph_end'][j] > word_dict_list[i]['ph_start'][j], f'{base_name} {i} {word_dict_list[i]} phone end_time < start_time'
    
    with open(tg_path.replace('.TextGrid','_tg.json'),'w',encoding='utf-8') as f:
        json.dump(word_dict_list,f,ensure_ascii=False,indent=4)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    # parser.add_argument('--overwrite', action='store_true', help='Enable overwrite')
    parser.add_argument("--language", type=str, help="The language of the sing", default='english')
    args = parser.parse_args()
    data_dir = args.data_dir
    language = args.language
    tg_dir = sorted(glob.glob(f"{data_dir}/*/*/*/*.TextGrid"))
    speech_dir = [i for i in tg_dir if 'Speech' in i]
    
    for tg in tqdm.tqdm(speech_dir,desc='Processing'):
        wav_path = tg.replace('.TextGrid','.wav')
        wav_duration = get_wav_time(wav_path)
        # compare the duration of the wav file and the textgrid file
        tg_data = textgrid.TextGrid.fromFile(tg)
        if round(wav_duration,3) != round(tg_data.maxTime,3):
            print(f'[Time Error]{tg} wav duration is not equal to textgrid duration,wav_duration:{wav_duration},tg_duration:{tg_data.maxTime}')
            continue
        dir_path, filename = os.path.split(tg)
        parent_dir, last_dir = os.path.split(dir_path)
        # check the last dir
        if last_dir != '朗读':
            print(f'[Dir Error]{tg} is not in 朗读')
            continue
        merge_small_silence(tg)
        tg2text(tg)
        # print({tg})
        tg2json(tg,language)
        # remove the txt file
        os.remove(tg.replace('.TextGrid','_word.txt'))
        os.remove(tg.replace('.TextGrid','_ph.txt'))
        
    print(f'finished!\ndata dir: \033[92m{data_dir}\033[0m\nlanguage: \033[92m{language}\033[0m')
        