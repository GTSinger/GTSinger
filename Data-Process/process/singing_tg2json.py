# python -u "check_tg_midi_foreign.py" --data-dir ~/Desktop/俄语_女声_第3周录音 --language russian
import glob
import os
import textgrid
import json
import argparse
import numpy as np
import argparse
import difflib
import tqdm
from data_process.utils.phonemes import ALL_PHONES_LIST, enum


# Check word list consistency between two TextGrid files
def check_word_list_consitency(tg1_path,tg2_path):
    tg1 = textgrid.TextGrid.fromFile(tg1_path)
    tg2 = textgrid.TextGrid.fromFile(tg2_path)
    tg1_words = [word.mark.strip().replace('â€™',"'").replace("’","'").replace('ã¼','ü').replace('·','') for word in tg1[0] if word.mark.strip() not in ['breathe','_NONE','breath']]
    tg2_words = [word.mark.strip().replace('â€™',"'").replace("’","'").replace('ã¼','ü').replace('·','') for word in tg2[0] if word.mark.strip() not in ['breathe','_NONE','breath']]
    if tg1_words != tg2_words:
        print(f'Error: \n\033[93m{tg1_path}\033[0m and \n\033[93m{tg2_path}\033[0m \nhave different words\n')
        list1 = tg1_words
        list2 = tg2_words
        basename1 = os.path.basename(tg1_path)
        basename2 = os.path.basename(tg2_path)
        s = difflib.SequenceMatcher(None, list1, list2)
        output1 = []
        output2 = []
        for tag, i1, i2, j1, j2 in s.get_opcodes():
            if tag == 'replace':
                output1.extend(['\033[93m' + item + '\033[0m' for item in list1[i1:i2]])
                output2.extend(['\033[93m' + item + '\033[0m' for item in list2[j1:j2]])
            elif tag == 'delete':
                output1.extend(['\033[93m' + item + '\033[0m' for item in list1[i1:i2]])
                output2.extend([''] * (i2-i1))
            elif tag == 'insert':
                output1.extend([''] * (j2-j1))
                output2.extend(['\033[93m' + item + '\033[0m' for item in list2[j1:j2]])
            elif tag == 'equal':
                output1.extend(['\033[92m' + item + '\033[0m' for item in list1[i1:i2]])
                output2.extend(['\033[92m' + item + '\033[0m' for item in list2[j1:j2]])

        print(basename1)
        for item in output1:
            print(item, end = ' ')
        print('\n')
        print(basename2)
        for item in output2:
            print(item, end = ' ')
        print('\n')

# Batch check word list consistency across TextGrid files in a directory
def batch_check_word_list(data_dir):
    for tg_path in sorted((glob.glob(f"{data_dir}/*/*/*/*.TextGrid"))):
        if 'speech' not in tg_path and 'control' not in tg_path:
            # This indicates the TextGrid is for vocal techniques
            tech_base_name = os.path.basename(tg_path)
            sex_name = tech_base_name.split('_')[0]
            number_name = tech_base_name.split('_')[-1]
            wujiqiao_base_name = f'{sex_name}_control_{number_name}' if len(tech_base_name.split('_'))==3 else f'{sex_name}_control.TextGrid'
            tg_path_list = tg_path.split('/')
            wujiqiao_path = '/'.join(tg_path_list[:-2]) + '/control/' + wujiqiao_base_name
            check_word_list_consitency(tg_path,wujiqiao_path)
    print('################finished check word list!########################')
    return True

def tg2text(tg_path):
    """generate text file from textgrid file

    Args:
        tg_path (str): the path of the textgrid file
    """
    # print(tg_path)
    tg = textgrid.TextGrid.fromFile(tg_path)
    word_intervals = tg[0]
    ph_intervals = tg[1]
    tech_intervals = tg[2]
    note_intervals = tg[3]
    
    assert len(word_intervals) == len(tech_intervals)
    assert word_intervals[-1].maxTime==tech_intervals[-1].maxTime==note_intervals[-1].maxTime==ph_intervals[-1].maxTime
    
    base_name = os.path.basename(tg_path)
    word_name = base_name.replace('.TextGrid','_word.txt')
    tech_name = base_name.replace('.TextGrid','_tech.txt')
    note_name = base_name.replace('.TextGrid','_note.txt')
    ph_name = base_name.replace('.TextGrid','_ph.txt')
    dir_path = os.path.dirname(tg_path)
    with open(os.path.join(dir_path,word_name),'w') as f:
        for word in word_intervals:
            mark = word.mark.strip().replace('â€™',"'").replace("’","'").replace('ã¼','ü').replace('√º','ü').replace('√§','ä').replace('체','ü').replace('채','ä').replace('ã¤','ä').replace('√ü','ß')
            if mark is None or mark in ['','silence','sil','_None']:
                mark = "_NONE"
            elif mark in ['breath','breathe','bre']:
                mark='breathe'
            mark=mark.strip()
            write_line = f'{mark} || {str(word.minTime)} || {str(word.maxTime)}\n'
            f.write(write_line)
    with open(os.path.join(dir_path,ph_name),'w') as f:
        for ph in ph_intervals:
            mark = ph.mark.strip()
            if mark is None or mark in ['','silence','sil','_None']:
                mark = "_NONE"
            elif mark in ['breath','breathe','bre']:
                mark='breathe'
            mark=mark.strip()
            write_line = f'{mark} || {str(ph.minTime)} || {str(ph.maxTime)}\n'
            f.write(write_line)
    with open(os.path.join(dir_path,tech_name),'w') as f:
        for i, tech in enumerate(tech_intervals):
            mark = tech.mark
            if '[' in mark or ']' in mark:
                print(tg_path,'idx:',i,'word:',word_intervals[i].mark,'tech',mark)
            if '#' in mark and '[' not in mark and ']' not in mark:
                mark = mark.replace('#',',')
            if mark in ['',' ',0,'0']:
                mark='0'
            mark=mark.replace('，',',')
            mark=mark.strip()
            line = mark + ' || ' + str(word_intervals[i].minTime) + ' || ' + str(word_intervals[i].maxTime) + '\n'
            f.write(line)
    with open(os.path.join(dir_path,note_name),'w') as f:
        for note in note_intervals:
            mark = note.mark.strip()
            write_line = f'{mark} || {str(note.minTime)} || {str(note.maxTime)}\n'
            f.write(write_line)
            
def check_phones(ph_path,language):
    with open(ph_path,'r') as f:
        ph_lines = f.readlines()
    all_phones = ALL_PHONES_LIST[enum[language]]
    for line in ph_lines:
        # line = line.strip()
        # print(line)
        ph = line.split(' || ')[0].strip()
        # print(ph)
        if ph not in all_phones and ph not in ['_NONE','breathe','breath','spn']:
            print(f'Error: {ph_path}\n{ph} not in phone list')
        
def tg2json(tg_path, language='english'):
    name_list = tg_path.split('/')[-4:]
    base_name = '/'.join(name_list)
    word_dict_list = []
    # First, convert the TextGrid file to text files to get the corresponding text file paths
    word_path = tg_path.replace('.TextGrid', '_word.txt')
    ph_path = tg_path.replace('.TextGrid', '_ph.txt')
    tech_path = tg_path.replace('.TextGrid', '_tech.txt')
    note_path = tg_path.replace('.TextGrid', '_note.txt')
    with open(word_path, 'r') as f:
        word_lines = f.readlines()
    with open(ph_path, 'r') as f:
        ph_lines = f.readlines()
    with open(tech_path, 'r') as f:
        tech_lines = f.readlines()
    with open(note_path, 'r') as f:
        note_lines = f.readlines()
    
    # First, handle word information
    word_id = -1
    for word in word_lines:
        word = word.strip()
        word_dict = {}
        mark, min, max = word.split(' || ')
        mark = mark.strip()  # This part is important
        word_dict["word"] = mark
        word_dict["start_time"] = round(float(min), 3)
        word_dict["end_time"] = round(float(max), 3)
        if mark not in ["breathe", "breath", "_NONE"]:
            word_id = word_id + 1
        word_dict['word_id'] = word_id
        word_dict["note"] = []
        word_dict["note_start"] = []
        word_dict["note_end"] = []
        word_dict["ph"] = []
        word_dict["ph_start"] = []
        word_dict["ph_end"] = []
        word_dict_list.append(word_dict)
        
    # Next, handle phone information
    idx = 0
    for i, ph in enumerate(ph_lines):
        ph = ph.strip()
        next_ph = ph_lines[i+1].strip() if i+1 < len(ph_lines) else None
        mark, min, max = ph.split(' || ')
        next_mark = next_min = next_max = None
        if next_ph is not None:
            next_mark, next_min, next_max = next_ph.split(' || ')
        mark = mark.strip()
        # First, handle '_NONE' and 'breathe' cases
        if mark in ["breathe", "breath", "_NONE"]:
            word_dict_list[idx]["ph"].append(mark)
            word_dict_list[idx]["ph_start"].append(round(float(word_dict_list[idx]["start_time"]), 3))
            word_dict_list[idx]["ph_end"].append(round(float(word_dict_list[idx]["end_time"]), 3))
            idx = idx + 1
        # Normal phones
        else:
            if next_max is not None and float(next_max) < float(word_dict_list[idx]["end_time"]) + 0.005:
                word_dict_list[idx]["ph"].append(mark)
                # Add phone's start
                if len(word_dict_list[idx]["ph_start"]) == 0:  # This is the first phone
                    word_dict_list[idx]["ph_start"].append(round(float(word_dict_list[idx]["start_time"]), 3))
                else:
                    word_dict_list[idx]["ph_start"].append(round(float(min), 3))
                # Add phone's end
                if next_min is not None and float(next_min) - float(max) < 0.005:
                    word_dict_list[idx]["ph_end"].append(round(float(next_min), 3))
                elif next_min is None:
                    word_dict_list[idx]["ph_end"].append(round(float(word_dict_list[idx]["end_time"]), 3))
                else:
                    print(f'{base_name} {i} {ph} {next_ph}, the phone is not aligned')
            else:
                # When next_max is greater than end_time or next_max does not exist, it means the current phone is the last phone of this word, and we need to operate on idx
                word_dict_list[idx]["ph"].append(mark)
                # Add phone's start
                if len(word_dict_list[idx]["ph_start"]) == 0:
                    word_dict_list[idx]["ph_start"].append(round(float(word_dict_list[idx]["start_time"]), 3))
                else:
                    word_dict_list[idx]["ph_start"].append(round(float(min), 3))
                # Add phone's end, since it's the last phone, align it directly with the word's end time
                word_dict_list[idx]["ph_end"].append(round(float(word_dict_list[idx]["end_time"]), 3))
                idx = idx + 1
    assert idx == len(word_dict_list), f"ph not aligned, {idx} != {len(word_dict_list)}"
    
    # Next, handle tech information
    idx = 0
    for tech in tech_lines:
        tech = tech.strip()
        mark, min, max = tech.split(' || ')
        # First, remove all glissando techniques, as glissando is directly recognized by midi
        mark = mark.strip()
        mark = mark.replace(',6', '').replace('6', '0')
        for s in mark:
            if s not in ['0', '1', '2', '3', '4', '5', ',']:
                print(f'{base_name} {idx} {tech}, wrong tech mark')
        word_dict_list[idx]["tech"] = mark
        idx = idx + 1
    assert idx == len(word_dict_list), f"tech not aligned, {idx} != {len(word_dict_list)}"
    
    # Next, handle note information
    # Note alignment
    idx = 0
    key_num = 0
    for note in note_lines:
        note = note.strip()
        mark, min, max = note.split(' || ')
        mark = int(mark)
        if mark > 300:
            # Divide by 10 and round
            mark = mark // 10
        assert 30 < mark < 90 or mark == 0, f'{mark} is not correct'
        min = round(float(min), 3)
        max = round(float(max), 3)
        if max - min < 0.02:
            assert False, f'{max - min} too short note'
        # Check if word and note are aligned
        if min >= word_dict_list[idx]["start_time"] - 0.01 and max <= word_dict_list[idx]["end_time"] + 0.01:
            # First, remove the same pitch within a word
            if key_num > 0 and mark == word_dict_list[idx]["note"][-1]:
                word_dict_list[idx]["note_end"][-1] = max  # Update end_time to the latest max
            else:
                word_dict_list[idx]["note"].append(mark)
                if key_num == 0:  # If it is the first note, start_time is aligned with the word's start_time
                    word_dict_list[idx]["note_start"].append(round(float(word_dict_list[idx]["start_time"]), 3))
                else:
                    word_dict_list[idx]["note_start"].append(round(float(word_dict_list[idx]["note_end"][-1]), 3))
                word_dict_list[idx]["note_end"].append(max)
                key_num = key_num + 1
            
            # If a word has more than 4 keys
            if key_num > 6:
                print(f'[ERROR] too many notes in a word\n{base_name}, {word_dict_list[idx]}')
            # Reaching the boundary of the word
            if max >= word_dict_list[idx]["end_time"] - 0.01 and max <= word_dict_list[idx]["end_time"] + 0.01:
                word_dict_list[idx]["note_end"][-1] = word_dict_list[idx]["end_time"]  # Ensure alignment
                key_num = 0  # Reset key_num
                idx = idx + 1
        else:  # If there is an alignment problem
            assert False, f'Alignment problem exists, {base_name}, {idx}, note start time: {min}, note end time: {max}, word start time: {word_dict_list[idx]["start_time"]}, word end time: {word_dict_list[idx]["end_time"]}'
    
    assert idx == len(word_dict_list), f"note not aligned, {idx} != {len(word_dict_list)}"
    
    # Check and adjust alignment between word and phone
    assert word_dict_list[0]['start_time'] == 0, f'{base_name} start_time is not 0'
    for i in range(len(word_dict_list) - 1):
        assert word_dict_list[i]['end_time'] == word_dict_list[i + 1]['start_time'], f'{base_name} {i}, {word_dict_list[i]} word and word not aligned'
        for j in range(len(word_dict_list[i]['ph']) - 1):
            word_dict_list[i]['ph_end'][j] = word_dict_list[i]['ph_start'][j + 1]
        for j in range(len(word_dict_list[i]['note']) - 1):
            word_dict_list[i]['note_end'][j] = word_dict_list[i]['note_start'][j + 1]
        
        # Check '_NONE' and 'breathe' in phones
        for ph in word_dict_list[i]['ph']:
            if ph not in ALL_PHONES_LIST[enum[language]] and ph not in ['<SP>', '<AP>']:
                print(f'[ph error] {base_name}, {word_dict_list[i]}, {ph} not in phone list')
    
    # Traverse each word, check if each phone's duration is greater than 0.02s
    for i in range(len(word_dict_list)):
        for j in range(len(word_dict_list[i]['ph'])):
            if round(word_dict_list[i]['ph_end'][j] - word_dict_list[i]['ph_start'][j], 3) < 0.02:
                print(f'[ph duration too small]\n{base_name}{word_dict_list[i]}, ph duration is too small (<0.02)')
                break
        # Check every non-<SP><AP> word has a note
        if word_dict_list[i]["word"] not in ['<SP>', '<AP>']:
            if 0 in word_dict_list[i]['note']:
                print(f'[no note but word]\n{base_name}, {word_dict_list[i]}')
    
    with open(tg_path.replace('.TextGrid', '_tg.json'), 'w') as f:
        json.dump(word_dict_list, f, ensure_ascii=False, indent=4)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument("--language", type=str, help="The language of the sing", default='english')
    args = parser.parse_args()
    data_dir = args.data_dir
    language = args.language
    
    wav_list = glob.glob(f"{data_dir}/*/*/*/*.wav")
    
    singing_dir_list = [i for i in wav_list if 'speech' not in i]
    tg_dir = [i.replace('.wav','.TextGrid') for i in singing_dir_list]
    ph_tg_dir = [i.replace('.wav','_ph.txt') for i in singing_dir_list]
    
    batch_check_word_list(data_dir)
    
    for musicxml in tqdm.tqdm(tg_dir,total=len(tg_dir)):
        try:
            tg2text(musicxml)
        except Exception as e:
            print(musicxml)
        
    for tgph_path in tqdm.tqdm(ph_tg_dir,total=len(ph_tg_dir)):
        check_phones(tgph_path,language=language)
    
    bad = 0
    for musicxml in tqdm.tqdm(tg_dir,total=len(tg_dir)):
        # print(musicxml)
        try:
            tg2json(musicxml,language=language)
        except AssertionError as ae:
            bad += 1
            print(bad,musicxml)
            print(f"{bad}:Assertion error occurred: {ae}")
        except Exception as e:
            bad += 1
            print(bad,musicxml)
            print(f"{bad}:An exception occurred: {e}")