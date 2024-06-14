# 根据textgrid 分割音频以及相应的textgrid
import argparse
import json
from collections import OrderedDict
from tqdm import tqdm
import glob
import os
import audio
import librosa
import glob
import difflib
import data_process.utils.seg as seg

# process the segementation of textgrid
def seg_tg(item_name, sen_id, start_pos, end_pos, word_list, target_dir):
    seg_wav_path = item_name + '#' + str(sen_id)
    target_dir = f'{target_dir}/{item_name}'
    os.makedirs(target_dir, exist_ok=True)
    textgrid1 = word_list[start_pos: end_pos+1]
    zero=textgrid1[0]['start_time']
    import copy
    word_list_new=copy.deepcopy(textgrid1)
    assert zero==textgrid1[0]['ph_start'][0]
    for i,word in enumerate(word_list_new):

        del word['word_id']

        word['start_time']= round(word['start_time']-zero,3)
        word['end_time']=round(word['end_time']-zero,3)
        for ii in range(len(word['ph_start'])):
            word['ph_start'][ii]=round(word['ph_start'][ii]-zero,3)
            word['ph_end'][ii]=round(word['ph_end'][ii]-zero,3)

        from collections import OrderedDict
        desired_order = ['word','start_time','end_time','ph','ph_start','ph_end']
        word_sort = OrderedDict((k, word[k]) for k in desired_order)
        word_list_new[i]=word_sort

    with open(f'{target_dir}/{seg_wav_path}.json', 'w', encoding="utf8") as f:
        json.dump(word_list_new, f, indent=4, ensure_ascii=False)

    # save tg file
    import textgrid
    new_word=textgrid.IntervalTier(name="word")
    new_ph=textgrid.IntervalTier(name="phone")

    zero=float(word_list_new[0]['start_time'])

    for word in word_list_new:
        word_in=textgrid.Interval(minTime=word['start_time']-zero, maxTime=word['end_time']-zero,mark=word['word'])
        new_word.addInterval(word_in)
        for i,ph in enumerate(word['ph']):
            try:
                ph_in=textgrid.Interval(minTime=word['ph_start'][i]-zero,maxTime=word['ph_end'][i]-zero,mark=ph)
            except:
                print(f'{target_dir}/{seg_wav_path}_tg.json')
                assert False
            new_ph.addInterval(ph_in)

    #保存tg
    tg = textgrid.TextGrid()
    tg.append(new_word)
    tg.append(new_ph)

    tg.write(f'{target_dir}/{seg_wav_path}.TextGrid')



# based on the control group to segment the speech

def seg_speech(wav_path, target_path):
    target_dir='/'.join(target_path.split('/')[:-1])
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    item_name = os.path.basename(wav_path)[:-4]
    wav_input, sr = librosa.core.load(wav_path, sr=None)
    # set the max and min duration of the segment
    max_dur = 12
    min_dur = 6
    max_sil = 0.1
    wordph_path = wav_path.replace(".wav", "_tg.json")
    if not os.path.exists(wordph_path):
        assert False,f'{wordph_path} does not exist'
    # print(wordph_path)

    with open(wordph_path, "r") as f:
        word_tech_list = json.load(f)

    tech_name = wordph_path.split('/')[-2]
    tg_name = wordph_path.split('/')[-1]
    ori_path='/'.join(wordph_path.split('/')[:-2])+'/Control_Group/'+tg_name.replace(tech_name,'Control')
    ori_path=ori_path.replace('Speech','Singing')
    # print(ori_path)
    with open(ori_path, "r") as f:
        word_list = json.load(f)
    
    list1=[word['word'] for word in word_list if word['word'] not in [ "_NONE",'<SP>','<AP>']]
    list2=[word['word'] for word in word_tech_list if word['word'] not in [ "_NONE",'<SP>','<AP>']]
   
    if len(list1)!=len(list2):
        basename1 = os.path.basename(ori_path)
        basename2 = os.path.basename(wordph_path)
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

        print(ori_path)
        for item in output1:
            print(item, end = ' ')
        print('\n')
        print(wordph_path)
        for item in output2:
            print(item, end = ' ')
        print('\n')

    jsonfile=ori_path.replace('_tg.json','_seg.json')
    with open(jsonfile, 'r') as f:
        segments=json.load(f)

    start_time=0
    end_time=0
    flag=0
    for idx, segment in enumerate(segments):
        if seg.check_empty(segment):
            continue

        for word in segment:
            if word["word"] not in [ "_NONE",'<SP>','<AP>']:
                # start_word_id = word["word_id"]
                break

        check=0
        start_id=0
        for i,word_i in enumerate(word_tech_list):
            if i<flag:
                continue
            # print(i,flag,start_id)
            lseg=[seg['word'] for seg in segment if seg['word'] not in [ "_NONE",'<SP>','<AP>']]
            lenseg=len(lseg)

            if check==1 and segment[-1]['word'] in ['<AP>','<SP>',"_NONE"]:
                # print(segment[-1]['word'],word_tech_list[i]['word'])
                if segment[-2]['word'] in ['<AP>','<SP>',"_NONE"]:
                    assert False,'too many sil'
                li=[seg['word'] for seg in word_tech_list[start_id:i+1] if seg['word'] not in [ "_NONE",'<SP>','<AP>']]
                # print(li,lseg)
                if len(li)==len(lseg):
                    # for 
                    if i<len(word_tech_list)-1 and word_tech_list[i+1]['word']in [ "_NONE",'<SP>','<AP>']:
                        end_time=word_tech_list[i+1]['end_time']
                        flag=i+2
                        end_pos=i+1
                        break
                    else:
                        end_time=word_tech_list[i]['end_time']
                        flag=i+1
                        end_pos=i
                        break

            if check==1 and segment[-1]['word'] not in [ "_NONE",'<SP>','<AP>']:
                # print(segment[-1]['word'],word_tech_list[i])
                li=[seg['word'] for seg in word_tech_list[start_id:i+1] if seg['word'] not in [ "_NONE",'<SP>','<AP>']]
                # print(li,lseg)
                if len(li)==len(lseg):
                    end_time=word_tech_list[i]['end_time']
                    flag=i+1
                    end_pos=i
                    # print(flag)
                    break

            if i==len(word_tech_list)-1:
                end_word=[seg['word'] for seg in word_tech_list[start_id:] if seg['word'] not in [ "_NONE",'<SP>','<AP>']]
                assert False,f'{wordph_path} no end,{i}{lseg},\n,{end_word}'

            if check==0 and segment[0]['word'] in [ "_NONE",'<SP>','<AP>']:
                if segment[1]['word'] in [ "_NONE",'<SP>','<AP>']:
                    assert False,f'too many breathe'
                # if segment[1]['word']==word_tech_list[i]['word']:
                    # add the corresponding silence
                if i>0 and word_tech_list[i-1]['word'] in [ "_NONE",'<SP>','<AP>']:
                    start_time=word_tech_list[i-1]['start_time']
                    flag=i+lenseg-1
                    check=1
                    start_id=i-1
                    start_pos=i -1                       
                else:
                    start_time=word_tech_list[i]['start_time']
                    flag=i+lenseg-1
                    check=1
                    start_id=i
                    start_pos=i
                continue

            if check==0 and segment[0]['word'] not in [ "_NONE",'<SP>','<AP>']:
                # print(segment[0]['word'])
                start_time=word_tech_list[i]['start_time']
                flag=i+lenseg-1
                check=1
                start_id=i
                start_pos=i
                continue

        current_dur=end_time-start_time
        # if current_dur < 1:
        #     print(f'false,\n{wordph_path},\n{current_dur},{end_time},{start_time}')
            # assert False

        seg.seg_wav(item_name, idx, sr, start_time, end_time, wav_input, target_dir)
        seg_tg(item_name, idx, start_pos, end_pos, word_tech_list, target_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    args = parser.parse_args()
    data_dir = args.data_dir
    target_dir=data_dir.replace('Speech','Speech_seg')

    # do the segmentation for all the speech audio
    for wav_path in tqdm(glob.glob(f"{data_dir}/*/*/*/*.wav")):
        path=target_dir+'/'.join(wav_path.split('/')[-4:])
        # print(wav_path)
        seg_speech(wav_path, path)

    max_dur = 0
    min_dur = 10

    # check the duration of the segmented audio
    for wav_path in glob.glob(f"{target_dir}/*/*/*/*/*.wav"):
        dur = librosa.get_duration(filename=wav_path)
        # print(dur)
        # if dur > 12:
            # print(wav_path.replace('.wav','_tg.json'),'\ndur:',dur)
        if dur > max_dur:
            max_dur = dur
        if dur < 1:
            print(wav_path.replace('.wav','.json'),'\ndur:',dur)
        if dur < min_dur:
            min_dur = dur
    print('max',max_dur)
    print('min',min_dur)

