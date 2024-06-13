import argparse
import json
from collections import OrderedDict
from tqdm import tqdm
import glob
import os
import audio
import librosa
import glob
import data_process.utils.seg as seg


# process the segementation of textgrid
def seg_tg(item_name, sen_id, start_pos, end_pos, word_list, target_dir):
    seg_wav_path = item_name + '#' + str(sen_id)
    target_dir = f'{target_dir}/{item_name}'
    os.makedirs(target_dir, exist_ok=True)
    textgrid = word_list[start_pos: end_pos+1]
    zero=textgrid[0]['start_time']
    assert zero==textgrid[0]['note_start'][0]
    assert zero==textgrid[0]['ph_start'][0]
    for word in textgrid:
        word['start_time']= round(word['start_time']-zero,3)
        word['end_time']=round(word['end_time']-zero,3)
        for i in range(len(word['ph_start'])):
            word['ph_start'][i]=round(word['ph_start'][i]-zero,3)
            word['ph_end'][i]=round(word['ph_end'][i]-zero,3)
        for i in range(len(word['note_start'])):
            word['note_start'][i]=round(word['note_start'][i]-zero,3)
            word['note_end'][i]=round( word['note_end'][i]-zero,3)  

    with open(f'{target_dir}/{seg_wav_path}_tg.json', 'w', encoding="utf8") as f:
        json.dump(textgrid, f, indent=4, ensure_ascii=False)

# Based on the segmentation of the textgrid, segment the wav file
def seg_long(wav_path, target_path):
    target_dir='/'.join(target_path.split('/')[:-1])
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    item_name = os.path.basename(wav_path)[:-4]
    # print(item_name)
    wav_input, sr = librosa.core.load(wav_path, sr=None)
    # 分割最大最小时长
    max_dur = 10
    min_dur = 7
    max_sil = 0.1
    wordph_path = wav_path.replace(".wav", "_tg.json")
    if not os.path.exists(wordph_path):
        print(wordph_path)
        return
    # print(wordph_path)
    segments = []
    seg_start_pos = []  # 每个segment start time from musicxml
    seg_end_pos = []  # 每个segment end time from musicxml

    with open(wordph_path, "r") as f:
        word_list = json.load(f)

    current_seg = []

    cur_start_time = 0
    cur_end_time = 0
    for idx, word in enumerate(word_list):

        current_dur = cur_end_time - cur_start_time

        # if len(current_seg)==0 and word["word"] in ["breathe", "breath", "_NONE","<AP>","<SP>"]:
        #     cur_end_time = cur_end_time + (float(word["end_time"]) - float(word["start_time"]))
        #     cur_start_time = cur_end_time
        #     continue

        true_len=len([word for word in current_seg if word not in ["breathe", "breath", "_NONE","<AP>","<SP>"]])

        if true_len>1 and word["word"] in ["breathe", "breath", "_NONE","<AP>","<SP>"] and current_dur > max_dur:
            if len(current_seg) > 0: # alreay have some words in the segment
                current_seg.append(word)
                segments.append(current_seg)
                seg_end_pos.append(idx)
                current_seg = []
            cur_end_time = cur_end_time + (float(word["end_time"]) - float(word["start_time"]))
            cur_start_time = cur_end_time
            continue
        if current_dur > min_dur and true_len > 1 and word["word"] in ["breathe", "breath", "_NONE","<AP>","<SP>"] and (float(word["end_time"]) - float(word["start_time"])) > max_sil:
            if len(current_seg) > 0: # alreay have some words in the segment
                current_seg.append(word)
                segments.append(current_seg)
                seg_end_pos.append(idx)
                current_seg = []
            cur_end_time = cur_end_time + (float(word["end_time"]) - float(word["start_time"]))
            cur_start_time = cur_end_time
            continue
        
        current_seg.append(word)
        cur_end_time = cur_end_time + (float(word["end_time"]) - float(word["start_time"]))
        if len(seg_end_pos) == len(seg_start_pos): # matches the start time and end time
            seg_start_pos.append(idx)
        # process the last segment
        if idx == len(word_list) -1:
            # print(current_seg)
            # print(current_seg[-1]['end_time']-current_seg[0]['start_time'])
            uword=[word['word'] for word in current_seg if word["word"] not in ["breathe", "breath", "_NONE","<AP>","<SP>"]]
            if len(uword)>1:
                # for word in current_seg:
                # if word["word"] not in ["breathe", "breath", "_NONE","<AP>","<SP>"]:
                if current_seg[-1]['end_time']-current_seg[0]['start_time']<min_dur or len(current_seg)==1:
                    segments[-1]+=current_seg
                    # print(seg_end_pos[-1])
                    seg_end_pos[-1]=idx
                    current_seg = []
                    # break
                else:
                    segments.append(current_seg)
                    seg_end_pos.append(idx)
                    current_seg = []
                    # break
            else:
                segments[-1]+=current_seg
                # print(seg_end_pos[-1])
                seg_end_pos[-1]=idx
                current_seg = []
                # break
    jsonfile=wordph_path.replace('_tg.json','_seg.json')
   
    with open(jsonfile,'w') as f:
        json.dump(segments,f)

    for idx, segment in enumerate(segments):
        if seg.check_empty(segment):
            continue
        start_pos, end_pos = seg_start_pos[idx], seg_end_pos[idx]
        # start_word_id = segment[0]["word_id"]
        for word in segment:
            if word["word"] not in ["breathe", "breath", "_NONE","<AP>","<SP>"]:
                # start_word_id = word["word_id"]
                break

        # end_word_id = segment[-1]["word_id"]
        start_time = float(segment[0]["start_time"])
        end_time = float(segment[-1]["end_time"])
        current_dur=end_time-start_time
        if current_dur < min_dur-1.1:
            print([word['word'] for word in segment])
            print('false',current_dur)
            assert False

        # do the segmentation
        seg.seg_wav(item_name, idx, sr, start_time, end_time, wav_input, target_dir)
        seg_tg(item_name, idx, start_pos, end_pos, word_list, target_dir)


# cut the technique group based on the segmentation of the control group
def seg_tech(wav_path, target_path):
    target_dir='/'.join(target_path.split('/')[:-1])
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    item_name = os.path.basename(wav_path)[:-4]
    # print(item_name)
    wav_input, sr = librosa.core.load(wav_path, sr=None)
    # set the max and min duration
    max_dur = 12
    min_dur = 6
    max_sil = 0.1
    wordph_path = wav_path.replace(".wav", "_tg.json")
    if not os.path.exists(wordph_path):
        print(wordph_path)
        assert False

    with open(wordph_path, "r") as f:
        word_tech_list = json.load(f)

    tech_name = wordph_path.split('/')[-2]
    tg_name = wordph_path.split('/')[-1]
    ori_path='/'.join(wordph_path.split('/')[:-2])+'/Control_Group/'+tg_name.replace(tech_name,'Control')
    # print(ori_path)
    with open(ori_path, "r") as f:
        word_list = json.load(f)
    
    w=[word['word'] for word in word_list if word['word'] not in ["breathe", "breath", "_NONE","<AP>","<SP>"]]
    t=[word['word'] for word in word_tech_list if word['word'] not in ["breathe", "breath", "_NONE","<AP>","<SP>"]]
    assert len(w) ==len(t),f'wrong align ,{wav_path}{len(t)},{len(w)},\n,{(t)},\n,{(w)},'

    jsonfile=ori_path.replace('_tg.json','_seg.json')
    with open(jsonfile, 'r') as f:
        segments=json.load(f)

    start_time=0
    end_time=0
    flag=0
    for idx, segment in enumerate(segments):
        if seg.check_empty(segment):
            continue
        # start_pos, end_pos = seg_start_pos[idx], seg_end_pos[idx]
        # start_word_id = segment[0]["word_id"]
        for word in segment:
            if word["word"] not in ["breathe", "breath", "_NONE","<AP>","<SP>"]:
                # start_word_id = word["word_id"]
                break

        check=0
        start_id=0
        for i,word_i in enumerate(word_tech_list):
            if i<flag:
                continue
            # print(i,flag,start_id)
            lseg=[seg['word'] for seg in segment if seg['word'] not in ["breathe", "breath", "_NONE","<AP>","<SP>"]]
            lenseg=len(lseg)
            # print([seg['word'] for seg in segment])

            # process the silence at the end

            if check==1 and segment[-1]['word'] in ["breathe", "breath", "_NONE","<AP>","<SP>"]:
                if segment[-2]['word'] in ["breathe", "breath", "_NONE","<AP>","<SP>"]:
                    assert False,'too many sil'
                if segment[-2]['word']==word_tech_list[i]['word']:
                    li=[seg['word'] for seg in word_tech_list[start_id:i+1] if seg['word'] not in ["breathe", "breath", "_NONE","<AP>","<SP>"]]
                    # print(li,lseg)
                    if len(li)==len(lseg):
                        err=0
                        for ii in range(len(li)):
                            if li[ii]!=lseg[ii]:
                                err+=1
                        if err>4:
                            assert False,"not align"
                        # for 
                        if i<len(word_tech_list)-1 and word_tech_list[i+1]['word']in ["breathe", "breath", "_NONE","<AP>","<SP>"]:
                            # print(wav_path)
                            end_time=word_tech_list[i+1]['end_time']
                            flag=i+2
                            end_pos=i+1
                            break
                        else:
                            end_time=word_tech_list[i]['end_time']
                            flag=i+1
                            end_pos=i
                            break

            if check==1 and segment[-1]['word'] not in ["breathe", "breath", "_NONE","<AP>","<SP>"] and segment[-1]['word']==word_tech_list[i]['word']:
                # print(segment[-1]['word'],word_tech_list[i])
                li=[seg['word'] for seg in word_tech_list[start_id:i+1] if seg['word'] not in ["breathe", "breath", "_NONE","<AP>","<SP>"]]
                if len(li)==len(lseg):
                    err=0
                    for ii in range(len(li)):
                        if li[ii]!=lseg[ii]:
                            err+=1
                    if err>4:
                        print(li,lseg)
                        assert False,"not align"
                    end_time=word_tech_list[i]['end_time']
                    flag=i+1
                    end_pos=i
                    # print(flag)
                    break

            # if the last word is not the same
            if check==1 and segment[-1]['word'] not in ["breathe", "breath", "_NONE","<AP>","<SP>"] and i>2 and segment[-2]['word']==word_tech_list[i-1]['word'] and segment[-3]['word']==word_tech_list[i-2]['word']:
                # print(segment[-1]['word'],word_tech_list[i])
                li=[seg['word'] for seg in word_tech_list[start_id:i+1] if seg['word'] not in ["breathe", "breath", "_NONE","<AP>","<SP>"]]
                # print(li,lseg)
                if len(li)==len(lseg):
                    err=0
                    for ii in range(len(li)):
                        if li[ii]!=lseg[ii]:
                            err+=1
                    if err>2:
                        assert False,"not align"
                    end_time=word_tech_list[i]['end_time']
                    flag=i+1
                    end_pos=i
                    # print(flag)
                    break

            # if we cannot find the end of the segment
            if i==len(word_tech_list)-1:
                end_word=[seg['word'] for seg in word_tech_list[start_id:] if seg['word'] not in ["breathe", "breath", "_NONE","<AP>","<SP>"]]
                assert False,f'{wordph_path}\n no end,{i}{lseg},\n,{end_word}'


            if check==0 and segment[0]['word'] in ["breathe", "breath", "_NONE","<AP>","<SP>"]:
                if segment[1]['word'] in ["breathe", "breath", "_NONE","<AP>","<SP>"]:
                    assert False,f'too many breathe'
                if segment[1]['word']==word_tech_list[i]['word']:
                    # add the corresponding silence
                    if i>0 and word_tech_list[i-1]['word'] in ["breathe", "breath", "_NONE","<AP>","<SP>"]:
                        # print(wav_path)
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

            if check==0 and segment[0]['word'] not in ["breathe", "breath", "_NONE","<AP>","<SP>"] and segment[0]['word']==word_tech_list[i]['word']:
                # print(segment[0]['word'])
                start_time=word_tech_list[i]['start_time']
                flag=i+lenseg-1
                check=1
                start_id=i
                start_pos=i
                continue

            # if the first word is wrong
            if check==0 and segment[0]['word'] not in ["breathe", "breath", "_NONE","<AP>","<SP>"] and i<len(word_tech_list)-2 and segment[1]['word']==word_tech_list[i+1]['word'] and segment[2]['word']==word_tech_list[i+2]['word']:
                start_time=word_tech_list[i]['start_time']
                flag=i+lenseg-1
                check=1
                start_id=i
                start_pos=i
                continue               

        #         # end_time=word_i[i+lenseg]['end_time']
        #         # flag=i+lenseg
        # # start_time = float(segment[0]["start_time"])
        # # end_time = float(segment[-1]["end_time"])
        # if current_dur < min_dur-2:
        #     print('false',ori_path,current_dur,end_time,start_time)
        #     assert False

        # do the segmentation
        seg.seg_wav(item_name, idx, sr, start_time, end_time, wav_input, target_dir)
        seg_tg(item_name, idx, start_pos, end_pos, word_tech_list, target_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    args = parser.parse_args()
    data_dir = args.data_dir
    target_dir=data_dir.replace('Singing','Singing_seg')

    # do the segmentation for the control group
    for wav_path in tqdm(glob.glob(f"{data_dir}/*/*/*/*.wav")):
        if 'Control_Group' in wav_path:
            path=target_dir+'/'.join(wav_path.split('/')[-4:])
            # print(wav_path)
            seg_long(wav_path, path)

    # segment the technique group
    for wav_path in tqdm(glob.glob(f"{data_dir}/*/*/*/*.wav")):
        if 'Control_Group' not in wav_path:
            path=target_dir+'/'.join(wav_path.split('/')[-4:])
            # print(wav_path)
            seg_tech(wav_path, path)

    max_dur = 10
    min_dur = 10

    # check the duration of the segment
    for wav_path in glob.glob(f"{target_dir}/*/*/*/*/*.wav"):
        dur = librosa.get_duration(filename=wav_path)
        # print(dur)
        # if dur > 12:
            # print(wav_path.replace('.wav','_tg.json'),'\ndur:',dur)
        if dur > max_dur:
            max_dur = dur
        if dur < 5:
            print(wav_path.replace('.wav','_tg.json'),'\ndur:',dur)
        if dur < min_dur:
            min_dur = dur
    print('max',max_dur)
    print('min',min_dur)

