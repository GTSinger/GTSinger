import glob
import os
import textgrid
import json
import argparse
import tqdm

def cal_dur(self, duration, divisions, tempo):
    return float(duration) / float(divisions) * 60.0 / float(tempo)

def tg_hier(tg_path):
    word_dict_list = []
    tg = textgrid.TextGrid.fromFile(tg_path)
    word_intervals = tg[0]
    ph_intervals = tg[1]
    mix_intervals= tg[2]
    falsetto_intervals= tg[3]
    breathe_intervals= tg[4]
    pharyngeal_intervals= tg[5]
    vibrato_intervals= tg[6]
    glissando_intervals= tg[7]

    # get word information
    for word in word_intervals:
        mark = word.mark.strip()
        min=word.minTime
        max=word.maxTime
        word_dict = {}
        word_dict["word"] = mark
        word_dict["start_time"] = round(float(min),3)
        word_dict["end_time"] = round(float(max),3)
        word_dict["note"]=[] 
        word_dict['note_dur']=[]
        word_dict['note_start']=[]
        word_dict['note_end']=[]
        word_dict['ph']=[]
        word_dict['ph_start']=[]
        word_dict['ph_end']=[]    
        word_dict['mix']=[]
        word_dict['falsetto']=[]
        word_dict['breathe']=[]
        word_dict['pharyngeal']=[]
        word_dict['vibrato']=[]
        word_dict['glissando']=[] 
        word_dict['tech']=''
        word_dict_list.append(word_dict)

    mix_list=[]
    for t in mix_intervals:
        mix_list.append(t.mark)

    falsetto_list=[]
    for t in falsetto_intervals:
        falsetto_list.append(t.mark)

    breathe_list=[]
    for t in breathe_intervals:
        breathe_list.append(t.mark)

    pharyngeal_list=[]
    for t in pharyngeal_intervals:
        pharyngeal_list.append(t.mark)

    vibrato_list=[]
    for t in vibrato_intervals:
        vibrato_list.append(t.mark)

    glissando_list=[]
    for t in glissando_intervals:
        glissando_list.append(t.mark)    

    # do phoneme alignment
    idx = 0
    for n,ph in enumerate(ph_intervals):
        if idx >= len(word_dict_list):
            print(f'{ph}{idx}, word_dict_list length is {len(word_dict_list)}')
            for element in word_dict_list:
                print(f'{element["word"]},{element["ph"]}')
            assert False
        # ph = ph.strip()
        mark=ph.mark.strip()
        min=round(float(ph.minTime),3)
        max=round(float(ph.maxTime),3)
        
        if min>=word_dict_list[idx]['start_time'] and max<=word_dict_list[idx]['end_time']:
            word_dict_list[idx]['ph'].append(mark)
            word_dict_list[idx]['ph_start'].append(min)
            word_dict_list[idx]['ph_end'].append(max)
            word_dict_list[idx]['mix'].append(mix_list[n])
            word_dict_list[idx]['falsetto'].append(falsetto_list[n])
            word_dict_list[idx]['breathe'].append(breathe_list[n])
            word_dict_list[idx]['pharyngeal'].append(pharyngeal_list[n])
            word_dict_list[idx]['vibrato'].append(vibrato_list[n])
            word_dict_list[idx]['glissando'].append(glissando_list[n])
        else:
            idx = idx + 1
            if min>=word_dict_list[idx]['start_time'] and max<=word_dict_list[idx]['end_time']:
                word_dict_list[idx]['ph'].append(mark)
                word_dict_list[idx]['ph_start'].append(min)
                word_dict_list[idx]['ph_end'].append(max) 
                word_dict_list[idx]['mix'].append(mix_list[n])
                word_dict_list[idx]['falsetto'].append(falsetto_list[n])
                word_dict_list[idx]['breathe'].append(breathe_list[n])
                word_dict_list[idx]['pharyngeal'].append(pharyngeal_list[n])
                word_dict_list[idx]['vibrato'].append(vibrato_list[n])
                word_dict_list[idx]['glissando'].append(glissando_list[n])
        if word_dict_list[idx]['mix']=='1':
            word_dict_list[idx]['tech']+=' 1'
        if word_dict_list[idx]['falsetto']=='1':
            word_dict_list[idx]['tech']+=' 2'
        if word_dict_list[idx]['breathe']=='1':
            word_dict_list[idx]['tech']+=' 3'
        if word_dict_list[idx]['pharyngeal']=='1':
            word_dict_list[idx]['tech']+=' 4'
        if word_dict_list[idx]['vibrato']=='1':
            word_dict_list[idx]['tech']+=' 5'
        if word_dict_list[idx]['glissando']=='1':
            word_dict_list[idx]['tech']+=' 6'
          

    assert idx == len(word_dict_list)-1, f"ph not align, {idx} != {len(word_dict_list)}"

    # get tempo and note information
    xml_fn=tg_path.replace('.Textgrid','.musicxml')
    with open(xml_fn, "r") as f:
        score_list = json.load(f)
    divisions = int(score_list[0]["divisions"])
    tempo = float(score_list[1]["sound"]["@tempo"])
    note_list = score_list[2:]

    idx=0
    for note in note_list:
        if note['lyric']== '' and note['slur']==[]:
            word_dict_list[idx]['note'].append(0)
            word_dict_list[idx]['note_dur'].append(cal_dur(note["duration"], divisions, tempo))
        elif note['lyric']==word_dict_list[idx]['word'] or (note['lyric']== '' and note['slur']!=[]):
            word_dict_list[idx]['note'].append(note['pitch'])
            word_dict_list[idx]['note_dur'].append(cal_dur(note["duration"], divisions, tempo))            
        if note['slur']==[] or note['slur']==['stop']:
            idx+=1

    for word in word_dict_list:
        time=word['start_time']
        for note in word['note']:
            dur=word['note_dur']/(word['end_time']-word['start_time'])
            word['note_start']=time
            word['note_end']=time+dur
            time=time+dur

    with open(tg_path.replace(f".TextGrid", ".json"), 'w', encoding="utf8") as f:
        json.dump(word_dict_list, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data directory')
    args = parser.parse_args()
    data_dir = args.data_dir

    for tg in sorted((glob.glob(f"{data_dir}/*/*/*/*/*.TextGrid"))):
        # print(musicxml)
        tg_hier(tg)
