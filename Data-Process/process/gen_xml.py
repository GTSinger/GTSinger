# 检查musicxml，同时合并tied rest等
import xmltodict
import pretty_midi
from tqdm import tqdm
import glob
import json
import os
import shutil
import textgrid

ALL_SHENGMU = ['b', 'c', 'ch', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 'sh', 't', 'x', 'z', 'zh']
ALL_YUNMU = ['a', 'ai', 'an', 'ang', 'ao',  'e', 'ei', 'en', 'eng', 'er',  'i', 'ia', 'ian', 'iang', 'iao',
             'ie', 'in', 'ing', 'iong', 'iou', 'o', 'ong', 'ou', 'u', 'ua', 'uai', 'uan', 'uang', 'uei',
             'uen', 'uo', 'v', 'van', 've', 'vn']

special_list = []

def type2duration(type, divisions):
    if type == "eighth":
        return divisions // 2
    elif type == "16th":
        return divisions // 4
    elif type == "32nd":
        return divisions // 8
    elif type == "64th":
        return divisions // 16
    else:
        assert False, f"{type} type not considered"

def music_hier(xml_path, postfix):
    notes_list = []
    attributes = {}
    directions = {}
    with open(xml_path, "r") as f:
        xmldict = xmltodict.parse(f.read())
    if "score-partwise" in xmldict:
        xmldict = xmldict["score-partwise"]
    parts = xmldict["part"]
    if type(parts) != list:
        parts = [parts]
    part_id = []
    word_id = -1
    all_note_id = 0
    tied_stack = []
    slur_stack = []
    for p, part in enumerate(parts): # music parts
        measures = part["measure"] # music bar
        for m, measure in enumerate(measures):
            part_id.append(p)
            # partwise music meta
            if "attributes" in measure and m == 0 and p==0: # 一般只有第一个小节的attributes 包含时值信息
                attributes.update(measure["attributes"])
                if type(measure["direction"]) is list:
                    direction = measure["direction"][0]
                else:
                    direction = measure["direction"]
                directions.update(direction)
                notes_list.append(attributes)
                notes_list.append(directions)
            notes = measure["note"]
            if type(notes) != list:
                notes = [notes]
            for n, note in enumerate(notes):
                # print("#######")
                note_dict ={}
                note_dict["part_id"] = p
                note_dict["measure_id"] = m
                note_dict["note_id"] = n
                note_dict["rest"] = 1 if "rest" in note else 0
                note_dict["grace"] = 1 if "grace" in note else 0 # 处理倚音(grace)
                # print(note_dict)
                if "duration" in note:
                    note_dict["duration"] = int(note["duration"])
                else:
                    note_dict["duration"] = type2duration(note["type"], int(attributes["divisions"]))
                # note_dict["beam"] = []
                note_dict["lyric"] = '' # 注意这里在check_lyric.py里面就消除了list的可能性
                note_dict["pitch"] = 0
                # note_dict["notations"] = []
                note_dict["tied"] = []
                note_dict["slur"] = []
                if "pitch" in note:
                    pitch = note["pitch"]["step"] + note["pitch"]["octave"]
                    pitch = pretty_midi.note_name_to_number(pitch)
                    if "alter" in note["pitch"]:
                        pitch = int(note["pitch"]["alter"]) + pitch # 处理半音升降调
                    note_dict["pitch"] = pitch
                # if "beam" in note:                                # beam音梁只是方便乐谱书写，没有实际意思
                #     beam = note["beam"]
                #     if type(beam) != list:
                #         beam = [beam]
                #     note_dict["beam"] = beam
                if "lyric" in note:
                    lyric = note["lyric"]
                    text = lyric["text"]
                    if text is None:
                        assert False, "lyric is not None"
                    if type(text) == str:
                        note_dict["lyric"] = text
                    elif type(text) == dict:
                        note_dict["lyric"] = text["#text"]
                    else:
                        assert False, f"error lyric {text}"
                    word_id = word_id + 1
                    note_dict["word_id"] = word_id
                    # if type(lyric) != list:
                    #     lyric = [lyric]
                    # for l in lyric:
                    #     note_dict["lyric"].append(l)
                elif "grace" in note:
                    # 针对倚音，它对应的一般是后面一个字，但是这个地方其实无所谓
                    note_dict["word_id"] = word_id + 1
                else:
                    # 这样针对slur和tied，word_id实际上就是前面最靠近他们的word
                    # 这样的word必然是他们应该对应的note
                    # 同时rest已经由rest=1标注了
                    note_dict["word_id"] = word_id

                if "notations" in note:
                    notations = note["notations"]
                    if type(notations) != list:
                        notations = [notations]
                    for nota in notations:
                        if "tied" in nota:
                            tied = nota["tied"]
                            if type(tied) is not list:
                                tied = [tied]
                            for tie in tied:
                                note_dict["tied"].append(tie["@type"])
                                if tie["@type"] != "stop":
                                    tied_stack.append(tie["@type"])
                                else:
                                    check = False
                                    while (len(tied_stack) > 0):
                                        if tied_stack[-1] == "start":
                                            check = True  # 匹配成功
                                            tied_stack.pop()
                                            break
                                        else:
                                            tied_stack.pop()
                                    assert check, "tied stack can not match stop"
                        if "slur" in nota:
                            slurs = nota['slur']
                            if type(slurs) is not list:
                                slurs = [slurs]
                            for slur in slurs:
                                note_dict["slur"].append(slur["@type"])
                                if slur["@type"] != "stop":
                                    slur_stack.append(slur["@type"])
                                else:
                                    check = False
                                    while (len(slur_stack) > 0):
                                        if slur_stack[-1] == "start":
                                            check = True  # 匹配成功
                                            slur_stack.pop()
                                            break
                                        else:
                                            slur_stack.pop()
                                    assert check, "slur stack can not match stop"

                note_dict["all_note_id"] = all_note_id
                all_note_id = all_note_id + 1
                notes_list.append(note_dict)
    with open(xml_path.replace(f".{postfix}", "_test.json"), 'w', encoding="utf8") as f:
        json.dump(notes_list, f, indent=4, ensure_ascii=False)

def refine_music(json_path):
    with open(json_path, "r") as f:
        score_list = json.load(f)
    note_list = score_list[2:]
    refined_notes = []
    tied_stack = []
    for idx, note in enumerate(note_list):
        tied = note["tied"]
        # 合并rest
        if note["rest"] == 1 and len(refined_notes) > 0 and refined_notes[-1]["rest"] == 1:
            last_note = refined_notes[-1].copy()
            last_note["duration"] = refined_notes[-1]["duration"] + note["duration"]
            refined_notes[-1] = last_note
            if len(note["slur"]) > 0:
                print(note)
            continue
        # 没有tied
        if len(tied) == 0:
            note["all_note_id"] = len(refined_notes)
            refined_notes.append(note)
            # note_durs.append(note["duration"])
        # tied 只有start，这意味着这个note不会被合并到其他note
        elif all([tie == "start" for tie in tied]):
            note["all_note_id"] = len(refined_notes)
            refined_notes.append(note)
            # note_durs.append(note["duration"])
            for tie in tied:
                tied_stack.append(tie)
        # tied 需要合并到其他note
        else:
            if not (len(note["lyric"])==0 and note["pitch"] == refined_notes[-1]["pitch"]):
                print(json_path)
                note_id = note["note_id"]
                measure_id = note["measure_id"]
                print(f"error match measure: {measure_id + 1}, note: {note_id + 1}")
                # assert False, "error"
            last_note = refined_notes[-1].copy()
            last_note["duration"] = refined_notes[-1]["duration"] + note["duration"]
            last_note["slur"] = refined_notes[-1]["slur"] + note["slur"]
            refined_notes[-1] = last_note


    # 检查信息没有损失
    prev_total_dur = 0
    prev_slur_list = []
    re_total_dur = 0
    re_slur_list = []
    for note in note_list:
        prev_total_dur = prev_total_dur + note["duration"]
        prev_slur_list = prev_slur_list + note["slur"]
    for note in refined_notes:
        re_total_dur = re_total_dur + note["duration"]
        re_slur_list = re_slur_list + note["slur"]
    # re_total_dur = sum(note_durs)
    # print(prev_slur_list)
    # print(re_slur_list)
    if prev_slur_list != re_slur_list:
        print(prev_slur_list)
        print(re_slur_list)
    assert prev_total_dur == re_total_dur and prev_slur_list == re_slur_list, f"error: {prev_total_dur}, {re_total_dur}"

    refined_notes = score_list[:2] + refined_notes
    with open(json_path.replace(f"_test.json", "_re.json"), 'w', encoding="utf8") as f:
        json.dump(refined_notes, f, indent=4, ensure_ascii=False)

def for_loop(data_dir, postfix):
    special_list = []
    for musicxml in sorted(tqdm(glob.glob(f"{data_dir}/*/*/*/*/*.{postfix}"))): # .musicxml --> .xml
        if musicxml in special_list:
            continue
        if postfix in ["musicxml", "xml"]:
            try:
                music_hier(musicxml, postfix)
            except:
                print(musicxml)
        else:
            assert False, f"invalid postfix {postfix}"

if __name__ == '__main__':
    # for_loop("/data_disk/singing-data-preprocess-main/m+t检查1220", "TextGrid")
    data_dir = "/Users/aaron/Downloads/华为数据/华为女声第一周"
    for_loop(data_dir, "musicxml")
    remove_list = []
    for musicxml in sorted(tqdm(glob.glob(f"{data_dir}/*/*/*/*/*_test.json"))):
        # if not os.path.exists(musicxml.replace("_re", "_tg")):
        #     print(musicxml)
        # if "_tg.json" not in musicxml and "_re.json" not in musicxml and musicxml not in remove_list:
        refine_music(musicxml)



