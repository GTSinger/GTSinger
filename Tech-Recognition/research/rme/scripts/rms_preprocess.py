# 将 rms 中的真实乐谱处理成真实的时间戳，并处理一下其他问题
# 这个代码只是把真实的note dur找出来并放进meta，并不包含这个meta是怎么生成出来的。
# %%
import json
import os

from tqdm import tqdm

def build_tson(meta_fn, target_dir, wav_dir=None):
    ALL_SHENMU = ['b', 'c', 'ch', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 'sh', 't', 'x', 'z',
                  'zh']
    ALL_YUNMU = ['a', 'ai', 'an', 'ang', 'ao', 'e', 'ei', 'en', 'eng', 'er', 'i', 'ia', 'ian', 'iang', 'iao',
                 'ie', 'in', 'ing', 'iong', 'iou', 'o', 'ong', 'ou', 'u', 'ua', 'uai', 'uan', 'uang', 'uei',
                 'uen', 'uo', 'v', 'van', 've', 'vn']
    metas = json.load(open(meta_fn, "r"))
    for item in tqdm(metas):
        label_data = {
            "label": [],
            "key": []
        }
        item_name = item["item_name"]
        singer, piece, name = item_name.split("#")
        os.makedirs(f"{target_dir}/{singer}#{piece}", exist_ok=True)
        target_tson_path = f"{target_dir}/{singer}#{piece}/{piece}#{name}.tson"

        # os.makedirs(f"{wav_dir}/{singer}#{piece}", exist_ok=True)
        # target_wav_path = f"{wav_dir}/{singer}#{piece}/{piece}#{name}.wav"
        # wav_path = f"/mnt/bn/ailabrenyi/datasets/opensource_sing/230227seg/{singer}/{piece}/{piece}#{name}.wav"
        # shutil.copyfile(wav_path, target_wav_path)
        text = []
        for t in item["txt"]:
            if t in ["breathe", "_NONE"]:
                text.append("sil")
            else:
                text.append(t)
        label_data["txt"] = "".join(text)
        notes = item["pitches"]
        ph = item["ph"]
        ph_durs = item["ph_durs"]
        num_words = max(item["note2words"]) + 1
        word_key = [[] for _ in range(num_words)]
        for i in range(len(item["note2words"])):
            word_id = item["note2words"][i]
            word_key[word_id].append(notes[i])
        ph_key = [[] for _ in range(num_words)]
        for i in range(len(item["ph2words"])):
            word_id = item["ph2words"][i]
            ph_key[word_id].append([ph[i], ph_durs[i]])

        assert len(word_key) == len(ph_key), f"{item_name}"
        index = 0
        start_time = 0
        key_start_time = 0
        for id in range(len(word_key)):
            wk = word_key[id]
            pk = ph_key[id]
            if len(pk) == 0:
                assert False, f"nonvalid word {id} {item_name}"
            elif len(pk) == 1:
                assert len(wk) == len(pk), f"note ph mismatch, {id} {item_name}"
                if pk[0][0] in ["breathe", "_NONE"]:
                    label_data["label"].append({"phone": f"sil", "phoneEnd": int((start_time + pk[0][1]) * 1e7),
                                                "phoneStart": int(start_time * 1e7), "index": index})
                    label_data["key"].append({"phone": 0, "phoneEnd": int((start_time + pk[0][1]) * 1e7),
                                                "phoneStart": int(start_time * 1e7), "index": index})
                    index = index + 1
                    start_time = start_time + pk[0][1]
                else:
                    label_data["label"].append({"phone": f"C0{pk[0][0]}", "phoneEnd": int((start_time + pk[0][1]) * 1e7),
                                                "phoneStart": int(start_time * 1e7), "index": index})
                    label_data["key"].append({"phone": f"{wk[0]}", "phoneEnd": int((start_time + pk[0][1]) * 1e7),
                                              "phoneStart": int(start_time * 1e7), "index": index})
                    start_time = start_time + pk[0][1]
                    index = index + 1
            else:
                if pk[0][0] in ALL_SHENMU:
                    assert len(pk) - len(wk) == 1, f"note ph mismatch, {id} {item_name}"
                    prev_start_time = start_time
                    label_data["label"].append(
                        {"phone": f"C0{pk[0][0]}", "phoneEnd": int((start_time + pk[0][1]) * 1e7),
                         "phoneStart": int(start_time * 1e7), "index": index})
                    start_time = start_time + pk[0][1]
                    cur_ph = ""
                    for i, p in enumerate(pk[1:]):
                        if p[0] != cur_ph:
                            label_data["label"].append(
                                {"phone": f"C0{p[0]}", "phoneEnd": int((start_time + p[1]) * 1e7),
                                 "phoneStart": int(start_time * 1e7), "index": index})
                            cur_ph = p[0]
                        else:
                            label_data["label"][-1]["phoneEnd"] = int((start_time + p[1]) * 1e7)
                        if i == 0:
                            label_data["key"].append({"phone": f"{wk[i]}", "phoneEnd": int((start_time + p[1]) * 1e7),
                                                      "phoneStart": int(prev_start_time * 1e7), "index": index})
                        else:
                            label_data["key"].append({"phone": f"{wk[i]}", "phoneEnd": int((start_time + p[1]) * 1e7),
                                                      "phoneStart": int(start_time * 1e7), "index": index})
                        start_time = start_time + p[1]
                else:
                    assert len(pk) == len(wk), f"note ph mismatch, {id} {item_name}"
                    cur_ph = ""
                    for i, p in enumerate(pk):
                        if p[0] != cur_ph:
                            label_data["label"].append(
                                {"phone": f"C0{p[0]}", "phoneEnd": int((start_time + p[1]) * 1e7),
                                 "phoneStart": int(start_time * 1e7), "index": index})
                            cur_ph = p[0]
                        else:
                            label_data["label"][-1]["phoneEnd"] = int((start_time + p[1]) * 1e7)
                        label_data["key"].append({"phone": f"{wk[i]}", "phoneEnd": int((start_time + p[1]) * 1e7),
                                                  "phoneStart": int(start_time * 1e7), "index": index})
                        start_time = start_time + p[1]
                index = index + 1
        json.dump(label_data, open(target_tson_path, 'w'), indent=2, ensure_ascii=False)

def preprocess(old_meta_path, new_meta_path, tson_dir):
    metafile_path = old_meta_path
    items_list = json.load(open(metafile_path))
    items = {}
    item_names = []
    deleted = set()
    for r in tqdm(items_list, desc=f'| Processing'):
        item_name = r['item_name']
        if item_name in items:
            print(f'warning: item name {item_name} duplicated')
        items[item_name] = r
        item_names.append(item_name)

        wav_fn = items[item_name]['wav_fn']
        wav_fn = os.path.relpath(wav_fn, '/home/zy/huawei/data/')
        wav_fn = os.path.join('/mnt/sdb/liruiqi/datasets', wav_fn)
        items[item_name]['wav_fn'] = wav_fn

        # parse tson
        spk, song_name, sen_id = item_name.split('#')
        tson_path = f"{tson_dir}/{spk}#{song_name}/{song_name}#{sen_id}.tson"
        tson_item = json.load(open(tson_path))
        real_note_durs = []
        for k in tson_item['key']:
            dur = (k['phoneEnd'] - k['phoneStart']) / 1e7
            if dur <= 0.09:
                print(f'skip {item_name} for invalid note dur: {dur}')
                deleted.add(item_name)
            real_note_durs.append(dur)
        items[item_name]['real_note_durs'] = real_note_durs

    print(f'total num of skipped: {len(deleted)}')
    meta_out = [items[item_name] for item_name in item_names if item_name not in deleted]
    json.dump(meta_out, open(new_meta_path, 'w'), ensure_ascii=False, indent=2)

    return meta_out, item_names, items, deleted

# %%
if __name__ == '__main__':
    old_meta_path = "/mnt/sdb/liruiqi/datasets/230227seg/meta.json"
    new_meta_path = "/mnt/sdb/liruiqi/datasets/230227seg/meta_new.json"
    tson_dir = "/mnt/sdb/liruiqi/SingingDictation/data/processed/rms_tson"
    # tson_dir = "/mnt/sdb/liruiqi/SingingDictation/data/processed/rms_tson_new"  # 这个是不跳过那些短于0.09的
    # build_tson(old_meta_path, tson_dir)

    meta_out, item_names, items, deleted = preprocess(old_meta_path, new_meta_path, tson_dir)
