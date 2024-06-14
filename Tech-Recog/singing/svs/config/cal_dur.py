import os
import torch
import json
with open("/home/zy/huawei/data/230227seg/meta.json", 'r') as f:
    item_lst = json.load(f)
print(" ")

def group_hidden_by_segs(h, seg_ids, max_len):
    """
    :param h: [B, T, H]
    :param seg_ids: [B, T]
    :return: h_ph: [B, T_ph, H]
    """
    B, T, H = h.shape
    h_gby_segs = h.new_zeros([B, max_len + 1, H]).scatter_add_(1, seg_ids[:, :, None].repeat([1, 1, H]), h)
    all_ones = h.new_ones(h.shape[:2])
    cnt_gby_segs = h.new_zeros([B, max_len + 1]).scatter_add_(1, seg_ids, all_ones).contiguous()
    h_gby_segs = h_gby_segs[:, 1:]
    cnt_gby_segs = cnt_gby_segs[:, 1:]
    h_gby_segs = h_gby_segs # / torch.clamp(cnt_gby_segs[:, :, None], min=1)
    return h_gby_segs

def process_item(item):
    word_durs = torch.tensor(item['word_durs']).reshape([1,-1,1])
    note_durs = torch.tensor(item['note_durs']).reshape([1,-1,1])
    note2word = torch.tensor(item['note2words']).reshape([1,-1])+1
    word_note_durs = group_hidden_by_segs(note_durs, note2word, max_len=note2word.max().item())
    is_not_breath_mask = torch.tensor([False if (t in ['breathe', '_NONE']) else True for t in item['txt']]).reshape([1,-1,1])
    l1_dur_errors = (word_durs - word_note_durs).abs().sum()
    l2_dur_errors = (word_durs - word_note_durs).pow(2).sum()
    word_durs_ = word_durs.clone()
    word_durs_[~is_not_breath_mask] = 0
    num_not_breathe_words = is_not_breath_mask.sum()
    total_durs = word_durs_.sum()
    return l1_dur_errors, l2_dur_errors, total_durs, num_not_breathe_words

singer_dict = {}
for item in item_lst:
    spk_name = item['singer']
    item_name = '#'.join(item['item_name'].split("#")[:-1])
    if spk_name not in singer_dict:
        singer_dict[spk_name] = {}
    if item_name not in singer_dict[spk_name]:
        singer_dict[spk_name][item_name] = {}

import tqdm
for item in tqdm.tqdm(item_lst):
    item_name = '#'.join(item['item_name'].split("#")[:-1])
    spk_name = item['singer']
    l1_dur_errors, l2_dur_errors, total_durs, num_not_breathe_words = process_item(item)
    
    singer_dict[spk_name][item_name]['l1_dur_errors'] = singer_dict[spk_name][item_name].get('l1_dur_errors',0) + l1_dur_errors
    singer_dict[spk_name][item_name]['l2_dur_errors'] = singer_dict[spk_name][item_name].get('l2_dur_errors',0) + l2_dur_errors
    singer_dict[spk_name][item_name]['total_durs'] = singer_dict[spk_name][item_name].get('total_durs',0) + total_durs
    singer_dict[spk_name][item_name]['num_not_breathe_words'] = singer_dict[spk_name][item_name].get('num_not_breathe_words',0) + num_not_breathe_words


result_dict = {}
for spk_name in list(singer_dict.keys()):
    if spk_name not in result_dict:
        result_dict[spk_name] = []
    for item_name in list(singer_dict[spk_name].keys()):
        l1_error_rate = singer_dict[spk_name][item_name]['l1_dur_errors'] / singer_dict[spk_name][item_name]['total_durs']
        l2_error_rate = singer_dict[spk_name][item_name]['l2_dur_errors'] / singer_dict[spk_name][item_name]['total_durs']
        result_dict[spk_name].append( [item_name, l1_error_rate.item(), l2_error_rate.item()] )
    
    result_dict[spk_name] = sorted(result_dict[spk_name], key=lambda x: x[1]) # sort by l1 dur_error
    # result_dict[spk_name] = sorted(result_dict[spk_name], key=lambda x: x[2]) # sort by l2 dur_error
with open("/home/renyi/hjz/NATSpeech/singing/svs/sorted_by_l1_error.json", 'w') as fp:
    json.dump(result_dict, fp, indent=2, separators=(',', ': '), ensure_ascii=False)
print(" ")


