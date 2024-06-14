# multi-singer/speaker/lingual svs
from singing.multi_svs.base_task import AuxDecoderMIDITask
import torch
from tasks.tts.dataset_utils import FastSpeechWordDataset, FastSpeechDataset
import random
from utils.commons.dataset_utils import collate_1d_or_2d
from singing.multi_svs.module.multi_singer import MultiSinger, PitchDiff
from utils.nn.model_utils import print_arch
import torch.nn.functional as F
from utils.commons.tensor_utils import tensors_to_scalars
from utils.audio.pitch_utils import norm_interp_f0, denorm_f0
from utils.commons.hparams import hparams
from singing.multi_svs.base_task import f0_to_figure
from utils.nn.schedulers import NoneSchedule
import numpy as np
import miditoolkit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.audio.pitch_utils import norm_f0
from tasks.tts.vocoder_infer.base_vocoder import get_vocoder_cls

def spec_to_figure(spec, vmin=None, vmax=None, title=''):
    if isinstance(spec, torch.Tensor):
        spec = spec.cpu().numpy()
    fig = plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
    return fig

def norm_interp_midif0(f0, uv, pitch_norm='log', f0_mean=None, f0_std=None):
    is_torch = isinstance(f0, torch.Tensor)
    if is_torch:
        device = f0.device
        f0 = f0.data.cpu().numpy()
    f0 = norm_f0(f0, uv, pitch_norm, f0_mean, f0_std)
    if sum(uv) == len(f0):
        f0[uv] = 0
    elif sum(uv) > 0:
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
    if is_torch:
        uv = torch.FloatTensor(uv)
        f0 = torch.FloatTensor(f0)
        f0 = f0.to(device)
        uv = uv.to(device)
    return f0, uv


def equal_batch(batch, batch_1):
    bsz = batch["txt_tokens"].shape[0]
    bsz_1 = batch_1["txt_tokens"].shape[0]
    if bsz == bsz_1:
        return batch, batch_1
    min_bsz = min(bsz, bsz_1)
    choice = random.sample([_ for _ in range(bsz)], min_bsz)
    choice1 = random.sample([_ for _ in range(bsz_1)], min_bsz)
    batch = {k: v[choice] for k, v in batch.items() if type(v) is not list and type(v) is not int}
    batch_1 = {k: v[choice1] for k, v in batch_1.items() if type(v) is not list and type(v) is not int}
    return batch, batch_1

class Aishell3Dataset(FastSpeechWordDataset):
    def __init__(self, prefix, shuffle):
        super().__init__(prefix, shuffle)  

    def f0_aug(self, f0, p=0.2):
        if random.random() < p:
            return f0
        else:
            semitone = random.randint(-5, 5)
            return f0 * (2 ** (semitone / 12))

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample["random_f0"] = self.f0_aug(sample["f0"])
        return sample

    def collater(self, samples):
        batch = super().collater(samples)
        batch["random_f0"] = collate_1d_or_2d([s['random_f0'] for s in samples], 0.0)
        return batch

class SpeechSingerDataset(FastSpeechWordDataset):
    def __init__(self, prefix, shuffle):
        super(SpeechSingerDataset, self).__init__(prefix, shuffle)
        self.indexed_ds_speech = None
        self.data_dir_speech = hparams["binary_data_dir_speech"]
        self.use_speech = False
        self.sizes_speech = np.load(f'{self.data_dir_speech}/{self.prefix}_lengths.npy')

    def _get_item(self, index):
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
        if index < len(self.indexed_ds):
            self.use_speech = False
            return self.indexed_ds[index]
        self.use_speech = True
        if self.indexed_ds_speech is None:
            self.indexed_ds_speech = IndexedDataset(f'{self.data_dir_speech}/{self.prefix}')
        index = index - len(self.indexed_ds)
        return self.indexed_ds_speech[index]

    def __getitem__(self, index):
        item = self._get_item(index)
        max_frames = hparams['max_frames']
        spec = torch.Tensor(item['mel'])[:max_frames]
        max_frames = spec.shape[0] // hparams['frames_multiple'] * hparams['frames_multiple']
        spec = spec[:max_frames]
        phone = torch.LongTensor(item['phone'][:hparams['max_input_tokens']])
        sample = {
            "id": index,
            "item_name": item['item_name'],
            "text": item['txt'],
            "txt_token": phone,
            "mel": spec,
            "mel_nonpadding": spec.abs().sum(-1) > 0,
        }
        max_frames = hparams['max_frames']
        spec = sample['mel']
        T = spec.shape[0]
        phone = sample['txt_token']
        sample['energy'] = (spec.exp() ** 2).sum(-1).sqrt()
        sample['mel2ph'] = mel2ph = torch.LongTensor(item['mel2ph'])[:T] if 'mel2ph' in item else None
        if hparams['use_pitch_embed']:
            assert 'f0' in item
            if hparams.get('normalize_pitch', False):
                f0 = item["f0"]
                if len(f0 > 0) > 0 and f0[f0 > 0].std() > 0:
                    f0[f0 > 0] = (f0[f0 > 0] - f0[f0 > 0].mean()) / f0[f0 > 0].std() * hparams['f0_std'] + \
                                 hparams['f0_mean']
                    f0[f0 > 0] = f0[f0 > 0].clip(min=60, max=500)
                pitch = f0_to_coarse(f0)
                pitch = torch.LongTensor(pitch[:max_frames])
            else:
                pitch = torch.LongTensor(item.get("pitch"))[:max_frames] if "pitch" in item else None
            f0, uv = norm_interp_f0(item["f0"][:max_frames], hparams)
            uv = torch.FloatTensor(uv)
            f0 = torch.FloatTensor(f0)
        else:
            f0 = uv = torch.zeros_like(mel2ph)
            pitch = None
        sample["f0"], sample["uv"], sample["pitch"] = f0, uv, pitch
        sample["yingram"] = torch.FloatTensor(item["yingram"])[:, 21:]
        sample["yingram_nospk"] = torch.FloatTensor(item["yingram_nospk"])[:, 21:]
        if self.use_speech:
            sample["infer_spk_embed"] = torch.Tensor(item['infer_spk_embed'])
            # sample["spk_id"] = item['spk_id']
            # sample["infer_spk_id"] = item["spk_id"]
            sample["spk_id"] = 0
            sample["infer_spk_id"] = 0
        else:
            # sample["infer_spk_embed"] = torch.Tensor(np.load("/home/renyi/hjz/NeuralSeq/data/processed/aishell3_1/mfa_inputs/1/0001000_SSB0623_SSB06230346.npy.npy"))
            sample["infer_spk_embed"] = None
            # sample["spk_id"] = hparams["num_spk"] + item["spk_id"]
            # sample["infer_spk_id"] = hparams["num_spk"] + item["spk_id"] # 0819修改，注意运行前面的pretrain model要改回去
            sample["spk_id"] = item['spk_id']
            sample["infer_spk_id"] = item["spk_id"]
        yin_len = sample["yingram"].shape[0]
        yin_len = yin_len // hparams.get('frames_multiple', 1) * hparams.get('frames_multiple', 1)
        mel_len = sample["mel"].shape[0]
        sample["yingram"] = sample["yingram"][:yin_len][:, :]
        sample["yingram_nospk"] = sample["yingram_nospk"][:yin_len][:, :]
        sample["mel"] = sample["mel"][:yin_len]
        sample["mel2ph"] = sample["mel2ph"][:yin_len]
        sample["f0"] = sample["f0"][:yin_len]
        sample["uv"] = sample["uv"][:yin_len]
        sample["energy"] = sample["energy"][:yin_len]
        if "pitch_midi" in item:
            sample['pitch_midi'] = torch.LongTensor(item['pitch_midi'])[:hparams['max_frames']]
            sample['midi_dur'] = torch.FloatTensor(item['midi_dur'])[:hparams['max_frames']]
            sample['is_slur'] = torch.LongTensor(item['is_slur'])[:hparams['max_frames']]
            sample['word_boundary'] = torch.LongTensor(item['word_boundary'])[:hparams['max_frames']]
        if hparams.get("energy_type") == "multiband":
            energy0 = (sample["mel"][:, :20].exp() ** 2).sum(-1).sqrt()
            energy1 = (sample["mel"][:, 20:40].exp() ** 2).sum(-1).sqrt()
            energy2 = (sample["mel"][:, 40:60].exp() ** 2).sum(-1).sqrt()
            energy3 = (sample["mel"][:, 60:80].exp() ** 2).sum(-1).sqrt()
            sample["mb_energy"] = torch.stack([energy0, energy1, energy2, energy3], dim=-1)
        return sample

    def collater(self, samples):
        mel_lengths = torch.LongTensor([s['mel'].shape[0] for s in samples])
        batch = super(SpeechSingerDataset, self).collater(samples)
        infer_spk_ids = torch.LongTensor([s['infer_spk_id'] for s in samples])
        batch['infer_spk_ids'] = infer_spk_ids
        if samples[0]["infer_spk_embed"] is None:
            batch["infer_spk_embed"] = None
        else:
            batch["infer_spk_embed"] = torch.stack([s['infer_spk_embed'] for s in samples])
        batch["mel_lengths"] = mel_lengths
        batch['yingram'] = utils.collate_2d([s['yingram'] for s in samples], 0.0)
        batch["yingram_nospk"] = utils.collate_2d([s['yingram_nospk'] for s in samples], 0.0)
        if "pitch_midi" in samples[0]:
            batch['pitch_midi'] = utils.collate_1d([s['pitch_midi'] for s in samples], 0)
            batch['midi_dur'] = utils.collate_1d([s['midi_dur'] for s in samples], 0)
            batch['is_slur'] = utils.collate_1d([s['is_slur'] for s in samples], 0)
            batch['word_boundary'] = utils.collate_1d([s['word_boundary'] for s in samples], 0)
        if hparams.get("energy_type") == "multiband":
            batch["mb_energy"] = utils.collate_2d([s['mb_energy'] for s in samples], 0.0)
        return batch

    def num_tokens2(self, index):
        index = index - len(self)
        return self.sizes_speech[index]

    def ordered_indices2(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self.sizes_speech))
            if self.sort_by_len:
                indices = indices[np.argsort(np.array(self.sizes_speech)[indices], kind='mergesort')]
        else:
            indices = np.arange(len(self.sizes_speech))
        return indices

class SpeechDataset(FastSpeechWordDataset):
    def __getitem__(self, index):
        sample = super(SpeechDataset, self).__getitem__(index)
        item = self._get_item(index)
        norm_f0 = item["f0"]
        f0_mean = norm_f0[norm_f0 > 0].mean()
        f0_std = norm_f0[norm_f0 > 0].std()
        norm_f0[norm_f0 > 0] = (norm_f0[norm_f0 > 0] - f0_mean) / f0_std
        sample["norm_f0"] = torch.FloatTensor(norm_f0)
        sample["f0_mean"] = f0_mean
        sample["f0_std"] = f0_std
        return sample
    
    def collater(self, samples):
        batch = super(SpeechDataset, self).collater(samples)
        mel_len = batch["mels"].shape[1]
        norm_f0 = utils.collate_1d([s['norm_f0'] for s in samples], 0.0)
        batch["norm_f0"] = norm_f0[:, :mel_len]
        batch["ph2words"] = batch["ph2word"]
        f0_mean = torch.Tensor([s['f0_mean'] for s in samples])
        f0_std = torch.Tensor([s['f0_std'] for s in samples])
        batch["f0_mean"] = f0_mean
        batch["f0_std"] = f0_std
        return batch

class SingingDataset(FastSpeechDataset):
    def get_expanded_note(self, note, note_dur, item):
        phs = item["phs"]
        ALL_SHENGMU = ['b', 'c', 'ch', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 'sh', 't', 'x', 'z', 'zh']
        for i, ph in enumerate(phs):
            if ph in ALL_SHENGMU:
                note_dur[i] = 0.0
        note_frames = (note_dur * 24000 / 128 + 0.5)
        note_frames = note_frames.astype(np.int)
        expanded_note = np.repeat(note, note_frames).astype(np.float)
        return expanded_note

    def get_avg_f0(self, f0, mel2word):
        num_word = max(mel2word)
        mel2word = np.array(mel2word)
        mask = (f0 > 0).astype(float)
        avg_word_f0 = []
        for n in range(num_word):
            word_mask = (mel2word == n+1)
            word_f0 = f0[np.where((mask * word_mask) > 0)[0]]
            if len(word_f0) == 0:
                avg_word_f0.append(0)
            else:
                avg_word_f0.append(np.median(word_f0))
        avg_word_f0 = torch.FloatTensor(avg_word_f0)
        mel2word = torch.LongTensor(mel2word)
        expanded_avg_f0 = torch.gather(F.pad(avg_word_f0, [1, 0]), 0, mel2word)
        return expanded_avg_f0

    def get_word_mask_f0(self, f0, mel2word):
        masked_f0 = f0
        mel2word = np.array(mel2word)
        num_word = max(mel2word)
        if int(num_word * 0.3) < 1:
            selected_words = random.sample(list(range(1, num_word+1)), 1)
        else:
            selected_words = random.sample(list(range(1, num_word+1)), int(num_word * 0.3))
        mask = np.zeros_like(mel2word)
        for word in selected_words:
            mask[mel2word==word] = 1
            masked_f0[mel2word==word] = 0
        masked_f0 = torch.FloatTensor(masked_f0)
        return masked_f0, mask
    
    def get_span_mask_f0(self, f0, mel2word):
        masked_f0 = f0
        mel2word = np.array(mel2word)
        num_word = max(mel2word)
        if int(num_word * 0.5) < 1:
            selected_words = random.sample(list(range(1, num_word+1)), 1)
        else:
            selected_words = random.sample(list(range(1, num_word+1)), int(num_word * 0.5))
        mask = np.zeros_like(mel2word)
        random_ratio = random.random() * 0.8 + 0.1 # 10% - 90% left
        for word in selected_words:
            num_frames = (mel2word==word).astype(float).sum()
            if num_frames < 10:
                continue
            start_frame = random.randint(0, int(num_frames*(1-random_ratio)))
            start_index = np.where(mel2word==word)[0][0] + start_frame
            end_index = start_index + int(num_frames*random_ratio)
            local_f0 = f0[start_index:end_index]
            local_uv = (local_f0 > 0).astype(float)
            local_mid_f0 = local_f0[np.where(local_uv > 0)[0]]
            if len(local_mid_f0) > 0:
                local_mid = np.median(local_mid_f0)
            else:
                local_mid = 0
            masked_f0[start_index:end_index] = local_mid
            mask[start_index:end_index] = 1
        masked_f0 = torch.FloatTensor(masked_f0)
        return masked_f0, mask

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        item = self._get_item(index)
        max_frames = sample['mel'].shape[0]
        sample["mel2word"] = torch.LongTensor(item["mel2word"])[:max_frames]
        sample["ph2word"] = torch.LongTensor(item['ph2word'])
        mid_f0 = self.get_avg_f0(item["f0"][:max_frames], item["mel2word"][:max_frames])
        sample["mid_f0"] = torch.FloatTensor(mid_f0)
        if "notes" in item and self.prefix != "train":
            note = np.array(item["notes"])
            note_dur = np.array(item["notes_dur"])
            expanded_note = self.get_expanded_note(note, note_dur, item)
            midi_f0 = 440 * (2 ** ((expanded_note - 69)/12.0)) * (expanded_note > 0).astype(np.float)
            sample["midi_f0"] = torch.FloatTensor(midi_f0)
            if len(sample["midi_f0"]) > len(sample["f0"]):
                sample["midi_f0"] = sample["midi_f0"][:len(sample["f0"])]
            else:
                pad = torch.zeros([len(sample["f0"]) - len(sample["midi_f0"])])
                sample["midi_f0"] = torch.cat([sample["midi_f0"], pad], dim=0)
            # sample["midi_f0"][sample["uv"] > 0] = 0
            # sample["mid_f0"] = self.get_avg_f0(sample["midi_f0"].numpy(), item["mel2word"][:max_frames])
            sample["mid_f0"] = sample["midi_f0"]
            sample["midi_f0"], _ = norm_interp_midif0(sample["midi_f0"], item["f0"][:len(sample["f0"])]==0)
        # wm_f0, wm = self.get_word_mask_f0(item["f0"][:max_frames], item["mel2word"][:max_frames])
        # sm_f0, sm = self.get_span_mask_f0(item["f0"][:max_frames], item["mel2word"][:max_frames])
        # sample["wm_f0"] = torch.FloatTensor(wm_f0)
        # sample["wm"] = torch.LongTensor(wm)
        # sample["sm_f0"] = torch.FloatTensor(sm_f0)
        # sample["sm"] = torch.LongTensor(sm)
        return sample

    def collater(self, samples):
        batch = super().collater(samples)
        mel2word = collate_1d_or_2d([s['mel2word'] for s in samples], 0)
        batch['mel2word'] = mel2word
        ph2word = collate_1d_or_2d([s['ph2word'] for s in samples], 0)
        batch['ph2word'] = ph2word
        batch['word_lengths'] = torch.LongTensor([max(s["ph2word"]) for s in samples])
        p = random.random()
        if p < 0.1:
            pass # without cond
        else:
            batch["mid_f0s"] = collate_1d_or_2d([s['mid_f0'] for s in samples], 0.0)
        if "midi_f0" in samples[0]:
            batch["midi_f0s"] = collate_1d_or_2d([s['midi_f0'] for s in samples], 0.0)
            batch["mid_f0s_"] = collate_1d_or_2d([s['mid_f0'] for s in samples], 0.0)
            # batch["wm_f0s_"] = collate_1d_or_2d([s['wm_f0'] for s in samples], 0.0)
            # batch["sm_f0s_"] = collate_1d_or_2d([s['sm_f0'] for s in samples], 0.0)
        # elif p < 0.6:
        #     batch["mid_f0s"] = collate_1d_or_2d([s['mid_f0'] for s in samples], 0.0)
        #     batch["wm_f0s"] = collate_1d_or_2d([s['wm_f0'] for s in samples], 0.0)
        #     batch["wms"] = collate_1d_or_2d([s['wm'] for s in samples], 0.0)
        # elif p < 0.8:
        #     batch["wm_f0s"] = collate_1d_or_2d([s['wm_f0'] for s in samples], 0.0)
        #     batch["wms"] = collate_1d_or_2d([s['wm'] for s in samples], 0.0)
        # else:
        #     batch["sm_f0s"] = collate_1d_or_2d([s['sm_f0'] for s in samples], 0.0)
        #     batch["sms"] = collate_1d_or_2d([s['sm'] for s in samples], 0.0)
        return batch

class XiaomaDataset(FastSpeechDataset):
    def get_expanded_note(self, midi_fn):
        mf = miditoolkit.MidiFile(midi_fn)
        instru = mf.instruments[0]
        notes = instru.notes
        note_list = []
        for note in notes:
            pitch = note.pitch
            note_start = note.start / 220 * 0.5
            note_end = note.end / 220 * 0.5
            note_dur = note_end - note_start
            note_frames = int(note_dur * 24000 / 128)
            note_list.extend([pitch]*note_frames)
        return np.array(note_list)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        item = self._get_item(index)
        item_name = item["item_name"]
        song = item_name[4:-5]
        sent = item_name[-4:]
        midi_fn = f"/data_disk/xiaoma/splits/{song}/{sent}.mid"

        # note = item["pitch_midi"]
        # note_dur = item["midi_dur"]
        expanded_note = self.get_expanded_note(midi_fn)
        # expanded_note = self.get_expanded_note(note, note_dur, item["ph"])
        midi_f0 = 440 * (2 ** ((expanded_note - 69)/12.0)) * (expanded_note > 0).astype(np.float)
        sample["midi_f0"] = torch.FloatTensor(midi_f0)
        if len(sample["midi_f0"]) > len(sample["f0"]):
            sample["midi_f0"] = sample["midi_f0"][:len(sample["f0"])]
        else:
            pad = torch.zeros([len(sample["f0"]) - len(sample["midi_f0"])])
            sample["midi_f0"] = torch.cat([sample["midi_f0"], pad], dim=0)
        return sample
    def collater(self, samples):
        batch = super().collater(samples)
        batch["midi_f0s"] = collate_1d_or_2d([s['midi_f0'] for s in samples], 0.0)
        return batch

class MultiSpeakerTask(AuxDecoderMIDITask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = Aishell3Dataset

    def build_model(self):
        dict_size = len(self.token_encoder)
        self.model = MultiSpeaker(dict_size, hparams)
        print_arch(self.model)
    
    def run_model(self, sample, infer=False):
        txt_tokens, mel2ph, spk_embed, f0, uv = sample["txt_tokens"], sample["mel2ph"], sample["spk_ids"], sample["f0"], \
        sample["uv"]
        output = self.model(txt_tokens, mel2ph, spk_embed, f0, uv, None, random_f0=sample["random_f0"])
        losses = {}
        self.add_pitch_loss(output, sample, losses)
        self.add_mel_loss(output['mel_out'], sample["mels"], losses)
        return losses, output
    
    def validation_start(self):
        pass

    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(sample, infer=True)
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        mel_out = (model_out['mel_out'])
        outputs = tensors_to_scalars(outputs)
        if batch_idx < hparams['num_valid_plots']:
            self.plot_mel(batch_idx, sample['mels'], mel_out)
            gt_f0 = denorm_f0(sample['f0'], sample["uv"])
            pred_f0 = denorm_f0(model_out['f0_restore'], sample["uv"])
            ref_f0 = denorm_f0(sample["random_f0"], sample["uv"])
            self.logger.add_figure(
                f'f0_{batch_idx}',
                f0_to_figure(gt_f0[0], ref_f0[0], pred_f0[0]),
                self.global_step)
        return outputs
    
    def add_pitch_loss(self, output, sample, losses):
        mel2ph = sample['mel2ph']  # [B, T_s]
        f0 = sample['f0']
        uv = sample['uv']
        nonpadding = (mel2ph != 0).float()
        f0_pred = output['f0_restore']
        nonpadding = nonpadding * (uv == 0).float()
        losses['f0'] = (F.l1_loss(f0_pred, f0, reduction='none') * nonpadding).sum() \
                       / nonpadding.sum() * 0.1

class MultiSingerTask(MultiSpeakerTask):
    def __init__(self):
        super().__init__()
        random.seed(1234)
        self.dataset_cls = SingingDataset

    def build_model(self):
        dict_size = len(self.token_encoder)
        self.model = MultiSinger(dict_size, hparams)
        print_arch(self.model)
    
    def run_model(self, sample, infer=False):
        txt_tokens, mel2ph, spk_embed, f0, uv = sample["txt_tokens"], sample["mel2ph"], sample["spk_embed"], sample["f0"], \
        sample["uv"]
        ph2word, mel2word, word_len = sample.get("ph2word"), sample.get("mel2word"), sample.get('word_lengths').max()

        output = self.model(txt_tokens, ph2word, word_len, mel2ph, mel2word, spk_embed, f0, uv, None, infer=infer, midi_f0=sample.get("midi_f0s"), mid_f0=sample.get("mid_f0s"), wm_f0=sample.get("wm_f0s"), wm=sample.get("wms"), sm_f0=sample.get("sm_f0s"), 
        sm=sample.get("sms"))
        losses = {}
        # losses["l_fd"] = output["l_fd"]
        # self.add_uv_loss(output, sample, losses)
        self.add_mel_loss(output['mel_out'], sample["mels"], losses)
        return losses, output
    
    def add_uv_loss(self, output, sample, losses):
        mel2ph = sample['mel2ph']  # [B, T_s]
        uv = sample['uv']
        nonpadding = (mel2ph != 0).float()
        uv_pred = output['uv_pred']
        losses['uv'] = (F.binary_cross_entropy_with_logits(
            uv_pred[:, :, 0], uv, reduction='none') * nonpadding).sum() \
                        / nonpadding.sum() * hparams['lambda_uv']
    def validation_start(self):
        self.vocoder = get_vocoder_cls(hparams['vocoder'])()
        # pass

    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(sample, infer=True)
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        outputs = tensors_to_scalars(outputs)
        if batch_idx < hparams['num_valid_plots']:
            sr = hparams["audio_sample_rate"]
            gt_f0 = denorm_f0(sample['f0'], sample["uv"])
            wav_gt = self.vocoder.spec2wav(sample["mels"][0].cpu(), f0=gt_f0[0].cpu())
            self.logger.add_audio(f'wav_gt_{batch_idx}', wav_gt, self.global_step, sr)
            wav_pred = self.vocoder.spec2wav(model_out['mel_out'][0].cpu(), f0=model_out["f0_denorm_pred"][0].cpu())
            self.logger.add_audio(f'wav_pred_{batch_idx}', wav_pred, self.global_step, sr)
            self.plot_mel(batch_idx, sample['mels'], model_out['mel_out'][0], f'mel_{batch_idx}')
            self.logger.add_figure(
                f'f0_{batch_idx}',
                f0_to_figure(model_out["f0_denorm"][0], model_out["midi_f0"][0], model_out["f0_denorm_pred"][0]),
                self.global_step)
            self.logger.add_figure(
                f'cond_{batch_idx}',
                f0_to_figure(model_out["f0_denorm"][0], None, sample["mid_f0s_"][0]),
                self.global_step)
            if 'attn' in model_out:
                self.logger.add_figure(
                    f'attn_{batch_idx}', spec_to_figure(model_out['attn'][0]), self.global_step)
        return outputs

class PitchDiffTask(MultiSingerTask):
    def __init__(self):
        super().__init__()

    def build_model(self):
        dict_size = len(self.token_encoder)
        self.model = PitchDiff(dict_size, hparams)
        pretrain_singer = torch.load(hparams["pretrain_singer"], map_location="cpu")["state_dict"]["model"]
        self.model.p_singer.load_state_dict(pretrain_singer)
        for k, v in self.model.named_parameters():
            if "p_singer" in k:
                v.requires_grad = False
        print_arch(self.model)
    
    def run_model(self, sample, infer=False):
        txt_tokens, mel2ph, spk_embed, f0, uv = sample["txt_tokens"], sample["mel2ph"], sample["spk_embed"], sample["f0"], \
        sample["uv"]
        ph2word, mel2word, word_len = sample.get("ph2word"), sample.get("mel2word"), sample.get('word_lengths').max()

        output = self.model(txt_tokens, ph2word, word_len, mel2ph, mel2word, spk_embed, f0, uv, None, infer=infer, midi_f0=sample.get("midi_f0s"), mid_f0=sample.get("mid_f0s"), wm_f0=sample.get("wm_f0s"), wm=sample.get("wms"), sm_f0=sample.get("sm_f0s"), 
        sm=sample.get("sms"))
        losses = {}
        losses["l_fd"] = output["l_fd"]
        return losses, output

    def build_optimizer(self, model):
        self.optimizer = optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=0.0002,
            betas=(0.9, 0.98))

        return optimizer
    
    def build_scheduler(self, optimizer):
        return NoneSchedule(optimizer, 0.0002)