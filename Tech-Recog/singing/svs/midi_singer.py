from singing.svs.module.midi_singer import MIDISinger, DiffPostnet
from singing.svs.base_gen_task import AuxDecoderMIDITask
from utils.commons.hparams import hparams
from tasks.tts.dataset_utils import FastSpeechDataset
import torch
from utils.commons.dataset_utils import collate_1d_or_2d
import torch.nn.functional as F
from utils.commons.ckpt_utils import load_ckpt

class MIDIDataset(FastSpeechDataset):
    def __getitem__(self, index):

        sample = super(MIDIDataset, self).__getitem__(index)
        item = self._get_item(index)
        max_frames = sample['mel'].shape[0]
        print(item.keys())
        note = torch.LongTensor(item['ep_pitches'][:hparams['max_input_tokens']])
        note_dur = torch.FloatTensor(item['ep_notedurs'][:hparams['max_input_tokens']])
        note_type = torch.LongTensor(item['ep_types'][:hparams['max_input_tokens']])
        sample["mel2word"] = torch.LongTensor(item.get("mel2word"))[:max_frames]
        sample["note"], sample["note_dur"], sample["note_type"] = note, note_dur, note_type
        return sample
    
    def collater(self, samples):
        if len(samples) == 0:
            return {}
        batch = super(MIDIDataset, self).collater(samples)
        notes = collate_1d_or_2d([s['note'] for s in samples], 0.0)
        note_durs = collate_1d_or_2d([s['note_dur'] for s in samples], 0.0)
        note_types = collate_1d_or_2d([s['note_type'] for s in samples], 0.0)
        mel2word = collate_1d_or_2d([s['mel2word'] for s in samples], 0)
        batch['mel2word'] = mel2word
        batch["notes"], batch["note_durs"], batch["note_types"] = notes, note_durs, note_types
        return batch

class MIDISingerTask(AuxDecoderMIDITask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = MIDIDataset

    def build_tts_model(self):
        dict_size = len(self.token_encoder)
        self.model = MIDISinger(dict_size, hparams)

    def run_model(self, sample, infer=False):
        txt_tokens = sample["txt_tokens"]
        mel2ph = sample["mel2ph"]
        spk_id = sample["spk_ids"]
        f0, uv = sample["f0"], sample["uv"]
        notes, note_durs, note_types = sample["notes"], sample["note_durs"], sample["note_types"]
        target = sample["mels"]
        output = self.model(txt_tokens, mel2ph=mel2ph, spk_embed=None, spk_id=spk_id,f0=f0, uv=uv, infer=infer, note=notes, note_dur=note_durs, note_type=note_types)
        losses = {}
        # print(sample.keys())
        # print(target.size(),output['mel_out'].size())
        target=target[:,:output['mel_out'].size()[1],:]
        # print(mel2ph)
        self.add_mel_loss(output['mel_out'], target, losses)
        self.add_dur_loss(output['dur'], mel2ph, txt_tokens, losses=losses)
        self.add_pitch_loss(output, sample, losses)
        return losses, output

    def add_pitch_loss(self, output, sample, losses):
        mel2ph = sample['mel2ph']  # [B, T_s]
        f0 = sample['f0']
        uv = sample['uv']
        nonpadding = (mel2ph != 0).float()
        if hparams["f0_gen"] == "diff":
            losses["fdiff"] = output["fdiff"]
            losses['uv'] = (F.binary_cross_entropy_with_logits(
                output["uv_pred"][:, :, 0], uv, reduction='none') * nonpadding).sum() / nonpadding.sum() * hparams['lambda_uv']
        elif hparams["f0_gen"] == "gmdiff":
            losses["gdiff"] = output["gdiff"]
            losses["mdiff"] = output["mdiff"]

class MIDISingerDiffJointTask(MIDISingerTask):
    def build_tts_model(self):
        dict_size = len(self.token_encoder)
        self.model = MIDISinger(dict_size, hparams)
        self.postnet = DiffPostnet()

    def run_model(self, sample, infer=False):
        txt_tokens = sample["txt_tokens"]
        mel2ph = sample["mel2ph"]
        spk_id = sample["spk_ids"]
        f0, uv = sample["f0"], sample["uv"]
        notes, note_durs, note_types = sample["notes"], sample["note_durs"], sample["note_types"]
        target = sample["mels"]
        output = self.model(txt_tokens, mel2ph=mel2ph, spk_embed=None, spk_id=spk_id,f0=f0, uv=uv, infer=infer, note=notes, note_dur=note_durs, note_type=note_types)
        losses = {}
        # print(sample.keys())
        # print(target.size(),output['mel_out'].size())
        target = target[:, :output['mel_out'].size()[1], :]  # maybe size problem
        # print(mel2ph)
        self.add_mel_loss(output['mel_out'], target, losses)
        self.add_dur_loss(output['dur'], mel2ph, txt_tokens, losses=losses)
        self.add_pitch_loss(output, sample, losses)

        # postnet
        coarse_mel = output["mel_out"]
        output["coarse_mel"] = coarse_mel
        self.postnet(target, infer, output, spk_id)
        losses["diff"] = output["diff"]
        return losses, output

class FlowPostnetTask(MIDISingerTask):
    def __init__(self):
        super(FlowPostnetTask, self).__init__()

    def build_model(self):
        self.build_pretrain_model()
        self.model = FlowPostnet()
        return self.model

    def build_pretrain_model(self):
        dict_size = len(self.token_encoder)
        self.pretrain = MIDISinger(dict_size, hparams)
        from utils.commons.ckpt_utils import load_ckpt
        load_ckpt(self.pretrain, hparams['fs2_ckpt_dir'], 'model', strict=True) 
        for k, v in self.pretrain.named_parameters():
            v.requires_grad = False    

    def run_model(self, sample, infer=False):
        txt_tokens = sample["txt_tokens"]
        mel2ph = sample["mel2ph"]
        spk_id = sample["spk_ids"]
        f0, uv = sample["f0"], sample["uv"]
        notes, note_durs, note_types = sample["notes"], sample["note_durs"], sample["note_types"]
        target = sample["mels"]
        output = self.model(txt_tokens, mel2ph=mel2ph, spk_embed=None, spk_id=spk_id,f0=f0, uv=uv, infer=infer, note=notes, note_dur=note_durs, note_type=note_types)
        coarse_mel = output["mel_out"]
        output["coarse_mel"] = coarse_mel
        self.model(target, infer, output, spk_id)
        losses = {}
        losses["postflow"] = output["postflow"]
        return losses, output

    def build_optimizer(self, model):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hparams['lr'],
            betas=(0.9, 0.98),
            eps=1e-9)
        return self.optimizer

    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, hparams['decay_steps'], gamma=0.5)

class DiffPostnetTask(FlowPostnetTask):
    def build_model(self):
        self.build_pretrain_model()
        self.model = DiffPostnet()
    
    def run_model(self, sample, infer=False):
        txt_tokens = sample["txt_tokens"]
        mel2ph = sample["mel2ph"]
        spk_id = sample["spk_ids"]
        f0, uv = sample["f0"], sample["uv"]
        notes, note_durs, note_types = sample["notes"], sample["note_durs"], sample["note_types"]
        target = sample["mels"]
        output = self.pretrain(txt_tokens, mel2ph=mel2ph, spk_embed=None, spk_id=spk_id,f0=f0, uv=uv, infer=infer, note=notes, note_dur=note_durs, note_type=note_types)
        coarse_mel = output["mel_out"]
        output["coarse_mel"] = coarse_mel
        self.model(target, infer, output, spk_id)
        losses = {}
        losses["diff"] = output["diff"]
        return losses, output