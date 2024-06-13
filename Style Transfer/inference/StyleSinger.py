from resemblyzer import VoiceEncoder
import torch
from utils.ckpt_utils import load_ckpt
from utils.hparams import hparams
from modules.StyleSinger.stylesinger import  StyleSinger
import os
import numpy as np
from tasks.tts.vocoder_infer.base_vocoder import BaseVocoder, get_vocoder_cls
from resemblyzer import VoiceEncoder
from modules.hifigan.hifigan_nsf import HifiGanGenerator
from data_gen.tts.emotion import inference as EmotionEncoder
from data_gen.tts.emotion.inference import embed_utterance as Embed_utterance
from data_gen.tts.emotion.inference import preprocess_wav
import torch
from utils.hparams import set_hparams
from utils.text.text_encoder import build_token_encoder
from utils.audios import librosa_wav2spec
from utils.pitch_utils import norm_interp_f0


class StyleSingerInfer:
    def __init__(self, hparams, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hparams = hparams
        self.device = device
        self.data_dir = hparams['processed_data_dir']
        self.ph_encoder = build_token_encoder(f'{self.data_dir}/phone_set.json')
        self.model = self.build_model()
        self.model.eval()
        self.model.to(self.device)
        self.vocoder: BaseVocoder = get_vocoder_cls(hparams)()

    def build_model(self):
        model = StyleSinger(self.ph_encoder)
        model.eval()
        from utils.commons.ckpt_utils import load_ckpt
        load_ckpt(model, hparams['exp_name'], 'model',strict=False)
        return model

    def forward_model(self, inp):
        sample = self.input_to_batch(inp)
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_id')
        if hparams['emo']:
            emo_embed = sample.get('emo_embed')
        else:
            emo_embed=None
        ref_mels = sample['mels']
        ref_f0=sample['f0']
        notes, note_durs, note_types = sample["notes"], sample["note_durs"], sample["note_types"]
        model_out = self.model(sample['txt_tokens'], spk_embed=spk_embed, emo_embed=emo_embed, ref_mels=ref_mels, ref_f0=ref_f0,
                                global_steps=20000, infer=True, note=notes, note_dur=note_durs, note_type=note_types)
    
        f0_pred=model_out['f0_denorm'].cpu().numpy()
        mel_pred = model_out["mel_out"].cpu()
        mel_pred_mask = np.abs(mel_pred).sum(-1) > 0
        mel_pred = mel_pred[mel_pred_mask]
        mel_pred = np.clip(mel_pred, hparams['mel_vmin'], hparams['mel_vmax'])
        if f0_pred is not None:
            if len(f0_pred) > len(mel_pred_mask):
                f0_pred = f0_pred[:len(mel_pred_mask)]
            f0_pred = f0_pred[mel_pred_mask]
        wav_pred = self.vocoder.spec2wav(mel_pred,f0=f0_pred)
        return wav_pred

    def build_vocoder(self):
        base_dir = self.hparams['vocoder_ckpt']
        config_path = f'{base_dir}/config.yaml'
        config = set_hparams(config_path, global_hparams=False)
        vocoder = HifiGanGenerator(config)
        load_ckpt(vocoder, base_dir, 'model_gen')
        return vocoder

    def run_vocoder(self, c):
        c = c.transpose(2, 1)
        y = self.vocoder(c)[:, 0]
        return y

    def process_audio(self, wav_fn):
        wav2spec_dict = librosa_wav2spec(
            wav_fn,
            fft_size=self.hparams['fft_size'],
            hop_size=self.hparams['hop_size'],
            win_length=self.hparams['win_size'],
            num_mels=self.hparams['audio_num_mel_bins'],
            fmin=self.hparams['fmin'],
            fmax=self.hparams['fmax'],
            sample_rate=self.hparams['audio_sample_rate'],
            loud_norm=self.hparams['loud_norm'])
        mel = wav2spec_dict['mel']
        wav = wav2spec_dict['wav'].astype(np.float16)
        return wav, mel

    def preprocess_input(self, inp):
        # processed ph
        ph_token = self.ph_encoder.encode(' '.join(inp["ph"]))

        # processed ref audio
        ref_audio = inp['ref_audio']
        voice_encoder = VoiceEncoder().cuda()
        EmotionEncoder.load_model(self.hparams['emotion_encoder_path'])
        wav, mel = self.process_audio(ref_audio)
        inp['mel']=mel
        inp['spk_embed'] = voice_encoder.embed_utterance(wav)
        processed_wav = preprocess_wav(ref_audio)
        inp['emo_embed'] = Embed_utterance(processed_wav, using_partials=True)
        
        inp.update({
            'item_name': inp['name'],
            'ph_token': ph_token,
            'wav_fn': ref_audio
        })
        
        # parselmouth
        time_step = hparams['hop_size'] / hparams['audio_sample_rate'] * 1000
        f0_min = 80
        f0_max = 800
        if hparams['hop_size'] == 128:
            pad_size = 4
        elif hparams['hop_size'] == 256:
            pad_size = 2
        else:
            assert False
        import parselmouth
        f0 = parselmouth.Sound(wav, hparams['audio_sample_rate']).to_pitch_ac(
            time_step=time_step / 1000, voicing_threshold=0.6,
            pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']
        lpad = pad_size * 2
        rpad = len(mel) - len(f0) - lpad
        f0 = np.pad(f0, [[lpad, rpad]], mode='constant')
        delta_l = len(mel) - len(f0)
        assert np.abs(delta_l) <= 8
        if delta_l > 0:
            f0 = np.concatenate([f0, [f0[-1]] * delta_l], 0)
        inp['f0']=f0 = f0[:len(mel)]
        
        return inp

    def input_to_batch(self, item):
        item_names = [item['item_name']]
        ph = [item['ph']]

        txt_tokens = torch.LongTensor(item['ph_token'])[None, :].to(self.device)
        mels = torch.FloatTensor(item['mel'])[None, :].to(self.device)
        spk_embed = torch.FloatTensor(item['spk_embed'])[None, :].to(self.device)
        # spk_id= torch.LongTensor([item['spk_id']]).to(self.device)
        emo_embed = torch.FloatTensor(item['emo_embed'])[None, :].to(self.device)

        note = torch.LongTensor(item['note'])[None, :].to(self.device)
        note_dur = torch.FloatTensor(item['note_dur'])[None, :].to(self.device)
        note_type = torch.LongTensor(item['note_type'])[None, :].to(self.device)
        f0, uv = norm_interp_f0(item["f0"], hparams)
        uv = torch.FloatTensor(uv).to(self.device)
        f0 = torch.FloatTensor(f0).to(self.device)

        batch = {
            'item_name': item_names,
            'ph': ph,
            'mels': mels,
            'txt_tokens': txt_tokens,
            # 'spk_id': spk_id,
            'spk_embed': spk_embed,
            'emo_embed': emo_embed,
            'f0': f0,
            'notes': note,
            'note_types': note_type,
            'note_durs': note_dur,
        }
        
        return batch

    def postprocess_output(self, output):
        return output

    def infer_once(self, inp):
        inp = self.preprocess_input(inp)
        output = self.forward_model(inp)
        output = self.postprocess_output(output)
        return output

    @classmethod
    def example_run(cls):
        from utils.audio import save_wav

        set_hparams()
        inp = {
            # target infromation
            'name': 'test',
            'ph': [
            "zh",
            "i",
            "i",
            "d",
            "uan",
            "j",
            "in",
            "x",
            "iou",
            "uen",
            "sh",
            "i",
            "breathe",
            "b",
            "ing",
            "l",
            "ian",
            "l",
            "i",
            "sh",
            "uang",
            "zh",
            "i",
            "breathe",
            "n",
            "an",
            "j",
            "i",
            "t",
            "uo",
            "breathe"
        ],
            'note':  [
            59,
            59,
            61,
            62,
            62,
            71,
            71,
            69,
            69,
            66,
            64,
            64,
            0,
            64,
            64,
            69,
            69,
            66,
            66,
            64,
            64,
            62,
            62,
            0,
            59,
            59,
            66,
            66,
            61,
            61,
            0
        ],
            'note_dur': [
            0.410958904109589,
            0.410958904109589,
            0.410958904109589,
            0.821917808219178,
            0.821917808219178,
            0.410958904109589,
            0.410958904109589,
            0.2054794520547945,
            0.2054794520547945,
            0.6164383561643836,
            0.821917808219178,
            0.821917808219178,
            0.32355999999999874,
            0.821917808219178,
            0.821917808219178,
            0.410958904109589,
            0.410958904109589,
            0.2054794520547945,
            0.2054794520547945,
            0.6164383561643836,
            0.6164383561643836,
            0.821917808219178,
            0.821917808219178,
            0.3099999999999987,
            0.821917808219178,
            0.821917808219178,
            0.6164383561643836,
            0.6164383561643836,
            0.6164383561643836,
            0.6164383561643836,
            0.33856999999999715
        ],
            'note_type': [
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            2,
            1,
            2,
            2,
            2,
            2,
            2,
            2,
            1
        ],
            # reference information
            'ref_audio': 'test/test.wav'
        }

        infer_ins = cls(hparams)
        out = infer_ins.infer_once(inp)
        os.makedirs('infer_out', exist_ok=True)
        save_wav(out, f'infer_out/test.wav', hparams['audio_sample_rate'])
        print(f'Save at infer_out/test.wav.')
    
if __name__ == '__main__':
    StyleSingerInfer.example_run()