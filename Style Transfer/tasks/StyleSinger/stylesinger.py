import matplotlib
matplotlib.use('Agg')
from data_gen.tts.data_gen_utils import get_pitch
from modules.fastspeech.tts_modules import mel2ph_to_dur
import matplotlib.pyplot as plt
from utils import audio
from utils.pitch_utils import  denorm_f0
from tasks.tts.vocoder_infer.base_vocoder import BaseVocoder, get_vocoder_cls
from utils.plot import spec_to_figure
from utils.hparams import hparams
import torch
import torch.optim
import torch.nn.functional as F
import torch.utils.data
from tasks.StyleSinger.dataset import StyleSinger_dataset
from modules.StyleSinger.stylesinger import StyleSinger
import torch.distributions
import numpy as np
import utils
import os
from tasks.tts.fs2 import FastSpeech2Task
from multiprocessing.pool import Pool
from utils.plot import spec_to_figure, f0_to_figure


class StyleSingerTask(FastSpeech2Task):
    def __init__(self):
        super(StyleSingerTask, self).__init__()
        self.dataset_cls = StyleSinger_dataset

    def build_tts_model(self):
        self.model = StyleSinger(self.phone_encoder)

    def build_model(self):
        self.build_tts_model()
        if hparams['load_ckpt'] != '':
            self.load_ckpt(hparams['load_ckpt'], strict=False)
        utils.num_params(self.model)
        return self.model

    def run_model(self, model, sample, return_output=False):
        # print(sample)
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        target = sample['mels']  # [B, T_s, 80]
        mel2ph = sample['mel2ph']  # [B, T_s]
        f0 = sample['f0']  # [B, T_s]
        uv = sample['uv']  # [B, T_s] 0/1
        notes, note_durs, note_types = sample["notes"], sample["note_durs"], sample["note_types"]

        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_id')
        
        if hparams['emo']:
            emo_embed = sample.get('emo_embed')
        else:
            emo_embed=None

        output = model(txt_tokens, mel2ph=mel2ph, spk_embed=spk_embed, emo_embed=emo_embed,
                        ref_mels=target, ref_f0=f0, f0=f0, uv=uv, tgt_mels=target, global_steps=self.global_step, infer=False, note=notes, note_dur=note_durs, note_type=note_types)
        losses = {}
        if hparams['decoder']=='diffsinger':
            losses['diff'] = output['diff']
        if hparams['style']:
            if self.global_step > hparams['forcing']:
                losses['gloss'] = output['gloss'] 
            if self.global_step > hparams['rq_start']:
                losses['rq_loss'] = output['rq_loss'] 
        self.add_mel_loss(output['mel_out'], target, losses)
        self.add_dur_loss(output['dur'], mel2ph, txt_tokens, losses=losses)
        self.add_pitch_loss(output, sample, losses)

        if not return_output:
            return losses
        else:
            return losses, output
        
    def add_pitch_loss(self, output, sample, losses):
        mel2ph = sample['mel2ph']  # [B, T_s]
        f0 = sample['f0']
        uv = sample['uv']
        nonpadding = (mel2ph != 0).float()
        if hparams["f0_gen"] == "gmdiff":
            losses["gdiff1"] = output["gdiff1"]
            losses["mdiff1"] = output["mdiff1"]
            losses["gdiff2"] = output["gdiff2"]
            losses["mdiff2"] = output["mdiff2"]
        elif hparams["f0_gen"] == "conv":
            self.add_f0_loss(output['pitch_pred'], f0, uv, losses, nonpadding=nonpadding) # output['pitch_pred']: [B, T, 2], f0: [B, T], uv: [B, T]

    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs['losses'] = {}
        try:
            outputs['losses'], model_out = self.run_model(self.model, sample, return_output=True)
        except:
            print(sample['item_name'])
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']

        mel_out = self.model.out2mel(model_out['mel_out'])
        outputs = utils.tensors_to_scalars(outputs)
        if self.global_step % hparams['valid_infer_interval'] == 0 \
                and batch_idx < hparams['num_valid_plots']:
            vmin = hparams['mel_vmin']
            vmax = hparams['mel_vmax']
            if self.vocoder is None:
                self.vocoder: BaseVocoder = get_vocoder_cls(hparams)()
            if self.global_step > 0:
                spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_id')
                if hparams['emo']:
                    emo_embed = sample.get('emo_embed')
                else:
                    emo_embed=None
                ref_mels = sample['mels']
                mel2ph = sample['mel2ph']  # [B, T_s]
                ref_f0=sample['f0']
                notes, note_durs, note_types = sample["notes"], sample["note_durs"], sample["note_types"]
                model_out = self.model(sample['txt_tokens'], spk_embed=spk_embed, emo_embed=emo_embed, ref_mels=ref_mels, ref_f0=ref_f0,
                                        global_steps=self.global_step, infer=True, note=notes, note_dur=note_durs, note_type=note_types)
                f0_pred=model_out['f0_denorm'].cpu().numpy()
                mel_pred = model_out["mel_out"].cpu()
                mel_pred_mask = np.abs(mel_pred).sum(-1) > 0
                mel_pred = mel_pred[mel_pred_mask]
                mel_pred = np.clip(mel_pred, hparams['mel_vmin'], hparams['mel_vmax'])
                if f0_pred is not None:
                    if len(f0_pred) > len(mel_pred_mask):
                        f0_pred = f0_pred[:len(mel_pred_mask)]
                    f0_pred = f0_pred[mel_pred_mask]
                self.logger.add_figure(
                    f'mel_{batch_idx}',
                    spec_to_figure(model_out['mel_out'][0], vmin, vmax), self.global_step)
                wav_pred = self.vocoder.spec2wav(mel_pred,f0=f0_pred)
                self.logger.add_audio(f'wav_{batch_idx}', wav_pred, self.global_step, hparams['audio_sample_rate'])
                
                mel_gt=sample['mels'].cpu()
                mel_gt_mask = np.abs(mel_gt).sum(-1) > 0
                mel_gt = mel_gt[mel_gt_mask]
                mel_gt = np.clip(mel_gt, hparams['mel_vmin'], hparams['mel_vmax'])
                f0_gt=denorm_f0(sample['f0'], sample['uv'], hparams).cpu().numpy()
                f0_gt = f0_gt[mel_gt_mask]
                
                wav_gt = self.vocoder.spec2wav(mel_gt,f0=f0_gt)
                self.logger.add_audio(f'wav_gt_{batch_idx}', wav_gt, self.global_step,
                      hparams['audio_sample_rate'])
                self.logger.add_figure(
                    f'f0_{batch_idx}',
                    f0_to_figure(f0_gt, None, f0_pred),
                    self.global_step)
        return outputs

    ############
    # infer
    ############
    def test_start(self):
        self.saving_result_pool = Pool(min(int(os.getenv('N_PROC', os.cpu_count())), 16))
        self.saving_results_futures = []
        self.results_id = 0
        self.result_f0s = []
        self.gen_dir = os.path.join(
            hparams['work_dir'],
            f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
        self.result_f0s_path = os.path.join(
            hparams['work_dir'], f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}', "result_f0s.npy")
        self.vocoder: BaseVocoder = get_vocoder_cls(hparams)()

    def test_step(self, sample, batch_idx):
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_id')
        if hparams['emo']:
            emo_embed = sample.get('emo_embed')
        else:
            emo_embed=None
        txt_tokens = sample['txt_tokens']
        ref_f0=sample['f0']
        mel2ph, uv, f0 = None, None, None
        ref_mels = sample['mels']
        if hparams['use_gt_dur']:
            mel2ph = sample['mel2ph']
        if hparams['use_gt_f0']:
            f0 = sample['f0']
            uv = sample['uv']
        notes, note_durs, note_types = sample["notes"], sample["note_durs"], sample["note_types"]
        global_steps = 200000
        
        run_model = lambda: self.model(
            txt_tokens, spk_embed=spk_embed, emo_embed=emo_embed, mel2ph=mel2ph, 
            f0=f0, uv=uv, ref_mels=ref_mels, ref_f0=ref_f0, global_steps=global_steps, infer=True, note=notes, note_dur=note_durs, note_type=note_types)
        outputs = run_model()
        sample['outputs'] = self.model.out2mel(outputs['mel_out'])
        sample['mel2ph_pred'] = outputs['mel2ph']
        sample['f0'] = denorm_f0(sample['f0'], sample['uv'], hparams)
        if hparams['pitch_type'] == 'ph':
            sample['f0'] = torch.gather(F.pad(sample['f0'], [1, 0]), 1, sample['mel2ph'])
        sample['f0_pred'] = outputs.get('f0_denorm')

        return self.after_infer(sample)

    def after_infer(self, predictions, sil_start_frame=0):

        predictions = utils.unpack_dict_to_list(predictions)
        assert len(predictions) == 1, 'Only support batch_size=1 in inference.'
        prediction = predictions[0]
        prediction = utils.tensors_to_np(prediction)
        item_name = prediction.get('item_name')
        text = prediction.get('text')
        ph_tokens = prediction.get('txt_tokens')

        mel_gt = prediction["mels"]
        mel_gt_mask = np.abs(mel_gt).sum(-1) > 0
        mel_gt = mel_gt[mel_gt_mask]
        mel_pred = prediction["outputs"]
        mel_pred_mask = np.abs(mel_pred).sum(-1) > 0
        mel_pred = mel_pred[mel_pred_mask]
        mel_gt = np.clip(mel_gt, hparams['mel_vmin'], hparams['mel_vmax'])
        mel_pred = np.clip(mel_pred, hparams['mel_vmin'], hparams['mel_vmax'])

        mel2ph_gt = prediction.get("mel2ph")
        mel2ph_gt = mel2ph_gt if mel2ph_gt is not None else None
        mel2ph_pred = prediction.get("mel2ph_pred")
        f0_gt = prediction.get("f0")
        f0_pred = prediction.get("f0_pred")
        self.result_f0s.append({"gt": f0_gt, "pred": f0_pred})

        str_phs = None
        if self.phone_encoder is not None and 'txt_tokens' in prediction:
            str_phs = self.phone_encoder.decode(prediction['txt_tokens'], strip_padding=True)

        if f0_pred is not None:
            f0_gt = f0_gt[mel_gt_mask]
            if len(f0_pred) > len(mel_pred_mask):
                f0_pred = f0_pred[:len(mel_pred_mask)]
            f0_pred = f0_pred[mel_pred_mask]

        wav_pred = self.vocoder.spec2wav(mel_pred, f0=f0_pred)
        wav_pred[:sil_start_frame * hparams['hop_size']] = 0
        gen_dir = self.gen_dir
        base_fn = f'[{self.results_id:06d}][{item_name}][%s]'

        base_fn = base_fn.replace(' ', '_')
        if not hparams['profile_infer']:
            os.makedirs(gen_dir, exist_ok=True)
            os.makedirs(f'{gen_dir}/wavs', exist_ok=True)
            os.makedirs(f'{gen_dir}/plot', exist_ok=True)
            if hparams.get('save_mel_npy', False):
                os.makedirs(f'{gen_dir}/npy', exist_ok=True)
            self.saving_results_futures.append(
                self.saving_result_pool.apply_async(self.save_result, args=[
                    wav_pred, mel_pred, base_fn % 'SVS', gen_dir, str_phs, mel2ph_pred]))

            if mel_gt is not None and hparams['save_gt']:
                wav_gt = self.vocoder.spec2wav(mel_gt, f0=f0_gt)
                self.saving_results_futures.append(
                    self.saving_result_pool.apply_async(self.save_result, args=[
                        wav_gt, mel_gt, base_fn % 'Ref', gen_dir, str_phs, mel2ph_gt]))
                if hparams['save_f0']:
                    import matplotlib.pyplot as plt
                    f0_pred_, _ = get_pitch(wav_pred, mel_pred, hparams)
                    f0_gt_, _ = get_pitch(wav_gt, mel_gt, hparams)
                    fig = plt.figure()
                    plt.plot(f0_pred_, label=r'$\hat{f_0}$')
                    plt.plot(f0_gt_, label=r'$f_0$')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(f'{gen_dir}/plot/[F0][{item_name}]{text}.png', format='png')
                    plt.close(fig)

        self.results_id += 1
        return {
            'item_name': item_name,
            'text': text,
            'ph_tokens': self.phone_encoder.decode(ph_tokens.tolist()),
            'wav_fn_pred': base_fn % 'SVS',
            'wav_fn_gt': base_fn % 'Ref',
        }

    @staticmethod
    def save_result(wav_out, mel, base_fn, gen_dir, str_phs=None, mel2ph=None):
        audio.save_wav(wav_out, f'{gen_dir}/wavs/{base_fn}.wav', hparams['audio_sample_rate'],
                       norm=hparams['out_wav_norm'])
        fig = plt.figure(figsize=(14, 10))
        spec_vmin = hparams['mel_vmin']
        spec_vmax = hparams['mel_vmax']
        heatmap = plt.pcolor(mel.T, vmin=spec_vmin, vmax=spec_vmax)
        fig.colorbar(heatmap)
        f0, _ = get_pitch(wav_out, mel, hparams)
        f0 = f0 / 10 * (f0 > 0)
        plt.plot(f0, c='red', linewidth=3, alpha=0.6)
        if mel2ph is not None and str_phs is not None:
            decoded_txt = str_phs.split(" ")
            dur = mel2ph_to_dur(torch.LongTensor(mel2ph)[None, :], len(decoded_txt))[0].numpy()
            dur = [0] + list(np.cumsum(dur))
            for i in range(len(dur) - 1):
                shift = (i % 20) + 1
                plt.text(dur[i], shift, decoded_txt[i])
                plt.hlines(shift, dur[i], dur[i + 1], colors='b' if decoded_txt[i] != '|' else 'black')
                plt.vlines(dur[i], 0, 5, colors='b' if decoded_txt[i] != '|' else 'black',
                           alpha=1, linewidth=1)
        plt.tight_layout()
        plt.savefig(f'{gen_dir}/plot/{base_fn}.png', format='png')
        plt.close(fig)

    def test_end(self, outputs):
        np.save(self.result_f0s_path, self.result_f0s)

        return {}
