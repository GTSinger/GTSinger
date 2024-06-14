from utils.audio.io import save_wav
import matplotlib.pyplot as plt
from tasks.tts.fs import FastSpeechTask
from tasks.tts.vocoder_infer.base_vocoder import get_vocoder_cls
import os
import torch.nn.functional as F
import torch
from utils.audio.pitch_utils import denorm_f0
from tqdm import tqdm
import numpy as np
from utils.commons.hparams import hparams
from utils.commons.tensor_utils import tensors_to_scalars
from utils.commons.multiprocess_utils import MultiprocessManager
from tasks.tts.vocoder_infer.base_vocoder import BaseVocoder
from utils.audio.align import mel2token_to_dur

def mel2ph_to_dur(mel2ph, T_txt, max_dur=None):
    B, _ = mel2ph.shape
    dur = mel2ph.new_zeros(B, T_txt + 1).scatter_add(1, mel2ph, torch.ones_like(mel2ph))
    dur = dur[:, 1:]
    if max_dur is not None:
        dur = dur.clamp(max=max_dur)
    return dur

def f0_to_figure(f0_gt, f0_cwt=None, f0_pred=None):
    fig = plt.figure(figsize=(12, 8))
    f0_gt = f0_gt.cpu().numpy()
    plt.plot(f0_gt, color='r', label='gt')
    if f0_cwt is not None:
        f0_cwt = f0_cwt.cpu().numpy()
        plt.plot(f0_cwt, color='b', label='ref')
    if f0_pred is not None:
        f0_pred = f0_pred.cpu().numpy()
        plt.plot(f0_pred, color='green', label='pred')
    plt.legend()
    return fig

# 这个task只是作为base_task
class AuxDecoderMIDITask(FastSpeechTask):
    def __init__(self):
        super().__init__()
        self.dataset_cls = None


    def build_tts_model(self):
        self.model = None


    def add_dur_loss(self, dur_pred, mel2ph, txt_tokens, wdb=None, losses=None):
        """
        :param dur_pred: [B, T], float, log scale
        :param mel2ph: [B, T]
        :param txt_tokens: [B, T]
        :param losses:
        :return:
        """
        B, T = txt_tokens.shape
        nonpadding = (txt_tokens != 0).float()
        # print(mel2ph,'------------------')
        dur_gt = mel2ph_to_dur(mel2ph, T).float() * nonpadding
        is_sil = torch.zeros_like(txt_tokens).bool()
        for p in self.sil_ph:
            is_sil = is_sil | (txt_tokens == self.token_encoder.encode(p)[0])
        is_sil = is_sil.float()  # [B, T_txt]

        # phone duration loss
        if hparams['dur_loss'] == 'mse':
            losses['pdur'] = F.mse_loss(dur_pred, (dur_gt + 1).log(), reduction='none')
            losses['pdur'] = (losses['pdur'] * nonpadding).sum() / nonpadding.sum()
            dur_pred = (dur_pred.exp() - 1).clamp(min=0)
        else:
            raise NotImplementedError

        # use linear scale for sent and word duration
        if hparams['lambda_word_dur'] > 0:
            idx = F.pad(wdb.cumsum(axis=1), (1, 0))[:, :-1]
            # word_dur_g = dur_gt.new_zeros([B, idx.max() + 1]).scatter_(1, idx, midi_dur)  # midi_dur can be implied by add gt-ph_dur
            word_dur_p = dur_pred.new_zeros([B, idx.max() + 1]).scatter_add(1, idx, dur_pred)
            word_dur_g = dur_gt.new_zeros([B, idx.max() + 1]).scatter_add(1, idx, dur_gt)
            wdur_loss = F.mse_loss((word_dur_p + 1).log(), (word_dur_g + 1).log(), reduction='none')
            word_nonpadding = (word_dur_g > 0).float()
            wdur_loss = (wdur_loss * word_nonpadding).sum() / word_nonpadding.sum()
            losses['wdur'] = wdur_loss * hparams['lambda_word_dur']
        if hparams['lambda_sent_dur'] > 0:
            sent_dur_p = dur_pred.sum(-1)
            sent_dur_g = dur_gt.sum(-1)
            sdur_loss = F.mse_loss((sent_dur_p + 1).log(), (sent_dur_g + 1).log(), reduction='mean')
            losses['sdur'] = sdur_loss.mean() * hparams['lambda_sent_dur']
    
    def validation_start(self):
        self.vocoder = get_vocoder_cls(hparams["vocoder"])()

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
                f0_to_figure(gt_f0[0], None, model_out["f0_denorm_pred"][0]),
                self.global_step)
        return outputs

    def test_start(self):
        self.saving_result_pool = MultiprocessManager(int(os.getenv('N_PROC', os.cpu_count())))
        self.result_f0s_path = os.path.join(
            hparams['work_dir'], f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}', "result_f0s.npy")
        self.result_f0s = []
        self.saving_results_futures = []
        self.gen_dir = os.path.join(
            hparams['work_dir'], f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
        self.vocoder: BaseVocoder = get_vocoder_cls(hparams['vocoder'])()
        os.makedirs(self.gen_dir, exist_ok=True)
        os.makedirs(f'{self.gen_dir}/wavs', exist_ok=True)
        os.makedirs(f'{self.gen_dir}/plot', exist_ok=True)
        if hparams.get('save_mel_npy', False):
            os.makedirs(f'{self.gen_dir}/mel_npy', exist_ok=True)

    def test_step(self, sample, batch_idx):
        mel2ph = sample['mel2ph']
        # if not hparams['use_gt_dur']:
        sample["mel2ph"] = None   
        _, outputs = self.run_model(sample, infer=True)
        sample["mel2ph"] = mel2ph
        f0 = denorm_f0(sample['f0'], sample['uv'])[0].cpu().numpy()
        f0_pred = outputs.get('f0_denorm_pred')[0].cpu().numpy()
        # f0_cond = outputs.get('midi_f0')[0].cpu().numpy()
        self.result_f0s.append({"gt": f0, "pred": f0_pred})
        item_name = sample['item_name'][0]
        tokens = sample['txt_tokens'][0].cpu().numpy()
        mel_gt = sample['mels'][0].cpu().numpy()
        mel_pred = outputs['mel_out'][0].cpu().numpy()
        str_phs = self.token_encoder.decode(tokens, strip_padding=True)
        mel2ph = sample["mel2ph"][0].cpu().numpy()
        mel2ph_pred = outputs.get("mel2ph")
        if mel2ph_pred is not None:
            mel2ph_pred = mel2ph_pred[0].cpu().numpy()
        base_fn = f'{item_name}[%s]'
        base_fn = base_fn.replace(' ', '_')
        gen_dir = self.gen_dir
        wav_pred = self.vocoder.spec2wav(mel_pred, f0=f0_pred)
        self.saving_result_pool.add_job(self.save_result, args=[
            wav_pred, mel_pred, base_fn % 'P', gen_dir, str_phs, mel2ph_pred, f0, f0_pred, None])
        if hparams['save_gt']:
            wav_gt = self.vocoder.spec2wav(mel_gt, f0=f0)
            self.saving_result_pool.add_job(self.save_result, args=[
                wav_gt, mel_gt, base_fn % 'G', gen_dir, str_phs, mel2ph, f0, f0_pred, None])
        print(f"Pred_shape: {mel_pred.shape}, gt_shape: {mel_gt.shape}")
        return {}

    @staticmethod
    def save_result(wav_out, mel, base_fn, gen_dir, str_phs=None, mel2ph=None, gt_f0=None, pred_f0=None, cond_f0=None):
        save_wav(wav_out, f'{gen_dir}/wavs/{base_fn}.wav', hparams['audio_sample_rate'],
                 norm=hparams['out_wav_norm'])
        fig = plt.figure(figsize=(14, 10))
        spec_vmin = hparams['mel_vmin']
        spec_vmax = hparams['mel_vmax']
        heatmap = plt.pcolor(mel.T, vmin=spec_vmin, vmax=spec_vmax)
        fig.colorbar(heatmap)
        if mel2ph is not None and str_phs is not None:
            decoded_txt = str_phs.split(" ")
            dur = mel2token_to_dur(torch.LongTensor(mel2ph)[None, :], len(decoded_txt))[0].numpy()
            dur = [0] + list(np.cumsum(dur))
            for i in range(len(dur) - 1):
                shift = (i % 20) + 1
                plt.text(dur[i], shift, decoded_txt[i])
                plt.hlines(shift, dur[i], dur[i + 1], colors='b' if decoded_txt[i] != '|' else 'black')
                plt.vlines(dur[i], 0, 5, colors='b' if decoded_txt[i] != '|' else 'black',
                            alpha=1, linewidth=1)
        plt.tight_layout()
        plt.savefig(f'{gen_dir}/plot/{base_fn}.png', format='png', dpi=1000)
        plt.close(fig)
        # f0 绘制
        if pred_f0 is not None:
            fig = plt.figure()
            plt.plot(pred_f0, label=r'$f0_p$')
            plt.plot(gt_f0, label=r'$f0_g$')
            if cond_f0 is not None:
                plt.plot(cond_f0, label=r'$f0_c$')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'{gen_dir}/plot/[F0][{base_fn}].png', format='png')
            plt.close(fig)

    def test_end(self, outputs):
        np.save(self.result_f0s_path, self.result_f0s)
        for _1, _2 in tqdm(self.saving_result_pool.get_results(), total=len(self.saving_result_pool)):
            pass
        return {}

    ##############
    # utils
    ##############
    @staticmethod
    def expand_f0_ph(f0, mel2ph):
        f0 = denorm_f0(f0, None, hparams)
        f0 = F.pad(f0, [1, 0])
        f0 = torch.gather(f0, 1, mel2ph)  # [B, T_mel]
        return f0