# score singer task
from utils.audio.io import save_wav
import matplotlib.pyplot as plt
from tqdm import tqdm
from singing.svs.module.score_singer import FsWordSinger, F0GenSinger, FlowPostnet, DiffPostnet
from singing.svs.module.diff.shallow_diffusion_tts import GaussianDiffusion
from singing.svs.module.diff.net import DiffNet
from singing.svs.module.diff.candidate_decoder import FFT
from multiprocessing.pool import Pool
import os
import numpy as np
import torch.nn.functional as F
import torch
import utils
from utils.commons.hparams import hparams
from utils.plot.plot import spec_to_figure
from utils.audio.pitch_utils import denorm_f0
from singing.svs.base_gen_task import AuxDecoderMIDITask
from tasks.tts.vocoder_infer.base_vocoder import BaseVocoder, get_vocoder_cls

def mel2ph_to_dur(mel2ph, T_txt, max_dur=None):
    B, _ = mel2ph.shape
    dur = mel2ph.new_zeros(B, T_txt + 1).scatter_add(1, mel2ph, torch.ones_like(mel2ph))
    dur = dur[:, 1:]
    if max_dur is not None:
        dur = dur.clamp(max=max_dur)
    return dur

def dur_to_figure(dur_gt, dur_pred, txt, mels=None, vmin=-5.5, vmax=1):
    dur_gt = dur_gt.cpu().numpy()
    dur_pred = dur_pred.cpu().numpy()
    dur_gt = np.cumsum(dur_gt).astype(int)
    dur_pred = np.cumsum(dur_pred).astype(int)
    fig = plt.figure(figsize=(12, 6))
    for i in range(len(dur_gt)):
        shift = (i % 8) + 1
        plt.text(dur_gt[i], shift * 4, txt[i])
        plt.text(dur_pred[i], 40 + shift * 4, txt[i])
        plt.vlines(dur_gt[i], 0, 40, colors='b')  # blue is gt
        plt.vlines(dur_pred[i], 40, 80, colors='r')  # red is pred
    plt.xlim(0, max(dur_gt[-1], dur_pred[-1]))
    if mels is not None:
        mels = mels.cpu().numpy()
        plt.pcolor(mels.T, vmin=vmin, vmax=vmax)
    return fig

def f0_to_figure(f0_gt, f0_cwt=None, f0_pred=None):
    fig = plt.figure(figsize=(12, 8))
    f0_gt = f0_gt.cpu().numpy()
    plt.plot(f0_gt, color='r', label='gt')
    if f0_cwt is not None:
        f0_cwt = f0_cwt.cpu().numpy()
        plt.plot(f0_cwt, color='b', label='cwt')
    if f0_pred is not None:
        f0_pred = f0_pred.cpu().numpy()
        plt.plot(f0_pred, color='green', label='pred')
    plt.legend()
    return fig

class ScoreSingerTask(AuxDecoderMIDITask):
    def __init__(self):
        super(ScoreSingerTask, self).__init__()
        torch.manual_seed(hparams["seed"])

    def build_model(self):
        self.model = FsWordSinger(self.phone_encoder)
        utils.print_arch(self.model)

    def run_model(self, model, sample, return_output=False, infer=False):
        print('------------')
        print(self.dataset_cls)
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        ph2word = sample["ph2words"]
        word_len = sample["word_lengths"].max()
        mel2word = sample["mel2word"]
        f0 = sample["f0"]
        uv = sample["uv"]
        target = sample['mels']  # [B, T_s, 80]
        mel2ph = sample['mel2ph']
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        note_tokens = sample["note"]
        note_durs = sample["note_durs"]
        note_types = sample["type"]
        note2words = sample["note2words"]
        energy = sample["mb_energy"] if hparams["energy_type"] == "multiband" else sample["energy"]
        # 改为由infer参数来决定是否infer
        output = model(txt_tokens, ph2word=ph2word, word_len=word_len,
                       mel2word=mel2word, mel2ph=mel2ph, spk_embed=spk_embed, infer_spk_embed=spk_embed,
                       f0=f0 if not infer else None, uv=uv, tgt_mels=target, infer=infer,
                       note_tokens=note_tokens, note_durs=note_durs, note_types=note_types, note2words=note2words, mel2notes=None, energy=energy, note_attns=None)

        losses = {}
        self.add_mel_loss(output['mel_out'], target, losses)
        if hparams["two_stage"]:
            self.add_pitch_loss(output, sample, losses)
        self.word_dur_loss(output["dur"], mel2word, sample["word_lengths"], txt_tokens, losses)
        if hparams["input_type"] == "phdur":
            self.ph_dur_loss(output["dur"], mel2ph, txt_tokens, losses)
        if not return_output:
            return losses
        else:
            return losses, output

    def add_f0_loss(self, p_pred, f0, uv, losses, nonpadding, postfix=''):
        assert p_pred[..., 0].shape == f0.shape
        if hparams['use_uv'] and hparams['pitch_type'] == 'frame':
            assert p_pred[..., 1].shape == uv.shape, (p_pred.shape, uv.shape)
            losses[f'uv{postfix}'] = (F.binary_cross_entropy_with_logits(
                p_pred[:, :, 1], uv, reduction='none') * nonpadding).sum() \
                                     / nonpadding.sum() * hparams['lambda_uv']
            nonpadding = nonpadding * (uv == 0).float()
        f0_pred = p_pred[:, :, 0]
        pitch_loss_fn = F.l1_loss if hparams['pitch_loss'] == 'l1' else F.mse_loss
        losses[f'f0{postfix}'] = (pitch_loss_fn(f0_pred, f0, reduction='none') * nonpadding).sum() \
                                 / nonpadding.sum() * hparams['lambda_f0']

    def word_dur_loss(self, dur_pred, mel2word, word_len, txt_tokens, losses=None):
        T = word_len.max()
        dur_gt = mel2ph_to_dur(mel2word, T).float()
        nonpadding = (torch.arange(T).to(dur_pred.device)[None, :] < word_len[:, None]).float()
        dur_pred = dur_pred * nonpadding
        dur_gt = dur_gt * nonpadding
        if hparams['dur_scale'] == 'log':
            dur_gt = (dur_gt + 1).log()
        wdur = F.l1_loss(dur_pred, dur_gt, reduction='none')
        wdur = (wdur * nonpadding).sum() / nonpadding.sum()
        losses['wdur'] = wdur
        if hparams['lambda_sent_dur'] > 0:
            assert hparams['dur_scale'] == 'linear'
            sdur = F.l1_loss(dur_pred.sum(-1), dur_gt.sum(-1))
            losses['sdur'] = sdur * hparams['lambda_sent_dur']

    def ph_dur_loss(self, dur_pred, mel2ph, txt_tokens, losses=None):
        """

        :param dur_pred: [B, T], float, log scale
        :param mel2ph: [B, T]
        :param txt_tokens: [B, T]
        :param losses:
        :return:
        """
        B, T = txt_tokens.shape
        nonpadding = (txt_tokens != 0).float()
        dur_gt = mel2ph_to_dur(mel2ph, T).float() * nonpadding
        losses['pdur'] = F.l1_loss(dur_pred, (dur_gt + 1).log(), reduction='none')
        losses['pdur'] = (losses['pdur'] * nonpadding).sum() / nonpadding.sum()
        losses['pdur'] = losses['pdur'] * hparams['lambda_ph_dur']

    def plot_dur(self, batch_idx, sample, model_out):
        T = sample['word_lengths'].max()
        dur_gt = mel2ph_to_dur(sample['mel2word'], T)[0]
        dur_pred = model_out['dur'][0]
        if hparams['dur_scale'] == 'log':
            dur_pred = dur_pred.exp() - 1
        dur_pred = torch.clamp(torch.round(dur_pred), min=0).long()
        txt = sample['words'][0]
        self.logger.add_figure(
            f'dur_{batch_idx}', dur_to_figure(dur_gt, dur_pred, txt, sample['mels'][0]),
            self.global_step)

    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(
            self.model, sample, return_output=True, infer=True)
        outputs['nsamples'] = sample['nsamples']
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs = utils.tensors_to_scalars(outputs)
        if self.vocoder is None:
            self.vocoder = get_vocoder_cls(hparams)()
        if batch_idx < hparams['num_valid_plots']:
            self.plot_mel(batch_idx, sample['mels'], model_out['mel_out'])
            if hparams["dur_model"] == "flow":
                pass
            else:
                pass
                # self.plot_dur(batch_idx, sample, model_out)
            gt_f0 = denorm_f0(sample['f0'], sample["uv"], hparams)
            pred_f0 = model_out.get('f0_denorm_pred')
            # pred_f0 = gt_f0
            self.logger.add_figure(
                f'f0_{batch_idx}',
                f0_to_figure(gt_f0[0], None, pred_f0[0]),
                self.global_step)
            self.plot_wav(batch_idx, sample['mels'], model_out['mel_out'], is_mel=True, gt_f0=gt_f0, f0=pred_f0)
            if 'attn' in model_out:
                self.logger.add_figure(
                    f'attn_{batch_idx}', spec_to_figure(model_out['attn'][0]), self.global_step)
            if 'ls_attn' in model_out:
                self.logger.add_figure(
                    f'lsattn_{batch_idx}', spec_to_figure(model_out['ls_attn'][0]), self.global_step)
            if hparams["use_energy_embed"]:
                gt_e = sample["energy"]
                pred_e = (model_out["mel_out"].exp() ** 2).sum(-1).sqrt()
                self.logger.add_figure(
                    f'e_{batch_idx}', f0_to_figure(gt_e[0], None, pred_e[0]), self.global_step)
        return outputs

    def test_start(self):
        self.f0_dicts = []
        self.saving_result_pool = Pool(min(int(os.getenv('N_PROC', os.cpu_count())), 16))
        self.saving_results_futures = []
        self.results_id = 0
        self.gen_dir = os.path.join(
            hparams['work_dir'],
            f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
        self.vocoder: BaseVocoder = get_vocoder_cls(hparams)()

    def test_step(self, sample, batch_idx):
        mel2word, uv, f0 = None, None, None
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        ph2word = sample["ph2words"]
        word_len = sample["word_lengths"].max()
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        note_tokens = sample["note"]
        note_durs = sample["note_durs"]
        note_types = sample["type"]
        note2words = sample["note2words"]
        ref_mels = None
        if hparams['use_gt_dur']:
            mel2word = sample['mel2word']
        outputs = self.model(txt_tokens, ph2word=ph2word, word_len=word_len,
                   mel2word=mel2word, mel2ph=None
                   , spk_embed=spk_embed, infer_spk_embed=spk_embed,
                   f0=None, uv=None, infer=True,
                   note_tokens=note_tokens, note_durs=note_durs, note_types=note_types,note2words=note2words, mel2notes=None,min_note=sample["min_note"])
        sample['outputs'] = outputs['mel_out']
        # sample['mel2word_pred'] = outputs['mel2word']
        if hparams.get('pe_enable') is not None and hparams['pe_enable']:
            sample['f0'] = self.pe(sample['mels'])['f0_denorm_pred']  # pe predict from GT mel
            sample['f0_pred'] = self.pe(sample['outputs'])['f0_denorm_pred']  # pe predict from Pred mel
        else:
            sample['f0'] = denorm_f0(sample['f0'], sample['uv'], hparams)
            sample['f0_pred'] = outputs.get('f0_denorm_pred')
        sample["e"] = sample["energy"]
        sample["pred_e"] = (outputs["mel_out"].exp() ** 2).sum(-1).sqrt()
        return self.after_infer(sample)

    def after_infer(self, predictions):
        if self.saving_result_pool is None and not hparams['profile_infer']:
            self.saving_result_pool = Pool(min(int(os.getenv('N_PROC', os.cpu_count())), 16))
            self.saving_results_futures = []
        predictions = utils.unpack_dict_to_list(predictions)
        t = tqdm(predictions)
        for num_predictions, prediction in enumerate(t):
            for k, v in prediction.items():
                if type(v) is torch.Tensor:
                    prediction[k] = v.cpu().numpy()

            item_name = prediction.get('item_name')

            # remove paddings
            mel_gt = prediction["mels"]
            mel_gt_mask = np.abs(mel_gt).sum(-1) > 0
            mel_gt = mel_gt[mel_gt_mask]
            mel2word_gt = prediction.get("mel2word")
            mel2word_gt = mel2word_gt[mel_gt_mask] if mel2word_gt is not None else None
            mel_pred = prediction["outputs"]
            mel_pred_mask = np.abs(mel_pred).sum(-1) > 0
            mel_pred = mel_pred[mel_pred_mask]
            mel_gt = np.clip(mel_gt, hparams['mel_vmin'], hparams['mel_vmax'])
            mel_pred = np.clip(mel_pred, hparams['mel_vmin'], hparams['mel_vmax'])

            mel2word_pred = prediction.get("mel2word_pred")
            if mel2word_pred is not None:
                if len(mel2word_pred) > len(mel_pred_mask):
                    mel2word_pred = mel2word_pred[:len(mel_pred_mask)]
                mel2word_pred = mel2word_pred[mel_pred_mask]

            f0_gt = prediction.get("f0")
            f0_pred = prediction.get("f0_pred")
            if f0_pred is not None:
                f0_gt = f0_gt[mel_gt_mask]
                if len(f0_pred) > len(mel_pred_mask):
                    f0_pred = f0_pred[:len(mel_pred_mask)]
                f0_pred = f0_pred[mel_pred_mask]
            f0_dict = {
                "item_name": item_name,
                "ref": f0_gt,
                "syn": f0_pred
            }
            self.f0_dicts.append(f0_dict)
            str_phs = prediction["words"]
            gen_dir = os.path.join(hparams['work_dir'],
                                   f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
            wav_pred = self.vocoder.spec2wav(mel_pred, f0=f0_pred)

            os.makedirs(gen_dir, exist_ok=True)
            os.makedirs(f'{gen_dir}/wavs', exist_ok=True)
            os.makedirs(f'{gen_dir}/plot', exist_ok=True)
            self.saving_results_futures.append(
                self.saving_result_pool.apply_async(self.save_result, args=[
                    wav_pred, mel_pred, 'P', item_name, None, gen_dir, str_phs, mel2word_pred, f0_gt, f0_pred, prediction.get("note"),
                prediction.get("note_durs"), prediction.get("type")]))

            if mel_gt is not None and hparams['save_gt']:
                wav_gt = self.vocoder.spec2wav(mel_gt, f0=f0_gt)
                self.saving_results_futures.append(
                    self.saving_result_pool.apply_async(self.save_result, args=[
                        wav_gt, mel_gt, 'G', item_name, None, gen_dir, None, mel2word_gt, f0_gt, f0_pred, prediction.get("note"),
                prediction.get("note_durs"), prediction.get("type")]))
            t.set_description(
                f"Pred_shape: {mel_pred.shape}, gt_shape: {mel_gt.shape}")

        return {}

    def test_end(self, outputs):
        self.saving_result_pool.close()
        [f.get() for f in tqdm(self.saving_results_futures)]
        self.saving_result_pool.join()
        np.save(os.path.join(self.gen_dir, "f0_dicts"), self.f0_dicts)
        return {}

    @staticmethod
    def save_result(wav_out, mel, prefix, item_name, text, gen_dir, str_phs=None, mel2ph=None, gt_f0=None, pred_f0=None,
                    pitch_midi=None,
                    midi_dur=None, is_slur=None):
        item_name = item_name.replace('/', '-')
        base_fn = f'[{item_name}][{prefix}]'

        if text is not None:
            base_fn += text
        # base_fn += ('-' + hparams['exp_name'])

        save_wav(wav_out, f'{gen_dir}/wavs/{base_fn}.wav', hparams['audio_sample_rate'],
                       norm=hparams['out_wav_norm'])
        fig = plt.figure(figsize=(14, 10))
        spec_vmin = hparams['mel_vmin']
        spec_vmax = hparams['mel_vmax']
        heatmap = plt.pcolor(mel.T, vmin=spec_vmin, vmax=spec_vmax)
        fig.colorbar(heatmap)
        # if hparams.get('pe_enable') is not None and hparams['pe_enable']:
        gt_f0 = (gt_f0 - 100) / (800 - 100) * 80 * (gt_f0 > 0)
        pred_f0 = (pred_f0 - 100) / (800 - 100) * 80 * (pred_f0 > 0)
        if prefix == "P":
            f0 = pred_f0
        else:
            f0 = gt_f0
        plt.plot(f0, c='red', linewidth=2, alpha=0.6)
            # plt.plot(gt_f0, c='red', linewidth=1, alpha=0.6)
        # else:
        #     f0, _ = get_pitch(wav_out, mel, hparams)
        #     f0 = (f0 - 100) / (800 - 100) * 80 * (f0 > 0)
        #     plt.plot(f0, c='white', linewidth=1, alpha=0.6)
        if mel2ph is not None and str_phs is not None:
            decoded_txt = str_phs
            dur = mel2ph_to_dur(torch.LongTensor(mel2ph)[None, :], len(decoded_txt))[0].numpy()
            dur = [0] + list(np.cumsum(dur))
            for i in range(len(dur) - 1):
                # postfix = f"_{pitch_midi[i]}_{is_slur[i]}"
                shift = (i % 20) + 1
                plt.text(dur[i], shift, decoded_txt[i])
                plt.hlines(shift, dur[i], dur[i + 1], colors='b' if decoded_txt[i] != '|' else 'black')
                plt.vlines(dur[i], 0, 5, colors='b' if decoded_txt[i] != '|' else 'black',
                           alpha=1, linewidth=1)
        plt.tight_layout()
        plt.savefig(f'{gen_dir}/plot/{base_fn}.png', format='png', dpi=1000)
        plt.close(fig)

class F0GenSingerTask(ScoreSingerTask):
    def __init__(self):
        super(F0GenSingerTask, self).__init__()

    def build_model(self):
        self.model = F0GenSinger(self.phone_encoder)
        # for k, v in self.model.named_parameters():
        #     if "uv_predictor" not in k:
        #         v.requires_grad = False
        utils.print_arch(self.model)

    def run_model(self, model, sample, return_output=False, infer=False):
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        ph2word = sample["ph2words"]
        word_len = sample["word_lengths"].max()
        mel2word = sample["mel2word"]
        f0 = sample["f0"]
        uv = sample["uv"]
        target = sample['mels']  # [B, T_s, 80]
        mel2ph = sample['mel2ph']
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        note_tokens = sample["note"]
        note_durs = sample["note_durs"]
        note_types = sample["type"]
        note2words = sample["note2words"]
        energy = sample["mb_energy"] if hparams["energy_type"] == "multiband" else sample["energy"]
        # 改为由infer参数来决定是否infer
        output = model(txt_tokens, ph2word=ph2word, word_len=word_len,
                       mel2word=mel2word, mel2ph=mel2ph, spk_embed=spk_embed, infer_spk_embed=spk_embed,
                       f0=f0 if not infer else None, uv=uv if not infer else None, tgt_mels=target, infer=infer,
                       note_tokens=note_tokens, note_durs=note_durs, note_types=note_types, note2words=note2words, mel2notes=None, energy=energy, note_attns=None)

        losses = {}
        if hparams["f0_gen"] == "diff":
            if "fdiff" in output:
                losses["fdiff"] = output["fdiff"]
            nonpadding = (mel2word != 0).float()
            losses[f'uv'] = (F.binary_cross_entropy_with_logits(
            output["uv_pred"][:, :, 0], uv, reduction='none') * nonpadding).sum() \
                                    / nonpadding.sum() * hparams['lambda_uv']
        elif hparams["f0_gen"] == "mdiff":
            if "mdiff" in output:
                losses["mdiff"] = output["mdiff"]
            nonpadding = (mel2word != 0).float()
            f0_pred = output["f0_pred"][:, :, 0]
            uv = sample['uv']
            pitch_loss_fn = F.l1_loss if hparams['pitch_loss'] == 'l1' else F.mse_loss
            losses[f'f0'] = (pitch_loss_fn(f0_pred, f0, reduction='none') * nonpadding).sum() \
                                        / nonpadding.sum() * hparams['lambda_f0']
        elif hparams["f0_gen"] == "gmdiff":
            if 'gdiff' in output:
                losses["gdiff"] = output["gdiff"]
                losses["mdiff"] = output["mdiff"]
                # losses["nll"] = output["nll"] * 0.1
        self.add_mel_loss(output['mel_out'], target, losses)
        # losses["sl"] = output["scale_loss"]
        self.word_dur_loss(output["dur"], mel2word, sample["word_lengths"], txt_tokens, losses)
        if not return_output:
            return losses
        else:
            return losses, output

DIFF_DECODERS = {
    'wavenet': lambda hp: DiffNet(hp['audio_num_mel_bins']),
    'fft': lambda hp: FFT(
        hp['hidden_size'], hp['dec_layers'], hp['dec_ffn_kernel_size'], hp['num_heads']),
}

# diffsinger part
class DiffSingerATask(ScoreSingerTask):
    def __init__(self):
        super(DiffSingerATask, self).__init__()

    def build_model(self):
        self.model = GaussianDiffusion(
            phone_encoder=self.phone_encoder,
            out_dims=80, denoise_fn=DIFF_DECODERS[hparams['diff_decoder_type']](hparams),
            timesteps=hparams['timesteps'],
            K_step=hparams['K_step'],
            loss_type=hparams['diff_loss_type'],
            spec_min=hparams['spec_min'], spec_max=hparams['spec_max'],
        )
        if hparams['fs2_ckpt'] != '':
            utils.load_ckpt(self.model.fs2, hparams['fs2_ckpt'], 'model', strict=True)
            # self.model.fs2.decoder = None
            for k, v in self.model.fs2.named_parameters():
                v.requires_grad = False
        utils.print_arch(self.model)

    def run_model(self, model, sample, return_output=False, infer=False):
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        ph2word = sample["ph2words"]
        word_len = sample["word_lengths"].max()
        mel2word = sample["mel2word"]
        f0 = sample["f0"]
        uv = sample["uv"]
        target = sample['mels']  # [B, T_s, 80]
        mel2ph = sample['mel2ph']
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        note_tokens = sample["note"]
        note_durs = sample["note_durs"]
        note_types = sample["type"]
        note2words = sample["note2words"]
        energy = sample["mb_energy"] if hparams["energy_type"] == "multiband" else sample["energy"]
        # 改为由infer参数来决定是否infer
        output = model(txt_tokens, ph2word=ph2word, word_len=word_len,
                       mel2word=mel2word, mel2ph=mel2ph, spk_embed=spk_embed, infer_spk_embed=spk_embed,
                       f0=f0 if not infer else None, uv=uv if not infer else None
                       , tgt_mels=target, infer=infer,
                       note_tokens=note_tokens, note_durs=note_durs, note_types=note_types, note2words=note2words, mel2notes=None, energy=energy, note_attns=None)

        losses = {}
        # if hparams["f0_gen"] == "diff":
        #     if "fdiff" in output:
        #         losses["fdiff"] = output["fdiff"]
        #     nonpadding = (mel2ph != 0).float()
        #     losses[f'uv'] = (F.binary_cross_entropy_with_logits(
        #     output["uv_pred"][:, :, 0], uv, reduction='none') * nonpadding).sum() \
        #                             / nonpadding.sum() * hparams['lambda_uv']
        # elif hparams["f0_gen"] == "mdiff":
        #     if "fdiff" in output:
        #         losses["fdiff"] = output["fdiff"]
        # elif hparams["f0_gen"] == "gmdiff":
        #     if 'gdiff' in output:
        #         losses["gdiff"] = output["gdiff"]
        #         losses["mdiff"] = output["mdiff"]
        # self.add_pitch_loss(output, sample, losses)
        if 'diff_loss' in output:
            losses['diff'] = output['diff_loss']
        # self.word_dur_loss(output["dur"], mel2word, sample["word_lengths"], txt_tokens, losses)
        if not return_output:
            return losses
        else:
            return losses, output

    def build_optimizer(self, model):
        self.optimizer = optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            weight_decay=hparams['weight_decay'])
        return optimizer
    
    def build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, hparams['decay_steps'], gamma=0.5)
        # return None

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx):
        if optimizer is None:
            return
        optimizer.step()
        optimizer.zero_grad()
        if self.scheduler is not None:
            self.scheduler.step(self.global_step // hparams['accumulate_grad_batches'])

class DiffSingerBTask(DiffSingerATask):
    def run_model(self, model, sample, return_output=False, infer=False):
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        ph2word = sample["ph2words"]
        word_len = sample["word_lengths"].max()
        mel2word = sample["mel2word"]
        f0 = sample["f0"]
        uv = sample["uv"]
        target = sample['mels']  # [B, T_s, 80]
        mel2ph = sample['mel2ph']
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        note_tokens = sample["note"]
        note_durs = sample["note_durs"]
        note_types = sample["type"]
        note2words = sample["note2words"]
        energy = sample["mb_energy"] if hparams["energy_type"] == "multiband" else sample["energy"]
        # 改为由infer参数来决定是否infer
        output = model(txt_tokens, ph2word=ph2word, word_len=word_len,
                       mel2word=mel2word, mel2ph=mel2ph, spk_embed=spk_embed, infer_spk_embed=spk_embed,
                       f0=f0, uv=uv, tgt_mels=target, infer=infer,
                       note_tokens=note_tokens, note_durs=note_durs, note_types=note_types, note2words=note2words, mel2notes=None, energy=energy, note_attns=None)

        losses = {}
        if 'diff_loss' in output:
            losses['diff'] = output['diff_loss']
        # self.add_mel_loss(output['mel_out'], target, losses)
        self.word_dur_loss(output["dur"], mel2word, sample["word_lengths"], txt_tokens, losses)
        if not return_output:
            return losses
        else:
            return losses, output

    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs['losses'] = {}
        outputs['losses'], model_out = self.run_model(
            self.model, sample, return_output=True, infer=True
            )
        outputs['nsamples'] = sample['nsamples']
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs = utils.tensors_to_scalars(outputs)
        if self.vocoder is None:
            self.vocoder = get_vocoder_cls(hparams)()
        if batch_idx < hparams['num_valid_plots']:
            self.plot_mel(batch_idx, sample['mels'], model_out['mel_out'])
            if hparams["dur_model"] == "flow":
                pass
            else:
                self.plot_dur(batch_idx, sample, model_out)
            gt_f0 = denorm_f0(sample['f0'], sample["uv"], hparams)
            pred_f0 = self.pe(model_out['mel_out'])['f0_denorm_pred']  # pe predict from Pred mel
            self.logger.add_figure(
                f'f0_{batch_idx}',
                f0_to_figure(gt_f0[0], None, pred_f0[0]),
                self.global_step)
            self.plot_wav(batch_idx, sample['mels'], model_out['mel_out'], is_mel=True, gt_f0=gt_f0, f0=pred_f0)
            if 'attn' in model_out:
                self.logger.add_figure(
                    f'attn_{batch_idx}', spec_to_figure(model_out['attn'][0]), self.global_step)
            self.logger.add_figure(
                f'lsattn_{batch_idx}', spec_to_figure(model_out['ls_attn'][0]), self.global_step)
        return outputs
# postnet 方案，着重spk 建模
# diff-based postnet for mel
# flow postnet for mel
class FlowPostnetTask(ScoreSingerTask):
    def __init__(self):
        super(FlowPostnetTask, self).__init__()

    def build_model(self):
        self.build_pretrain_model()
        self.model = FlowPostnet()
        utils.print_arch(self.model)
        return self.model

    def build_pretrain_model(self):
        self.pretrain = F0GenSinger(self.phone_encoder)
        utils.load_ckpt(self.pretrain, hparams['fs2_ckpt'], 'model', strict=True) 
        for k, v in self.pretrain.named_parameters():
            v.requires_grad = False    

    def run_model(self, model, sample, return_output=False, infer=False):
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        ph2word = sample["ph2words"]
        word_len = sample["word_lengths"].max()
        mel2word = sample["mel2word"]
        f0 = sample["f0"]
        uv = sample["uv"]
        target = sample['mels']  # [B, T_s, 80]
        mel2ph = sample['mel2ph']
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        note_tokens = sample["note"]
        note_durs = sample["note_durs"]
        note_types = sample["type"]
        note2words = sample["note2words"]
        energy = sample["mb_energy"] if hparams["energy_type"] == "multiband" else sample["energy"]
        # 改为由infer参数来决定是否infer
        output = self.pretrain(txt_tokens, ph2word=ph2word, word_len=word_len,
                       mel2word=mel2word, mel2ph=mel2ph, spk_embed=spk_embed, infer_spk_embed=spk_embed,
                       f0=f0 if not infer else None, uv=uv if not infer else None
                       , tgt_mels=target, infer=infer,
                       note_tokens=note_tokens, note_durs=note_durs, note_types=note_types, note2words=note2words, mel2notes=sample["mel2notes"], energy=energy, note_attns=sample["note_attns"])

        coarse_mel = output["mel_out"]
        output["coarse_mel"] = coarse_mel
        self.model(target, infer, output, spk_embed)
        losses = {}
        losses["postflow"] = output["postflow"]
        if not return_output:
            return losses
        else:
            return losses, output

    # def test_start(self):
    #     super().test_start()
    #     self.pretrain = F0GenSinger(self.phone_encoder).to(self.model.device)
    #     utils.load_ckpt(self.pretrain, hparams['fs2_ckpt'], 'model', strict=True)

    def test_step(self, sample, batch_idx):
        mel2word, uv, f0 = None, None, None
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        ph2word = sample["ph2words"]
        word_len = sample["word_lengths"].max()
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        note_tokens = sample["note"]
        note_durs = sample["note_durs"]
        note_types = sample["type"]
        note2words = sample["note2words"]
        ref_mels = None
        if hparams['use_gt_dur']:
            mel2word = sample['mel2word']

        # hparams["f0_gen"]="diff"
        # hparams["f0_residual_layers"]=12
        # # # hparams["note_model"] = "attn"
        # hparams["fs2_ckpt"] = "/home/renyi/hjz/NeuralSeq/checkpoints/staff/f0diff_dur2"
        # self.pretrain = F0GenSinger(self.phone_encoder).to(txt_tokens.device)
        # utils.load_ckpt(self.pretrain, hparams['fs2_ckpt'], 'model', strict=True)
        run_model = lambda: self.pretrain(txt_tokens, ph2word=ph2word, word_len=word_len,
                   mel2word=mel2word, mel2ph=None
                   , spk_embed=spk_embed, infer_spk_embed=spk_embed,
                   f0=None, uv=None, infer=True,
                   note_tokens=note_tokens, note_durs=note_durs, note_types=note_types,note2words=note2words, mel2notes=None,min_note=None)
        outputs = run_model()
        self.model(ref_mels, True, outputs, spk_embed)
        sample['outputs'] = (outputs['mel_out'])
        # sample['mel2word_pred'] = outputs['mel2word']
        if hparams.get('pe_enable') is not None and hparams['pe_enable']:
            sample['f0'] = self.pe(sample['mels'])['f0_denorm_pred']  # pe predict from GT mel
            sample['f0_pred'] = self.pe(sample['outputs'])['f0_denorm_pred']  # pe predict from Pred mel
        else:
            sample['f0'] = denorm_f0(sample['f0'], sample['uv'], hparams)
            sample['f0_pred'] = outputs.get('f0_denorm_pred')
        sample["e"] = sample["energy"]
        sample["pred_e"] = (outputs["mel_out"].exp() ** 2).sum(-1).sqrt()
        return self.after_infer(sample)

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
        utils.print_arch(self.model)
    
    def run_model(self, model, sample, return_output=False, infer=False):
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        ph2word = sample["ph2words"]
        word_len = sample["word_lengths"].max()
        mel2word = sample["mel2word"]
        f0 = sample["f0"]
        uv = sample["uv"]
        target = sample['mels']  # [B, T_s, 80]
        mel2ph = sample['mel2ph']
        spk_embed = sample.get('spk_embed') if not hparams['use_spk_id'] else sample.get('spk_ids')
        note_tokens = sample["note"]
        note_durs = sample["note_durs"]
        note_types = sample["type"]
        note2words = sample["note2words"]
        energy = sample["mb_energy"] if hparams["energy_type"] == "multiband" else sample["energy"]
        # 改为由infer参数来决定是否infer
        output = self.pretrain(txt_tokens, ph2word=ph2word, word_len=word_len,
                       mel2word=mel2word, mel2ph=mel2ph, spk_embed=spk_embed, infer_spk_embed=spk_embed,
                       f0=f0 if not infer else None, uv=uv, tgt_mels=target, infer=infer,
                       note_tokens=note_tokens, note_durs=note_durs, note_types=note_types, note2words=note2words, mel2notes=None, energy=energy, note_attns=None)

        coarse_mel = output["mel_out"]
        output["coarse_mel"] = coarse_mel
        self.model(target, infer, output, spk_embed)
        losses = {}
        losses["diff"] = output["diff"]
        if not return_output:
            return losses
        else:
            return losses, output
