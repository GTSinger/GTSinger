import os
import sys
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.functional.classification import binary_auroc, binary_recall, binary_f1_score, binary_precision, binary_accuracy
import matplotlib.pyplot as plt
from tqdm import tqdm
import mir_eval
import pretty_midi
import glob
import warnings

from utils import seed_everything
from utils.commons.hparams import hparams
from tasks.tts.tts_utils import parse_dataset_configs
from utils.audio.io import save_wav
from utils.audio.align import mel2token_to_dur
from utils.audio.pitch_utils import denorm_f0, boundary2Interval, midi_to_hz, save_midi, \
    validate_pitch_and_itv, midi_melody_eval, melody_eval_pitch_and_itv
from utils.commons.tensor_utils import tensors_to_scalars
from utils.commons.ckpt_utils import load_ckpt
from utils.nn.model_utils import print_arch
from utils.commons.multiprocess_utils import MultiprocessManager
from utils.audio.pitch_utils import midi_onset_eval, midi_offset_eval, midi_pitch_eval
from utils.commons.multiprocess_utils import multiprocess_run_tqdm
from utils.commons.losses import sigmoid_focal_loss
# from utils.commons.gpu_mem_track import MemTracker

from tasks.tts.speech_base import SpeechBaseTask
from research.singtech.te_dataset import TEDataset
from research.singtech.modules.te import TechExtractor

# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# gpu_tracker = MemTracker()

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

class TETask(SpeechBaseTask):
    def __init__(self, *args, **kwargs):
        super(SpeechBaseTask, self).__init__(*args, **kwargs)
        self.dataset_cls = TEDataset
        self.vocoder = None
        self.saving_result_pool = None
        self.saving_results_futures = None
        self.max_tokens, self.max_sentences, \
            self.max_valid_tokens, self.max_valid_sentences = parse_dataset_configs()
        seed_everything(hparams['seed'])

        # UserWarning: No positive samples in targets, true positive value should be meaningless. Returning zero tensor in true positive score
        warnings.filterwarnings("ignore", category=UserWarning)

    def build_model(self):
        self.build_tts_model()
        if hparams['load_ckpt'] != '':
            load_ckpt(self.model, hparams['load_ckpt'])
        print_arch(self.model)
        return self.model

    def build_tts_model(self):
        model_name = hparams.get('model', None)
        self.model = TechExtractor(hparams)

    def _training_step(self, sample, batch_idx, _):
        loss_output, _ = self.run_model(sample)
        total_loss = sum([v for v in loss_output.values() if isinstance(v, torch.Tensor) and v.requires_grad])
        loss_output['batch_size'] = sample['nsamples']
        return total_loss, loss_output

    def run_model(self, sample, infer=False):
        # gpu_tracker.track()
        mel = sample['mels']
        pitch_coarse = sample['pitch_coarse']
        uv = sample['uv'].long()
        mel_nonpadding = sample['mel_nonpadding']
        ph_bd = sample['ph_bd']
        breathiness = sample.get('breathiness', None)
        energy = sample.get('energy', None)
        zcr = sample.get('zcr', None)
        variance = {'breathiness': breathiness, 'energy': energy, 'zcr': zcr}
        techs = sample['techs']

        output = self.model(mel=mel, ph_bd=ph_bd, pitch=pitch_coarse, uv=uv, variance=variance,
                            non_padding=mel_nonpadding, train=not infer)
        losses = {}
        if not infer:
            # try:
            self.add_tech_loss(output['tech_logits'], techs, losses)
            # except RuntimeError as err:
            #     _, exc_value, exc_tb = sys.exc_info()
            #     tb = traceback.extract_tb(exc_tb)[-1]
            #     print(f'skip {sample["item_name"][:4]}{"..." if len(sample["item_name"]) > 4 else ""}, '
            #           f'{err}: {exc_value} in {tb[0]}:{tb[1]} "{tb[2]}" in {tb[3]}')
                # losses['tech'] = 0.0
                # print('-' * 30)
                # note_bd_logits = output['note_bd_logits']
                # from research.rme.modules.me8 import regulate_boundary
                # note_bd_pred = regulate_boundary(note_bd_logits, 0.5, 17, word_bd, 8)
                # print("note_bd_pred.sum(-1)", note_bd_pred.sum(-1))
                # print("notes.shape", notes.shape)
                # print("note_bd.sum", note_bd.sum(-1))
        # gpu_tracker.track()
        return losses, output

    def add_tech_loss(self, tech_logits, techs, losses):
        bsz, T, num_techs = tech_logits.shape
        tech_losses = F.binary_cross_entropy_with_logits(tech_logits, techs.float(), reduction='none')  # [B, T, C]
        tech_losses = tech_losses.reshape(-1, num_techs).mean(0)    # [C]
        lambda_tech = hparams.get('lambda_tech', 1.0)
        lambdas_tech = hparams.get('lambdas_tech', '')
        if lambdas_tech != '' and '-' in lambdas_tech:
            lambda_tech = [float(i) for i in lambdas_tech.split('-')]
            assert len(lambda_tech) == hparams.get('tech_num', 6), f"{len(lambda_tech)} {hparams.get('tech_num', 6)}"
        else:
            lambda_tech = [lambda_tech for _ in range(hparams.get('tech_num', 6))]
        losses['mix_tech'] = tech_losses[0] * lambda_tech[0]
        losses['falsetto_tech'] = tech_losses[1] * lambda_tech[1]
        losses['breathy_tech'] = tech_losses[2] * lambda_tech[2]
        losses['pharyngeal_tech'] = tech_losses[3] * lambda_tech[3]
        losses['vibra_tech'] = tech_losses[4] * lambda_tech[4]
        losses['glissando_tech'] = tech_losses[5] * lambda_tech[5]
        if hparams.get('tech_focal_loss', None) not in ['none', None, 0]:
            gamma = float(hparams.get('tech_focal_loss', None))
            focal_loss = sigmoid_focal_loss(
                tech_logits, techs.float(), alpha=-1, gamma=gamma, reduction='mean')
            losses['tech_fc'] = focal_loss * hparams.get('lambda_tech_focal', 1.0)

    def validation_start(self):
        # self.vocoder = get_vocoder_cls(hparams["vocoder"])()
        self.vocoder = None
        # torch.cuda.empty_cache()

    def validation_step(self, sample, batch_idx):
        outputs = {}
        outputs['losses'] = {}
        with torch.no_grad():
            outputs['losses'], model_out = self.run_model(sample, infer=False)
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']
        if batch_idx < hparams['num_valid_stats']:
            mel = sample['mels']
            pitch_coarse = sample['pitch_coarse']
            uv = sample['uv'].long()
            mel_nonpadding = sample['mel_nonpadding']
            breathiness = sample.get('breathiness', None)
            energy = sample.get('energy', None)
            zcr = sample.get('zcr', None)
            variance = {'breathiness': breathiness, 'energy': energy, 'zcr': zcr}
            ph_bd = sample['ph_bd']
            techs = sample['techs']
            gt_f0 = denorm_f0(sample['f0'], sample["uv"])

            with torch.no_grad():
                # no note_bd
                output = self.model(mel=mel, ph_bd=ph_bd, pitch=pitch_coarse, uv=uv, variance=variance,
                                    non_padding=mel_nonpadding, train=True)
                tech_logits = output['tech_logits']
                tech_probs = torch.sigmoid(tech_logits)
                tech_pred = output['tech_pred']
                threshold = hparams.get('tech_threshold', 0.8)

                if torch.sum(techs) > 0:
                    outputs['losses']['tech_auroc'] = binary_auroc(tech_logits, techs, threshold)
                    outputs['losses']['tech_p'] = binary_precision(tech_logits, techs, threshold)
                    outputs['losses']['tech_r'] = binary_recall(tech_logits, techs, threshold)
                    outputs['losses']['tech_f'] = binary_f1_score(tech_logits, techs, threshold)
                outputs['losses']['tech_a'] = binary_accuracy(tech_logits, techs, threshold)

                tech_names = ['mix_tech', 'falsetto_tech', 'breathy_tech', 'pharyngeal_tech', 'vibra_tech', 'glissando_tech']
                for tech_idx, tech_name in enumerate(tech_names):
                    if torch.sum(techs[:, :, tech_idx]) > 0:
                        outputs['losses'][f'{tech_name}_auroc'] = binary_auroc(tech_logits[:, :, tech_idx], techs[:, :, tech_idx], threshold)
                        outputs['losses'][f'{tech_name}_p'] = binary_precision(tech_logits[:, :, tech_idx], techs[:, :, tech_idx], threshold)
                        outputs['losses'][f'{tech_name}_r'] = binary_recall(tech_logits[:, :, tech_idx], techs[:, :, tech_idx], threshold)
                        outputs['losses'][f'{tech_name}_f'] = binary_f1_score(tech_logits[:, :, tech_idx], techs[:, :, tech_idx], threshold)
                    outputs['losses'][f'{tech_name}_a'] = binary_accuracy(tech_logits[:, :, tech_idx], techs[:, :, tech_idx], threshold)
                    if batch_idx < hparams['num_valid_plots']:
                        self.logger.add_figure(
                            f'tech_{tech_name}_{batch_idx}',
                            f0_tech_to_figure(gt_f0[0].data.cpu().numpy(), ph_bd[0].data.cpu().numpy(),
                                              tech_pred[0, :, tech_idx].data.cpu().numpy(),
                                              techs[0, :, tech_idx].data.cpu().numpy(),
                                              tech_probs[0, :, tech_idx].data.cpu().numpy(),
                                              tech_name, fig_name=sample['item_name']),
                            self.global_step)

            self.save_valid_result(sample, batch_idx, model_out)
        outputs = tensors_to_scalars(outputs)
        return outputs

    def validation_end(self, outputs):
        # torch.cuda.empty_cache()
        return super(TETask, self).validation_end(outputs)

    def save_valid_result(self, sample, batch_idx, model_out):
        pass

    def test_start(self):
        self.saving_result_pool = MultiprocessManager(int(os.getenv('N_PROC', os.cpu_count())))
        self.saving_results_futures = []
        self.gen_dir = os.path.join(
            hparams['work_dir'], f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
        os.makedirs(self.gen_dir, exist_ok=True)
        os.makedirs(f'{self.gen_dir}/plot', exist_ok=True)

    def test_step(self, sample, batch_idx):
        _, outputs = self.run_model(sample, infer=True)
        ph_bd = sample['ph_bd'][0].cpu()
        tech_gt = sample['techs'][0].cpu()
        f0 = denorm_f0(sample['f0'], sample['uv'])[0].cpu()
        tech_logits = outputs['tech_logits'][0].cpu()
        tech_probs = torch.sigmoid(outputs['tech_logits'])[0].cpu()
        tech_pred = outputs['tech_pred'][0].cpu()
        threshold = hparams.get('tech_threshold', 0.8)

        item_name = sample['item_name'][0]
        gen_dir = self.gen_dir
        self.saving_result_pool.add_job(self.save_result, args=[
            item_name, gen_dir, f0, ph_bd, tech_gt, tech_probs, tech_pred, tech_logits, threshold])
        return {}

    @staticmethod
    def save_result(item_name, gen_dir, gt_f0=None, ph_bd=None, tech_gt=None, tech_probs=None, tech_pred=None,
                    tech_logits=None, threshold=0.8):
        if torch.sum(tech_gt) > 0:
            tech_auroc = binary_auroc(tech_logits, tech_gt, threshold)
            tech_p = binary_precision(tech_logits, tech_gt, threshold)
            tech_r = binary_recall(tech_logits, tech_gt, threshold)
            tech_f = binary_f1_score(tech_logits, tech_gt, threshold)
        else:
            tech_auroc = np.nan
            tech_p = np.nan
            tech_r = np.nan
            tech_f = np.nan
        tech_a = binary_accuracy(tech_logits, tech_gt, threshold)
        res = [tech_auroc, tech_p, tech_r, tech_f, tech_a]

        tech_names = ['mix_tech', 'falsetto_tech', 'breathy_tech', 'pharyngeal_tech', 'vibra_tech', 'glissando_tech']
        for tech_idx, tech_name in enumerate(tech_names):
            if torch.sum(tech_gt[:, tech_idx]) > 0:
                tech_auroc = binary_auroc(tech_logits[:, tech_idx], tech_gt[:, tech_idx], threshold)
                tech_p = binary_precision(tech_logits[:, tech_idx], tech_gt[:, tech_idx], threshold)
                tech_r = binary_recall(tech_logits[:, tech_idx], tech_gt[:, tech_idx], threshold)
                tech_f = binary_f1_score(tech_logits[:, tech_idx], tech_gt[:, tech_idx], threshold)
            else:
                tech_auroc = np.nan
                tech_p = np.nan
                tech_r = np.nan
                tech_f = np.nan
            tech_a = binary_accuracy(tech_logits[:, tech_idx], tech_gt[:, tech_idx], threshold)
            res = res + [tech_auroc, tech_p, tech_r, tech_f, tech_a]

            fig = f0_tech_to_figure(gt_f0.numpy(), ph_bd.numpy(), tech_pred[:, tech_idx].numpy(), tech_gt[:, tech_idx].numpy(),
                                    tech_probs[:, tech_idx].numpy(), tech_name, fig_name=item_name,
                                    save_path=f'{gen_dir}/plot/{item_name}[{tech_name}].png')
            plt.close(fig)

        res = [res, item_name]

        return res

    def test_end(self, outputs):
        res = []
        item_names = []
        for r_id, r in tqdm(self.saving_result_pool.get_results(), total=len(self.saving_result_pool)):
            res.append(r[0])
            item_names.append(r[1])

        res = np.array(res)     # [N, 5 + 5x6]
        results = []
        for i in range(res.shape[1]):
            _res = res[:, i]
            results.append(np.mean(_res[~np.isnan(_res)]))

        scores = {}
        tech_auroc, tech_p, tech_r, tech_f, tech_a = results[:5]
        tech_names = ['mix_tech', 'falsetto_tech', 'breathy_tech', 'pharyngeal_tech', 'vibra_tech', 'glissando_tech']
        for tech_idx, tech_name in enumerate(tech_names):
            scores[tech_name] = results[(tech_idx+1)*5: (tech_idx+2)*5]

        # print(f"overall |     auroc: {tech_auroc:.3f}")
        # print(f"overall | precision: {tech_p:.3f}")
        # print(f"overall |    recall: {tech_r:.3f}")
        # print(f"overall |        f1: {tech_f:.3f}")
        # print(f"overall |  accuracy: {tech_a:.3f}")
        # print('-' * 90)
        # print(f"overall |     auroc: {tech_auroc:.3f}" + f" || mix_tech     |     auroc: {scores['mix_tech'][0]:.3f}" +    f" || falsetto_tech |     auroc: {scores['falsetto_tech'][0]:.3f}" + f" || breathy_tech |     auroc: {scores['breathy_tech'][0]:.3f}")
        # print(f"overall | precision: {tech_p:.3f}" +     f" || mix_tech     | precision: {scores['mix_tech'][1]:.3f}" +    f" || falsetto_tech | precision: {scores['falsetto_tech'][1]:.3f}" + f" || breathy_tech | precision: {scores['breathy_tech'][1]:.3f}")
        # print(f"overall |    recall: {tech_r:.3f}" +     f" || mix_tech     |    recall: {scores['mix_tech'][2]:.3f}" +    f" || falsetto_tech |    recall: {scores['falsetto_tech'][2]:.3f}" + f" || breathy_tech |    recall: {scores['breathy_tech'][2]:.3f}")
        # print(f"overall |        f1: {tech_f:.3f}" +     f" || mix_tech     |        f1: {scores['mix_tech'][3]:.3f}" +    f" || falsetto_tech |        f1: {scores['falsetto_tech'][3]:.3f}" + f" || breathy_tech |        f1: {scores['breathy_tech'][3]:.3f}")
        # print(f"overall |  accuracy: {tech_a:.3f}" +     f" || mix_tech     |  accuracy: {scores['mix_tech'][4]:.3f}" +    f" || falsetto_tech |  accuracy: {scores['falsetto_tech'][4]:.3f}" + f" || breathy_tech |  accuracy: {scores['breathy_tech'][4]:.3f}")
        # print('-' * 110)
        # print(" " * 30 + f"pharyngeal_tech  |     auroc: {scores['pharyngeal_tech'][0]:.3f}" + f" || vibra_tech   |     auroc: {scores['vibra_tech'][0]:.3f}" +   f" || glissando_tech    |     auroc: {scores['glissando_tech'][0]:.3f}")
        # print(" " * 30 + f"pharyngeal_tech  | precision: {scores['pharyngeal_tech'][1]:.3f}" + f" || vibra_tech   | precision: {scores['vibra_tech'][1]:.3f}" +   f" || glissando_tech    | precision: {scores['glissando_tech'][1]:.3f}")
        # print(" " * 30 + f"pharyngeal_tech  |    recall: {scores['pharyngeal_tech'][2]:.3f}" + f" || vibra_tech   |    recall: {scores['vibra_tech'][2]:.3f}" +   f" || glissando_tech    |    recall: {scores['glissando_tech'][2]:.3f}")
        # print(" " * 30 + f"pharyngeal_tech  |        f1: {scores['pharyngeal_tech'][3]:.3f}" + f" || vibra_tech   |        f1: {scores['vibra_tech'][3]:.3f}" +   f" || glissando_tech    |        f1: {scores['glissando_tech'][3]:.3f}")
        # print(" " * 30 + f"pharyngeal_tech  |  accuracy: {scores['pharyngeal_tech'][4]:.3f}" + f" || vibra_tech   |  accuracy: {scores['vibra_tech'][4]:.3f}" +   f" || glissando_tech    |  accuracy: {scores['glissando_tech'][4]:.3f}")

        print("=" * 20 + "Overall assessment" + "=" * 20)
        print(f"|  item     |   auroc   | precision |  recall   |    f1     |  accuracy |")
        print('-' * 73)
        print(f"|  overall  |   {tech_auroc:.3f}   |   {tech_p:.3f}   |   {tech_r:.3f}   |   {tech_f:.3f}   |   {tech_a:.3f}   |")
        print(f"|    mix_tech    |   {scores['mix_tech'][0]:.3f}   |   {scores['mix_tech'][1]:.3f}   |   {scores['mix_tech'][2]:.3f}   |   {scores['mix_tech'][3]:.3f}   |   {scores['mix_tech'][4]:.3f}   |")
        print(f"| falsetto_tech  |   {scores['falsetto_tech'][0]:.3f}   |   {scores['falsetto_tech'][1]:.3f}   |   {scores['falsetto_tech'][2]:.3f}   |   {scores['falsetto_tech'][3]:.3f}   |   {scores['falsetto_tech'][4]:.3f}   |")
        print(f"|  breathy_tech  |   {scores['breathy_tech'][0]:.3f}   |   {scores['breathy_tech'][1]:.3f}   |   {scores['breathy_tech'][2]:.3f}   |   {scores['breathy_tech'][3]:.3f}   |   {scores['breathy_tech'][4]:.3f}   |")
        print(f"|  pharyngeal_tech   |   {scores['pharyngeal_tech'][0]:.3f}   |   {scores['pharyngeal_tech'][1]:.3f}   |   {scores['pharyngeal_tech'][2]:.3f}   |   {scores['pharyngeal_tech'][3]:.3f}   |   {scores['pharyngeal_tech'][4]:.3f}   |")
        print(f"|  vibra_tech   |   {scores['vibra_tech'][0]:.3f}   |   {scores['vibra_tech'][1]:.3f}   |   {scores['vibra_tech'][2]:.3f}   |   {scores['vibra_tech'][3]:.3f}   |   {scores['vibra_tech'][4]:.3f}   |")
        print(f"|   glissando_tech    |   {scores['glissando_tech'][0]:.3f}   |   {scores['glissando_tech'][1]:.3f}   |   {scores['glissando_tech'][2]:.3f}   |   {scores['glissando_tech'][3]:.3f}   |   {scores['glissando_tech'][4]:.3f}   |")
        print()

        ## other specific stats

        # spk
        print()
        print("=" * 20 + "Speaker assessment" + "=" * 20)

        item_idxs = {'male': [], 'female': []}
        for item_idx, item_name in enumerate(item_names):
            if '男声' in item_name:
                item_idxs['male'].append(item_idx)
            else:
                item_idxs['female'].append(item_idx)
        for spk in ['male', 'female']:
            results = []
            for i in range(res.shape[1]):
                _res = res[item_idxs[spk], i]
                results.append(np.mean(_res[~np.isnan(_res)]))
            scores = {}
            tech_auroc, tech_p, tech_r, tech_f, tech_a = results[:5]
            tech_names = ['mix_tech', 'falsetto_tech', 'breathy_tech', 'pharyngeal_tech', 'vibra_tech', 'glissando_tech']
            for tech_idx, tech_name in enumerate(tech_names):
                scores[tech_name] = results[(tech_idx + 1) * 5: (tech_idx + 2) * 5]

            print(f"-" * 20 + ("华为男声" if spk == 'male' else '华为女声') + f" {len(item_idxs[spk])} items" + "-" * 20)
            print(f"|  item     |   auroc   | precision |  recall   |    f1     |  accuracy |")
            print('-' * 73)
            print(f"|  overall  |   {tech_auroc:.3f}   |   {tech_p:.3f}   |   {tech_r:.3f}   |   {tech_f:.3f}   |   {tech_a:.3f}   |")
            print(f"|    mix_tech    |   {scores['mix_tech'][0]:.3f}   |   {scores['mix_tech'][1]:.3f}   |   {scores['mix_tech'][2]:.3f}   |   {scores['mix_tech'][3]:.3f}   |   {scores['mix_tech'][4]:.3f}   |")
            print(f"| falsetto_tech  |   {scores['falsetto_tech'][0]:.3f}   |   {scores['falsetto_tech'][1]:.3f}   |   {scores['falsetto_tech'][2]:.3f}   |   {scores['falsetto_tech'][3]:.3f}   |   {scores['falsetto_tech'][4]:.3f}   |")
            print(f"|  breathy_tech  |   {scores['breathy_tech'][0]:.3f}   |   {scores['breathy_tech'][1]:.3f}   |   {scores['breathy_tech'][2]:.3f}   |   {scores['breathy_tech'][3]:.3f}   |   {scores['breathy_tech'][4]:.3f}   |")
            print(f"|  pharyngeal_tech   |   {scores['pharyngeal_tech'][0]:.3f}   |   {scores['pharyngeal_tech'][1]:.3f}   |   {scores['pharyngeal_tech'][2]:.3f}   |   {scores['pharyngeal_tech'][3]:.3f}   |   {scores['pharyngeal_tech'][4]:.3f}   |")
            print(f"|  vibra_tech   |   {scores['vibra_tech'][0]:.3f}   |   {scores['vibra_tech'][1]:.3f}   |   {scores['vibra_tech'][2]:.3f}   |   {scores['vibra_tech'][3]:.3f}   |   {scores['vibra_tech'][4]:.3f}   |")
            print(f"|   glissando_tech    |   {scores['glissando_tech'][0]:.3f}   |   {scores['glissando_tech'][1]:.3f}   |   {scores['glissando_tech'][2]:.3f}   |   {scores['glissando_tech'][3]:.3f}   |   {scores['glissando_tech'][4]:.3f}   |")
            print()

        # per song
        print()
        print("=" * 20 + "Per sentence assessment" + "=" * 20)

        item_idxs = {}
        for item_idx, item_name in enumerate(item_names):
            item_name_ = item_name.split('#')
            sentence_name = '#'.join(item_name_[:-1])
            if sentence_name in item_idxs:
                item_idxs[sentence_name].append(item_idx)
            else:
                item_idxs[sentence_name] = [item_idx]
        # print(item_idxs)
        for sentence_name in sorted(item_idxs.keys()):
            results = []
            for i in range(res.shape[1]):
                _res = res[item_idxs[sentence_name], i]
                results.append(np.mean(_res[~np.isnan(_res)]))
            scores = {}
            tech_auroc, tech_p, tech_r, tech_f, tech_a = results[:5]
            tech_names = ['mix_tech', 'falsetto_tech', 'breathy_tech', 'pharyngeal_tech', 'vibra_tech', 'glissando_tech']
            for tech_idx, tech_name in enumerate(tech_names):
                scores[tech_name] = results[(tech_idx + 1) * 5: (tech_idx + 2) * 5]

            print(f"-" * 20 + f"{sentence_name}" + f" {len(item_idxs[sentence_name])} items" + "-" * 20)
            print(f"|  item     |   auroc   | precision |  recall   |    f1     |  accuracy |")
            print('-' * 73)
            print(f"|  overall  |   {tech_auroc:.3f}   |   {tech_p:.3f}   |   {tech_r:.3f}   |   {tech_f:.3f}   |   {tech_a:.3f}   |")
            print(f"|    mix_tech    |   {scores['mix_tech'][0]:.3f}   |   {scores['mix_tech'][1]:.3f}   |   {scores['mix_tech'][2]:.3f}   |   {scores['mix_tech'][3]:.3f}   |   {scores['mix_tech'][4]:.3f}   |")
            print(f"| falsetto_tech  |   {scores['falsetto_tech'][0]:.3f}   |   {scores['falsetto_tech'][1]:.3f}   |   {scores['falsetto_tech'][2]:.3f}   |   {scores['falsetto_tech'][3]:.3f}   |   {scores['falsetto_tech'][4]:.3f}   |")
            print(f"|  breathy_tech  |   {scores['breathy_tech'][0]:.3f}   |   {scores['breathy_tech'][1]:.3f}   |   {scores['breathy_tech'][2]:.3f}   |   {scores['breathy_tech'][3]:.3f}   |   {scores['breathy_tech'][4]:.3f}   |")
            print(f"|  pharyngeal_tech   |   {scores['pharyngeal_tech'][0]:.3f}   |   {scores['pharyngeal_tech'][1]:.3f}   |   {scores['pharyngeal_tech'][2]:.3f}   |   {scores['pharyngeal_tech'][3]:.3f}   |   {scores['pharyngeal_tech'][4]:.3f}   |")
            print(f"|  vibra_tech   |   {scores['vibra_tech'][0]:.3f}   |   {scores['vibra_tech'][1]:.3f}   |   {scores['vibra_tech'][2]:.3f}   |   {scores['vibra_tech'][3]:.3f}   |   {scores['vibra_tech'][4]:.3f}   |")
            print(f"|   glissando_tech    |   {scores['glissando_tech'][0]:.3f}   |   {scores['glissando_tech'][1]:.3f}   |   {scores['glissando_tech'][2]:.3f}   |   {scores['glissando_tech'][3]:.3f}   |   {scores['glissando_tech'][4]:.3f}   |")
            print()

        return {}

def bd_to_idxs(bd):
    # bd [T]
    idxs = []
    for idx in range(len(bd)):
        if bd[idx] == 1:
            idxs.append(idx)
    return idxs

def f0_tech_to_figure(f0_gt, ph_bd, tech_pred, tech_gt, tech_probs, tech_name, fig_name='', save_path=None):
    fig = plt.figure(figsize=(12, 8))
    plt.plot(f0_gt, color='r', label='gt f0')
    ph_idxs = [0] + bd_to_idxs(ph_bd) + [len(ph_bd)]
    t_pred = np.zeros(f0_gt.shape[0])
    t_gt = np.zeros(f0_gt.shape[0])
    t_logits = np.zeros(f0_gt.shape[0])
    # print('-'*40)
    # print('f0_gt.shape[0]', f0_gt.shape[0])
    # print('ph_idxs', ph_idxs)
    for i in range(len(ph_idxs)-1):
        t_pred[ph_idxs[i]: ph_idxs[i + 1]] = tech_pred[i] * 200
        t_gt[ph_idxs[i]: ph_idxs[i + 1]] = tech_gt[i] * 200
        t_logits[ph_idxs[i]: ph_idxs[i + 1]] = tech_probs[i] * 200
    plt.plot(t_gt, color='blue', label=f"gt {tech_name}")
    plt.plot(t_pred, color='green', label=f"pred {tech_name}")
    plt.plot(t_logits, color='orange', label=f"logits {tech_name}")
    plt.title(fig_name)
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path, format='png')
    return fig



