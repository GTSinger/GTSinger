import matplotlib

from utils.tts_utils import sequence_mask, select_attn, get_focus_rate, get_phone_coverage_rate, get_diagonal_focus_rate

matplotlib.use('Agg')

from multiprocessing.pool import Pool
from tasks.base_task import data_loader
from utils.common_schedulers import RSQRTSchedule
from vocoders.base_vocoder import get_vocoder_cls, BaseVocoder

import os
import numpy as np
from tqdm import tqdm
import torch.distributed as dist

from modules.fastspeech import transformer_tts
from tasks.base_task import BaseTask, BaseDataset
from utils.hparams import hparams
from utils.indexed_datasets import IndexedDataset
from utils.text_encoder import TokenTextEncoder
import json

import matplotlib.pyplot as plt
import torch
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import utils
import logging
from utils import audio


class TransTTSDataset(BaseDataset):
    def __init__(self, prefix, shuffle=False):
        super().__init__(shuffle)
        self.data_dir = hparams['binary_data_dir']
        self.prefix = prefix
        self.hparams = hparams
        self.sizes = np.load(f'{self.data_dir}/{self.prefix}_lengths.npy')
        self.indexed_ds = None

    def _get_item(self, index):
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
        return self.indexed_ds[index]

    def __getitem__(self, index):
        item = self._get_item(index)
        spec = torch.Tensor(item['mel'])[:self.hparams['max_frames']]
        sample = {
            "id": index,
            "item_name": item['item_name'],
            "text": item['txt'],
            "txt_token": torch.LongTensor(item['phone']),
            "mel": spec,
        }
        return sample

    def collater(self, samples):
        if len(samples) == 0:
            return {}

        id = torch.LongTensor([s['id'] for s in samples])
        item_name = [s['item_name'] for s in samples]
        text = [s['text'] for s in samples]
        txt_tokens = utils.collate_1d([s['txt_token'] for s in samples])
        target = utils.collate_2d([s['mel'] for s in samples])
        prev_output_mels = utils.collate_2d([s['mel'] for s in samples], 0, shift_right=True)
        prev_output_mels[:, 0] = self.hparams['mel_vmin']
        # sort by descending source length
        txt_lengths = torch.LongTensor([s['txt_token'].numel() for s in samples])
        mel_lengths = torch.LongTensor([s['mel'].shape[0] for s in samples])

        batch = {
            'id': id,
            'item_name': item_name,
            'nsamples': len(samples),
            'text': text,
            'txt_tokens': txt_tokens,
            'txt_lengths': txt_lengths,
            'mels': target,
            'mel_lengths': mel_lengths,
            'prev_output_mels': prev_output_mels
        }
        return batch


class TransformerTtsTask(BaseTask):
    def __init__(self, *args, **kwargs):
        self.dataset_cls = TransTTSDataset
        self.max_tokens = hparams['max_tokens']
        self.max_sentences = hparams['max_sentences']
        self.max_eval_tokens = hparams['max_eval_tokens']
        if self.max_eval_tokens == -1:
            hparams['max_eval_tokens'] = self.max_eval_tokens = self.max_tokens
        self.max_eval_sentences = hparams['max_eval_sentences']
        if self.max_eval_sentences == -1:
            hparams['max_eval_sentences'] = self.max_eval_sentences = self.max_sentences
        self.vocoder = None
        self.phone_encoder = self.build_phone_encoder(hparams['binary_data_dir'])
        self.padding_idx = self.phone_encoder.pad()
        self.eos_idx = self.phone_encoder.eos()
        self.seg_idx = self.phone_encoder.seg()
        self.saving_result_pool = None
        self.saving_results_futures = None
        self.stats = {}
        super().__init__(*args, **kwargs)

    @data_loader
    def train_dataloader(self):
        train_dataset = self.dataset_cls(prefix=hparams['train_set_name'], shuffle=True)
        return self.build_dataloader(train_dataset, True, self.max_tokens, self.max_sentences,
                                     endless=hparams['endless_ds'])

    @data_loader
    def val_dataloader(self):
        valid_dataset = self.dataset_cls(prefix=hparams['valid_set_name'], shuffle=False)
        return self.build_dataloader(valid_dataset, False, self.max_eval_tokens, self.max_eval_sentences)

    @data_loader
    def test_dataloader(self):
        test_dataset = self.dataset_cls(prefix=hparams['test_set_name'], shuffle=False)
        self.test_dl = self.build_dataloader(
            test_dataset, False, self.max_eval_tokens,
            self.max_eval_sentences, batch_by_size=False)
        return self.test_dl

    def build_dataloader(self, dataset, shuffle, max_tokens=None, max_sentences=None,
                         required_batch_size_multiple=-1, endless=False, batch_by_size=True):
        devices_cnt = torch.cuda.device_count()
        if devices_cnt == 0:
            devices_cnt = 1
        if required_batch_size_multiple == -1:
            required_batch_size_multiple = devices_cnt

        def shuffle_batches(batches):
            np.random.shuffle(batches)
            return batches

        if max_tokens is not None:
            max_tokens *= devices_cnt
        if max_sentences is not None:
            max_sentences *= devices_cnt
        indices = dataset.ordered_indices()
        if batch_by_size:
            batch_sampler = utils.batch_by_size(
                indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences,
                required_batch_size_multiple=required_batch_size_multiple,
            )
        else:
            batch_sampler = []
            for i in range(0, len(indices), max_sentences):
                batch_sampler.append(indices[i:i + max_sentences])

        if shuffle:
            batches = shuffle_batches(list(batch_sampler))
            if endless:
                batches = [b for _ in range(1000) for b in shuffle_batches(list(batch_sampler))]
        else:
            batches = batch_sampler
            if endless:
                batches = [b for _ in range(1000) for b in batches]
        num_workers = dataset.num_workers
        if self.trainer.use_ddp:
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
            batches = [x[rank::num_replicas] for x in batches if len(x) % num_replicas == 0]
        return torch.utils.data.DataLoader(dataset,
                                           collate_fn=dataset.collater,
                                           batch_sampler=batches,
                                           num_workers=num_workers,
                                           pin_memory=False)

    def build_phone_encoder(self, data_dir):
        phone_list_file = os.path.join(data_dir, 'phone_set.json')
        phone_list = json.load(open(phone_list_file))
        return TokenTextEncoder(None, vocab_list=phone_list, replace_oov=',')

    def build_model(self):
        model = transformer_tts.TransformerTTS(self.phone_encoder)
        utils.print_arch(model)
        return model

    def build_scheduler(self, optimizer):
        return RSQRTSchedule(optimizer)

    def build_optimizer(self, model):
        self.optimizer = optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=hparams['lr'],
            betas=(hparams['optimizer_adam_beta1'], hparams['optimizer_adam_beta2']),
            weight_decay=hparams['weight_decay'])
        return optimizer

    def _training_step(self, sample, batch_idx, _):
        input = sample['txt_tokens']
        prev_output_mels = sample['prev_output_mels']
        target = sample['mels']
        output, _ = self.model(input, prev_output_mels, target)
        loss_output = self.loss(output, target)
        total_loss = sum(loss_output.values())
        return total_loss, loss_output

    def loss(self, decoder_output, target):
        # decoder_output : B x T x (mel+1)
        # target : B x T x mel
        predicted_mel = decoder_output[:, :, :-1]
        predicted_stop = decoder_output[:, :, -1]
        seq_mask, stop_mask = self.make_stop_target(target)

        l1_loss = F.l1_loss(predicted_mel, target, reduction='none')
        l2_loss = F.mse_loss(predicted_mel, target, reduction='none')
        weights = self.weights_nonzero_speech(target)
        l1_loss = (l1_loss * weights).sum() / weights.sum()
        l2_loss = (l2_loss * weights).sum() / weights.sum()
        stop_loss = (self.weighted_cross_entropy_with_logits(stop_mask, predicted_stop,
                                                             hparams['stop_token_weight']) * seq_mask).sum()
        stop_loss = stop_loss / (seq_mask.sum() + target.size(0) * (hparams['stop_token_weight'] - 1))

        return {
            'l1': l1_loss,
            'l2': l2_loss,
            'stop_loss': stop_loss,
        }

    def validation_step(self, sample, batch_idx):
        input = sample['txt_tokens']
        prev_output_mels = sample['prev_output_mels']
        target = sample['mels']
        output, attn_logits = self.model(input, prev_output_mels, target)
        outputs = {}
        outputs['losses'] = self.loss(output, target)
        outputs['total_loss'] = sum(outputs['losses'].values())
        outputs['nsamples'] = sample['nsamples']

        txt_lengths = sample['txt_lengths'].float()  # - 1 # exclude eos
        mel_lengths = sample['mel_lengths'].float()
        src_padding_mask = input.eq(0)  # | input.eq(self.eos_idx)  # also exclude eos
        src_seg_mask = input.eq(self.seg_idx)
        target_padding_mask = target.abs().sum(-1).eq(0)

        encdec_attn = select_attn(attn_logits)
        outputs['focus_rate'] = get_focus_rate(encdec_attn, src_padding_mask, target_padding_mask).mean()
        outputs['phone_coverage_rate'] = get_phone_coverage_rate(
            encdec_attn, src_padding_mask, src_seg_mask, target_padding_mask).mean()
        attn_ks = txt_lengths.float() / mel_lengths.float()
        outputs['diagonal_focus_rate'], diag_mask = get_diagonal_focus_rate(
            encdec_attn, attn_ks, mel_lengths, src_padding_mask, target_padding_mask)
        outputs['diagonal_focus_rate'] = outputs['diagonal_focus_rate'].mean()
        outputs = utils.tensors_to_scalars(outputs)
        return outputs

    def _validation_end(self, outputs):
        all_losses_meter = {
            'total_loss': utils.AvgrageMeter(),
            'fr': utils.AvgrageMeter(),
            'pcr': utils.AvgrageMeter(),
            'dfr': utils.AvgrageMeter(),
        }

        for output in outputs:
            for k, v in output['losses'].items():
                if k not in all_losses_meter:
                    all_losses_meter[k] = utils.AvgrageMeter()
                all_losses_meter[k].update(v, output['nsamples'])
            all_losses_meter['total_loss'].update(output['total_loss'], output['nsamples'])
            all_losses_meter['fr'].update(output['focus_rate'], output['nsamples'])
            all_losses_meter['pcr'].update(output['phone_coverage_rate'], output['nsamples'])
            all_losses_meter['dfr'].update(output['diagonal_focus_rate'], output['nsamples'])
        return {f'{k}': round(v.avg, 4) for k, v in all_losses_meter.items()}

    def test_start(self):
        self.saving_result_pool = Pool(8)
        self.saving_results_futures = []
        self.vocoder: BaseVocoder = get_vocoder_cls(hparams)()

    def test_step(self, sample, batch_idx):
        logging.info('inferring batch {} with {} samples'.format(batch_idx, sample['nsamples']))
        with utils.Timer('trans_tts', print_time=hparams['profile_infer']):
            txt_tokens = sample['txt_tokens']
            decoded_mel, encdec_attn, hit_eos, stop_logits = self.model.infer(txt_tokens)
            if not hparams['profile_infer']:
                txt_lengths = sample['txt_lengths'].float() - 1  # exclude eos
                mel_lengths = (1.0 - hit_eos[:, 1:].float()).sum(dim=-1) + 1
                mel_lengths = mel_lengths.float()
                src_padding_mask = txt_tokens.eq(0) | txt_tokens.eq(self.eos_idx)  # also exclude eos
                src_seg_mask = txt_tokens.eq(self.seg_idx)
                target_padding_mask = decoded_mel.abs().sum(-1).eq(0)
                focus_rate = get_focus_rate(encdec_attn, src_padding_mask, target_padding_mask)
                phone_coverage_rate = get_phone_coverage_rate(encdec_attn, src_padding_mask, src_seg_mask,
                                                                    target_padding_mask)
                attn_ks = txt_lengths.float() / mel_lengths.float()
                diagonal_focus_rate, diag_mask = get_diagonal_focus_rate(encdec_attn, attn_ks, mel_lengths,
                                                                               src_padding_mask,
                                                                               target_padding_mask)
                encdec_attn = encdec_attn[:, None]

        hit_eos = hit_eos[:, 1:]
        outputs = decoded_mel
        predict_lengths = (1.0 - hit_eos.float()).sum(dim=-1)
        outputs *= (1.0 - hit_eos.float())[:, :, None]

        sample['outputs'] = outputs
        sample['predict_lengths'] = predict_lengths
        sample['encdec_attn'] = encdec_attn
        self.after_infer(sample)

    def after_infer(self, predictions):
        predictions = utils.unpack_dict_to_list(predictions)
        for num_predictions, prediction in enumerate(tqdm(predictions)):
            for k, v in prediction.items():
                if type(v) is torch.Tensor:
                    prediction[k] = v.cpu().numpy()

            item_name = prediction.get('item_name')
            text = prediction.get('text').replace(":", "%3A")[:80]
            txt_tokens = prediction.get('txt_tokens')
            txt_lengths = prediction.get('txt_lengths')
            targets = prediction.get("mels")
            outputs = prediction["outputs"]
            str_phs = self.phone_encoder.decode(txt_tokens, strip_eos=True, strip_padding=True)
            targets = self.remove_padding(targets)  # speech
            outputs = self.remove_padding(outputs)

            if 'encdec_attn' in prediction:
                encdec_attn = prediction['encdec_attn']
                encdec_attn = encdec_attn[encdec_attn.max(-1).sum(-1).argmax(-1)]
                encdec_attn = encdec_attn.T[:txt_lengths, :len(targets)]
            else:
                encdec_attn = None

            gen_dir = os.path.join(hparams['work_dir'],
                                   f'generated_{self.trainer.global_step}_{hparams["gen_dir_name"]}')
            os.makedirs(gen_dir, exist_ok=True)
            os.makedirs(f'{gen_dir}/wavs', exist_ok=True)
            os.makedirs(f'{gen_dir}/plot', exist_ok=True)
            os.makedirs(f'{gen_dir}/attn_plot', exist_ok=True)
            if hparams.get('save_mel_npy', False):
                os.makedirs(f'{gen_dir}/npy', exist_ok=True)

            wav_pred = self.vocoder.spec2wav(outputs)
            self.saving_results_futures.append(
                self.saving_result_pool.apply_async(self.save_result, args=[
                    wav_pred, outputs, 'P', item_name, text, gen_dir, encdec_attn, str_phs]))

            if targets is not None and hparams['save_gt']:
                wav_gt = self.vocoder.spec2wav(targets)
                self.saving_results_futures.append(
                    self.saving_result_pool.apply_async(self.save_result, args=[
                        wav_gt, targets, 'G', item_name, text, gen_dir]))

            if hparams['profile_infer']:
                if 'gen_wav_time' not in self.stats:
                    self.stats['gen_wav_time'] = 0
                self.stats['gen_wav_time'] += len(wav_pred) / hparams['audio_sample_rate']
                print('gen_wav_time: ', self.stats['gen_wav_time'])

    @staticmethod
    def save_result(wav_out, mel, prefix, item_name, text, gen_dir, alignment=None, str_phs=None):
        base_fn = f'[{item_name}][{prefix}]'
        if text is not None:
            base_fn += text
        audio.save_wav(wav_out, f'{gen_dir}/wavs/{base_fn}.wav', hparams['audio_sample_rate'],
                       norm=hparams['out_wav_norm'])

        fig = plt.figure(figsize=(14, 10))
        heatmap = plt.pcolor(mel.T)
        fig.colorbar(heatmap)
        plt.tight_layout()
        plt.savefig(f'{gen_dir}/plot/{base_fn}.png', format='png')
        plt.close(fig)

        if alignment is not None:
            fig, ax = plt.subplots(figsize=(12, 16))
            im = ax.imshow(alignment, aspect='auto', origin='lower',
                           interpolation='none')
            decoded_txt = str_phs.split(" ")
            ax.set_yticks(np.arange(len(decoded_txt)))
            ax.set_yticklabels(list(decoded_txt), fontsize=6)
            fig.colorbar(im, ax=ax)
            fig.savefig(f'{gen_dir}/attn_plot/{base_fn}_attn.png', format='png')
            plt.close()
        if hparams.get('save_mel_npy', False):
            np.save(f'{gen_dir}/npy/{base_fn}', mel)

    def test_end(self, outputs):
        self.saving_result_pool.close()
        [f.get() for f in tqdm(self.saving_results_futures)]
        self.saving_result_pool.join()
        return {}

    ##########
    # utils
    ##########
    def remove_padding(self, x, padding_idx=0):
        return utils.remove_padding(x, padding_idx)

    def weights_nonzero_speech(self, target):
        # target : B x T x mel
        # Assign weight 1.0 to all labels except for padding (id=0).
        dim = target.size(-1)
        return target.abs().sum(-1, keepdim=True).ne(0).float().repeat(1, 1, dim)

    def make_stop_target(self, target):
        # target : B x T x mel
        seq_mask = target.abs().sum(-1).ne(0).float()
        seq_length = seq_mask.sum(1)
        mask_r = 1 - sequence_mask(seq_length - 1, target.size(1)).float()
        return seq_mask, mask_r

    def weighted_cross_entropy_with_logits(self, targets, logits, pos_weight=1):
        x = logits
        z = targets
        q = pos_weight
        l = 1 + (q - 1) * z
        return (1 - z) * x + l * (torch.log(1 + torch.exp(-x.abs())) + F.relu(-x))
