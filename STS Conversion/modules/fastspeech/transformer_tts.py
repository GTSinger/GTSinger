from modules.fastspeech.tts_modules import TransformerEncoderLayer, TransformerDecoderLayer, \
    DEFAULT_MAX_SOURCE_POSITIONS, DEFAULT_MAX_TARGET_POSITIONS
from modules.commons.common_layers import *
from modules.commons.gaussian import sample_from_gaussian
from utils.hparams import hparams
from utils.tts_utils import sequence_mask, fill_with_neg_inf2, select_attn


class TransformerEncoder(nn.Module):
    def __init__(self, embed_tokens, last_ln=True, num_layers=None, hidden_size=None, kernel_size=None, num_heads=2):
        super().__init__()
        self.num_layers = hparams['enc_layers'] if num_layers is None else num_layers
        self.hidden_size = hparams['hidden_size'] if hidden_size is None else hidden_size
        self.embed_tokens = embed_tokens
        self.num_heads = num_heads
        self.padding_idx = embed_tokens.padding_idx
        embed_dim = embed_tokens.embedding_dim
        self.dropout = hparams['dropout']
        self.embed_scale = math.sqrt(embed_dim)
        self.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        self.embed_positions = SinusoidalPositionalEmbedding(
            embed_dim, self.padding_idx,
            init_size=self.max_source_positions + self.padding_idx + 1,
        )
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(
                self.hidden_size, self.dropout, kernel_size=kernel_size, num_heads=num_heads) for _ in
            range(self.num_layers)
        ])
        self.last_ln = last_ln
        if last_ln:
            self.layer_norm = LayerNorm(embed_dim)

    def forward_embedding(self, txt_tokens):
        # embed tokens and positions
        embed = self.embed_scale * self.embed_tokens(txt_tokens)
        positions = self.embed_positions(txt_tokens)
        # x = self.prenet(x)
        x = embed + positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, embed

    def forward(self, txt_tokens):
        """

        :param txt_tokens: [B, T]
        :return: {
            'encoder_out': [T x B x C]
            'encoder_padding_mask': [B x T]
            'encoder_embedding': [B x T x C]
            'attn_w': []
        }
        """
        x, encoder_embedding = self.forward_embedding(txt_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = txt_tokens.eq(self.padding_idx).data

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask=encoder_padding_mask)

        if self.last_ln:
            x = self.layer_norm(x)
            x = x * (1 - encoder_padding_mask.float()).transpose(0, 1)[..., None]
        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
            'encoder_embedding': encoder_embedding,  # B x T x C
            'attn_w': []
        }


class TransformerDecoder(nn.Module):
    def __init__(self, padding_idx=0, num_layers=None, causal=True, dropout=None,
                 out_dim=None, hidden_size=None, kernel_size=None, in_dim=None):
        super().__init__()
        self.num_layers = hparams['dec_layers'] if num_layers is None else num_layers
        self.hidden_size = hparams['hidden_size'] if hidden_size is None else hidden_size
        self.prenet_hidden_size = hparams['prenet_hidden_size']
        self.prenet_dropout = hparams['prenet_dropout']
        self.padding_idx = padding_idx
        self.causal = causal
        self.dropout = hparams['dropout'] if dropout is None else dropout
        self.in_dim = hparams['audio_num_mel_bins'] if in_dim is None else in_dim
        self.out_dim = hparams['audio_num_mel_bins'] + 1 if out_dim is None else out_dim
        self.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS
        self.embed_positions = SinusoidalPositionalEmbedding(
            self.hidden_size, self.padding_idx,
            init_size=self.max_target_positions + self.padding_idx + 1,
        )
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(self.hidden_size, self.dropout, kernel_size=kernel_size) for i in
            range(self.num_layers)
        ])
        self.layer_norm = LayerNorm(self.hidden_size)
        self.project_out_dim = Linear(self.hidden_size, self.out_dim, bias=False)
        self.prenet_fc1 = Linear(self.in_dim, self.prenet_hidden_size)
        self.prenet_fc2 = Linear(self.prenet_hidden_size, self.prenet_hidden_size)
        self.prenet_fc3 = Linear(self.prenet_hidden_size, self.hidden_size, bias=False)

    def forward_prenet(self, x):
        mask = x.abs().sum(-1, keepdim=True).ne(0).float()

        prenet_dropout = self.prenet_dropout
        # prenet_dropout = random.uniform(0, 0.5) if self.training else 0
        x = self.prenet_fc1(x)
        x = F.relu(x)
        x = F.dropout(x, prenet_dropout, training=True)
        x = self.prenet_fc2(x)
        x = F.relu(x)
        x = F.dropout(x, prenet_dropout, training=True)
        x = self.prenet_fc3(x)
        x = F.relu(x)
        x = x * mask
        return x

    def forward(
            self,
            prev_output_mels,  # B x T x 80
            encoder_out=None,  # T x B x C
            encoder_padding_mask=None,  # B x T
            target_mels=None,
            incremental_state=None,
            return_hiddens=False,
            attn_out=None  # T x B x C
    ):
        # embed positions
        if incremental_state is not None:
            positions = self.embed_positions(
                prev_output_mels.abs().sum(-1),
                incremental_state=incremental_state
            )
            prev_output_mels = prev_output_mels[:, -1:, :]
            positions = positions[:, -1:, :]
            self_attn_padding_mask = None
        else:
            positions = self.embed_positions(
                target_mels.abs().sum(-1),
                incremental_state=incremental_state
            )
            self_attn_padding_mask = target_mels.abs().sum(-1).eq(0).data

        # convert mels through prenet
        x = self.forward_prenet(prev_output_mels)
        # embed positions
        x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        all_attn_logits = []
        h = []
        # decoder layers
        for layer in self.layers:
            if incremental_state is None and self.causal:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, attn_logits = layer(
                x,
                encoder_out=encoder_out,
                encoder_padding_mask=encoder_padding_mask,
                attn_out=attn_out,
                incremental_state=incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                reset_attn_weight=hparams.get('reset_attn_weight')
            )
            h.append(x)
            all_attn_logits.append(attn_logits)

        x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # B x T x C -> B x T x 81
        x = self.project_out_dim(x)

        # attn_logits = torch.stack(all_attn_logits, dim=1) # B x n_layers x head x target_len x src_len
        if return_hiddens:
            return x, all_attn_logits, h
        else:
            return x, all_attn_logits

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
                not hasattr(self, '_future_mask')
                or self._future_mask is None
                or self._future_mask.device != tensor.device
                or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(fill_with_neg_inf2(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]


class TransformerTTS(nn.Module):
    def __init__(self, dictionary, num_spk=0, causal_decoder=True,
                 enc_layers=None, dec_layers=None, hidden_size=None,
                 enc_kernel=None, dec_kernel=None,
                 return_dec_hiddens=False, out_dim=None, dec_in_dim=None):
        super().__init__()
        self.dictionary = dictionary
        self.padding_idx = dictionary.pad()
        self.enc_layers = hparams['enc_layers'] if enc_layers is None else enc_layers
        self.dec_layers = hparams['dec_layers'] if dec_layers is None else dec_layers
        self.hidden_size = hparams['hidden_size'] if hidden_size is None else hidden_size
        self.dec_in_dim = hparams['audio_num_mel_bins'] if dec_in_dim is None else dec_in_dim
        self.out_dim = hparams['audio_num_mel_bins'] + 1 if out_dim is None else out_dim
        self.causal_decoder = causal_decoder
        self.return_dec_hiddens = return_dec_hiddens
        self.enc_kernel = enc_kernel
        self.dec_kernel = dec_kernel
        self.encoder_embed_tokens = self.build_embedding(self.dictionary, self.hidden_size)
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.num_heads = [2] * hparams['enc_layers']
        if num_spk > 0:
            self.spk_embed = Embedding(num_spk, self.hidden_size)

    def build_embedding(self, dictionary, embed_dim):
        num_embeddings = len(dictionary)
        emb = Embedding(num_embeddings, embed_dim, self.padding_idx)
        return emb

    def build_encoder(self):
        return TransformerEncoder(self.encoder_embed_tokens, num_layers=self.enc_layers,
                                  hidden_size=self.hidden_size, kernel_size=self.enc_kernel)

    def build_decoder(self):
        return TransformerDecoder(padding_idx=self.padding_idx, causal=self.causal_decoder,
                                  num_layers=self.dec_layers, hidden_size=self.hidden_size,
                                  kernel_size=self.dec_kernel, out_dim=self.out_dim,
                                  in_dim=self.dec_in_dim)

    def forward_encoder(self, txt_tokens, spk_ids=None, *args, **kwargs):
        encoder_out = self.encoder(txt_tokens)
        if spk_ids is not None:
            encoder_out += self.spk_embed(spk_ids)[None, :, :]
        return encoder_out

    def forward_decoder(self, prev_output_mels, encoder_out, encoder_padding_mask=None, incremental_state=None):
        decoder_output, attn_logits = self.decoder(
            prev_output_mels, encoder_out, encoder_padding_mask, incremental_state=incremental_state)
        return decoder_output, attn_logits

    def forward(self, txt_tokens, prev_output_mels, target_mels=None, spk_ids=None, return_hiddens=False, *args,
                **kwargs):
        encoder_outputs = self.forward_encoder(txt_tokens, spk_ids)
        encoder_out = encoder_outputs['encoder_out']
        encoder_padding_mask = encoder_outputs['encoder_padding_mask'].data
        return self.decoder(prev_output_mels, encoder_out, encoder_padding_mask,
                            target_mels=target_mels, return_hiddens=return_hiddens)

    def infer(self, src_tokens, spk_ids=None):
        mel_bins = hparams['audio_num_mel_bins']
        bsz = src_tokens.size(0)
        max_input_len = src_tokens.size(1)

        decode_length = self.estimate_decode_length(max_input_len)
        encoder_outputs = self.forward_encoder(src_tokens, spk_ids)
        encoder_out = encoder_outputs['encoder_out']
        encoder_padding_mask = encoder_outputs['encoder_padding_mask']

        hit_eos = src_tokens.new(bsz, 1).fill_(0).bool()
        stop_logits = src_tokens.new(bsz, 0).fill_(0).float()
        stage = 0
        decoder_input = src_tokens.new(bsz, decode_length + 1, mel_bins).fill_(
            0).float()
        decoded_mel = src_tokens.new(bsz, 0, mel_bins).fill_(0).float()
        encdec_attn_logits = []

        for i in range(self.dec_layers):
            encdec_attn_logits.append(src_tokens.new(bsz, self.num_heads[i], 0, max_input_len).fill_(0).float())
        attn_pos = src_tokens.new(bsz).fill_(0).int()
        use_masks = []
        for i in range(self.dec_layers):
            use_masks.append(src_tokens.new(self.num_heads[i]).fill_(0).float())

        incremental_state = {}
        step = 0
        if hparams['attn_constraint']:
            for i, layer in enumerate(self.decoder.layers):
                enc_dec_attn_constraint_mask = src_tokens.new(bsz, self.num_heads[i], max_input_len).fill_(0).int()
                layer.set_buffer('enc_dec_attn_constraint_mask', enc_dec_attn_constraint_mask, incremental_state)

        def is_finished(step, decode_length, hit_eos, stage):
            finished = step >= decode_length
            finished |= (hit_eos[:, -1].sum() == hit_eos.size(0)).cpu().numpy()
            if hparams['attn_constraint']:
                finished &= stage != 0
            return finished

        while True:
            if is_finished(step, decode_length, hit_eos, stage):
                break

            decoder_output, attn_logits = self.forward_decoder(decoder_input[:, :step + 1], encoder_out,
                                                               encoder_padding_mask,
                                                               incremental_state=incremental_state)
            next_mel = decoder_output[:, -1:, :mel_bins]
            stop_logit = decoder_output[:, -1:, -1]
            stop_logits = torch.cat((stop_logits, stop_logit), dim=1)
            decoded_mel = torch.cat((decoded_mel, next_mel), dim=1)
            for i in range(self.dec_layers):
                encdec_attn_logits[i] = torch.cat((encdec_attn_logits[i], attn_logits[i]), dim=2)
            step += 1

            this_hit_eos = hit_eos[:, -1:]
            if hparams['attn_constraint']:
                this_hit_eos |= (attn_pos[:, None] >= (encoder_padding_mask < 1.0).float() \
                                 .sum(dim=-1, keepdim=True).int() - 5) & (torch.sigmoid(stop_logit) > 0.5)
            else:
                this_hit_eos |= torch.sigmoid(stop_logit) > 0.5
            hit_eos = torch.cat((hit_eos, this_hit_eos), dim=1)

            decoder_input[:, step] = next_mel[:, -1]

            if hparams['attn_constraint']:
                stage_change_step = 50
                all_prev_weights = []
                for i in range(self.dec_layers):
                    all_prev_weights.append(torch.softmax(encdec_attn_logits[i], dim=-1))  # bsz x head x L x L_kv

                # if the stage should change
                next_stage = (step == stage_change_step) | (step >= decode_length)
                next_stage |= (hit_eos[:, -1].sum() == hit_eos.size(0)).cpu().numpy()
                next_stage &= (stage == 0)

                # choose the diagonal attention
                if next_stage:
                    use_masks = []
                    for i in range(hparams['dec_layers']):
                        use_mask = (all_prev_weights[i][:, :, :step].max(dim=-1).values.mean(
                            dim=(0, 2)) > 0.6).float()  # [head]
                        use_masks.append(use_mask)
                    attn_pos = src_tokens.new(bsz).fill_(0).int()

                    # reseet when the stage changes
                    for layer in self.decoder.layers:
                        layer.clear_buffer(src_tokens, encoder_out, encoder_padding_mask, incremental_state)

                    encdec_attn_logits = []
                    for i in range(hparams['dec_layers']):
                        encdec_attn_logits.append(
                            src_tokens.new(bsz, self.num_heads[i], 0, max_input_len).fill_(0).float())
                    decoded_mel = src_tokens.new(bsz, 0, mel_bins).fill_(0).float()
                    decoder_input = src_tokens.new(bsz, decode_length + 1, hparams['audio_num_mel_bins']).fill_(
                        0).float()
                    hit_eos = src_tokens.new(bsz, 1).fill_(0).bool()
                    stage = stage + 1
                    step = 0

                prev_weights_mask1 = sequence_mask(
                    torch.max(attn_pos - 1, attn_pos.new(attn_pos.size()).fill_(0)).float(),
                    encdec_attn_logits[0].size(-1)).float()  # bsz x L_kv
                prev_weights_mask2 = 1.0 - sequence_mask(attn_pos.float() + 4,
                                                         encdec_attn_logits[0].size(-1)).float()  # bsz x L_kv
                enc_dec_attn_constraint_masks = []
                for i in range(self.dec_layers):
                    mask = (prev_weights_mask1 + prev_weights_mask2)[:, None, :] * use_masks[i][None, :,
                                                                                   None]  # bsz x head x L_kv
                    enc_dec_attn_constraint_masks.append(mask)
                # enc_dec_attn_constraint_masks = (prev_weights_mask1 + prev_weights_mask2)[:, None, None, :] * use_masks[None, :, None, None] # bsz x (n_layers x head) x 1 x L_kv

                for i, layer in enumerate(self.decoder.layers):
                    enc_dec_attn_constraint_mask = enc_dec_attn_constraint_masks[i]
                    layer.set_buffer('enc_dec_attn_constraint_mask', enc_dec_attn_constraint_mask,
                                     incremental_state)

                def should_move_on():
                    prev_weights = []
                    for i in range(self.dec_layers):
                        prev_weight = (all_prev_weights[i] * use_masks[i][None, :, None, None]).sum(dim=1)
                        prev_weights.append(prev_weight)
                    prev_weights = sum(prev_weights) / sum([mask.sum() for mask in use_masks])
                    # prev_weights = (prev_weights * use_masks[None, :, None, None]).sum(dim=1) / use_masks.sum()
                    move_on = (prev_weights[:, -3:].mean(dim=1).gather(1, attn_pos[:, None].long())).squeeze(-1) < 0.7
                    move_on &= torch.argmax(prev_weights[:, -1], -1) > attn_pos.long()
                    return attn_pos + move_on.int()

                if step > 3 and stage == 1:
                    attn_pos = should_move_on()

        # size = encdec_attn_logits.size()
        # encdec_attn_logits = encdec_attn_logits.view(size[0], size[1]*size[2], size[3], size[4])
        encdec_attn = select_attn(encdec_attn_logits)
        return decoded_mel, encdec_attn, hit_eos, stop_logits

    def estimate_decode_length(self, input_length):
        return input_length * 8 + 100


class TransformerTTSGuas(TransformerTTS):
    def build_decoder(self):
        dec_out_dim = hparams['audio_num_mel_bins'] * 2 + 1
        return TransformerDecoder(padding_idx=self.padding_idx, causal=self.causal_decoder,
                                  num_layers=self.dec_layers, hidden_size=self.hidden_size,
                                  kernel_size=self.dec_kernel, out_dim=dec_out_dim)

    def forward_decoder(self, prev_output_mels, encoder_out, encoder_padding_mask, incremental_state=None):
        gaus_temp = hparams['gaus_sample_temp']
        decoder_output, attn_logits = self.decoder(
            prev_output_mels, encoder_out, encoder_padding_mask, incremental_state=incremental_state)
        decoder_output = torch.cat(
            [sample_from_gaussian(decoder_output[:, :, :-1], gaus_temp), decoder_output[:, :, -1:]], -1)
        return decoder_output, attn_logits
