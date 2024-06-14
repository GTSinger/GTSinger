import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                                 is_causal=False, scale=None) -> torch.Tensor:
    # query [B, nhead, T, C], key [B, nhead, T, C]
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, kv_dim=None):
        super().__init__()
        if kv_dim is None:
            kv_dim = dim

        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_q = nn.Linear(dim, hidden_dim, bias=False)
        self.to_kv = nn.Linear(kv_dim, hidden_dim * 2, bias=False)

        self.to_out = nn.Sequential(nn.Linear(hidden_dim, dim, ),)

        self.sdpa = False
        if hasattr(F, 'scaled_dot_product_attention') and hasattr(torch.backends.cuda, 'sdp_kernel'):
            self.sdpa = True

    def forward(self, q, kv=None, mask=None):
        # q [B, T, C], kv [B, T, C]

        if kv is None:
            kv = q
        q = self.to_q(q)
        k, v = self.to_kv(kv).chunk(2, dim=2)
        q, k, v = map(
            lambda t: rearrange(t, "b t (h c) -> b h t c", h=self.heads), (q, k, v)
        )
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
        if self.sdpa:
            with torch.backends.cuda.sdp_kernel(enable_math=False):
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            out = scaled_dot_product_attention(q, k, v, attn_mask=mask)

        out = rearrange(out, "b h t c -> b t (h c) ", h=self.heads, )
        return self.to_out(out)
