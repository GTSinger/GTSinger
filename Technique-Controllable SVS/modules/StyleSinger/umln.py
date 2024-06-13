from modules.commons.common_layers import *
import random


class DistributionUncertainty(nn.Module):

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, hidden_size=256):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self._activated = True
        self.hidden_size = hidden_size
        self.affine_layer = LinearNorm(
            hidden_size,
            2 * hidden_size, # For both b (bias) g (gain)
        )
        self.factor=1.0

    def __repr__(self):
        return f'DSU(p={self.p}, alpha={self.alpha}, eps={self.eps})'

    def set_activation_status(self, status=True):
        self._activated = status
    
    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        if x.shape[0] == 1:
            t=torch.zeros(x.size()).to(x.device)
        else:
            t = (x.std(dim=0, keepdim=True) + self.eps)
            # print(x.var(dim=0, keepdim=True) ,t)
            t = t.repeat(x.shape[0], 1, 1)
        return t
    

    def forward(self, x, spk_embed):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x


        mu, sig = torch.mean(x, dim=-1, keepdim=True), torch.std(x, dim=-1, keepdim=True)
        x_normed = (x - mu) / (sig + 1e-6)  # [B, T, H_m]


        # Get Bias and Gain
        mu1, sig1 = torch.split(self.affine_layer(spk_embed), self.hidden_size, dim=-1)  # [B, 1, 2 * H_m] --> 2 * [B, 1, H_m]
        # print(mu1.shape,sig1.shape())

        sqrtvar_mu = self.sqrtvar(mu1)
        sqrtvar_std = self.sqrtvar(sig1)

        beta = self._reparameterize(mu1, sqrtvar_mu)
        gamma = self._reparameterize(sig1, sqrtvar_std)
        # print(mu1,'\nbeta',beta,'\n----------------\n')

        # Perform Scailing and Shifting
        return gamma * x_normed + beta # [B, T, H_m]
    