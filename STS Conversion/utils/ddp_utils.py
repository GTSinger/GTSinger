from torch.nn.parallel import DistributedDataParallel
from torch.nn.parallel.distributed import _find_tensors
import torch.optim
import torch.utils.data
import torch


class DDP(DistributedDataParallel):
    """
    Override the forward call in lightning so it goes to training and validation step respectively
    """
    def forward(self, *inputs, **kwargs):  # pragma: no cover
        self._sync_params()
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        assert len(self.device_ids) == 1
        if self.module.training:
            output = self.module.training_step(*inputs[0], **kwargs[0])
        elif self.module.testing:
            output = self.module.test_step(*inputs[0], **kwargs[0])
        else:
            output = self.module.validation_step(*inputs[0], **kwargs[0])
        if torch.is_grad_enabled():
            # We'll return the output object verbatim since it is a freeform
            # object. We need to find any tensors in this object, though,
            # because we need to figure out which parameters were used during
            # this forward pass, to ensure we short circuit reduction for any
            # unused parameters. Only if `find_unused_parameters` is set.
            if self.find_unused_parameters:
                self.reducer.prepare_for_backward(list(_find_tensors(output)))
            else:
                self.reducer.prepare_for_backward([])
        return output
