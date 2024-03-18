import os

import torch


class TransferVarTensor(object):
    """Return a copy of the input Variable or Tensor on specified device."""

    def __init__(self, device_id=-1):
        self.device_id = device_id

    def __call__(self, var_or_tensor: torch.Tensor):
        return var_or_tensor.cpu() if self.device_id == -1 \
            else var_or_tensor.cuda(self.device_id)


class TransferModulesOnly(object):
    """Transfer modules to cpu or specified gpu."""

    def __init__(self, device_id=-1):
        self.device_id = device_id

    def __call__(self, modules_or_both: list):
        # both means modules_and_optims, cause only transform model!
        self.transfer_modules(modules_or_both, self.device_id)

    def transfer_modules(self, modules_or_both, device_id=-1):
        """Transfer optimizers/modules to cpu or specified gpu.
        Args:
            modules_and_or_optims: A list, which members are only torch.nn.Module or None.
            device_id: gpu id, or -1 which means transferring to cpu
        """
        for item in modules_or_both:
            if isinstance(item, torch.optim.Optimizer):
                continue
            elif isinstance(item, (torch.nn.Module, torch.nn.DataParallel)):
                if device_id == -1:
                    item.cpu()
                else:
                    item.cuda(device=device_id)
            elif item is not None:
                print('[Warning] Invalid type {}'.format(
                    item.__class__.__name__))


def set_devices(sys_device_ids):
    """
    It sets some GPUs to be visible and returns some wrappers to transferring
    Variables/Tensors and Modules.
    Args:
        sys_device_ids: a tuple; which GPUs to use
            e.g.    sys_device_ids = (), only use cpu
                        sys_device_ids = (3,), use the 4th gpu
                        sys_device_ids = (0, 1, 2, 3,), use first 4 gpus
                        sys_device_ids = (0, 2, 4,), use the 1st, 3rd and 5th gpus
    Returns:
        TVT: a `TransferVarTensor` callable
        TMO: a `TransferModulesOnly` callable
    """
    # Set the CUDA_VISIBLE_DEVICES environment variable
    visible_devices = ''
    for i in sys_device_ids:
        visible_devices += '{}, '.format(i)
    os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices
    # Return wrappers.
    # Models and user defined Variables/Tensors would be transferred to the
    # first device.
    device_id = 0 if len(sys_device_ids) > 0 else -1
    TVT = TransferVarTensor(device_id)
    TMO = TransferModulesOnly(device_id)
    return TVT, TMO
