'''
Author: Shuailin Chen
Created Date: 2021-04-03
Last Modified: 2021-04-03
	content: 
'''
import torch
from torch._six import inf
from typing import Union, Iterable

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]


def get_params_norm(parameters: _tensor_or_tensors, norm_type: float = 2.0, sta_type=('total')) -> torch.Tensor:
    r"""get gradient norm of an iterable of parameters.
    @in     -parameters      - an iterable of Tensors or a single Tensor  that will have gradients normalized
    @in     -norm_type       -type of the used p-norm. Can be ``'inf'`` for infinity norm.
    @in     sta_type         -'total' or 'max' or 'mean'
    @ret    -Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        Warning('Not gradients')
        return 0
    device = parameters[0].grad.device
    # ret = []
    # for sta in sta_type:
    #     if 'max' == sta:
    #         ret.append(max(p.grad.abs().max().to(device) for p in parameters))

    if sta_type=='total':
        if norm_type == inf:
            total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
        else:
            total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
        ret = total_norm
    else:
        raise NotImplementedError
    return ret 


if __name__ == '__main__':
    a = torch.arange(9, dtype=torch.float)-4
    a = a.reshape(3,3)
    b = a+1
    a.grad = b
    # my_clip_grad_norm_(a, max_norm=100, norm_type=2)
    print('done')