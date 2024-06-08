import torch
from torch import Tensor

def get_gen_loss(
        crit_fake_pred: Tensor, 
        is_wgan: bool
)->Tensor:
    """get_gen_loss

    Args:
        crit_fake_pred (Tensor): result of the discriminator over over the generated samples
        is_wgan (bool): do we compute wgan loss

    Returns:
        Tensor: loss for the generator
    """
    if is_wgan:
        gen_loss = -1. * torch.mean(crit_fake_pred)
    else:
        crit_fake_pred = torch.clamp(crit_fake_pred, min=0.001, max=0.999)
        gen_loss = -torch.mean(torch.log(crit_fake_pred))
    return gen_loss

def get_crit_loss(
        crit_fake_pred : Tensor, 
        crit_real_pred : Tensor, 
        gp : Tensor, 
        c_lambda : Tensor, 
        is_wgan: bool
)->Tensor:
    """get_crit_loss

    Args:
        crit_fake_pred (Tensor) : result of the discriminator over generated samples
        crit_real_pred (Tensor) : result of the discriminator over real samples
        gp (Tensor) : gradient of the discriminator over a mixture of inputs (ensure Lipschitz)
        c_lambda (Tensor) : intensity of gradient penalty
        is_wgan (bool) : do we compute wgan loss ?

    Returns :
        (Tensor) : the loss

    
    Note
    ----
    In this function, in the first statement, there is no log because we use the Wasserstein distance and especially, the Villani result about Lipschitz functions.
    The convergence of discriminator will be the wasserstein distance
    """
    if is_wgan:
        # opposite of the Villani result
        crit_loss = torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + c_lambda * gp
    else:
        assert c_lambda == 0, 'When not using WGAN : c_lambda must be zero'
        # >> clamping to avoid infinite gradient when using torch.log
        crit_fake_pred = torch.clamp(crit_fake_pred, min=0.001, max=0.999)
        crit_real_pred = torch.clamp(crit_real_pred, min=0.001, max=0.999)
        crit_loss = -(torch.mean(torch.log(crit_real_pred))+torch.mean(torch.log(1-crit_fake_pred)))
    return crit_loss
