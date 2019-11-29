import torch
import torch.nn as nn
import pytorch_ssim
import agencyNet_720P_single
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss, None

class CharbonnierLossPlusSSIM(nn.Module):
    """Charbonnier Loss (L1) + SSIM loss"""

    def __init__(self, eps=1e-6, lambda_=1e-2):
        super(CharbonnierLossPlusSSIM, self).__init__()
        self.eps = eps
        self.lambda_ = lambda_

    def forward(self, x, y):
        diff = x - y
        loss_c = torch.mean(torch.sqrt(diff * diff + self.eps))
        loss_s = 1 - pytorch_ssim.ssim(x, y)
        loss = loss_c + self.lambda_* loss_s
        return loss, [loss_c, loss_s]

class MSSSIMLoss(nn.Module):
    """Charbonnier MSSSIM loss"""

    def __init__(self, eps=1e-6, lambda_=1e-2):
        super(MSSSIMLoss, self).__init__()
        self.eps = eps
        self.lambda_ = lambda_
        self.ms_ssim_module = MS_SSIM(win_size=11, win_sigma=1.5, data_range=1.0, size_average=True, channel=3)

    def forward(self, x, y):
        # diff = x - y
        # loss_c = torch.mean(torch.sqrt(diff * diff + self.eps))
        # loss_s = 1 - pytorch_ssim.ssim(x, y)
        # loss = loss_c + self.lambda_* loss_s
        loss = 1 - self.ms_ssim_module(x, y)
        return loss, None

class SSIMLoss(nn.Module):
    """Charbonnier MSSSIM loss"""

    def __init__(self, eps=1e-6, lambda_=1e-2):
        super(SSIMLoss, self).__init__()
        self.eps = eps
        self.lambda_ = lambda_
        
    def forward(self, x, y):
        # diff = x - y
        # loss_c = torch.mean(torch.sqrt(diff * diff + self.eps))
        # loss_s = 1 - pytorch_ssim.ssim(x, y)
        # loss = loss_c + self.lambda_* loss_s
        loss = 1 - pytorch_ssim.ssim(x, y)
        return loss, None

class CharbonnierLossPlusMSSSIM(nn.Module):
    """Charbonnier Loss (L1) + MSSSIM loss"""

    def __init__(self, eps=1e-6, lambda_= 1):
        super(CharbonnierLossPlusMSSSIM, self).__init__()
        self.eps = eps
        self.lambda_ = lambda_
        self.ms_ssim_module = MS_SSIM(win_size=11, win_sigma=1.5, data_range=1.0, size_average=True, channel=3)

    def forward(self, x, y):
        diff = x - y
        loss_c = torch.mean(torch.sqrt(diff * diff + self.eps))
        loss_m = 1 - self.ms_ssim_module(x, y)
        loss = loss_c + self.lambda_* loss_m
        return loss, [loss_c, loss_m]




class CharbonnierLossPlusSSIMPlusVMAF(nn.Module):
    """Charbonnier Loss (L1) + SSIM loss + VMAF loss"""

    def __init__(self, eps=1e-6, lambda_=1e-2, lambda_vmaf = 1e-3):
        super(CharbonnierLossPlusSSIMPlusVMAF, self).__init__()
        self.eps = eps
        self.lambda_ = lambda_
        self.lambda_vmaf = lambda_vmaf

        self.vmaf_model = agencyNet_720P_single.CNN_Net(in_channels=6, out_channels=1)
        self.vmaf_model.load_state_dict(torch.load('/DATA7_DB7/data/yxhuang/vmaf_data/model/VMAF_agency_720P_module_119_0100.pkl', map_location='cpu'))
        if torch.cuda.is_available():
            self.vmaf_model = self.vmaf_model.cuda()

    def forward(self, x, y):
        diff = x - y
        loss_c = torch.mean(torch.sqrt(diff * diff + self.eps))
        loss_s = 1 - pytorch_ssim.ssim(x, y)
        x_input = torch.cat((x,y),1)
        loss_v = 1 - self.vmaf_model(x_input)
        loss = loss_c + self.lambda_  * loss_s + self.lambda_vmaf * loss_v
        return loss, [loss_c, loss_s, loss_v]


# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp,
                                          grad_outputs=grad_outputs, create_graph=True,
                                          retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1)**2).mean()
        return loss
