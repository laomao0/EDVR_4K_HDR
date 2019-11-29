from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import torch
# X: (N,3,H,W) a batch of RGB images (0~255)
# Y: (N,3,H,W)  
X = torch.rand(4, 3, 512, 512)
Y = torch.rand(4, 3, 512, 512)
#Y = X

# ssim_val = ssim( X, Y, data_range=1.0, size_average=False) # return (N,)
# ms_ssim_val = ms_ssim( X, Y, data_range=1.0, size_average=False ) #(N,)

# # or set 'size_average=True' to get a scalar value as loss.
# ssim_loss = ssim( X, Y, data_range=1.0, size_average=True) # return a scalar
# ms_ssim_loss = ms_ssim( X, Y, data_range=1.0, size_average=True )

# or reuse windows with SSIM & MS_SSIM. 
ssim_module = SSIM(win_size=11, win_sigma=1.5, data_range=1.0, size_average=True, channel=3)
ms_ssim_module = MS_SSIM(win_size=11, win_sigma=1.5, data_range=1.0, size_average=True, channel=3)

ssim_loss = 1 - ssim_module(X, Y)
ms_ssim_loss = 1 - ms_ssim_module(X, Y)

X = torch.rand(4, 3, 512, 512)
Y = torch.rand(4, 3, 512, 512)