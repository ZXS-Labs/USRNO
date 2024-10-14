from operator import index
import random
import math
from torchvision.transforms import InterpolationMode
from scipy import ndimage as nd

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
from utils import make_coord
from utils import show_feature_map

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, InterpolationMode.BICUBIC)(
            transforms.ToPILImage()(img)))
@register('sr-implicit-downsampled-fast-ixi')
class SRImplicitDownsampledFast(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        img = img.squeeze(0)
        #print(idx)
        s = random.uniform(self.scale_min, self.scale_max)
        if self.inp_size is None:
            h_lr = math.floor(img.shape[-3] / s + 1e-9)
            w_lr = math.floor(img.shape[-2] / s + 1e-9)
            z_lr = math.floor(img.shape[-1] / s + 1e-9)
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            z_hr = round(z_lr * s)
            img = img[:h_hr, :w_hr, :z_hr] # assume round int
            img_down = F.grid_sample(img.unsqueeze(0).unsqueeze(0),make_coord([h_lr,w_lr,z_lr],flatten=False).flip(-1).unsqueeze(0),mode='bilinear',\
                                padding_mode='border', align_corners=False)
            crop_lr, crop_hr = img_down.squeeze(0), img
        else:
            h_lr = self.inp_size
            w_lr = self.inp_size
            z_lr = self.inp_size
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            z_hr = round(z_lr * s)
            x0 = random.randint(0, img.shape[-3] - w_hr)
            y0 = random.randint(0, img.shape[-2] - w_hr)
            z0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[x0: x0 + h_hr, y0: y0 + w_hr, z0: z0 + z_hr]
            crop_lr = F.grid_sample(crop_hr.unsqueeze(0).unsqueeze(0),make_coord([h_lr,w_lr,z_lr],flatten=False).flip(-1).unsqueeze(0),mode='bilinear',\
                                padding_mode='border', align_corners=False)
            crop_lr = crop_lr.squeeze(0)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-3)
                if vflip:
                    x = x.flip(-2)
                if dflip:
                    x = x.transpose(-3, -2)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord = make_coord([h_hr, w_hr, z_hr], flatten=False)
        hr_rgb = crop_hr

        if self.inp_size is not None:
            #hr_coord = hr_coord[torch.arange(0,h_hr,4), :, :][:, torch.arange(0,w_hr,4), :]
            #hr_rgb = hr_rgb[:, torch.arange(0,h_hr,4), :][:, :, (torch.arange(0,w_hr,4))]
            #print(hr_rgb.shape)
            #x0 = random.randint(0, hr_rgb.shape[-2] - w_lr)
            #y0 = random.randint(0, hr_rgb.shape[-1] - w_lr)
            #hr_rgb = hr_rgb[:, x0: x0 + w_lr, y0: y0 + w_lr]
            #hr_coord = hr_coord[x0: x0 + w_lr, y0: y0 + w_lr, :]
            
            idx = torch.tensor(np.random.choice(h_hr*w_hr*z_hr, h_lr*w_lr*z_lr, replace=False))
            hr_coord = hr_coord.view(-1, hr_coord.shape[-1])
            hr_coord = hr_coord[idx, :]
            hr_coord = hr_coord.view(h_lr, w_lr, z_lr, hr_coord.shape[-1])

            hr_rgb = crop_hr.contiguous().view(1, -1)
            hr_rgb = hr_rgb[:, idx]
            hr_rgb = hr_rgb.view(1, h_lr, w_lr, z_lr)
        
        cell = torch.tensor([2 / crop_hr.shape[-3], 2 / crop_hr.shape[-2], 2 / crop_hr.shape[-1]], dtype=torch.float32)
        #show_feature_map((hr_rgb-0.5)/0.5,'last',True)
        #show_feature_map((crop_lr-0.5)/0.5,'inp',True)
        #print(crop_hr.shape, crop_lr.shape)
        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }
@register('arssr_hcp_wapper')
class SRImplicitDownsampledFastHcp(Dataset):
    def __init__(self, dataset, inp_size=None):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        crop_lr, hr_coord, hr_rgb = self.dataset[idx]
        cell = torch.tensor([2 / hr_rgb.shape[-3], 2 / hr_rgb.shape[-2], 2 / hr_rgb.shape[-1]], dtype=torch.float32)
        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }
@register('sr-implicit-downsampled-fast-msd-lastdim')
class SRImplicitDownsampledFast(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        img = img.squeeze(0)
        #print(idx)
        s = random.uniform(self.scale_min, self.scale_max)
        if self.inp_size is None:
            h_lr = img.shape[-3]
            w_lr = img.shape[-2]
            h_hr = h_lr
            w_hr = w_lr 
            z_lr = math.floor(img.shape[-1] / s + 1e-9)
            z_hr = round(z_lr * s)
            img = img[:, :, :z_hr] # assume round int
            img_down = F.grid_sample(img.unsqueeze(0).unsqueeze(0),make_coord([h_lr, w_lr, z_lr],flatten=False).flip(-1).unsqueeze(0),mode='bilinear',\
                                padding_mode='border', align_corners=False)
            crop_lr, crop_hr = img_down.squeeze(0), img
        else:
            h_lr = self.inp_size
            w_lr = self.inp_size
            z_lr = self.inp_size
            h_hr = h_lr
            w_hr = w_lr 
            z_hr = round(z_lr * s)
            z0 = random.randint(0, img.shape[-1] - z_hr)
            x0 = random.randint(0, img.shape[-3] - x_hr)
            y0 = random.randint(0, img.shape[-2] - y_hr)
            crop_hr = img[:, :, z0: z0 + z_hr]
            crop_lr = F.grid_sample(crop_hr.unsqueeze(0).unsqueeze(0),make_coord([h_lr,w_lr,z_lr],flatten=False).flip(-1).unsqueeze(0),mode='bilinear',\
                                padding_mode='border', align_corners=False)
            crop_lr = crop_lr.squeeze(0)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-3)
                if vflip:
                    x = x.flip(-2)
                if dflip:
                    x = x.transpose(-3, -2)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord = make_coord([h_hr, w_hr, z_hr], flatten=False)
        hr_rgb = crop_hr

        if self.inp_size is not None:
            #hr_coord = hr_coord[torch.arange(0,h_hr,4), :, :][:, torch.arange(0,w_hr,4), :]
            #hr_rgb = hr_rgb[:, torch.arange(0,h_hr,4), :][:, :, (torch.arange(0,w_hr,4))]
            #print(hr_rgb.shape)
            #x0 = random.randint(0, hr_rgb.shape[-2] - w_lr)
            #y0 = random.randint(0, hr_rgb.shape[-1] - w_lr)
            #hr_rgb = hr_rgb[:, x0: x0 + w_lr, y0: y0 + w_lr]
            #hr_coord = hr_coord[x0: x0 + w_lr, y0: y0 + w_lr, :]
            
            idx = torch.tensor(np.random.choice(h_hr*w_hr*z_hr, h_hr*w_hr*z_lr, replace=False))
            hr_coord = hr_coord.view(-1, hr_coord.shape[-1])
            hr_coord = hr_coord[idx, :]
            hr_coord = hr_coord.view(h_lr, w_lr, z_lr, hr_coord.shape[-1])

            hr_rgb = crop_hr.contiguous().view(1, -1)
            hr_rgb = hr_rgb[:, idx]
            hr_rgb = hr_rgb.view(1, h_hr, h_hr, z_lr)
        
        cell = torch.tensor([2 / crop_hr.shape[-3], 2 / crop_hr.shape[-2], 2 / crop_hr.shape[-1]], dtype=torch.float32)
        #show_feature_map((hr_rgb-0.5)/0.5,'last',True)
        #show_feature_map((crop_lr-0.5)/0.5,'inp',True)
        #print(crop_hr.shape, crop_lr.shape)
        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }
@register('sr-implicit-downsampled-fast-msd')
class SRImplicitDownsampledFast(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        img = img.squeeze(0)
        #print(idx)
        s = random.uniform(self.scale_min, self.scale_max)
        if self.inp_size is None:
            h_lr = math.floor(img.shape[-3] / s + 1e-9)
            w_lr = math.floor(img.shape[-2] / s + 1e-9)
            z_lr = math.floor(img.shape[-1] / s + 1e-9)
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            z_hr = round(z_lr * s)
            img = img[:h_hr, :w_hr, :z_hr] # assume round int
            img_down = F.grid_sample(img.unsqueeze(0).unsqueeze(0),make_coord([h_lr,w_lr,z_lr],flatten=False).flip(-1).unsqueeze(0),mode='bilinear',\
                                padding_mode='border', align_corners=False)
            crop_lr, crop_hr = img_down.squeeze(0), img
        else:
            h_lr = self.inp_size
            w_lr = self.inp_size
            z_lr = self.inp_size
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            z_hr = round(z_lr * s)
            x0 = random.randint(0, img.shape[-3] - w_hr)
            y0 = random.randint(0, img.shape[-2] - w_hr)
            z0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[x0: x0 + h_hr, y0: y0 + w_hr, z0: z0 + z_hr]
            crop_lr = F.grid_sample(crop_hr.unsqueeze(0).unsqueeze(0),make_coord([h_lr,w_lr,z_lr],flatten=False).flip(-1).unsqueeze(0),mode='bilinear',\
                                padding_mode='border', align_corners=False)
            crop_lr = crop_lr.squeeze(0)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-3)
                if vflip:
                    x = x.flip(-2)
                if dflip:
                    x = x.transpose(-3, -2)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord = make_coord([h_hr, w_hr, z_hr], flatten=False)
        hr_rgb = crop_hr

        if self.inp_size is not None:
            #hr_coord = hr_coord[torch.arange(0,h_hr,4), :, :][:, torch.arange(0,w_hr,4), :]
            #hr_rgb = hr_rgb[:, torch.arange(0,h_hr,4), :][:, :, (torch.arange(0,w_hr,4))]
            #print(hr_rgb.shape)
            #x0 = random.randint(0, hr_rgb.shape[-2] - w_lr)
            #y0 = random.randint(0, hr_rgb.shape[-1] - w_lr)
            #hr_rgb = hr_rgb[:, x0: x0 + w_lr, y0: y0 + w_lr]
            #hr_coord = hr_coord[x0: x0 + w_lr, y0: y0 + w_lr, :]
            
            idx = torch.tensor(np.random.choice(h_hr*w_hr*z_hr, h_lr*w_lr*z_lr, replace=False))
            hr_coord = hr_coord.view(-1, hr_coord.shape[-1])
            hr_coord = hr_coord[idx, :]
            hr_coord = hr_coord.view(h_lr, w_lr, z_lr, hr_coord.shape[-1])

            hr_rgb = crop_hr.contiguous().view(1, -1)
            hr_rgb = hr_rgb[:, idx]
            hr_rgb = hr_rgb.view(1, h_lr, w_lr, z_lr)
        
        cell = torch.tensor([2 / crop_hr.shape[-3], 2 / crop_hr.shape[-2], 2 / crop_hr.shape[-1]], dtype=torch.float32)
        #show_feature_map((hr_rgb-0.5)/0.5,'last',True)
        #show_feature_map((crop_lr-0.5)/0.5,'inp',True)
        #print(crop_hr.shape, crop_lr.shape)
        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }

@register('sr-implicit-downsampled-fast-low-hcp')
class SRImplicitDownsampledFastLowHcp(Dataset):
    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        crop_lr = img[0]
        crop_hr = img[1]
        hr_coord = make_coord(crop_hr.shape, flatten=False)
        hr_rgb = crop_hr
        cell = torch.tensor([2 / crop_hr.shape[-3], 2 / crop_hr.shape[-2], 2 / crop_hr.shape[-1]], dtype=torch.float32)
        
        #show_feature_map((hr_rgb-0.5)/0.5,'last',True)
        #show_feature_map((crop_lr-0.5)/0.5,'inp',True)
        #print(crop_hr.shape, crop_lr.shape)
        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }

@register('sr-implicit-downsampled-fast-hcp')
class SRImplicitDownsampledFastHcp(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        # print(img.shape)
        #print(idx)
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-3] / s + 1e-9)
            w_lr = math.floor(img.shape[-2] / s + 1e-9)
            z_lr = math.floor(img.shape[-1] / s + 1e-9)
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            z_hr = round(z_lr * s)
            img = img[:h_hr, :w_hr, :z_hr] # assume round int
            # img_down = resize_fn(img, (h_lr, w_lr))
            img_down = torch.from_numpy(nd.interpolation.zoom(img, w_lr / w_hr, order=3))
            crop_lr, crop_hr = img_down, img
        else:
            h_lr = self.inp_size
            w_lr = self.inp_size
            z_lr = self.inp_size
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            z_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[-3] - w_hr)
            y0 = random.randint(0, img.shape[-2] - w_hr)
            z0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[x0: x0 + w_hr, y0: y0 + w_hr, z0: z0 + w_hr]
            crop_lr = torch.from_numpy(nd.interpolation.zoom(crop_hr, w_lr / w_hr, order=3))
            crop_lr = crop_lr.view(1,*crop_lr.shape)
        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-3)
                if vflip:
                    x = x.flip(-2)
                if dflip:
                    x = x.transpose(-3, -2)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord = make_coord(crop_hr.shape, flatten=False)
        hr_rgb = crop_hr

        if self.inp_size is not None:
            #hr_coord = hr_coord[torch.arange(0,h_hr,4), :, :][:, torch.arange(0,w_hr,4), :]
            #hr_rgb = hr_rgb[:, torch.arange(0,h_hr,4), :][:, :, (torch.arange(0,w_hr,4))]
            #print(hr_rgb.shape)
            #x0 = random.randint(0, hr_rgb.shape[-2] - w_lr)
            #y0 = random.randint(0, hr_rgb.shape[-1] - w_lr)
            #hr_rgb = hr_rgb[:, x0: x0 + w_lr, y0: y0 + w_lr]
            #hr_coord = hr_coord[x0: x0 + w_lr, y0: y0 + w_lr, :]
            
            idx = torch.tensor(np.random.choice(h_hr*w_hr*z_hr, h_lr*w_lr*z_lr, replace=False))
            #idx,_ = torch.sort(idx)
            hr_coord = hr_coord.view(-1, hr_coord.shape[-1])
            hr_coord = hr_coord[idx, :]
            hr_coord = hr_coord.view(h_lr, w_lr, z_lr, hr_coord.shape[-1])

            hr_rgb = crop_hr.contiguous().view(1, -1)
            hr_rgb = hr_rgb[:,idx]
            hr_rgb = hr_rgb.view(1, h_lr, w_lr, z_lr)
        
        cell = torch.tensor([2 / crop_hr.shape[-3], 2 / crop_hr.shape[-2], 2 / crop_hr.shape[-1]], dtype=torch.float32)
        
        #show_feature_map((hr_rgb-0.5)/0.5,'last',True)
        #show_feature_map((crop_lr-0.5)/0.5,'inp',True)
        #print(crop_hr.shape, crop_lr.shape)
        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }

@register('sr-implicit-downsampled-fast')
class SRImplicitDownsampledFast(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        #print(idx)
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            img = img[:, :h_hr, :w_hr] # assume round int
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        else:
            h_lr = self.inp_size
            w_lr = self.inp_size
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord = make_coord([h_hr, w_hr], flatten=False)
        hr_rgb = crop_hr

        if self.inp_size is not None:
            #hr_coord = hr_coord[torch.arange(0,h_hr,4), :, :][:, torch.arange(0,w_hr,4), :]
            #hr_rgb = hr_rgb[:, torch.arange(0,h_hr,4), :][:, :, (torch.arange(0,w_hr,4))]
            #print(hr_rgb.shape)
            #x0 = random.randint(0, hr_rgb.shape[-2] - w_lr)
            #y0 = random.randint(0, hr_rgb.shape[-1] - w_lr)
            #hr_rgb = hr_rgb[:, x0: x0 + w_lr, y0: y0 + w_lr]
            #hr_coord = hr_coord[x0: x0 + w_lr, y0: y0 + w_lr, :]
            
            idx = torch.tensor(np.random.choice(h_hr*w_hr, h_lr*w_lr, replace=False))
            #idx,_ = torch.sort(idx)
            hr_coord = hr_coord.view(-1, hr_coord.shape[-1])
            hr_coord = hr_coord[idx, :]
            hr_coord = hr_coord.view(h_lr, w_lr, hr_coord.shape[-1])

            hr_rgb = crop_hr.contiguous().view(crop_hr.shape[0], -1)
            hr_rgb = hr_rgb[:, idx]
            hr_rgb = hr_rgb.view(crop_hr.shape[0], h_lr, w_lr)
        
        cell = torch.tensor([2 / crop_hr.shape[-2], 2 / crop_hr.shape[-1]], dtype=torch.float32)
        
        #show_feature_map((hr_rgb-0.5)/0.5,'last',True)
        #show_feature_map((crop_lr-0.5)/0.5,'inp',True)
        #print(crop_hr.shape, crop_lr.shape)
        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }

@register('sr-implicit-paired-fast')
class SRImplicitPairedFast(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]
        #print(img_hr.shape, img_lr.shape)

        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]
            h_hr = s * h_lr
            w_hr = s * w_lr
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        #print(h_hr, w_hr, h_lr, w_lr)
        hr_coord = make_coord([h_hr, w_hr], flatten=False)
        hr_rgb = crop_hr
        
        if self.inp_size is not None:
            x0 = random.randint(0, h_hr - h_lr)
            y0 = random.randint(0, w_hr - w_lr)
            
            hr_coord = hr_coord[x0:x0+self.inp_size, y0:y0+self.inp_size, :]
            hr_rgb = crop_hr[:, x0:x0+self.inp_size, y0:y0+self.inp_size]
        
        cell = torch.tensor([2 / crop_hr.shape[-2], 2 / crop_hr.shape[-1]], dtype=torch.float32)

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }