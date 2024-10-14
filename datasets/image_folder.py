import os
import json
from PIL import Image
import random
import numpy as np
import torch
import h5py
from utils import make_coord
from torch.utils.data import Dataset
from torchvision import transforms
import SimpleITK as sitk
from datasets import register
from scipy import ndimage as nd
from tqdm import tqdm
@register('image-folder')
class ImageFolder(Dataset):

    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1):
        self.repeat = repeat

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)
            self.files.append(transforms.ToTensor()(
                Image.open(file).convert('RGB')))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]
        #print(x.shape,x.min(),x.max())
        return x

@register('arssr_hcp')
class ImgTrain(Dataset):
    def __init__(self, in_path_hr, is_train, sample_size=None):
        self.is_train = is_train
        if sample_size is not None:
            self.sample_size = sample_size
        self.patch_hr = []
        filenames = os.listdir(in_path_hr)
        for f in tqdm(filenames):
            img = sitk.ReadImage(os.path.join(in_path_hr, f))
            img_vol = sitk.GetArrayFromImage(img)
            self.patch_hr.append(img_vol)

    def __len__(self):
        return len(self.patch_hr)

    def __getitem__(self, item):
        patch_hr = self.patch_hr[item]
        # randomly get an up-sampling scale from [2, 4]
        s = np.round(random.uniform(2, 4 + 0.04), 1)
        # compute the size of HR patch according to the scale
        hr_h, hr_w, hr_d = (np.array([10, 10, 10]) * s).astype(int)
        # generate HR patch by cropping
        patch_hr = patch_hr[:hr_h, :hr_w, :hr_d]
        # simulated LR patch by down-sampling HR patch
        patch_lr = nd.interpolation.zoom(patch_hr, 1 / s, order=3)
        patch_lr = torch.from_numpy(patch_lr.astype(np.float32))
        patch_hr = torch.from_numpy(patch_hr.astype(np.float32))
        patch_lr = patch_lr.unsqueeze(0)
        # generate coordinate set
        xyz_hr = make_coord(patch_hr.shape, flatten=False)
        patch_hr = patch_hr.unsqueeze(0)
        # randomly sample voxel coordinates
        if self.is_train:
            xyz_hr = xyz_hr.view(-1, xyz_hr.shape[-1])
            sample_indices = torch.tensor(np.random.choice(len(xyz_hr), self.sample_size, replace=False))
            xyz_hr = xyz_hr[sample_indices,:]
            patch_hr = patch_hr.reshape(-1, 1)[sample_indices,:]

            xyz_hr = xyz_hr.view(20,20,20,-1)
            patch_hr = patch_hr.view(1,20,20,20)
        return patch_lr, xyz_hr, patch_hr


def loader_train(in_path_hr, batch_size, sample_size, is_train):
    """
    :param in_path_hr: the path of HR patches
    :param batch_size: N in Equ. 3
    :param sample_size: K in Equ. 3
    :param is_train:
    :return:
    """
    return data.DataLoader(
        dataset=ImgTrain(in_path_hr=in_path_hr, sample_size=sample_size, is_train=is_train),
        batch_size=batch_size,
        shuffle=is_train
    )

@register('image-folder-msd')
class ImageFolder_msd(Dataset):

    def __init__(self, root_path,repeat=1):
        self.repeat = repeat

        filenames = sorted(os.listdir(root_path))

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)
            #print(file)
            x = torch.from_numpy((sitk.GetArrayFromImage(sitk.ReadImage(file)).astype(np.float32)))
            minx = x.min()
            maxx = x.max()
            x = (x-minx)/(maxx - minx)
            #print(maxx)
            self.files.append(x.permute(1,2,0))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)].unsqueeze(0)
        #print(x.shape)
        return x
@register('image-folder-low-hcp')
class ImageFolder_low_hcp(Dataset):
    def __init__(self, root_path,repeat=1):
        self.repeat = repeat

        filenames = sorted(os.listdir(root_path))

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)
            #print(file)
            x = torch.from_numpy(h5py.File(file,'r')['data'][()])
            y = torch.from_numpy(h5py.File(file,'r')['target'][()])
            minx = y.min()
            maxx = y.max()
            x = (x-minx)/(maxx - minx)
            y = (y-minx)/(maxx - minx)
            #print(maxx)
            self.files.append((x,y))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        # x = self.files[idx // 256][:, :, idx % 256].unsqueeze(0)
        x = self.files[idx % len(self.files)]
        #print(x.shape)
        return x
@register('image-folder-hcp')
class ImageFolder_hcp(Dataset):

    def __init__(self, root_path,repeat=1):
        self.repeat = repeat

        filenames = sorted(os.listdir(root_path))

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)
            #print(file)

            x = torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(file)).astype(np.float32))
            minx = x.min()
            maxx = x.max()
            x = (x-minx)/(maxx - minx)
            img_cropped = x[:264,:264,:]
            x = np.pad(img_cropped,((0,0),(0,0),(4,4)),'constant')
            #print(maxx)
            self.files.append(x)

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        # x = self.files[idx // 256][:, :, idx % 256].unsqueeze(0)
        x = self.files[idx % len(self.files)]
        #print(x.shape)
        return x

@register('image-folder-ixi')
class ImageFolder_x(Dataset):

    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1):
        self.repeat = repeat

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)
            #print(file)
            x = torch.from_numpy(np.load(file).astype(np.float32))
            minx = x.min()
            maxx = x.max()
            x = (x-minx)/(maxx - minx)
            #print(maxx)
            self.files.append(x)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = self.files[idx]#240 240 96
        return x

def get_img_file(file_name):
    imagelist = []
    for parent, dirnames, filenames in os.walk(file_name):
        for filename in filenames:
            if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff', '.npy')):
                imagelist.append(os.path.join(parent, filename))
            #print(imagelist)
        return imagelist

@register('paired-image-folders-depth')    
class DRSRDataset(Dataset):
    def __init__(self, path, scale, dataset_name):
        self.scale = scale
        self.path = path
        self.DepthHR_files = sorted(get_img_file(path+'/DepthHR'))
        self.DepthLR_files = sorted(get_img_file(path+'/DepthLr'))
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.DepthHR_files)

    def __getitem__(self, index):
        Depth = np.load(self.DepthLR_files[index])
        if self.dataset_name == 'NYU' or self.dataset_name == 'RGBDD':
            GT = np.load(self.DepthHR_files[index])*100
        elif self.dataset_name == 'Middlebury' or self.dataset_name == 'Lu':
            GT = np.load(self.DepthHR_files[index])
        D_min = GT.min()
        D_max = GT.max()
        return torch.Tensor(Depth), torch.Tensor(GT), torch.tensor(D_min), torch.tensor(D_max)


@register('paired-image-folders-ixi')
class PairedImageFolders_x(Dataset):
    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = ImageFolder_x(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder_x(root_path_2, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]


@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]
