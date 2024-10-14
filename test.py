import argparse
import os
import math
from functools import partial
import torch.nn.functional as F
import SimpleITK as sitk

import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import structural_similarity
import datasets
import models
import utils
import lpips
def resampleSize(sitkImage, depth):
    euler3d = sitk.Euler3DTransform()

    xsize, ysize, zsize = sitkImage.GetSize()
    xspacing, yspacing, zspacing = sitkImage.GetSpacing()
    new_spacing_z = (zspacing*float(zsize))/(depth[2])
    new_spacing_x = (xspacing*float(xsize))/(depth[0])
    new_spacing_y = (yspacing*float(ysize))/(depth[1])

    origin = sitkImage.GetOrigin()
    direction = sitkImage.GetDirection()

    newsize = depth
    newspace = (new_spacing_x, new_spacing_y, new_spacing_z)
    sitkImage = sitk.Resample(sitkImage,newsize,euler3d,sitk.sitkBSpline3,origin,newspace,direction)
    ret = sitk.GetArrayFromImage(sitkImage)
    return ret
loss_vgg = lpips.LPIPS(net='vgg').cuda()
from utils import show_feature_map
def calc_lpips_vgg(img1, img2):
    shape = img1.shape
    ret = 0
    inp1 = torch.zeros((1,3,shape[0],shape[1],shape[2]))
    inp1[:,0,:,:,:] = img1
    inp1[:,1,:,:,:] = img1
    inp1[:,2,:,:,:] = img1
    inp2 = torch.zeros((1,3,shape[0],shape[1],shape[2]))
    inp2[:,0,:,:,:] = img2
    inp2[:,1,:,:,:] = img2
    inp2[:,2,:,:,:] = img2
    inp1 = inp1.cuda()
    inp2 = inp2.cuda()
    for i in range(shape[0]):
        ret+=loss_vgg(inp1[:,:,i,:,:], inp2[:,:,i,:,:]).item()
    
    for i in range(shape[1]):
        ret+=loss_vgg(inp1[:,:,:,i,:], inp2[:,:,:,i,:]).item()

    for i in range(shape[2]):
        ret+=loss_vgg(inp1[:,:,:,:,i], inp2[:,:,:,:,i]).item()
    return ret/(shape[0]+shape[1]+shape[2])
def calc_ssim(image, ground_truth):
    image = image.reshape(image.shape[-3:])
    ground_truth = ground_truth.reshape(ground_truth.shape[-3:])
    image = image.cpu().numpy()
    ground_truth = ground_truth.cpu().numpy()
    data_range = np.max(ground_truth) - np.min(ground_truth)
    return structural_similarity(image, ground_truth, data_range=data_range)

def batched_predict(model, inp, coord, cell, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[2]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, :, ql: qr, :, :], cell)
            ql = qr
            preds.append(pred)
        pred = torch.cat(preds, dim=3)
    return pred


def eval_psnr(loader, model, data_norm=None, scale=4, eval_type=None, eval_bsize=None, scale_max=4,
              verbose=False,mcell=False, name="a.nii.gz"):
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, 1, -1).cuda()

    if eval_type is None:
        metric_fn = utils.calc_psnr
        # metric_fn = utils.psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    elif eval_type.startswith('ixi'):
        scale = int(eval_type.split('-')[1])
        metric_fn = utils.calc_psnr
    else:
        raise NotImplementedError

    val_psnr = utils.Averager()
    val_ssim = utils.Averager()
    val_lpips = utils.Averager()
    pbar = tqdm(loader, leave=False, desc='val')
    cnt = 0
    
    #dataset = datasets.make({'name': 'ixi-paired','args':{'root_path_1': './data/IXI/IXI_HR/PD/test', 'root_path_2': './data/IXI/IXI_LR/bicubic_2x/PD/test'}})
    #dataset = datasets.make({'name': 'paired-image-folders','args':{'root_path_1': './data/benchmark/Set14/hr', 'root_path_2': './data/benchmark/Set14/image_SRF_2'}})
    for batch in pbar:
        cnt+=1
        
        # if cnt!=5712: continue

        for k, v in batch.items():
            batch[k] = v.cuda(non_blocking=True)

        inp = (batch['inp'] - inp_sub) / inp_div
        coord = batch['coord']
        cell = batch['cell']

        #x,y = dataset[0]
        #print((x.cuda()-batch['gt']).max(), (y.cuda()-batch['inp']).max())
        #print(y.max(), batch['inp'].max(), y.cuda().max()-batch['inp'].max())
        #break

        #print(coord.shape, cell)
        #print(scale)
        if mcell == False: 
            c = 1
        else : 
            c = max(scale/scale_max, 1)
        
        if eval_bsize is None:
            with torch.no_grad():
                pred = model(inp, coord, cell*c)
        else:
            pred = batched_predict(model, inp, coord, cell*c, eval_bsize)

        with torch.no_grad():
            pred = pred * gt_div + gt_sub
            pred.clamp_(0, 1)
            pred = pred.reshape(batch['gt'].shape[-3:])
            batch['gt'] = batch['gt'].reshape(batch['gt'].shape[-3:])
            # sitk.WriteImage(sitk.GetImageFromArray(pred),'./Data/result/'+str(cnt)+name)
            res = metric_fn(pred, batch['gt'])
            # res_ssim = calc_ssim(pred, batch['gt'])
            # res_lpips = calc_lpips_vgg(pred, batch['gt'])
        val_psnr.add(res.item(), inp.shape[0])
        # val_ssim.add(res_ssim, inp.shape[0])
        # val_lpips.add(res_lpips, inp.shape[0])
        if verbose:
            pbar.set_description('val {:.4f}'.format(val_psnr.item()))
        #print(scale)
        #print(pred.shape)
        #break
        # print(pred.min(), pred.max())
        # show_feature_map((pred - pred.min())/(pred.max() - pred.min()),'IXI/OUT/PD/60',name=str(cnt)+'_'+str(scale),rgb=False)
        # show_feature_map(batch['inp'],'IXI/OUT/PD/inp',name=str(cnt)+'_'+str(scale)+'_inp',rgb=False)
    return val_psnr,val_ssim,val_lpips


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='./configs/test_srno.yaml')
    parser.add_argument('--model')
    parser.add_argument('--scale_max', default='4')
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--mcell', default=False)
    args = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    #print(os.environ['CUDA_VISIBLE_DEVICES'])

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)
    spec = config['test_dataset']
    model_spec = torch.load(args.model)['model']
    print(model_spec['name'], model_spec['args'])
    model = models.make(model_spec, load_sd=True).cuda()
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        num_workers=8, pin_memory=True, shuffle=False)

    import time
    t1= time.time()
    val_psnr,val_ssim,val_lpips = eval_psnr(loader, model,
        scale = spec['wrapper']['args']['scale_max'], 
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        name = config.get('name'),
        scale_max = int(args.scale_max),
        verbose=True,
        mcell=bool(args.mcell))
    t2 =time.time()
    # print('result: {:.4f}'.format(res), utils.time_text(t2-t1))
    print("x"+str(spec['wrapper']['args']['scale_max'])+":","psnr:",val_psnr.item(),"ssim:",val_ssim.item(),"lpips_vgg:",val_lpips.item())
    
