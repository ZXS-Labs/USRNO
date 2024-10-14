import argparse
import os
import math
from functools import partial

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.image_folder import DRSRDataset

import datasets
import models
import utils

from utils import show_feature_map

def inference_net_eachDataset(test_path, model, dataset_name, scale, data_norm=None, mcell=False):

    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    # # 1. Load the best weight and create the dataloader for testing
    #print(dataset_name)
    testloader = DataLoader(DRSRDataset(test_path, scale, dataset_name),
                            batch_size=1)
    #model.load_state_dict(torch.load(net_Path))
    # 2. Compute the metrics
    metrics = torch.zeros(1, testloader.__len__())
    with torch.no_grad():
        model.eval()
        for i, (Depth, gt, D_min, D_max) in enumerate(testloader):
            Depth, gt, D_min, D_max = Depth.cuda(), gt.cuda(), D_min.cuda(), D_max.cuda()
            #print(Depth.shape, gt.shape)
            #print(D_min, D_max)
            inp = (Depth - inp_sub) / inp_div
            inp = inp.repeat(1,3,1,1)
            coord = utils.make_coord(gt.shape[2:], flatten=False).unsqueeze(0).cuda()
            cell = torch.tensor([2 / gt.shape[-2], 2 / gt.shape[-1]], dtype=torch.float32).unsqueeze(0).cuda()
            # print(gt.shape, inp.shape, coord.shape)

            if mcell == False: c = 1
            else: c = max(scale/4, 1)
            
            imgf_raw = model(inp, coord, cell*c)
            imgf_raw = (imgf_raw * gt_div + gt_sub).clamp_(0, 1)
            imgf = (imgf_raw * (D_max - D_min)) + D_min
            #print(inp.shape, gt.shape, gt.min(), gt.max(), imgf.shape, imgf.min(), imgf.max())
            
            if dataset_name == 'Middlebury' or dataset_name == 'Lu':
                imgf2image = imgf[0,2,:,:].clamp_(min=0, max=255)
                gt2image = gt[0,0,:,:].clamp_(min=0, max=255)
            elif dataset_name == 'NYU':
                #imgf2image = imgf[0,0, 6:-6, 6:-6]
                #gt2image = gt[0,0, 6:-6, 6:-6]
                imgf2image = imgf[0,2,:,:]
                gt2image = gt[0,0,:,:]
            else:
                imgf2image = imgf[0,0,:,:]
                gt2image = gt[0,0,:,:]
            #imgf2image, gt2image = (1/(imgf2image/100))*100, (1/(gt/100))*100
            #print(inp.shape, gt2image.shape, gt2image.min(), gt2image.max(), imgf2image.shape, imgf2image.min(), imgf2image.max())
            metrics[:, i] = utils.Rmse(imgf2image.cpu(), gt2image.cpu())

            #print(Depth.max(),Depth.min())
            # show_feature_map((imgf2image - gt2image).pow(2).unsqueeze(0).unsqueeze(0),'mid/OUT/pred',name=str(i)+'_'+str(scale),rgb=False,en=1)
            # show_feature_map((Depth * (D_max - D_min) + D_min).clamp_(min=0,max=255),'mid/OUT/inp',name=str(i)+'_'+str(scale)+'_inp',rgb=False,en=1)
            #break
        # print(metrics)
    return metrics.mean(dim=1).item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='./configs/test_srno-depth.yaml')
    parser.add_argument('--model')
    parser.add_argument('--scale_max', default='4')
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--mcell', default=False)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    #print(os.environ['CUDA_VISIBLE_DEVICES'])

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    #dataset = datasets.make(spec['dataset'])
    #dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    #loader = DataLoader(dataset, batch_size=spec['batch_size'],
    #    num_workers=8, pin_memory=True, shuffle=False)
    
    model_spec = torch.load(args.model)['model']
    print(model_spec['name'], model_spec['args'])
    model = models.make(model_spec, load_sd=True).cuda()
    
    dataset_name = config.get('eval_type').split('-')[0]
    scale = int(config.get('eval_type').split('-')[1])
    print(dataset_name, scale)

    import time
    t1= time.time()
    res = inference_net_eachDataset(spec['dataset']['args']['root_path'], model,
        data_norm=config.get('data_norm'),
        dataset_name=dataset_name,
        scale=scale,
        mcell=bool(args.mcell))
    t2 =time.time()
    print('x{} result: {:.4f}'.format(scale, res), utils.time_text(t2-t1))
    
