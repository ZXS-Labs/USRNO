import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

import models
from models.galerkin import simple_attn
from models import register
from utils import make_coord

def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            m.weight.uniform_(64/(np.pi*2), 128/(np.pi*2))

@register('srno')
class SRNO(nn.Module):

    def __init__(self, encoder_spec, width=256, blocks=16 , mod = False):
        super().__init__()
        self.width = width
        self.mod = mod
        if mod: 
            self.modm = nn.Conv2d(2, 64, 1)
            self.modm.apply(sine_init)
        self.encoder = models.make(encoder_spec)
        #self.conv00 = nn.Conv2d(64, 64, 3, padding = 1, groups = 64)
        self.conv00 = nn.Conv2d((64 + 2)*4+2+(64*4), self.width, 1)
        #self.fc0 = nn.Conv2d(6, self.width, 1) ## 8 -> 6
        #self.pe = nn.Conv2d(self.width, self.width, 7, padding = 3, groups = 256)

        self.conv0 = simple_attn(self.width, blocks)
        self.conv1 = simple_attn(self.width, blocks)
        #self.conv2 = simple_attn(self.width, blocks)
        #self.conv3 = simple_attn(self.width, blocks)

        
        self.fc1 = nn.Conv2d(self.width, 256, 1)
        self.fc2 = nn.Conv2d(256, 3, 1)
        #self.convs = nn.ModuleList([self.conv0, self.conv1])
        
    def gen_feat(self, inp):
        self.inp = inp
        self.feat = self.encoder(inp)
        return self.feat
        
    def query_rgb(self, coord, cell):      
        feat = (self.feat)
        grid = 0

        # prepare meta-data (coordinate)
        pos_lr = make_coord(feat.shape[-2:], flatten=False).cuda()\
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        rel_coords = []
        feat_s = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:

                coord_ = coord.clone()
                coord_[:, :, :, 0] += vx * rx + eps_shift
                coord_[:, :, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                feat_ = F.grid_sample(feat, coord_.flip(-1), mode='nearest', align_corners=False)

                # rel coord
                old_coord = F.grid_sample(pos_lr, coord_.flip(-1), mode='nearest', align_corners=False)
                rel_coord = coord.permute(0, 3, 1, 2) - old_coord
                rel_coord[:, 0, :, :] *= feat.shape[-2]
                rel_coord[:, 1, :, :] *= feat.shape[-1]

                # area
                area = torch.abs(rel_coord[:, 0, :, :] * rel_coord[:, 1, :, :])
                areas.append(area + 1e-9)

                rel_coords.append(rel_coord)
                feat_s.append(feat_)
                
        # cell
        rel_cell = cell.clone()
        rel_cell[:,0] *= feat.shape[-2]
        rel_cell[:,1] *= feat.shape[-1]

        # apply local ensemble
        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t

        feat_s_mod = []
        for index, area in enumerate(areas):
            feat_s[index] = feat_s[index] * (area / tot_area).unsqueeze(1)
            feat_s_mod.append(feat_s[index] * torch.sin(np.pi*2*self.modm(rel_coords[index])))

         
        grid = torch.cat([*rel_coords, *feat_s, *feat_s_mod,\
            rel_cell.unsqueeze(-1).unsqueeze(-1).repeat(1,1,coord.shape[1],coord.shape[2])],dim=1)

        x = self.conv00(grid) 
        x = self.conv0(x, 0)
        #x = x + self.posEncoding(coord, self.width, 2/cell)
        x = self.conv1(x, 1)
        #x = x + self.pe

        feat = x #+ bias
        ret = self.fc2(F.gelu(self.fc1(feat)))
        
        #show_feature_map(ret.permute(0,2,3,1) * 0.5 + 0.5, 'sr/edsr', rgb = True)
        #show_feature_map(ret * 0.5 + 0.5, 'sr/edsr',name = 'lres', rgb = False)

        ret = ret + F.grid_sample(self.inp, coord.flip(-1), mode='bilinear',\
                                padding_mode='border', align_corners=False)
        #show_feature_map(F.grid_sample(self.inp, coord.flip(-1), mode='bilinear',\
        #                        padding_mode='border', align_corners=False).permute(0,2,3,1) *0.5 +0.5, 'sr/edsr', rgb=True)
        #show_feature_map(ret+F.grid_sample(self.inp, coord.flip(-1), mode='bilinear',\
        #                        padding_mode='border', align_corners=False),'last',True)
        return ret #kornia.filters.median_blur(ret, (3,3))

    def forward(self, inp, coord = torch.FloatTensor(1, 128,128,2), cell=torch.FloatTensor(1, 2,)):
        #coord = coord.cuda()
        #cell = cell.cuda()
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)