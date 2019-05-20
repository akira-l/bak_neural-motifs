import os, sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

class residual(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(residual, self).__init__()

        self.conv1 = nn.Conv2d(inp_dim, out_dim, (3, 3), padding=(1, 1), stride=(stride, stride), bias=False)
        self.bn1   = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), padding=(1, 1), bias=False)
        self.bn2   = nn.BatchNorm2d(out_dim)
        
        self.skip  = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
            nn.BatchNorm2d(out_dim)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2   = self.bn2(conv2)

        skip  = self.skip(x)
        return self.relu(bn2 + skip)


class residual_up(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=2, with_bn=True):
        super(residual_up, self).__init__()

        self.upconv1 = nn.ConvTranspose2d(inp_dim, inp_dim, (3, 3), padding=1, output_padding=1, 
                                          stride=(stride, stride), bias=True, 
                                          dilation=1 if stride<=2 else stride // 2)
        self.upconv2 = nn.ConvTranspose2d(inp_dim, out_dim, (3, 3), padding=1, output_padding=1,
                                          stride=(stride, stride), bias=False,
                                          dilation=1 if stride<=2 else stride // 2)
        self.bn1     = nn.BatchNorm2d(out_dim)
        self.relu1   = nn.ReLU(inplace=True)


        self.upconv3 = nn.ConvTranspose2d(out_dim, out_dim, (3, 3),  padding=1, output_padding=1,
                                          stride=(stride, stride), bias=True, 
                                          dilation=1 if stride<=2 else stride // 2)
        self.upconv4 = nn.ConvTranspose2d(out_dim, out_dim, (3, 3),  padding=1, output_padding=1,
                                          stride=(stride, stride), bias=False, 
                                          dilation=1 if stride<=2 else stride // 2)
        self.bn2     = nn.BatchNorm2d(out_dim)
        
        self.skip  = nn.Sequential(
            nn.ConvTranspose2d(inp_dim, out_dim, (3, 3), padding=1, output_padding=1, 
                                stride=stride**4, bias=False, 
                                dilation=stride**4//2),
            nn.BatchNorm2d(out_dim)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.upconv1(x)
        conv2 = self.upconv2(conv1)
        bn1   = self.bn1(conv2)
        relu1 = self.relu1(bn1)

        conv3 = self.upconv3(relu1)
        conv4 = self.upconv4(conv3)
        bn2   = self.bn2(conv4)

        skip  = self.skip(x)
        return self.relu(bn2 + skip)


class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=2, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn   = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn   = self.bn(conv)
        relu = self.relu(bn)
        return relu


class fully_connected(nn.Module):
    def __init__(self, inp_dim, out_dim, with_bn=True):
        super(fully_connected, self).__init__()
        self.with_bn = with_bn

        self.linear = nn.Linear(inp_dim, out_dim)
        if self.with_bn:
            self.bn = nn.BatchNorm1d(out_dim)
        self.relu   = nn.ReLU(inplace=True)

    def forward(self, x):
        linear = self.linear(x)
        bn     = self.bn(linear) if self.with_bn else linear
        relu   = self.relu(bn)
        return relu

class offset_regr(nn.Module):
    def __init__(self, inp_dim, stride, resolution):
        super(offset_regr, self).__init__()
        self.buffer_conv1 = nn.Conv2d(inp_dim, 100, (3, 3), padding=(1, 1), stride=(stride, stride), bias=False)
        self.buffer_conv2 = nn.Conv2d(100, 20, (3, 3), padding=(1, 1), stride=(stride, stride), bias=False)

        self.linear = nn.Linear(20*resolution, resolution)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        conv_x = self.buffer_conv1(x)
        conv_x = self.buffer_conv2(conv_x)
        
        linear_x = self.linear(conv_x.view(conv_x.size(0), -1))
        return self.relu(linear_x)



def make_layer(k, inp_dim, out_dim, modules, layer=convolution, **kwargs):
    layers = [layer(k, inp_dim, out_dim, **kwargs)]
    for _ in range(1, modules):
        layers.append(layer(k, out_dim, out_dim, **kwargs))
    return nn.Sequential(*layers)



class BilinearModel(nn.Module):
    def __init__(self):
        super(BilinearModel, self).__init__()
        self.glo_conv1 = nn.Conv2d(in_channels=512, out_channels=256,
                                   kernel_size=3, stride=2, padding=1,
                                   dilation=1, bias=True)
        self.glo_conv2 = nn.Conv2d(in_channels=256, out_channels=256,
                                   kernel_size=3, stride=2, padding=1,
                                   dilation=1, bias=True)
        self.glo_conv3 = nn.Conv2d(in_channels=256, out_channels=256,
                                   kernel_size=3, stride=2, padding=1,
                                   dilation=1, bias=True)

        glo_feat_size = 6400
        self.rpn_fc = nn.Linear(4096, 4096)
        self.glo_fc = nn.Linear(glo_feat_size, 4096)

    def forward(self, rpn_ind, rpn_feat, glo_feat):
        batch_size = glo_feat.size(0)
        glo_trans = F.relu(self.glo_conv1(glo_feat))
        glo_trans = F.relu(self.glo_conv2(glo_trans))
        glo_trans = F.relu(self.glo_conv3(glo_trans))
        glo_bi_feat = self.glo_fc(glo_trans.view(batch_size, -1))
        rpn_bi_list = [self.rpn_fc(x) for x in rpn_feat]
        bi_feat_list = []
        rpn_counter = 0
        img_counter = 0
        for rpn_bi in rpn_bi_list:
            if rpn_ind[rpn_counter] != img_counter:
                img_counter += 1
            assert rpn_ind[rpn_counter] == img_counter
            rpn_bi_ = rpn_bi.unsqueeze(0).unsqueeze(0)
            bi_feat = torch.matmul(rpn_bi_.transpose(1,2).unsqueeze(3), 
                                 glo_bi_feat[img_counter].unsqueeze(0).unsqueeze(0).transpose(1,2).unsqueeze(2))
            bi_feat_list.append(bi_feat)
            rpn_counter += 1
        return torch.cat(bi_feat_list, 0)
        

class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()

    def forward(self, rpn_feat, glo_feat):
        pass


class CornerDecode(nn.Module):
    def __init__(self, **kwargs):
        super(CornerDecode, self).__init__()
        cur_dim = 4096
        out_dim = 512
        out_class = 50
        cur_mod = 2
        self.up_layer = make_layer(3, cur_dim, out_dim, cur_mod, layer=residual_up, **kwargs)
        self.tl_layer = make_layer(3, out_dim, out_class, cur_mod, layer=convolution, **kwargs)
        self.br_layer = make_layer(3, out_dim, out_class, cur_mod, layer=convolution, **kwargs)
        self.tl_offset_layer = offset_regr(out_dim, stride=2, resolution=64*64)
        self.br_offset_layer = offset_regr(out_dim, stride=2, resolution=64*64)
        self.tl_tag_layer = make_layer(3, out_dim, 1, cur_mod, layer=convolution, **kwargs)
        self.br_tag_layer = make_layer(3, out_dim, 1, cur_mod, layer=convolution, **kwargs)

    def forward(self, att_feat):
        decode_feat = self.up_layer(att_feat)
        pdb.set_trace()
        tl_heatmap = self.tl_layer(decode_feat)
        br_heatmap = self.br_layer(decode_feat)
        tl_tag = self.tl_tag_layer(decode_feat)
        br_tag = self.br_tag_layer(decode_feat)
        tl_offset = self.tl_offset_layer(decode_feat)
        br_offset = self.br_offset_layer(decode_feat)
        return decode_feat, tl_heatmap, br_heatmap, tl_tag, br_tag, tl_offset, br_offset


class AttModel(nn.Module):
    def __init__(self, model='bilinear'):
        super(AttModel, self).__init__()
        # model list: bilinear, transformer
        self.att_encode = BilinearModel() if model=='bilinear' else TransformerModel()
        self.corner_decode = CornerDecode()


    def forward(self, rpn_ind, rpn_feat, global_feat):
        att_feat = self.att_encode(rpn_ind, rpn_feat, global_feat)
        corner_feat, tl_heatmap, br_heatmap, tl_tag, br_tag, tl_offset, br_offset = self.corner_decode(att_feat)
        return corner_feat, tl_heatmap, br_heatmap, tl_tag, br_tag, tl_offset, br_offset


class bilinear_model(nn.Module):
    def __init__(self, triplet_size):
        super(bilinear_model, self).__init__()
        self.fc_outsize = triplet_size
 
        self.bn_conv = nn.BatchNorm3d(num_features=30,
                                  track_running_stats=True)
        self.conv1 = nn.Conv2d(in_channels=512,
                                out_channels=256,
                                kernel_size=(3,3),
                                stride=(2,2),
                                padding=(1,1),
                                dilation=(1,1),
                                bias=True)
        self.conv2 = nn.Conv2d(in_channels=256,
                                out_channels=128,
                                kernel_size=(3,3),
                                stride=(2,2),
                                padding=(1,1),
                                dilation=(1,1),
                                bias=True)
 
        self.conv_fc = nn.Linear(128*4, 64)
        self.dropout = nn.Dropout(p=0.5)
 
 
        self.fc_insize = 64*30*30
        self.cls_dropout = nn.Dropout(p=0.3)
        self.cls_fc = nn.Linear(self.fc_insize, self.fc_outsize)
 
        self.obj_dropout = nn.Dropout(p=0.3)
        self.obj_fc = nn.Linear(self.fc_insize, 400)
 
        self.rela_dropout = nn.Dropout(p=0.3)
        self.rela_fc = nn.Linear(self.fc_insize, 90)
 
    def forward(self, inputs, save_flag=False):
        in_size = torch.tensor(inputs.size()).tolist()
        re_inputs = inputs.view(in_size[0], in_size[1], in_size[2], 7, 7)
        bn_inputs = self.bn_conv(re_inputs)
        re_inputs = bn_inputs.view(in_size[0]*in_size[1], in_size[2], 7, 7)
        conv1_out = F.relu(self.conv1(re_inputs))
        conv2_out = F.relu(self.conv2(conv1_out))
 
        refc_in = conv2_out.view(in_size[0], in_size[1], -1)
        in_1 = self.dropout(self.conv_fc(refc_in))
        in_2 = in_1
        bin_op = torch.matmul(in_1.transpose(1,2).unsqueeze(3), in_2.transpose(1,2).unsqueeze(2))
        tri_in = bin_op.view(in_size[0], -1)
        tri_out = self.cls_dropout(self.cls_fc(tri_in))
        rela_out = self.obj_dropout(self.rela_fc(tri_in))
        obj_out = self.rela_dropout(self.obj_fc(tri_in))

        if save_flag:
            return tri_in, cls_out
        return tri_out, obj_out, rela_out


if __name__ == '__main__':
    att_model = AttModel()
    rpn_feat = torch.rand(4, 4096)
    overall_feat = torch.rand(2, 512, 37, 37)
    rpn_ind = torch.Tensor([0,1,1,1])
    tmp = att_model(rpn_ind, rpn_feat, overall_feat)