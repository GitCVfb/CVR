import sys
import random
import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from image_proc import *

from package_core.net_basics import *
from package_core.model_base import *
from package_core.losses import *
from package_core.flow_utils import *
from package_core.image_proc import *
from reblur_package import *

from forward_warp_package import *
import softsplat

from net_scale import *
from net_pwc import *
from net_rssr import *

from Convert_m_t import *

backwarp_tenGrid = {}
def backwarp(tenInput, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
        tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])

        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()
        # end

    tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

    return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)
# end

class ModelCVR(ModelBase):
    def __init__(self, opts):
        super(ModelCVR, self).__init__()
        
        self.opts = opts
        
        # create networks
        self.model_names=['flow', 'scale', 'syn']
        self.net_flow = PWCDCNet().cuda()
        self.net_scale = UNet_Esti_Scale().cuda()
        self.net_syn = UNet(20, 5).cuda()
        
        # load in initialized network parameters
        if not opts.is_training or opts.continue_train:
            self.load_checkpoint(opts.model_label)#load CVR or CVR* model
        else:
            self.load_checkpoint_of_RSSR(opts.model_label_rssr)#load pretrained RSSR model (cf. ICCV21)
        
        self.upsampleX4 = nn.Upsample(scale_factor=4, mode='bilinear')
        
        if self.opts.is_training:
            # initialize optimizers
            
            if self.opts.model_type == 'CVR':
                self.optimizer_G = torch.optim.Adam([
                    {'params': self.net_flow.parameters(), 'lr': 1e-5},
                    {'params': self.net_scale.parameters(), 'lr': 1e-5},
                    {'params': self.net_syn.parameters()}], lr=opts.lr)
            else:#self.opts.model_type == 'CVR*'
                self.optimizer_G = torch.optim.Adam([
                    {'params': self.net_flow.parameters(), 'lr': 1e-5},
                    {'params': self.net_syn.parameters()}], lr=opts.lr) 
            
            
            self.optimizer_names = ['G']
            self.build_lr_scheduler()
            
            # create losses
            self.loss_fn_perceptual = PerceptualLoss(loss=nn.L1Loss())
            self.loss_fn_L1 = L1Loss()
            self.loss_fn_tv_2C = VariationLoss(nc=2)
            
            ###Initializing VGG16 model for perceptual loss
            self.MSE_LossFn = nn.MSELoss()
            vgg16 = torchvision.models.vgg16(pretrained=True)
            self.vgg16_conv_4_3 = nn.Sequential(*list(vgg16.children())[0][:22])
            self.vgg16_conv_4_3.to('cuda')
            for param in self.vgg16_conv_4_3.parameters():
		            param.requires_grad = False

    def set_input(self, _input):
        im_rs, im_gs, gt_flow, im_gs_f = _input
        self.im_rs = im_rs.cuda()
        self.im_gs = im_gs
        self.gt_flow = gt_flow
        self.im_gs_f = im_gs_f

        if self.im_gs is not None:
            self.im_gs = self.im_gs.cuda()
        if self.im_gs_f is not None:
            self.im_gs_f = self.im_gs_f.cuda()
        if self.gt_flow is not None:
            self.gt_flow = self.gt_flow.cuda()

    def forward(self, tt):# tt is in [0, 1]
        
        im_rs0 = self.im_rs[:,0:3,:,:].clone()
        im_rs1 = self.im_rs[:,3:6,:,:].clone()

        B,_,H,W = self.im_rs.size()
        self.mask_fn = FlowWarpMask.create_with_implicit_mesh(B, 3, H, W)
        
        grid_rows = self.generate_2D_grid(H, W)[1]
        t_flow_offset = grid_rows.unsqueeze(0).unsqueeze(0)
        self.I0_t_flow_ref_to_m = -(t_flow_offset-H//2+0.0001)/(H//2)
        self.I1_t_flow_ref_to_m = -self.I0_t_flow_ref_to_m
        
        Fs_0_1=self.net_flow(im_rs0, im_rs1)
        Fs_1_0=self.net_flow(im_rs1, im_rs0)
        
        F_0_1 = self.upsampleX4(Fs_0_1[0])*20.0
        F_1_0 = self.upsampleX4(Fs_1_0[0])*20.0

        g_I1_F_1_0, _ = warp_image_flow(im_rs0, F_1_0)
        g_I0_F_0_1, _ = warp_image_flow(im_rs1, F_0_1)
        
        if self.opts.model_type == 'CVR':
            #Codes from lines 130 to 138 are used to implement CVR (NBMF-based)
            interp_out_var = self.net_scale(im_rs0, F_0_1, im_rs1, F_1_0)
            interp_out_var_0 = interp_out_var[:,0,:,:]
            interp_out_var_1 = interp_out_var[:,1,:,:]
        
            F_0_1 = F_0_1 + interp_out_var[:,2:4,:,:]
            F_1_0 = F_1_0 + interp_out_var[:,4:6,:,:]
        
            T_0_m = self.I0_t_flow_ref_to_m * interp_out_var_0.unsqueeze(1)
            T_1_m = self.I1_t_flow_ref_to_m * interp_out_var_1.unsqueeze(1)
        else:#self.opts.model_type == 'CVR*'
            #Codes from lines 141 to 142 are used to implement CVR* (ABMF-based)
            T_0_m = 0.5 * self.I0_t_flow_ref_to_m
            T_1_m = 0.5 * self.I1_t_flow_ref_to_m
        
        #######   #######   #######
        #GS image synthesis corresponding to time tt (tt in [0,1])
        gs_t, Fs_t = self.convert_m2t(self.im_rs, [F_0_1, F_1_0], [T_0_m, T_1_m], tt)#convert to time tt
        g_It_F_0_t = gs_t[0] #Initial intermediate GS frame candidates
        g_It_F_1_t = gs_t[1]
        F_0_t = Fs_t[0] #Initial bilateral motion field (BMF) corresponding to time tt
        F_1_t = Fs_t[1]

        X_in = torch.cat((im_rs0, im_rs1, F_0_1, F_1_0, F_0_t, F_1_t, g_It_F_0_t, g_It_F_1_t), dim=1)
        intrpOut = self.net_syn(X_in) #GS frame synthesis module
        
        delta_F_0_t = intrpOut[:, 0:2, :, :] #BMF residuals
        delta_F_1_t = intrpOut[:, 2:4, :, :]
        
        F_0_t_final = F_0_t + delta_F_0_t #Enhanced BMF (motion enhancement layer)
        F_1_t_final = F_1_t + delta_F_1_t

        V_0_t  = torch.sigmoid(intrpOut[:, 4:5, :, :]) #Time-aware bilateral occlusion mask (contextual aggregation layer)
        V_1_t  = 1.0 - V_0_t
        
        #Forward warping
        tenMetric_0_1 = torch.nn.functional.l1_loss(input=im_rs0, target=backwarp(tenInput=im_rs1, tenFlow=F_0_1), reduction='none').mean(1, True)
        tenMetric_1_0 = torch.nn.functional.l1_loss(input=im_rs1, target=backwarp(tenInput=im_rs0, tenFlow=F_1_0), reduction='none').mean(1, True)
        g_I0t = softsplat.FunctionSoftsplat(tenInput=im_rs0, tenFlow=F_0_t_final, tenMetric=-20.0 * tenMetric_0_1, strType='softmax') #Refined intermediate GS frame candidates
        g_I1t = softsplat.FunctionSoftsplat(tenInput=im_rs1, tenFlow=F_1_t_final, tenMetric=-20.0 * tenMetric_1_0, strType='softmax')
        
        t = (1 - 2e-2) * tt + 1e-2 #add a small time perturbation
        gs_t_final = ((1-t) * V_0_t * g_I0t + t * V_1_t * g_I1t) / ((1-t) * V_0_t + t * V_1_t)# Generate final intermediate GS frames at time tt

        out_mask=0
        out_image=[g_I1_F_1_0, g_I0_F_0_1, g_It_F_0_t, g_It_F_1_t, g_I0t, g_I1t]
        out_visible=[(1-t)*V_0_t/((1-t)*V_0_t+t*V_1_t), t*V_1_t/((1-t)*V_0_t+t*V_1_t)]
        out_flow=[F_0_1, F_1_0, F_0_t_final, F_1_t_final, delta_F_0_t, delta_F_1_t]

        if self.opts.is_training:
            gs_final_0, out_image_0, out_flow_0 = self.synthese_img_t(im_rs0, im_rs1, F_0_1, F_1_0, T_0_m, T_1_m, 0)
            gs_final_1, out_image_1, out_flow_1 = self.synthese_img_t(im_rs0, im_rs1, F_0_1, F_1_0, T_0_m, T_1_m, 1)
            
            return [gs_final_0, gs_t_final, gs_final_1], \
                   [out_image_0, out_image, out_image_1], \
                   out_mask, \
                   [out_flow_0, out_flow, out_flow_1]
        
        return gs_t_final, out_image, out_mask, out_flow, out_visible


    def optimize_parameters(self):
        img_rs_0 = self.im_rs[:,0:3,:,:].clone()
        img_rs_1 = self.im_rs[:,3:6,:,:].clone()
        im_gs_0 = self.im_gs[:,0:3,:,:].clone()
        im_gs_1 = self.im_gs[:,3:6,:,:].clone()
        im_gs_f_0 = self.im_gs_f[:,0:3,:,:].clone()
        im_gs_f_1 = self.im_gs_f[:,3:6,:,:].clone()
        
        img_gs_gt = [im_gs_0, im_gs_f_1, im_gs_1] #GT GS at times 0, 0.5, 1, respectively

        #===========================================================#
        #                   Initialize losses                       #
        #===========================================================#
        self.loss_L1_gs = torch.tensor([0.], requires_grad=True).cuda().float()
        self.loss_perceptual = torch.tensor([0.], requires_grad=True).cuda().float()
        self.loss_flow_smoothness = torch.tensor([0.], requires_grad=True).cuda().float()
        self.loss_L1_ccl = torch.tensor([0.], requires_grad=True).cuda().float()
        #===========================================================#
        #                       Compute losses                      #
        #===========================================================#
        self.pred_gs, self.pred_im, self.pred_mask, self.pred_flow = self.forward(0.5)
        
        for lv in range(3):
            self.loss_L1_gs += self.opts.lamda_L1 *\
                                self.loss_fn_L1(self.pred_gs[lv], img_gs_gt[lv], mean=True)
            self.loss_perceptual += self.opts.lamda_perceptual *\
                                self.loss_fn_perceptual.get_loss(self.pred_gs[lv], img_gs_gt[lv])

        self.loss_L1_ccl += self.opts.lamda_L1_ccl * self.loss_fn_L1(self.pred_im[1][4], img_gs_gt[1], mean=True)
        self.loss_L1_ccl += self.opts.lamda_L1_ccl * self.loss_fn_L1(self.pred_im[1][5], img_gs_gt[1], mean=True)
        for lv in range(0,2,3):
            self.loss_L1_ccl += self.opts.lamda_L1_ccl * self.loss_fn_L1(self.pred_im[lv][2], img_gs_gt[lv], mean=True)
            self.loss_L1_ccl += self.opts.lamda_L1_ccl * self.loss_fn_L1(self.pred_im[lv][3], img_gs_gt[lv], mean=True)
        
        if self.pred_flow is not None and self.opts.lamda_flow_smoothness>1e-6:
            for lv in range(4):
                self.loss_flow_smoothness += self.opts.lamda_flow_smoothness * self.loss_fn_tv_2C(self.pred_flow[1][lv], mean=True)
            for lv_ in range(0,2,3):
                for lv in range(2):
                    self.loss_flow_smoothness += self.opts.lamda_flow_smoothness * self.loss_fn_tv_2C(self.pred_flow[lv_][lv], mean=True)
        
        self.loss_L1_gs = self.loss_L1_gs /3.0
        self.loss_perceptual = self.loss_perceptual /3.0
        self.loss_L1_ccl = self.loss_L1_ccl /6.0
        self.loss_flow_smoothness = self.loss_flow_smoothness /6.0

        # sum them up
        self.loss_G = self.loss_L1_gs +\
                        self.loss_perceptual +\
                        self.loss_L1_ccl +\
                        self.loss_flow_smoothness

        # Optimize
        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step() 

    # save networks to file 
    def save_checkpoint(self, label):
        self.save_network(self.net_flow, 'flow', label, self.opts.log_dir)
        if self.opts.model_type == 'CVR':
            self.save_network(self.net_scale, 'scale',  label, self.opts.log_dir)
        self.save_network(self.net_syn,  'syn',  label, self.opts.log_dir)

    def load_checkpoint_of_RSSR(self, label):
        self.load_network(self.net_flow, 'flow', label, self.opts.log_dir_rssr)
        if self.opts.model_type == 'CVR':
            self.load_network(self.net_scale, 'scale',  label, self.opts.log_dir_rssr)

    def load_checkpoint(self, label):
        self.load_network(self.net_flow, 'flow', label, self.opts.log_dir)
        if self.opts.model_type == 'CVR':
            self.load_network(self.net_scale, 'scale',  label, self.opts.log_dir)
        self.load_network(self.net_syn,  'syn',  label, self.opts.log_dir)

    def get_current_scalars(self):
        losses = {}
        losses['loss_G'] = self.loss_G.item()
        losses['loss_L1_gs'] = self.loss_L1_gs.item()
        losses['loss_perceptual'] = self.loss_perceptual.item()
        losses['loss_flow_smoothness'] = self.loss_flow_smoothness.item()
        losses['loss_L1_ccl'] = self.loss_L1_ccl.item()
        return losses

    def get_current_visuals(self):
        output_visuals = {}

        output_visuals['im_rs'] = self.im_rs[:,-3:,:,:].clone()
        output_visuals['im_gs'] = self.im_gs_f[:,-3:,:,:].clone()
 
        return output_visuals

    def generate_2D_grid(self, H, W):
        x = torch.arange(0, W, 1).float().cuda() 
        y = torch.arange(0, H, 1).float().cuda()

        xx = x.repeat(H, 1)
        yy = y.view(H, 1).repeat(1, W)
    
        grid = torch.stack([xx, yy], dim=0) 

        return grid

    def convert_m2t(self, im_rs, pred_flow, pred_scale, t):
        _,_,H,W=im_rs.size()
        tt0 = H * (t+0.5)#forward
        tt1 = H * (t-0.5)#backward

        T_0_t = scale_from_m2t(pred_scale[0], H, W, tt0)
        T_1_t = scale_from_m2t(pred_scale[1], H, W, tt1)
        
        F_0_t = pred_flow[0] * T_0_t
        F_1_t = pred_flow[1] * T_1_t
        
        tenMetric_0_1 = torch.nn.functional.l1_loss(input=im_rs[:,0:3,:,:], target=backwarp(tenInput=im_rs[:,3:6,:,:], tenFlow=pred_flow[0]), reduction='none').mean(1, True)
        tenMetric_1_0 = torch.nn.functional.l1_loss(input=im_rs[:,3:6,:,:], target=backwarp(tenInput=im_rs[:,0:3,:,:], tenFlow=pred_flow[1]), reduction='none').mean(1, True)
        g_I0t = softsplat.FunctionSoftsplat(tenInput=im_rs[:,0:3,:,:], tenFlow=F_0_t, tenMetric=-20.0 * tenMetric_0_1, strType='softmax')
        g_I1t = softsplat.FunctionSoftsplat(tenInput=im_rs[:,3:6,:,:], tenFlow=F_1_t, tenMetric=-20.0 * tenMetric_1_0, strType='softmax')
        
        return [g_I0t, g_I1t], [F_0_t, F_1_t]

    def synthese_img_t(self, im_rs0, im_rs1, F_0_1, F_1_0, T_0_m, T_1_m, tt):
        #synthesis GS images at time tt
        gs_t, Fs_t = self.convert_m2t(self.im_rs, [F_0_1, F_1_0], [T_0_m, T_1_m], tt)
        g_It_F_0_t = gs_t[0]
        g_It_F_1_t = gs_t[1]
        F_0_t = Fs_t[0]
        F_1_t = Fs_t[1]

        X_in = torch.cat((im_rs0, im_rs1, F_0_1, F_1_0, F_0_t, F_1_t, g_It_F_0_t, g_It_F_1_t), dim=1)#20
        intrpOut = self.net_syn(X_in)

        delta_F_0_t = intrpOut[:, 0:2, :, :]
        delta_F_1_t = intrpOut[:, 2:4, :, :]
        
        F_0_t_final = F_0_t + delta_F_0_t
        F_1_t_final = F_1_t + delta_F_1_t
        V_0_t  = torch.sigmoid(intrpOut[:, 4:5, :, :])
        V_1_t  = 1.0 - V_0_t
        
        tenMetric_0_1 = torch.nn.functional.l1_loss(input=im_rs0, target=backwarp(tenInput=im_rs1, tenFlow=F_0_1), reduction='none').mean(1, True)
        tenMetric_1_0 = torch.nn.functional.l1_loss(input=im_rs1, target=backwarp(tenInput=im_rs0, tenFlow=F_1_0), reduction='none').mean(1, True)
        g_I0t = softsplat.FunctionSoftsplat(tenInput=im_rs0, tenFlow=F_0_t_final, tenMetric=-20.0 * tenMetric_0_1, strType='softmax')
        g_I1t = softsplat.FunctionSoftsplat(tenInput=im_rs1, tenFlow=F_1_t_final, tenMetric=-20.0 * tenMetric_1_0, strType='softmax')
        
        t = (1 - 2e-2) * tt + 1e-2
        gs_t_final = ((1-t) * V_0_t * g_I0t + t * V_1_t * g_I1t) / ((1-t) * V_0_t + t * V_1_t)

        out_image=[g_It_F_0_t, g_It_F_1_t, g_I0t, g_I1t]
        out_flow=[F_0_t_final, F_1_t_final, delta_F_0_t, delta_F_1_t]

        return gs_t_final, out_image, out_flow