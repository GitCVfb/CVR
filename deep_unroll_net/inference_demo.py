import os
import torch
import random
import argparse
import numpy as np
from skimage import io
import shutil

import cv2
import flow_viz

from package_core.generic_train_test import *
from forward_warp_package import *
from dataloader import *
from model_CVR import *
from frame_utils import *

##===================================================##
##********** Configure training settings ************##
##===================================================##
parser=argparse.ArgumentParser()
parser.add_argument('--batch_sz', type=int, default=1, help='batch size used for training')
parser.add_argument('--continue_train', type=bool, default=True, help='flags used to indicate if train model from previous trained weight')
parser.add_argument('--is_training', type=bool, default=False, help='flag used for selecting training mode or evaluation mode')
parser.add_argument('--n_chan', type=int, default=3, help='number of channels of input/output image')
parser.add_argument('--seq_len', type=int, default=2)
parser.add_argument('--shuffle_data', type=bool, default=False)
parser.add_argument('--crop_sz_H', type=int, default=448, help='cropped image size height')
parser.add_argument('--crop_sz_W', type=int, default=640, help='cropped image size width')

parser.add_argument('--model_type', type=str, required=True, help='CVR or CVR*')
parser.add_argument('--model_label', type=str, required=True)

parser.add_argument('--log_dir', type=str, required=True)
parser.add_argument('--results_dir', type=str, required=True)
parser.add_argument('--data_dir', type=str, required=True)

parser.add_argument('--is_Fastec', type=int, default=0)
 
opts=parser.parse_args()

##===================================================##
##****************** Create model *******************##
##===================================================##
model=ModelCVR(opts)

##===================================================##
##**************** Train the network ****************##
##===================================================##
class Demo(Generic_train_test):
    def test(self):
        with torch.no_grad():
            seq_lists = os.listdir(self.opts.data_dir)
            for seq in seq_lists:
                im_rs0_path = os.path.join(os.path.join(self.opts.data_dir, seq), 'rs_0.png')
                im_rs1_path = os.path.join(os.path.join(self.opts.data_dir, seq), 'rs_1.png')

                im_rs0 = torch.from_numpy(io.imread(im_rs0_path).transpose(2,0,1))[:3,:,:].unsqueeze(0).clone()
                im_rs1 = torch.from_numpy(io.imread(im_rs1_path).transpose(2,0,1))[:3,:,:].unsqueeze(0).clone()

                im_rs = torch.cat([im_rs0,im_rs1], dim=1).float()/255.
                
                if self.opts.is_Fastec==1:
                    im_rs0 = F.interpolate(im_rs[:,0:3,:,:],size=[448,640], mode='bilinear')#
                    im_rs1 = F.interpolate(im_rs[:,3:6,:,:],size=[448,640], mode='bilinear')#
                    im_rs  = torch.cat((im_rs0,im_rs1),1).clone()#
                    
                _input = [im_rs, None, None, None]
                B,C,H,W = im_rs.size()
                
                self.model.set_input(_input)
                pred_gs_f_final, pred_im_f, pred_mask_f, pred_flow_f, pred_visible_f = self.model.forward(0.5)
                pred_gs_m_final, pred_im_m, pred_mask_m, pred_flow_m, pred_visible_m = self.model.forward(1.0)
                
                if self.opts.is_Fastec==1:
                    pred_gs_f_final = F.interpolate(pred_gs_f_final,size=[480,640], mode='bilinear')#
                    pred_gs_m_final = F.interpolate(pred_gs_m_final,size=[480,640], mode='bilinear')#
                    for i in range(6):
                        pred_im_f[i] = F.interpolate(pred_im_f[i],size=[480,640], mode='bilinear')#
                        pred_flow_f[i] = F.interpolate(pred_flow_f[i],size=[480,640], mode='bilinear')#
                        pred_im_m[i] = F.interpolate(pred_im_m[i],size=[480,640], mode='bilinear')#
                        pred_flow_m[i] = F.interpolate(pred_flow_m[i],size=[480,640], mode='bilinear')#
                
                # save results
                im_gs_0_f = io.imread(os.path.join(os.path.join(self.opts.data_dir, seq), 'gs_0_f.png'))
                im_gs_0_m = io.imread(os.path.join(os.path.join(self.opts.data_dir, seq), 'gs_0_m.png'))
                im_gs_1_f = io.imread(os.path.join(os.path.join(self.opts.data_dir, seq), 'gs_1_f.png'))
                im_gs_1_m = io.imread(os.path.join(os.path.join(self.opts.data_dir, seq), 'gs_1_m.png'))
                im_rs_0 = io.imread(im_rs0_path)
                im_rs_1 = io.imread(im_rs1_path)

                im_gs_0_f = im_gs_0_f[:].copy()
                im_gs_0_m = im_gs_0_m[:].copy()
                im_gs_1_f = im_gs_1_f[:].copy()
                im_gs_1_m = im_gs_1_m[:].copy()
                im_rs_0 = im_rs_0[:].copy()
                im_rs_1 = im_rs_1[:].copy()

                io.imsave(os.path.join(self.opts.results_dir, seq+'_rs_0.png'), im_rs_0)
                io.imsave(os.path.join(self.opts.results_dir, seq+'_rs_1.png'), im_rs_1)
                #io.imsave(os.path.join(self.opts.results_dir, seq+'_pred_0_f.png'), (pred_im_f[2].clamp(0,1).cpu().numpy().transpose(0,2,3,1)[0]*255).astype(np.uint8))
                #io.imsave(os.path.join(self.opts.results_dir, seq+'_pred_1_f.png'), (pred_im_f[3].clamp(0,1).cpu().numpy().transpose(0,2,3,1)[0]*255).astype(np.uint8))
                
                #io.imsave(os.path.join(self.opts.results_dir, seq+'_pred_0_m.png'), (pred_im_m[2].clamp(0,1).cpu().numpy().transpose(0,2,3,1)[0]*255).astype(np.uint8))
                #io.imsave(os.path.join(self.opts.results_dir, seq+'_pred_1_m.png'), (pred_im_m[3].clamp(0,1).cpu().numpy().transpose(0,2,3,1)[0]*255).astype(np.uint8))
                
                io.imsave(os.path.join(self.opts.results_dir, seq+'_final_pred_f.png'), (pred_gs_f_final.clamp(0,1).cpu().numpy().transpose(0,2,3,1)[0]*255).astype(np.uint8))
                io.imsave(os.path.join(self.opts.results_dir, seq+'_final_pred_m.png'), (pred_gs_m_final.clamp(0,1).cpu().numpy().transpose(0,2,3,1)[0]*255).astype(np.uint8))

                print('saved', self.opts.results_dir, seq+'_pred_gs_img.png')
                
                im_gs_1_m_new=torch.from_numpy(im_gs_1_m[:,:,:3].transpose(2,0,1)).unsqueeze(0).cuda().float()/255.
                io.imsave(os.path.join(self.opts.results_dir, seq+'_gt_1m.png'), (im_gs_1_m_new.clamp(0,1).cpu().numpy().transpose(0,2,3,1)[0]*255).astype(np.uint8))
                im_gs_1_f_new=torch.from_numpy(im_gs_1_f[:,:,:3].transpose(2,0,1)).unsqueeze(0).cuda().float()/255.
                io.imsave(os.path.join(self.opts.results_dir, seq+'_gt_1f.png'), (im_gs_1_f_new.clamp(0,1).cpu().numpy().transpose(0,2,3,1)[0]*255).astype(np.uint8))
                
                save_BMF = False
                if save_BMF == True:
                    flow = pred_flow_f[2]
                    if self.opts.is_Fastec==1:
                        flow = F.interpolate(flow,size=[480,640], mode='bilinear')#
                    flow = flow.cpu().numpy().transpose(0,2,3,1)[0]
                    flow_image = flow_viz.flow_to_image(flow)
                    cv2.imwrite(os.path.join(self.opts.results_dir, seq+'_pred_flow_0_f.png'), flow_image)

                    flow = pred_flow_f[3]
                    if self.opts.is_Fastec==1:
                        flow = F.interpolate(flow,size=[480,640], mode='bilinear')#
                    flow = flow.cpu().numpy().transpose(0,2,3,1)[0]
                    flow_image = flow_viz.flow_to_image(flow)
                    cv2.imwrite(os.path.join(self.opts.results_dir, seq+'_pred_flow_1_f.png'), flow_image)
                    
                    flow = pred_flow_m[2]
                    if self.opts.is_Fastec==1:
                        flow = F.interpolate(flow,size=[480,640], mode='bilinear')#
                    flow = flow.cpu().numpy().transpose(0,2,3,1)[0]
                    flow_image = flow_viz.flow_to_image(flow)
                    cv2.imwrite(os.path.join(self.opts.results_dir, seq+'_pred_flow_0_m.png'), flow_image)

                    flow = pred_flow_m[3]
                    if self.opts.is_Fastec==1:
                        flow = F.interpolate(flow,size=[480,640], mode='bilinear')#
                    flow = flow.cpu().numpy().transpose(0,2,3,1)[0]
                    flow_image = flow_viz.flow_to_image(flow)
                    cv2.imwrite(os.path.join(self.opts.results_dir, seq+'_pred_flow_1_m.png'), flow_image)
                    
                
                save_intermediate_variable = False

                if save_intermediate_variable == True:
                    flow = pred_flow_f[4]
                    if self.opts.is_Fastec==1:
                        flow = F.interpolate(flow,size=[480,640], mode='bilinear')#
                    flow = flow.cpu().numpy().transpose(0,2,3,1)[0]
                    flow_image = flow_viz.flow_to_image(flow)
                    cv2.imwrite(os.path.join(self.opts.results_dir, seq+'_pred_flow_aux_0_f.png'), flow_image)
                    
                    flow = pred_flow_f[5]
                    if self.opts.is_Fastec==1:
                        flow = F.interpolate(flow,size=[480,640], mode='bilinear')#
                    flow = flow.cpu().numpy().transpose(0,2,3,1)[0]
                    flow_image = flow_viz.flow_to_image(flow)
                    cv2.imwrite(os.path.join(self.opts.results_dir, seq+'_pred_flow_aux_1_f.png'), flow_image)
                    
                    flow = pred_flow_m[4]
                    if self.opts.is_Fastec==1:
                        flow = F.interpolate(flow,size=[480,640], mode='bilinear')#
                    flow = flow.cpu().numpy().transpose(0,2,3,1)[0]
                    flow_image = flow_viz.flow_to_image(flow)
                    cv2.imwrite(os.path.join(self.opts.results_dir, seq+'_pred_flow_aux_0_m.png'), flow_image)
                    
                    flow = pred_flow_m[5]
                    if self.opts.is_Fastec==1:
                        flow = F.interpolate(flow,size=[480,640], mode='bilinear')#
                    flow = flow.cpu().numpy().transpose(0,2,3,1)[0]
                    flow_image = flow_viz.flow_to_image(flow)
                    cv2.imwrite(os.path.join(self.opts.results_dir, seq+'_pred_flow_aux_1_m.png'), flow_image)
                    
                    ##save BMF residual map
                    if self.opts.is_Fastec==1:
                        aux_flow_map_0_f_ = torch.norm(F.interpolate(pred_flow_f[4],size=[480,640], mode='bilinear'),dim=1).cpu().numpy().transpose(1,2,0)
                    else:
                        aux_flow_map_0_f_ = torch.norm(pred_flow_f[4],dim=1).cpu().numpy().transpose(1,2,0)
                    aux_flow_map_0_f = (aux_flow_map_0_f_ - np.min(aux_flow_map_0_f_)) / (np.max(aux_flow_map_0_f_) - np.min(aux_flow_map_0_f_)) * 255.
                    cv2.imwrite(os.path.join(self.opts.results_dir, seq+'_pred_flow_aux_0_f_gray.png'), aux_flow_map_0_f.astype(np.uint8))
                    
                    if self.opts.is_Fastec==1:
                        aux_flow_map_1_f_ = torch.norm(F.interpolate(pred_flow_f[5],size=[480,640], mode='bilinear'),dim=1).cpu().numpy().transpose(1,2,0)
                    else:
                        aux_flow_map_1_f_ = torch.norm(pred_flow_f[5],dim=1).cpu().numpy().transpose(1,2,0)
                    aux_flow_map_1_f = (aux_flow_map_1_f_ - np.min(aux_flow_map_1_f_)) / (np.max(aux_flow_map_1_f_) - np.min(aux_flow_map_1_f_)) * 255.
                    cv2.imwrite(os.path.join(self.opts.results_dir, seq+'_pred_flow_aux_1_f_gray.png'), aux_flow_map_1_f.astype(np.uint8))
                    
                    #save visible map
                    if self.opts.is_Fastec==1:
                        visible_map_0_f_ = np.absolute(F.interpolate(pred_visible_f[0],size=[480,640], mode='bilinear').cpu().numpy().transpose(0,2,3,1)[0])
                    else:
                        visible_map_0_f_ = np.absolute(pred_visible_f[0].cpu().numpy().transpose(0,2,3,1)[0])
                    visible_map_0_f_[0,0,0] = 0
                    visible_map_0_f_[0,1,0] = 1
                    visible_map_0_f = (visible_map_0_f_ - np.min(visible_map_0_f_)) / (np.max(visible_map_0_f_) - np.min(visible_map_0_f_)) * 255.
                    cv2.imwrite(os.path.join(self.opts.results_dir, seq+'_visible_0_f.png'), visible_map_0_f.astype(np.uint8))
                    
                    if self.opts.is_Fastec==1:
                        visible_map_1_f_ = np.absolute(F.interpolate(pred_visible_f[1],size=[480,640], mode='bilinear').cpu().numpy().transpose(0,2,3,1)[0])
                    else:
                        visible_map_1_f_ = np.absolute(pred_visible_f[1].cpu().numpy().transpose(0,2,3,1)[0])
                    visible_map_1_f_[0,0,0] = 0
                    visible_map_1_f_[0,1,0] = 1
                    visible_map_1_f = (visible_map_1_f_ - np.min(visible_map_1_f_)) / (np.max(visible_map_1_f_) - np.min(visible_map_1_f_)) * 255.
                    cv2.imwrite(os.path.join(self.opts.results_dir, seq+'_visible_1_f.png'), visible_map_1_f.astype(np.uint8))
                    
                    if self.opts.is_Fastec==1:
                        visible_map_0_m_ = np.absolute(F.interpolate(pred_visible_m[0],size=[480,640], mode='bilinear').cpu().numpy().transpose(0,2,3,1)[0])
                    else:
                        visible_map_0_m_ = np.absolute(pred_visible_m[0].cpu().numpy().transpose(0,2,3,1)[0])
                    visible_map_0_m_[0,0,0] = 0
                    visible_map_0_m_[0,1,0] = 1
                    visible_map_0_m = (visible_map_0_m_ - np.min(visible_map_0_m_)) / (np.max(visible_map_0_m_) - np.min(visible_map_0_m_)) * 255.
                    cv2.imwrite(os.path.join(self.opts.results_dir, seq+'_visible_0_m.png'), visible_map_0_m.astype(np.uint8))
                    
                    if self.opts.is_Fastec==1:
                        visible_map_1_m_ = np.absolute(F.interpolate(pred_visible_m[1],size=[480,640], mode='bilinear').cpu().numpy().transpose(0,2,3,1)[0])
                    else:
                        visible_map_1_m_ = np.absolute(pred_visible_m[1].cpu().numpy().transpose(0,2,3,1)[0])
                    visible_map_1_m_[0,0,0] = 0
                    visible_map_1_m_[0,1,0] = 1
                    visible_map_1_m = (visible_map_1_m_ - np.min(visible_map_1_m_)) / (np.max(visible_map_1_m_) - np.min(visible_map_1_m_)) * 255.
                    cv2.imwrite(os.path.join(self.opts.results_dir, seq+'_visible_1_m.png'), visible_map_1_m.astype(np.uint8))
                            
Demo(model, opts, None, None).test()


