import os
import torch
import random
import argparse
import numpy as np
from skimage import io
import shutil

import imageio
import cv2
import flow_viz

from package_core.generic_train_test import *
from forward_warp_package import *
from dataloader import *
from model_CVR import *
from frame_utils import *
from Convert_m_t import *

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
                
                self.model.set_input(_input)
                
                # save original RS images
                im_rs_0 = io.imread(im_rs0_path)
                im_rs_1 = io.imread(im_rs1_path)
                io.imsave(os.path.join(self.opts.results_dir, seq+'_rs_0.png'), im_rs_0)
                io.imsave(os.path.join(self.opts.results_dir, seq+'_rs_1.png'), im_rs_1)
                
                # generate GS images for any time, i.e., 0.0, 0.1, 0.2, ..., 0.9, 1.0
                preds_0=[]
                preds_0_tensor=[]

                copies = 11  #The number of recovered GS image sequence

                for t in range(0,copies):
                    #convert to GS of t-th moment
                    pred_gs_t, pred_im, pred_mask, pred_flow, _ = self.model.forward(t/(copies-1)) # 10X inversion
                    
                    if self.opts.is_Fastec==1:
                        for i in range(2):
                            pred_gs_t = F.interpolate(pred_gs_t,size=[480,640], mode='bilinear')#
                         
                    # save results
                    io.imsave(os.path.join(self.opts.results_dir, seq+'_pred_'+str(t)+'.png'), (pred_gs_t.clamp(0,1).cpu().numpy().transpose(0,2,3,1)[0]*255).astype(np.uint8))
                    
                    preds_0.append((pred_gs_t.clamp(0,1).cpu().numpy().transpose(0,2,3,1)[0]*255).astype(np.uint8))
                    preds_0_tensor.append(pred_gs_t)

                    print('saved', self.opts.results_dir, seq+'_pred_'+str(t)+'.png')
                    
                '''
                ## Computing the cropped areas, and then save them
                pred_imgs_rec=preds_0_tensor
                img_rec = pred_imgs_rec[0].clone()
                for t in range(1,copies):
                    img_rec *= (pred_imgs_rec[t] * 255)
                    img_rec = img_rec.clamp(0,1)
                
                x_min, y_min, x_max, y_max = self.cut_img_without_margin(img_rec.squeeze(0))
                print(x_min, y_min, x_max, y_max)
                
                preds_0_crop=[]
                for t in range(0,copies):
                    preds_0_crop.append((preds_0_tensor[t][:,:,y_min:y_max,x_min:x_max].clamp(0,1).cpu().numpy().transpose(0,2,3,1)[0]*255).astype(np.uint8))
                    # save results
                    io.imsave(os.path.join(self.opts.results_dir, seq+'_pred_crop_'+str(t)+'.png'), (preds_0_tensor[t][:,:,y_min:y_max,x_min:x_max].clamp(0,1).cpu().numpy().transpose(0,2,3,1)[0]*255).astype(np.uint8))
                '''
                #make gif
                make_gif_flag = True
                if make_gif_flag:
                    pred_imgs_gif=preds_0
                    
                    #imageio.mimsave(os.path.join(self.opts.results_dir, seq+'_duration_'+str(0.5)+'.gif'), pred_imgs_gif, duration = 0.5) # can modify the frame duration as needed
                    imageio.mimsave(os.path.join(self.opts.results_dir, seq+'_duration_'+str(0.1)+'.gif'), pred_imgs_gif, duration = 0.1)
                
                    #pred_imgs_gif_crop=preds_0_crop
                    #imageio.mimsave(os.path.join(self.opts.results_dir, seq+'_crop_duration_'+str(0.5)+'.gif'), pred_imgs_gif_crop, duration = 0.5)
                    #imageio.mimsave(os.path.join(self.opts.results_dir, seq+'_crop_duration_'+str(0.1)+'.gif'), pred_imgs_gif_crop, duration = 0.1)
                    
                    print('\n')





    def cut_img_without_margin(self, img_pre):
        """
        Extract the largest inscribed rectangle according to all corrected images with multiplied their corresponding masks 
        Input: tensor(3*H*W), image with black edges
        """
        
        img_bgr = img_pre.cpu().numpy().copy()
        
        img_bgr = (img_bgr * 255.0).astype(np.uint8)
        img_bgr_  = img_bgr.transpose(1,2,0).copy()
        img_bgr  = img_bgr.transpose(1,2,0)
        
        h, w, c = img_bgr.shape
        
        img_bgr  = (img_bgr * np.random.rand(h, w, c)).astype(np.uint8)
        #print(img_bgr[:,:,0])
        
        img_copy = img_bgr.copy()
        
        mask = np.zeros((h + 2, w + 2), np.uint8)  # Mask
        cv2.floodFill(img_copy, mask=mask, seedPoint=(0, 0), newVal=(0, 0, 255),
                      loDiff=(1, 1, 1), upDiff=(1, 1, 1))
        cv2.floodFill(img_copy, mask=mask, seedPoint=(w-1, 0), newVal=(0, 0, 255),
                      loDiff=(1, 1, 1), upDiff=(1, 1, 1))
        cv2.floodFill(img_copy, mask=mask, seedPoint=(0, h-1), newVal=(0, 0, 255),
                      loDiff=(1, 1, 1), upDiff=(1, 1, 1))
        cv2.floodFill(img_copy, mask=mask, seedPoint=(w-1, h-1), newVal=(0, 0, 255),
                      loDiff=(1, 1, 1), upDiff=(1, 1, 1))
                      
        for i in range(20,w-20,20):
            cv2.floodFill(img_copy, mask=mask, seedPoint=(i, 0), newVal=(0, 0, 255),
                      loDiff=(1, 1, 1), upDiff=(1, 1, 1))
            cv2.floodFill(img_copy, mask=mask, seedPoint=(i, h-1), newVal=(0, 0, 255),
                      loDiff=(1, 1, 1), upDiff=(1, 1, 1))
        for i in range(20,h-20,20):
            cv2.floodFill(img_copy, mask=mask, seedPoint=(0, i), newVal=(0, 0, 255),
                      loDiff=(1, 1, 1), upDiff=(1, 1, 1))
            cv2.floodFill(img_copy, mask=mask, seedPoint=(w-1, i), newVal=(0, 0, 255),
                      loDiff=(1, 1, 1), upDiff=(1, 1, 1))
                      
        mask_inv = np.where(mask > 0.5, 0, 1).astype(np.uint8)
        
        kernel = np.ones((9, 9), np.uint8)
        mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, kernel)
        
        #cv2.imwrite('images/mask_floodfill.png', mask_inv*255)
        
        _, contours, hierarchy = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        index=0
        c_point=0
        for i in range(len(contours)):
            c_points = np.squeeze(contours[i])
            if len(c_points)>c_point:
                c_point=len(c_points)
                index=i
        
        #cv2.drawContours(img_bgr_, contours[index], -1, (0, 255, 0), 3)
        #cv2.imwrite('images/points_contours.png', img_bgr_)

        contour = contours[index].reshape(len(contours[index]),2)

        rect = []
        for i in range(len(contour)):
            x1, y1 = contour[i]
            for j in range(len(contour)):
                x2, y2 = contour[j]
                area = abs(y2-y1)*abs(x2-x1)
                rect.append(((x1,y1), (x2,y2), area))

        all_rect = sorted(rect, key = lambda x : x[2], reverse = True)

        best_rect_found = False
        index_rect = 0
        nb_rect = len(all_rect)
    
        while not best_rect_found and index_rect < nb_rect:
            rect = all_rect[index_rect]
            (x1, y1) = rect[0]
            (x2, y2) = rect[1]

            valid_rect = True
        
            x = min(x1, x2)
            while x <max(x1,x2)+1 and valid_rect:
                if mask_inv[y1,x] == 0 or mask_inv[y2,x] == 0:
                    valid_rect = False
                x+=1

            y = min(y1, y2)
            while y <max(y1,y2)+1 and valid_rect:
                if mask_inv[y,x1] == 0 or mask_inv[y,x2] == 0:
                    valid_rect = False
                y+=1

            if valid_rect:
                best_rect_found = True

            index_rect+=1
            
        x_min = min(x1,x2)
        y_min = min(y1,y2)
        x_max = max(x1,x2)
        y_max = max(y1,y2)
        
        return x_min, y_min, x_max, y_max


    def draw_arrow(self, image, x, y, optical_flow):
        height, width, _ = image.shape
        start_point = np.array([x, y])
        end_point = start_point + optical_flow
        end_point = np.round(end_point).astype(np.int32)
        end_point = np.clip(end_point, 0, [width - 1, height - 1])
        assert end_point[0] < width
        assert end_point[1] < height
        cv2.arrowedLine(image, tuple(start_point), tuple(end_point), (0.0, 0.0, 0.0), thickness=2)

    def draw_all_arrows(self, img1, img2, optical_flow):
        assert img1.shape == img2.shape
        height, width, _ = img1.shape
        assert optical_flow.shape[0] == height
        assert optical_flow.shape[1] == width
        assert optical_flow.shape[2] == 2
        #assert img1.max() <= 1
        #assert img2.max() <= 1
        #blended_image = (img1 + img2) * 0.5
        blended_image = img1.copy()
        narrows_per_row = 17
        narrows_per_col = 15
        for y in np.arange(22, height, height // narrows_per_col):
            assert y < height
            for x in np.arange(26, width-10, width // narrows_per_row):
                assert x < width
                self.draw_arrow(blended_image, x, y, optical_flow[y, x, :])
        # blended_image = cv2.resize(blended_image, (0, 0), fx=2, fy=2)
        return blended_image  


Demo(model, opts, None, None).test()
