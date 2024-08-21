# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import glob
import argparse
import numpy as np
np.set_printoptions(threshold = 1000000000000, precision = 2)

import PIL.Image as pil
from PIL import ImageDraw, ImageFont
import matplotlib as mpl
import matplotlib.cm as cm

import torch
import torch.nn as nn
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
from evaluate_depth import STEREO_SCALE_FACTOR
import cv2
import time

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', 
                        default = None)
    
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',
                        default = "0328_r101_v135_cut0.66_fuiegan_a")
    
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument("--pred_metric_depth",
                        help='if set, predicts metric depth instead of disparity. (This only '
                              'makes sense for stereo-trained KITTI models).',
                        action='store_true')
    
    parser.add_argument("--use_flow", default=False, help='加入計算光流，偵測出鏡頭的移動方向')
    parser.add_argument("--nine_points", default=True, help='顯示圖中九宮格點位置的深度值')
    
    #parser.add_argument("--video", default=True, help='輸入為影片檔')
    parser.add_argument("--video_path", default='0', help='影片檔路徑')
    
    return parser.parse_args()

def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.pred_metric_depth and "stereo" not in args.model_name:
        print("Warning: The --pred_metric_depth flag only makes sense for stereo-trained KITTI "
              "models. For mono-trained models, output depths will not in metric space.")
        
    start = time.time()

    download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("models", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")
    #
    #
    # LOADING PRETRAINED MODEL#
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(101, False)
    
    # 修改
    encoder.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2,  padding=3, padding_mode='replicate')
    encoder.encoder.avgpool = nn.AdaptiveAvgPool2d(2)
    print(encoder)
    
    loaded_dict_enc = torch.load(encoder_path, map_location=device)
    #
    #
    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))
    
    print(depth_decoder)
    
    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()
    
    is_webcam = args.video_path.isdigit()
    print(is_webcam)
    
    if is_webcam:
        vid = cv2.VideoCapture(int(args.video_path))
    else:
        vid = cv2.VideoCapture(args.video_path)
        
    if not vid.isOpened():
        print('Could not open video "%s"' % args.video_path)
        exit(-1)
                
    target_fps   = round(vid.get(cv2.CAP_PROP_FPS))
    #target_fps=30
    
    frame_width  = round(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = round(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if is_webcam:
        num_frames = float('inf')
    else:
        num_frames = round(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    

    # FINDING INPUT IMAGES
    if args.video_path is not None:
        if args.video_path == '0':
            print(args.video_path)
            vid = cv2.VideoCapture(0)
            target_fps = round(vid.get(cv2.CAP_PROP_FPS))
            num_frames = float('inf')
            
            target_fps = 10
            out = cv2.VideoWriter('video_results/output.mp4', cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (frame_width, frame_height))
            out_origin = cv2.VideoWriter('video_results/input.mp4', cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (frame_width, frame_height))
        else:
            print(args.video_path)
            vid = cv2.VideoCapture(args.video_path)
            target_fps = round(vid.get(cv2.CAP_PROP_FPS))
            num_frames = round(vid.get(cv2.CAP_PROP_FRAME_COUNT))
            
        print('FPS:', target_fps, ' Flames:', num_frames)
        output_directory = 'video_results'
        
        
    
    else:    
        if os.path.isfile(args.image_path):
            # Only testing on a single image
            paths = [args.image_path]
            output_directory = os.path.dirname(args.image_path)
        elif os.path.isdir(args.image_path):
            # Searching folder for images
            paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
            output_directory = args.image_path
    
        else:
            raise Exception("Can not find args.image_path: {}".format(args.image_path))
    
        print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        # # Create some random colors
        # color = np.random.randint(0, 255, (100, 3))
        
        # feature_params = dict(maxCorners=100,
        #               qualityLevel=0.2,
        #               minDistance=7,
        #               blockSize=7)

        # # Parameters for lucas kanade optical flow
        # # maxLevel 為使用的金字塔層數
        # lk_params = dict(winSize=(15, 15),
        #                  maxLevel=4,
        #                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        

        # path = eval(repr(paths[0]).replace("\\", '/'))
        
        # old_frame = cv2.imread(path)
        
        # old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        # # 原點
        # p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        
        # # Create a mask image for drawing purposes 
        # mask = np.zeros_like(old_frame)
        
        # w = old_frame.shape[1]
        # h = old_frame.shape[0]
        # mid_x = int(w/2)
        # mid_y = int(h/2)
        
        # num = 0
        cv2.namedWindow("result", 0);
        cv2.resizeWindow("result", 640, 360);
        
        cv2.namedWindow("input", 0);
        cv2.resizeWindow("input", 640, 360);
        
        
        # for idx, image_path in enumerate(paths):
        frame_count = 1
        
        while True:
            ret, frame = vid.read()
            if ret == False:
                print('Over')
                break
            input_image = pil.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # if image_path.endswith("_disp.jpg"):
            #     # don't try to predict disparity for a disparity image!
            #     continue
        
            ###################################################################################################
            # if args.use_flow:
            #     path = eval(repr(paths[idx]).replace("\\", '/'))
            #     frame = cv2.imread(path)
            #     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #     p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                
            #     # 找點
            #     # 如果找不到追蹤點 重新找一個新的起點
            #     if p1 is None:
            #         p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
            #         #if p0 = None:
            #         p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            #     # 有找到的就留著
            #     good_new = p1[st == 1]
            #     good_old = p0[st == 1]
                
            #     v1_sum = 0
            #     v2_sum = 0
            #     k = 0
            #     # draw the tracks
                
            #     for i, (new, old) in enumerate(zip(good_new, good_old)):
            #         a, b = new.ravel()
            #         c, d = old.ravel()
                    
            #         # 計算向量和
            #         # print('舊坐標:', c , d)
            #         # print('新座標:', a,  b)
            #         v1, v2 = a-c, b-d
            #         #print('向量:(%.2f, %.2f)'%(v1, v2))
            #         v1_sum += v1
            #         v2_sum += v2
    
            #         a,b,c,d = int(a), int(b), int(c), int(d)
            #         mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            #         frame_dot = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
                        
            #             # 向量和取平均
            #     v1 = 0-v1_sum/(i+1)
            #     v2 = 0-v2_sum/(i+1)
            #     print('整張圖的向量:(%.2f, %.2f)'%(v1, v2))
            #     # 光流圖
            #     flow = cv2.add(frame_dot, mask)
            #     #cv2.imwrite('D:/download/video2img/flow/flow/%s.jpg'%str(num).zfill(10), flow)
                
            #     # 黑底箭頭圖
            #     black = np.zeros_like(old_frame)
            #     arrow = cv2.arrowedLine(black, (mid_x, mid_y), (mid_x + int(v1)*10, mid_y + int(v2)*10), (500, 500, 500), 5, 16, 0, 0.3)
                
            #     result = cv2.add(frame, arrow)
                
            #     #cv2.imshow('arrow', arrow)
            #     #cv2.imwrite('D:/download/video2img/flow/arrow/%s.jpg'%str(num).zfill(10), arrow)
                
            #     #cv2.imshow('result', result)
            #     #cv2.imwrite('D:/download/video2img/flow/result/%s.jpg'%str(num).zfill(10), result)
            
            #     k = cv2.waitKey(1) #& 0xff
            #     if k == 27:
            #         break
            #     # Now update the previous frame and previous points
            #     old_gray = frame_gray.copy()
            #     p0 = good_new.reshape(-1, 1, 2)
            #     num += 1
                
            #     arrow_pil = pil.fromarray(arrow)
            ##################################################################################################
            
            # Load image and preprocess
            #input_image = pil.open(image_path).convert('RGB')
            
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)
            
            min_depth = 0.01
            max_depth = 20

            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)
                 
            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving numpy file
            output_name = os.path.splitext(os.path.basename(args.video_path))[0]
            scaled_disp, depth = disp_to_depth(disp, min_depth, max_depth)
            print('='*100)

            if args.pred_metric_depth:
                name_dest_npy = os.path.join(output_directory, "{}_depth.npy".format(output_name))
                metric_depth = STEREO_SCALE_FACTOR * depth.cpu().numpy()
                np.save(name_dest_npy, metric_depth)
            else:
                name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
                #np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            #normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min()*0.8, vmax=vmax*0.8) #(seathru 2.5、1.2, FLsea 2.2、1.2)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            
            # 輸出處理
            colormapped_im = cv2.GaussianBlur(colormapped_im, (15, 15), 0)
            
            
            # ################## resize後的深度數值 ######################
            scaled_disp2, depth2 = disp_to_depth(disp_resized, min_depth, max_depth)
            
            ### 畫九宮格深度點 ###
            im = pil.fromarray(colormapped_im)

            # if args.use_flow:
            #     im = pil.blend(im, arrow_pil, 0.1)
            ##############################################################
                
            if args.nine_points:
                drawim = ImageDraw.Draw(im)
                text_font = ImageFont.truetype('‪C:\Windows\Fonts\ARLRDBD.TTF', 20, encoding='utf-8') 
                for a in range(0 ,3):
                    for b in range(0, 3):
                        w = int(depth2.size()[3]*(a+1)/4)
                        h = int(depth2.size()[2]*(b+1)/4)
                        drawim.ellipse((w-5, h-5, w+5 , h+5), fill = 'red', outline = 'black')
                        point_depth = '%.3f'%depth2.cpu().numpy()[0][0][h][w] + 'm'
                        drawim.text((w, h), point_depth , fill= 'green', outline = 'black', font = text_font)
            ##########################################################
             
            # 儲存圖片
            name_dest_im = os.path.join(output_directory, "%s_disp.jpeg"%(str(frame_count).zfill(5)))
            
            #im.save(name_dest_im)
            
            pix = np.array(im)[:, :, ::-1]
            
            # image = np.zeros((1080, 3840, 3))
            # image[0:1080, 0:1920] = pix
            # image[0:1080, 1920:3840] = frame
            # image = cv2.resize(image, (1280, 360))
            # cv2.imwrite("video_results/combin_%s.jpeg"%(str(frame_count).zfill(5)), image)
            # cv2.imshow('output', image)
            cv2.imshow("result", pix)
            cv2.imshow("input", frame)
            
            # 保存結果
            if is_webcam:
                out.write(pix)
                out_origin.write(frame)
            else:
                im.save(name_dest_im)
                cv2.imwrite("video_results/%s.jpg"%(str(frame_count).zfill(5)), frame)
            
            frame_count += 1
        
            if cv2.waitKey(1) == 27:
                break
            
            # print("   Processed {:d} of {:d} images - saved predictions to:".format(
            #     idx + 1, len(paths)))
            print("   - {}".format(name_dest_im))
            # print("   - {}".format(name_dest_npy))
            print('   FPS: %.2f'%(frame_count/(time.time() - start)))
            print('   最大深度: ', depth.cpu().numpy().max())
            print('   最小深度: ', depth.cpu().numpy().min())
            print()

    print('-> Done!')
    #out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
