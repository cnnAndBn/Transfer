import argparse
import cv2
import torch
from mmdet.apis import inference_detector, init_detector, show_result
import os.path as osp
import os
import shutil
from tqdm import tqdm
import json

def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('image_dir',help='image dir path')
    parser.add_argument('vis_dir', help='vis dir path')
    parser.add_argument('result_path', help='rsult json save path')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    args = parser.parse_args()
    return args


def main():
    
    clsThr = [0.8171,0.9825,0.9955,0.22,0.196]
    
    args = parse_args()

    if not osp.exists(args.image_dir):
        raise FileNotFoundError('Image dir not exist')

    if not osp.exists(args.vis_dir):
        os.mkdir(args.vis_dir)
    else:
        print('Warning: {} already exist.'.format(args.vis_dir))

    model = init_detector(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    resultDict = {}
    for prefixPath,dirs,files in tqdm(os.walk(args.image_dir)):

         for file_name in files:
             if file_name.startswith('.'):
                 continue
             try:   
               img = cv2.imread(osp.join(prefixPath, file_name))
               result = inference_detector(model, img)
             except:
                continue

             #sub_path = prefixPath.strip(args.image_dir)
             sub_path = '/'.join(prefixPath.split('/')[-2:])
             vis_file_name = sub_path.replace('/','#')+'#'+file_name
             vis_path = osp.join(args.vis_dir,vis_file_name)

             if isinstance(result, tuple):
                 bbox_result, segm_result = result
             else:
                 bbox_result, segm_result = result, None
            
             dictKey = '/'.join([sub_path,file_name])
             resultDict[dictKey] = bbox_result 
             #show_result(img, result, model.CLASSES, score_thr=args.score_thr, wait_time=1, show=False, out_file=vis_path)
   
    savedDict = {}
    #convert into non numpy format for export json 
    for f,perImgDet in resultDict.items():

            imgDet = []
            for idx,dets in enumerate(perImgDet):
               cls = idx +1

               for det in dets:
                if det[-1]>clsThr[idx]:
                    imgDet.append([int(det[0]),int(det[1]),int(det[2]),int(det[3]),float(det[4]),cls])  #coordinate,score,cls

            savedDict[f]=imgDet

    with open(args.result_path, 'w') as output_json_file:
            json.dump(savedDict, output_json_file)

   

if __name__ == '__main__':
    main()

