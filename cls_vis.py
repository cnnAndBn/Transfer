import argparse
import cv2
import torch
from mmdet.apis import inference_detector, init_detector
import os.path as osp
import os
import shutil
import mmcv
import numpy as np



def show_result(img,   
                result,
                class_names,
                score_thrs=[],
                wait_time=0,
                show=True,
                out_file=None,
                key_cls_list=None,
                ):
    """Added by qianxiao ， same as official show_result function,
       but extract specified class result into single folder

    Args:
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        class_names (list[str] or tuple[str]): A list of class names.
        score_thr (float): The threshold to visualize the bboxes and masks.
        wait_time (int): Value of waitKey param.
        show (bool, optional): Whether to show the image with opencv or not.
        out_file (str, optional): If specified, the visualization result will
            be written to the out file instead of shown in a window.

    Returns:
        np.ndarray or None: If neither `show` nor `out_file` is specified, the
            visualized image is returned, otherwise None is returned.
    """
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    img = img.copy()
    orig_img = img.copy()
    if isinstance(result, tuple):
        bbox_result = result[0]
    bboxes = np.vstack(bbox_result)

    # draw bounding boxes
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]

    labels = np.concatenate(labels)

    for cls,thr in zip(key_cls_list,score_thrs):

        vis_dir = osp.dirname(out_file)
        vis_name = osp.basename(out_file)

        copied_orig_img = orig_img.copy()
        ids = np.nonzero(np.logical_and(labels==int(cls),bboxes[:, -1]>float(thr)))
        if ids[0].size!=0:
            cls_labels = labels[ids]
            cls_bboxes = bboxes[ids]
            mmcv.imshow_det_bboxes(
                copied_orig_img,
                cls_bboxes,
                cls_labels,
                class_names=class_names,
                score_thr=float(thr),
                show=show,
                wait_time=wait_time,
                out_file=None)

            cls_vis_path = osp.join(vis_dir, cls, vis_name)

            try:
               cv2.imwrite(cls_vis_path, copied_orig_img)
            except:
                print(cls_vis_path)

    if not (show or out_file):
        return img


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection visulization by category')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('image_dir',help='image dir path')
    parser.add_argument('vis_dir', help='vis dir path')
    parser.add_argument('key_vis_cls',help='visualize the specified class into seperate folder')
    parser.add_argument('key_cls_score_thr', help='bbox score threshold of each class')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if not osp.exists(args.image_dir):
        raise FileNotFoundError('Image dir not exist')

    key_vis_cls_list = args.key_vis_cls.split(',')
    key_cls_score_thr_list = args.key_cls_score_thr.split(',')
    assert len(key_cls_score_thr_list) == len(key_vis_cls_list),'cls threshold and cls not equal ({} : {})'\
        .format(len(key_cls_score_thr_list),len(key_vis_cls_list))

    if not osp.exists(args.vis_dir):
        os.mkdir(args.vis_dir)
    else:
        shutil.rmtree(args.vis_dir)
        os.mkdir(args.vis_dir)

    #make seperate folder
    for cls in key_vis_cls_list:
        cls_vis_dir = osp.join(args.vis_dir,cls)
        if not osp.exists(cls_vis_dir):
            os.mkdir(cls_vis_dir)
        else:
            shutil.rmtree(cls_vis_dir)
            os.mkdir(cls_vis_dir)

    model = init_detector(args.config, args.checkpoint, device=torch.device('cuda', args.device))

    model.CLASSES=['back','official','personal','black','other']

    for prefix,dirs,files in os.walk(args.image_dir):

        for f in files:
            if f.startswith('.'):
                continue
            try:
               imgPath = osp.join(prefix,f)
               img = cv2.imread(imgPath)
               result = inference_detector(model, img)
            except:
               print('{} can not be read,skipped'.format(imgPath))
               continue

            save_name = prefix.replace(args.image_dir,'').replace('/','_')
            save_name = '_'.join((save_name,f))
            vis_path = osp.join(args.vis_dir,save_name)
           
            show_result(img, result, model.CLASSES, score_thrs=key_cls_score_thr_list, wait_time=1,show=False,
                        out_file=vis_path,key_cls_list=key_vis_cls_list)


if __name__ == '__main__':
    main()
