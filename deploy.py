from mmcls.models import ImageClassifier
from mmcv.runner import load_checkpoint
from mmcls.datasets.pipelines import Compose
from mmcv.parallel import collate,scatter
import torch
import numpy as np

def inference_model(model, img, pipeline):
    """Inference image(s) with the classifier.

    Args:
        model (nn.Module): The loaded classifier.
        img (str/ndarray): The image filename or loaded image.
        pipeline(list) : The preprocessing steps

    Returns:
        result (dict): The classification results that contains
        `pred_label` and `pred_score`.
    """

    device = next(model.parameters()).device  # model device
    # build the data pipeline
    if isinstance(img, str):
        if pipeline[0]['type'] != 'LoadImageFromFile':
            pipeline.insert(0, dict(type='LoadImageFromFile'))
        data = dict(img_info=dict(filename=img), img_prefix=None)
    else:
        if pipeline[0]['type'] == 'LoadImageFromFile':
            pipeline.pop(0)
        data = dict(img=img)
    test_pipeline = Compose(pipeline)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    # forward the model
    with torch.no_grad():
        scores = model(return_loss=False, **data)
        pred_score = np.max(scores, axis=1)[0]
        pred_label = np.argmax(scores, axis=1)[0]
        result = {'pred_label': pred_label, 'pred_score': float(pred_score)}
    return result


if __name__ == '__main__':

    modelWeightPath = '/home/yons/workspace/mmlab/mmclassification/tools/work_dirs/resnet50_baseline/epoch_22.pth'
    imgPath = '/media/yons/myspace/dataset/IT/cls_v1/val/order/0bdb5e37679f4b6a9ec802c1f5250b89.jpg'

    modelConfig = dict(
        backbone=dict(
            type='ResNeSt',
            depth=50,
            num_stages=4,
            out_indices=(3,),
            style='pytorch'),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(
            type='LinearClsHead',
            num_classes=4,
            in_channels=2048,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk=(1,),
        ))

    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='Resize', size=(224, 224)),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='ImageToTensor', keys=['img']),
        dict(type='Collect', keys=['img'])]
    device = 'cuda:0'

    #Initialize and load model
    funRouter = ImageClassifier(**modelConfig)
    checkpoint = load_checkpoint(funRouter,modelWeightPath,map_location=device)
    funRouter.eval()


    #Do inference , {'device:0','discount':1,'order':2, 'other': 3}
    result = inference_model(funRouter,imgPath,test_pipeline)
    clsIndx = result['pred_label']




