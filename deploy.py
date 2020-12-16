import os.path as osp
import mmcv
import numpy as np
import torch
from mmdet.models.detectors import CascadeRCNN
from mmcv.runner import load_checkpoint
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
import scipy
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from ntsnet.model import attention_net
from ntsnet.config import *
from PIL import Image,ImageDraw,ImageFont
import cv2


def imgAddChCharactor(img,text,left,top,color=(255,0,0),size=20):

      if isinstance(img,np.ndarray):
          img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

      draw = ImageDraw.Draw(img)
      font = ImageFont.truetype('/home/yons/anaconda3/envs/balanceGS/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf/STIXGeneralItalic.ttf',size,encoding='utf-8')
      draw.text((left,top),text,color,font)
      return cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)




def deployShow(img,showInfo):

    img = mmcv.imread(img)
    img = img.copy()

    for item in showInfo:
        cv2.rectangle(img, (item[0],item[1]), (item[2],item[3]), (0,0,255), thickness=2)
        #cv2.putText(img, item[4], (item[0],item[1] +18), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,0,0))
        img = imgAddChCharactor(img,item[4],item[0],item[1] +18)
    return img


class LoadImage(object):

    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
        else:
            results['filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

class DeviceLocator(object):

    def __init__(self,config,checkpoint,clsThr=None,device=0):
        '''
        Args:
            config[str]: the config file path for the model
            checkpoint[str]: the model weight file path
            clsThr[list]: the threshold for each class to output
            device[int]:the gpu id to run model
        '''

        if not osp.exists(config):
             raise FileNotFoundError('{} not found.'.format(config))
        if not osp.exists(checkpoint):
            raise FileNotFoundError('{} not found.'.format(checkpoint))

        self.config = mmcv.Config.fromfile(config)
        self.deviceLocator = CascadeRCNN(**self.config.model,test_cfg=self.config.test_cfg)

        checkpoint = load_checkpoint(self.deviceLocator, checkpoint)
        self.deviceLocator.CLASSES = checkpoint['meta']['CLASSES']

        if clsThr == None:
             self.clsThr = [0.5]*len(self.deviceLocator.CLASSES)
             print(Warning('Threshold not specified, using default 0.5'))
        else:
            self.clsThr = clsThr

        self.device = 'cuda:{}'.format(device)
        self.deviceLocator.to(self.device)
        self.deviceLocator.eval()


    def infer(self,img):
        '''
        Args:
            img: image path or a image array

        Returns:

        '''

        test_pipeline = [LoadImage()] + self.config.test_pipeline
        test_pipeline = Compose(test_pipeline)
        # prepare data
        data = dict(img=img)
        data = test_pipeline(data)
        data = scatter(collate([data], samples_per_gpu=1), [torch.device(self.device)])[0]
        # forward the model
        with torch.no_grad():
            result = self.deviceLocator(return_loss=False, rescale=True, **data)


        # filter bb according to threshold
        filterResult = []
        for cls , bbs in enumerate(result[0]):  # result:tuple (bb,segm)
            thr = self.clsThr[cls]
            filterResult.append(bbs[np.nonzero(bbs[:, -1] >= thr)])

        return filterResult



class DeviceRecognizer(object):

    def __init__(self,topN,checkpoint,id2name,device=0):
        '''
        device type classifer
        Args:
            topN: topN part crop from the whole image
            checkpoint: the trained weight file path
            id2name: device id to device type name
            device: gpu id to run model
        '''

        if not osp.exists(checkpoint):
            raise FileNotFoundError('{} not found.'.format(checkpoint))

        if not osp.exists(id2name):
            raise FileNotFoundError('{} not found.'.format(id2name))

        self.recognizer = attention_net(topN=topN,pretrained=False)
        ckcheckpointpt = torch.load(checkpoint)
        self.recognizer.load_state_dict(ckcheckpointpt['net_state_dict'])
        self.recognizer = self.recognizer.to('cuda:{}'.format(device))
        self.recognizer.eval()
        self.device = device

        with open(id2name,'r') as f:
            m = f.readlines()
        self.id2nameMapper = {}
        for it in m:
            self.id2nameMapper[it.strip('\n').split(' ')[1]] = it.strip('\n').split(' ')[0]



    def infer(self,img,bbs):
        '''

          Args:
              img:  the whole image
              bbs:  the bboxes from locator,in format [x1,y1,x2,y2]

          Returns: list of tuple (clsID,clsName,maxProb)

        '''

        if type(img)==str and not osp.exists(img):
            raise FileNotFoundError('{} not found'.format(img))

        image = scipy.misc.imread(img)
        # data = dict(img=img)
        # image = LoadImage()(data)

        results = []
        for bb in bbs:

            patch = image[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2])]
            patch = Image.fromarray(patch, mode='RGB')
            patch = transforms.Resize(INPUT_SIZE, Image.BILINEAR)(patch)
            patch = transforms.ToTensor()(patch)
            patch = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(patch).unsqueeze(0)
            patch = patch.to(torch.device('cuda:0'))

            _, concat_logits, _, _, _ = self.recognizer(patch)
            prob = F.softmax(concat_logits, dim=1)
            maxProb,maxId = torch.max(prob, 1)
            classID = int(maxId.data.item())+1
            className = self.id2nameMapper[str(classID)]
            maxProb=maxProb.data.cpu().item()
            results.append((classID,className,maxProb))

        return results





def test():

    #configuration of locator
    configPathLoc = './cascade_mask_rcnn_r50_fpn.py'
    weightPathLoc = './epoch_48.pth'
    #testImgPath = './sanxin_note3_1b1bd746-fbb7-11ea-9f10-50eb71937875.jpg'
    testImgPath = './lenovo_vibex2_2_0909.jpg'
    device = 0
    clsThr = [0.8171,0.9825,0.9955,0.22,0.196]

    #configuration of recognizer
    weightPathRec = './125.ckpt'
    id2namefile = './deviceID2Name.txt'


    #initialize model
    locator = DeviceLocator(config=configPathLoc,checkpoint=weightPathLoc,clsThr=clsThr,device=device)
    recognizer = DeviceRecognizer(topN=PROPOSAL_NUM,checkpoint=weightPathRec,id2name=id2namefile,device=device)

    #infer
    location = locator.infer(testImgPath)


    toFinegrain = np.concatenate((location[0],location[1])) # only back and officialFront to finegrain
    finalRes = []
    showInfo = []
    if toFinegrain.shape[0] !=0:
        finegrainLabels = recognizer.infer(testImgPath,toFinegrain)
        locs = toFinegrain.tolist()
        for id,(loc , label) in enumerate(zip(locs,finegrainLabels),1):
            item = {}
            item['category'] = label[1]     #device name
            item['identifier'] = label[0]
            item['id_in_image'] = id
            item['prob'] = loc[-1]*label[-1] # finalProb = detProb*finegrainProb
            showInfo.append([int(k) for k in loc[:4]]+[item['category']])
            finalRes.append(item)

        showImg = deployShow(testImgPath, showInfo)
        cv2.imwrite('./result_img.jpg',showImg)

    return finalRes,showImg









if __name__ == '__main__':
    test()
