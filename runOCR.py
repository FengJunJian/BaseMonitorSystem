import onnxruntime

from PIL import Image
from PIL import ImageDraw,ImageFont
import cv2
#import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
import pyclipper
import copy
import time
import math
from loguru import logger
import sys
from glob import glob
import os
import json

#Detection
from runYOLO import mainYOLO,Init,ncolors,Ship_classNames,COCO_classNames
#
#####################################textDector
class textDector(object):
    def __init__(self,onnxPath):
        self.use_onnx=True
        self.det_algorithm='DB'
        self.predictor=onnxruntime.InferenceSession(onnxPath)
        self.input_tensor=self.predictor.get_inputs()[0]
        self.unclip_ratio=1.5
        self.dstShape = (640, 960)  # （640，960）
        self.std= [0.229, 0.224, 0.225],
        self.mean= [0.485, 0.456, 0.406],
        self.scale= 1./255.
        self.max_candidates=1000
        self.min_size=3
        self.score_mode = 'fast'
        self.box_thresh=0.6

    def get_mini_boxes(self,contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        return box, min(bounding_box[1])
    def box_score_fast(self,bitmap, _box):
        '''
        box_score_fast: use bbox mean score as the mean score
        '''
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]
    def box_score_slow(self,bitmap, contour):
        '''
        box_score_slow: use polyon mean score as the mean score
        '''
        h, w = bitmap.shape[:2]
        contour = contour.copy()
        contour = np.reshape(contour, (-1, 2))

        xmin = np.clip(np.min(contour[:, 0]), 0, w - 1)
        xmax = np.clip(np.max(contour[:, 0]), 0, w - 1)
        ymin = np.clip(np.min(contour[:, 1]), 0, h - 1)
        ymax = np.clip(np.max(contour[:, 1]), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin

        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]
    def unclip(self, box):
        unclip_ratio = self.unclip_ratio
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded
    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        '''

        bitmap = _bitmap
        height, width = bitmap.shape

        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            if self.score_mode == "fast":
                score = self.box_score_fast(pred, points.reshape(-1, 2))
            else:
                score = self.box_score_slow(pred, contour)
            if self.box_thresh > score:
                continue

            box = self.unclip(points).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype(np.int16))
            scores.append(score)
        return np.array(boxes, dtype=np.int16), scores
    def postprocess(self,pred, shape_list, thresh=0.3):
        # pred = outs_dict['maps']
        pred = pred[:, 0, :, :]
        segmentation = pred > thresh  # self.thresh
        #np.where(segmentation)
        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            # if self.dilation_kernel is not None:
            #     mask = cv2.dilate(
            #         np.array(segmentation[batch_index]).astype(np.uint8),
            #         self.dilation_kernel)
            # else:
            mask = segmentation[batch_index]
            boxes, scores = self.boxes_from_bitmap(pred[batch_index], mask,
                                              src_w, src_h)

            boxes_batch.append({'points': boxes})
        return boxes_batch

    def preprocess(self,img):
        assert isinstance(img, (np.ndarray, list, str))
        if isinstance(img, str):
            # imread
            ori_im=cv2.imread(img)
            if ori_im is None:
                raise "error in loading image:{}".format(img)
        elif isinstance(img, np.ndarray) and len(img.shape) == 2:
            ori_im = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            ori_im=img
        # else:
        #     raise ValueError('Error Type:{}'.format(img))
        self.ori_im=ori_im
        ori_im=ori_im.astype(np.float32)
        ori_im*=self.scale
        ori_im-=self.mean
        ori_im/=self.std
        #ori_im = cv2.imread(img_path)
        # 'DetResizeForTest': {
        #                 'limit_side_len': args.det_limit_side_len,
        #                 'limit_type': args.det_limit_type,
        #             }
        #         }, {
        #             'NormalizeImage': {
        #                 'std': [0.229, 0.224, 0.225],
        #                 'mean': [0.485, 0.456, 0.406],
        #                 'scale': '1./255.',
        #                 'order': 'hwc'
        #             }
        #self.dstShape
        shape_list = np.array(
            [ori_im.shape[0], ori_im.shape[1], self.dstShape[0] / ori_im.shape[0], self.dstShape[1] / ori_im.shape[1]])
        image = cv2.resize(ori_im, (self.dstShape[1],self.dstShape[0]))
        image = np.expand_dims(image, 0)
        shape_list = np.expand_dims(shape_list, 0)
        image = image.astype(np.float32).transpose([0, 3, 1, 2])
        inputDict = {'x': image}
        return inputDict,shape_list
        #pass
    def detector(self,img):
        inputDict,shape_list=self.preprocess(img)
        self.resultMap = self.predictor.run(output_names=None, input_feed=inputDict)
        post_result = self.postprocess(self.resultMap[0], shape_list)
        dt_boxes = post_result[0]['points']
        dt_boxes = self.filter_tag_det_res(dt_boxes, self.ori_im.shape)
        dt_boxes = self.sorted_boxes(dt_boxes)
        img_crop_list = []
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(self.ori_im, tmp_box)
            img_crop_list.append(img_crop)
        return img_crop_list,dt_boxes
    ######################################################
    #####################################################filter_tag_det_res
    def order_points_clockwise(self,pts):
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(self,points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points
    def filter_tag_det_res(self,dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes
    #####################################################
    ######################################################sortbox
    def sorted_boxes(self,dt_boxes):
        """
        Sort text boxes in order from top to bottom, left to right
        args:
            dt_boxes(array):detected text boxes with shape [4, 2]
        return:
            sorted boxes(array) with shape [4, 2]
        """
        num_boxes = dt_boxes.shape[0]
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)

        for i in range(num_boxes - 1):
            if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                    (_boxes[i + 1][0][0] < _boxes[i][0][0]):
                tmp = _boxes[i]
                _boxes[i] = _boxes[i + 1]
                _boxes[i + 1] = tmp
        return _boxes
    ######################################################
    #####################################get_rotate_crop_image
    def get_rotate_crop_image(self,img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        assert len(points) == 4, "shape of points must be 4*2"
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img
#####################################

#####################################textRecognizer
class textRecognizer(object):
    def __init__(self,onnxPath,character_dict_path,use_space_char=True):
        self.use_onnx=True
        self.rec_algorithm='CRNN'
        self.character_str=[]
        self.predictor=onnxruntime.InferenceSession(onnxPath)
        self.input_tensor=self.predictor.get_inputs()[0]
        self.output_tensors=None#self.predictor.get_outputs()[0].name
        with open(character_dict_path, "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode('utf-8').strip("\n").strip("\r\n")
                self.character_str.append(line)
        if use_space_char:
            self.character_str.append(" ")
        dict_character = list(self.character_str)
        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        #dict_character = list(self.character_str)
        self.character=dict_character
    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character
    def resize_norm_img(self,img, max_wh_ratio,imgC=3, imgH=32, imgW=320):
        #imgC, imgH, imgW = self.rec_image_shape
        assert imgC == img.shape[2]
        imgW = int((32 * max_wh_ratio))
        if self.use_onnx:
            w = self.input_tensor.shape[3:][0]
            if w is not None and w > 0:
                imgW = w

        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im
    def textRecognizer(self,img_list,rec_batch_num=6):
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))
        rec_res = [['', 0.0]] * img_num
        batch_num = rec_batch_num
        #st = time.time()

        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            max_wh_ratio = 0
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                #CRNN
                norm_img = self.resize_norm_img(img_list[indices[ino]],
                                                max_wh_ratio)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()

            if self.use_onnx:
                input_dict = {}
                input_dict[self.input_tensor.name] = norm_img_batch
                outputs = self.predictor.run(self.output_tensors,
                                             input_dict)
                preds = outputs[0]
            else:
                self.input_tensor.copy_from_cpu(norm_img_batch)
                self.predictor.run()
                outputs = []
                for output_tensor in self.output_tensors:
                    output = output_tensor.copy_to_cpu()
                    outputs.append(output)

                if len(outputs) != 1:
                    preds = outputs
                else:
                    preds = outputs[0]
            rec_result = self.postprocess(preds)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]

        return rec_res#, time.time() - st
    def postprocess(self,preds, label=None, *args, **kwargs):
        if isinstance(preds, tuple):
            preds = preds[-1]
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        if label is None:
            return text
        label = self.decode(label)
        return text, label
    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = [0]#self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if is_remove_duplicate:
                    # only for predict
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[
                            batch_idx][idx]:
                        continue
                char_list.append(self.character[int(text_index[batch_idx][
                    idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list)))
        return result_list
######################################
CHONNXocrDict={
    'det_model_dir':'detModel/ocr/det/ch/ch_PP-OCRv2_det_infer/det.onnx',
    'rec_model_dir':'detModel/ocr/rec/ch/ch_PP-OCRv2_rec_infer/rec.onnx',
    'cls_model_dir':'detModel/ocr/cls/ch_ppocr_mobile_v2.0_cls_infer/cls.onnx',
    'rec_char_dict_path':'detModel/ocr/ppocr_keys_v1.txt',
    }


def mainOCR(config):
    logger.add('OCRlog.txt', mode='a')
    saveDir = config.saveDir
    imgPath=config.img
    videoPath = config.video
    modelpath = config.modelpath
    flagShow=config.flagShow
    #flagResize=config.flagResize
    modelname = config.modelname
    flagCamera=config.flagCamera
    className=None
    if flagCamera:
        className= COCO_classNames
    else:
        className = Ship_classNames # - 1
    NUM = len(className)
    colors = np.array(ncolors(NUM))
    if not modelpath:
        modelpath='.'
    try:
        Session=Init(modelname,className)
    except:
        logger.error("Error for Reading the detection model:%s" % (modelname))
        sys.exit(1)
    # logger.warning("imgPath is None!")
    # sys.exit(1)
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    logger.info(str([imgPath]))
    TextDec = textDector(os.path.join(modelpath, CHONNXocrDict['det_model_dir']))
    TextRec = textRecognizer(os.path.join(modelpath, CHONNXocrDict['rec_model_dir']),
                             os.path.join(modelpath, CHONNXocrDict['rec_char_dict_path']))
    logger.info('Text detector and recognizer are loaded!')
    ft = ImageFont.truetype(os.path.join(modelpath, 'fonts/simsun.ttc'), size=36)  # 30

    if imgPath:
        imgPaths=None
        if isinstance(imgPath,str):
            if '%.jpg' in imgPath or '%.png' in imgPath:
                ind = imgPath.rfind('%')
                imgPath = ''.join([imgPath[:ind], '*', imgPath[ind + 1:]])
                imgPaths = glob(imgPath)
            else:
                imgPaths = [imgPath]
        else:
            raise ValueError('Input Image paths error:{}'.format(imgPath))
        for indx,imgN in enumerate(imgPaths):
            logger.info("{}:Reading image:{}".format(indx,imgN))
            basename = os.path.basename(imgN)#name.ext
            basename = os.path.splitext(basename)[0]#name
            img=cv2.imread(imgN)
            # if flagResize:
            #     img=cv2.resize(img,(1920,1080))
            outImg, dets = mainYOLO(img, Session, colors)
            img_crop_list, dt_boxes = TextDec.detector(img)  # detector
            rec_res = TextRec.textRecognizer(img_crop_list)  # recognizer
            # if self.args.save_crop_res:
            #     self.draw_crop_rec_res(self.args.crop_res_save_dir, img_crop_list,
            #                            rec_res)
            filter_boxes, filter_rec_res = [], []
            for box, rec_reuslt in zip(dt_boxes, rec_res):
                text, score = rec_reuslt
                if score >= 0.4:  # self.drop_score:
                    filter_boxes.append(box)
                    filter_rec_res.append(rec_reuslt)

            boxes = np.array(filter_boxes)
            txts = [line[0] for line in filter_rec_res]
            scores = [line[1].tolist() for line in filter_rec_res]

            #image_arr = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
            #image = Image.open(img).convert('RGB')

            image = Image.fromarray(cv2.cvtColor(outImg, cv2.COLOR_BGR2RGB))
            #image = Image.fromarray(image_arr)

            imageC = image.copy()
            draw = ImageDraw.ImageDraw(imageC)
            outputs = []
            for i, box in enumerate(boxes):
                if scores[i] < 0.1:
                    continue
                output = [tuple(b) for b in box.tolist()]
                output.extend([scores[i], txts[i]])
                draw.polygon([tuple(b) for b in box], outline=(255, 0, 10))  #
                w, h = ft.getsize(txts[i])
                draw.text((box[0][0], box[0][1] - h), txts[i], font=ft, fill='red')
                outputs.append(output)

            if flagShow:
                #logger.warning("input key 'q' to quit and key 'c' to next VideoFile!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ")
                outImg = cv2.cvtColor(np.asarray(imageC), cv2.COLOR_RGB2BGR)
                cv2.imshow("VideoResult", outImg)
                waitK = cv2.waitKey(2)
                #del outImg
                if waitK == ord('q'):
                    logger.warning("System quit with the button 'q' pressed!!!!")
                    sys.exit(1)
                elif waitK == ord('c'):
                    logger.warning("Next VideoFile with the button 'c' pressed!!!!")
                    break

            savingfile = os.path.join(saveDir, basename + 'OCR.jpg')
            imageC.save(savingfile)
            logger.info('Saving image:{}'.format(savingfile))
            savingfile = os.path.join(saveDir, basename + 'OCR.json')
            logger.info("Saving the json file %s for Image!\n" % (savingfile))
            with open(savingfile, 'w') as f:
                jsonContent={'texts':outputs,'objects':dets.tolist()}
                json.dump(jsonContent, f)  #{'objects':[[xmin,ymin,xmax,ymax,score,cls],]
                # 'texts':[[(x1,y1),(x2,y2),(x3,y3),(x4,y4),score,text],] }

    VideoWriter = None
    waitK = 0
    videoJson = {}
    if flagCamera:#采用摄像头
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Error:camera doesn't open!")
            sys.exit(1)
        totalFrame = int(0)
        logger.warning("Total frame number is %d" % (totalFrame))
        if saveDir:
            if not os.path.exists(saveDir):
                os.makedirs(saveDir)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            fps = round(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            basename = os.path.basename('Camera'+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
            #basename = os.path.splitext(basename)[0]
            savingfile = os.path.join(saveDir, basename + 'OCR.avi')
            logger.info("write the video results:%s" % (savingfile))
            VideoWriter = cv2.VideoWriter(savingfile, fourcc, fps, (width, height), True)

        logger.info('reading the video!')

        total_time = 0
        count=1
        while True:#for count in range(totalFrame):  # 开始识别
            ret, frame = cap.read()
            if not ret:
                break
            logger.info("Video detecting.....")
            a = time.time()

            outImg, dets = mainYOLO(frame, Session, colors)
            img_crop_list, dt_boxes = TextDec.detector(frame)  # detector
            rec_res = TextRec.textRecognizer(img_crop_list)  # recognizer
            # if self.args.save_crop_res:
            #     self.draw_crop_rec_res(self.args.crop_res_save_dir, img_crop_list,
            #                            rec_res)
            filter_boxes, filter_rec_res = [], []
            for box, rec_reuslt in zip(dt_boxes, rec_res):
                text, score = rec_reuslt
                if score >= 0.5:  # self.drop_score:
                    filter_boxes.append(box)
                    filter_rec_res.append(rec_reuslt)
            # return filter_boxes, filter_rec_res
            # [[box.tolist(), res] for box, res in zip(dt_boxes, rec_res)]
            # plt.imshow(resultMap[0][0,0])
            # print(filter_boxes)
            # print(filter_rec_res)
            boxes = np.array(filter_boxes)
            txts = [line[0] for line in filter_rec_res]
            scores = [line[1].tolist() for line in filter_rec_res]

            # image_arr = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(cv2.cvtColor(outImg, cv2.COLOR_BGR2RGB))
            # image = Image.open(img).convert('RGB')

            imageC = image.copy()
            draw = ImageDraw.ImageDraw(imageC)
            outputs = []
            for i, box in enumerate(boxes):
                if scores[i] < 0.1:
                    continue
                output = [tuple(b) for b in box.tolist()]
                output.extend([scores[i], txts[i]])
                draw.polygon([tuple(b) for b in box], outline=(255, 0, 10))  #
                w, h = ft.getsize(txts[i])
                draw.text((box[0][0], box[0][1] - h), txts[i], font=ft, fill='red')
                outputs.append(output)
            b = time.time()
            logger.info("frame %d finished:%fs" % (count, b - a))
            total_time += (b - a)
            # videoJson.update({count:dets.tolist()})
            outImg = cv2.cvtColor(np.asarray(imageC), cv2.COLOR_RGB2BGR)
            #if flagShow:
            logger.warning("input key 'q' to quit!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ")
            cv2.imshow("VideoResult", outImg)
            waitK = cv2.waitKey(2)
            if waitK == ord('q'):
                logger.warning("System quit with the button 'q' pressed!!!!")
                sys.exit(1)
            # elif waitK == ord('c'):
            #     logger.warning("Next VideoFile with the button 'c' pressed!!!!")
            #     break
            if VideoWriter:
                logger.info("writing the video!")
                VideoWriter.write(outImg)
            cap.read()
            count=count+1
        # savingfile = os.path.join(saveDir, basename + '.json')
        # logger.info("Saving the json file %s for Video!" % (savingfile))
        # with open(savingfile, 'w') as f:
        #     json.dump(videoJson, f)  # (xmin,ymin,xmax,ymax,score,cls)
        cap.release()
        VideoWriter.release()
        logger.info("video detection average tiem per frame:%fs" % (total_time / count))
    elif videoPath:
        ##############################
        if '%.MOV' in videoPath or '%.mov' in videoPath or '%.avi' in videoPath or'%.mp4' in videoPath:
            ind=videoPath.rfind('%')
            videoPath=''.join([videoPath[:ind],'*',videoPath[ind+1:]])
            #videoPath=videoPath.replace('#','*')
            videoPaths=glob(videoPath)
        else:
            videoPaths=[videoPath]
        logger.info('videoPath:%s' % (videoPath))
        for i in range(len(videoPaths)):
            videoP=videoPaths[i]
            cap=cv2.VideoCapture(videoP)
            if not cap.isOpened():
                logger.error("Error:Videofile %s doesn't exist!"%(videoP))
                continue
                #sys.exit(1)
            totalFrame=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.warning("Total frame number is %d"%(totalFrame))
            if saveDir:
                if not os.path.exists(saveDir):
                    os.makedirs(saveDir)
                fourcc=cv2.VideoWriter_fourcc(*'XVID')
                fps=round(cap.get(cv2.CAP_PROP_FPS))
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                basename = os.path.basename(videoP)
                basename=os.path.splitext(basename)[0]
                savingfile = os.path.join(saveDir, basename+'OCR.avi')
                logger.info("write the video results:%s"%(savingfile))
                VideoWriter=cv2.VideoWriter(savingfile,fourcc,fps,(width,height),True)

            logger.info('reading the video!')

            total_time=0
            for count in range(totalFrame):#开始识别
                ret, frame = cap.read()
                if not ret:
                    break
                logger.info("Video detecting.....")
                a = time.time()


                outImg, dets = mainYOLO(frame, Session, colors)
                img_crop_list, dt_boxes = TextDec.detector(frame)  # detector
                rec_res = TextRec.textRecognizer(img_crop_list)  # recognizer
                # if self.args.save_crop_res:
                #     self.draw_crop_rec_res(self.args.crop_res_save_dir, img_crop_list,
                #                            rec_res)
                filter_boxes, filter_rec_res = [], []
                for box, rec_reuslt in zip(dt_boxes, rec_res):
                    text, score = rec_reuslt
                    if score >= 0.5:  # self.drop_score:
                        filter_boxes.append(box)
                        filter_rec_res.append(rec_reuslt)
                # return filter_boxes, filter_rec_res
                # [[box.tolist(), res] for box, res in zip(dt_boxes, rec_res)]
                # plt.imshow(resultMap[0][0,0])
                # print(filter_boxes)
                # print(filter_rec_res)
                boxes = np.array(filter_boxes)
                txts = [line[0] for line in filter_rec_res]
                scores = [line[1].tolist() for line in filter_rec_res]

                # image_arr = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(cv2.cvtColor(outImg, cv2.COLOR_BGR2RGB))
                # image = Image.open(img).convert('RGB')
                # image = Image.fromarray(image_arr)

                imageC = image.copy()
                draw = ImageDraw.ImageDraw(imageC)
                outputs = []
                for i, box in enumerate(boxes):
                    if scores[i] < 0.1:
                        continue
                    output = [tuple(b) for b in box.tolist()]
                    output.extend([scores[i], txts[i]])
                    draw.polygon([tuple(b) for b in box], outline=(255, 0, 10))  #
                    w, h = ft.getsize(txts[i])
                    draw.text((box[0][0], box[0][1] - h), txts[i], font=ft, fill='red')
                    outputs.append(output)
                b = time.time()
                logger.info("frame %d finished:%fs" % (count,b - a))
                total_time+=(b-a)
                #videoJson.update({count:dets.tolist()})
                outImg = cv2.cvtColor(np.asarray(imageC), cv2.COLOR_RGB2BGR)
                if flagShow:
                    logger.warning("input key 'q' to quit and key 'c' to next VideoFile!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ")
                    cv2.imshow("VideoResult", outImg)
                    waitK=cv2.waitKey(2)
                    if waitK == ord('q'):
                        logger.warning("System quit with the button 'q' pressed!!!!")
                        sys.exit(1)
                    elif waitK==ord('c'):
                        logger.warning("Next VideoFile with the button 'c' pressed!!!!")
                        break
                if VideoWriter:
                    logger.info("writing the video!")
                    VideoWriter.write(outImg)
                cap.read()
            #savingfile = os.path.join(saveDir, basename + '.json')
            # logger.info("Saving the json file %s for Video!" % (savingfile))
            # with open(savingfile, 'w') as f:
            #     json.dump(videoJson, f)  # (xmin,ymin,xmax,ymax,score,cls)
            cap.release()
            VideoWriter.release()
            logger.info("video detection average tiem per frame:%fs" % (total_time / count))

        ##############################
            #savingfile = os.path.join(saveDir, basename + 'OCR.avi')
            #imageC.save(savingfile)
            # logger.info('Saving Video:{}'.format(savingfile))
            # savingfile = os.path.join(saveDir, basename + 'OCR.json')
            # logger.info("Saving the json file %s for Videos!\n" % (savingfile))
            # with open(savingfile, 'w') as f:
            #     json.dump(outputs, f)  # [(x1,y1),(x2,y2),(x3,y3),(x4,y4),score,text]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, default='', help='image file, eg:--img PATH/demo.jpg')
    # parser.add_argument('--video', type=str, default='', help='video file, eg:--video PATH/demo.mp4')
    # parser.add_argument('--modelname', type=str, default='',
    #                     help='file of detection model, eg:PATH/demo.model')  # './FasterRCNN.model'
    # parser.add_argument('--flagShow', action='store_true', help='whether to show the results,eg:--flagShow', )
    parser.add_argument('--saveDir', type=str, default='',
                        help='if want to save the result of image or video, Please place the Path,eg:--saveDir Path')
    parser.add_argument('--modelpath', type=str, default='', help='path to the ship ocr model')
    # parser.add_argument('--log', type=str, default='',
    #                     help='Using default name if None')
    # parser.add_argument('-h','--h' action='store_true', help='whether to show the results,eg:--flagShow', )
    config = parser.parse_args()
    #img_path='H:\ShipDataSet\ShipText\\1643180070.jpg'
    imageC, outputs=mainOCR(config)
    imageC.show()
    print(outputs)#[(x1,y1),(x2,y2),(x3,y3),(x4,y4),score,text]


