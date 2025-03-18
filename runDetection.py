import cv2
import os
import sys
import numpy as np
import json
import time
from runYOLO import mainYOLO,Init,ncolors,Ship_classNames
#from runFasterRCNN import mainFasterRCNN
from loguru import logger
import datetime
#from DatasetClass import Ship_classNames
from tqdm import tqdm
from glob import glob

def main(config):
    #parser.print_help()
    #config = parser.parse_args()
    imgPath = config.img  # MVI_1592_VIS_00462
    videoPath = config.video
    modelname = config.modelname
    flagShow = config.flagShow
    saveDir = config.saveDir
    flagResize=config.flagResize
    log = config.log
    #showFun=config.showFun
    NUM = len(Ship_classNames)  # - 1
    colors = np.array(ncolors(NUM))

    if not log:
        log='log'+datetime.datetime.now().strftime('%Y%m%d%H%M%S')+'.log'
    logger.add(log, mode='a')
    logger.info(str([imgPath,videoPath,modelname,flagShow,saveDir,log]))
    # ImageWriter=None
    if not videoPath and not imgPath:
        #parser.print_help()
        logger.warning("Both videoPath and imgPath are None!")
        sys.exit(1)
        #raise ValueError('Error:None of img and video')
    logger.info("Reading the detection model:%s" % (modelname))
    try:
        Session=Init(modelname)
    except:
        logger.error("Error for Reading the detection model:%s" % (modelname))
        sys.exit(1)
    img=None
    if imgPath:
        if '%.jpg' in imgPath or '%.png' in imgPath or '%.JPG' in imgPath:
            ind=imgPath.rfind('%')
            imgPath=''.join([imgPath[:ind],'*',imgPath[ind+1:]])
            imgPaths=glob(imgPath)
        else:
            imgPaths=[imgPath]
        logger.info('imgPath:%s with %d images'%(imgPath,len(imgPaths)))
        for i in tqdm(range(len(imgPaths))):
            imgP=imgPaths[i]
            img = cv2.imread(imgP)
            if img is None:
                logger.error("Error:The imgPath %s doesn't exist!%"%(imgP))
                continue
                #sys.exit(1)

            if flagResize:
                img=cv2.resize(img,(1920,1080))

            logger.info("detecting..." )
            a=time.time()
            # outImg=img
            # dets=np.array([])
            #for j in tqdm(range(100)):
            outImg,dets=mainYOLO(img,Session,colors)#Detection det:(cxmin,ymin,xmax,ymax,score,clsId)
            # cv2.imshow('a',outImg)
            # cv2.waitKey(2)
            #cv2.waitKey(100)
                # outImg, dets = mainFasterRCNN(img, Session, colors)  # Detection det:(cxmin,ymin,xmax,ymax,score,clsId)
            b=time.time()
            logger.info("detection finished:%fs"%(b-a))
            # if showFun:
            #     showFun(outImg)

            if flagShow:
                cv2.imshow("imgResult",outImg)
                cv2.waitKey(10)
            if saveDir:
                if not os.path.exists(saveDir):
                    os.makedirs(saveDir)
                basename=os.path.basename(imgP)
                savingfile=os.path.join(saveDir, basename)
                logger.info("Saving the image%s"%(savingfile))
                cv2.imwrite(savingfile,outImg)
                cv2.waitKey(2)
                basename=os.path.splitext(basename)[0]
                savingfile = os.path.join(saveDir, basename+'.json')
                logger.info("Saving the json file %s for Image!" % (savingfile))
                with open(savingfile,'w') as f:
                    json.dump(dets.tolist(),f)#(xmin,ymin,xmax,ymax,score,cls)
            del img
            del outImg
            del dets

    VideoWriter = None
    waitK=0
    videoJson={}
    if videoPath:
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
                savingfile = os.path.join(saveDir, basename+'.avi')
                logger.info("write the video results:%s"%(savingfile))
                VideoWriter=cv2.VideoWriter(savingfile,fourcc,fps,(width,height),True)

            logger.info('reading the video!')

            total_time=0
            for count in tqdm(range(totalFrame)):
                ret, frame = cap.read()
                if not ret:
                    break
                logger.info("Video detecting.....")
                a = time.time()
                outImg, dets = mainYOLO(frame, Session, colors)
                b = time.time()
                logger.info("frame %d finished:%fs" % (count,b - a))
                total_time+=(b-a)
                videoJson.update({count:dets.tolist()})
                if flagShow:
                    showImg=None
                    logger.warning("input key 'q' to quit and key 'c' to next VideoFile!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ")
                    showImg=cv2.resize(outImg,None,fx=0.5,fy=0.5)
                    cv2.imshow("VideoResult", showImg)
                    waitK=cv2.waitKey(2)
                    del showImg
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
            savingfile = os.path.join(saveDir, basename + '.json')
            logger.info("Saving the json file %s for Video!" % (savingfile))
            with open(savingfile, 'w') as f:
                json.dump(videoJson, f)  # (xmin,ymin,xmax,ymax,score,cls)
            cap.release()
            VideoWriter.release()
            logger.info("video detection average tiem per frame:%fs" % (total_time / count))
        cv2.destroyAllWindows()

class ShipDetection:
    def __init__(self,config):
        # self.imgPath = config.img  # MVI_1592_VIS_00462
        # self.videoPath = config.video
        self.modelname = config.modelname
        self.flagShow = config.flagShow
        self.saveDir = config.saveDir
        self.flagResize = config.flagResize
        self.log = config.log
        self.NUM = len(Ship_classNames)  # - 1
        self.colors = np.array(ncolors(self.NUM))
        self.Session=None
        if not self.log:
            self.log = 'log' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.log'
        logger.add(self.log, mode='a')
        logger.info(str([self.modelname, self.flagShow, self.saveDir, self.log]))
        # ImageWriter=None
        # if not videoPath and not imgPath:
        #     # parser.print_help()
        #     logger.warning("Both videoPath and imgPath are None!")
        #     sys.exit(1)

        logger.info("Reading the detection model:%s" % (self.modelname))
        try:
            self.Session = Init(self.modelname)
        except:
            logger.error("Error for Reading the detection model:%s" % (self.modelname))

    def mainImg(self,image):
        if image is None:
            logger.error("Error:The imgPath %s doesn't exist!%" % (image))
            return None
        img=image.copy()
        #img = cv2.imread(imgP)

        if self.flagResize:
            img=cv2.resize(img,(1920,1080))

        logger.info("detecting..." )
        a=time.time()
        outImg,dets=mainYOLO(img,self.Session,self.colors)#Detection det:(cxmin,ymin,xmax,ymax,score,clsId)
        b=time.time()
        logger.info("detection finished:%fs"%(b-a))

        if self.flagShow:
            cv2.imshow("imgResult",outImg)
            cv2.waitKey(3)
        # if self.saveDir:
            # if not os.path.exists(self.saveDir):
            #     os.makedirs(self.saveDir)
            # basename=os.path.basename(imgP)
            # savingfile=os.path.join(saveDir, basename)
            # logger.info("Saving the image%s"%(savingfile))
            # cv2.imwrite(savingfile,outImg)
            # cv2.waitKey(2)
            # basename=os.path.splitext(basename)[0]
            # savingfile = os.path.join(saveDir, basename+'.json')
            # logger.info("Saving the json file %s for Image!" % (savingfile))
            # with open(savingfile,'w') as f:
            #     json.dump(dets.tolist(),f)#(xmin,ymin,xmax,ymax,score,cls)
        del img
        #del dets
        return outImg,dets


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--img', type=str, default='', help='image file, eg:--img PATH/demo.jpg')
    parser.add_argument('--video',  type=str, default='', help='video file, eg:--video PATH/demo.mp4')
    parser.add_argument('--modelname', type=str, default='', help='file of detection model, eg:PATH/demo.model')#'./FasterRCNN.model'
    parser.add_argument('--flagShow',action='store_true',help='whether to show the results,eg:--flagShow',)
    parser.add_argument('--saveDir',type=str,default='',help='if want to save the result of image or video, Please place the Path,eg:--saveDir Path')
    parser.add_argument('--log', type=str, default='',
                        help='Using default name if None')
    #parser.add_argument('-h','--h' action='store_true', help='whether to show the results,eg:--flagShow', )
    config = parser.parse_args()
    # config = argparse.Namespace(img = 'MVI_1592_VIS_00462.jpg', video='DSC_6252.MOV',modelname='detModel/YOLOS.model',flagShow=True,saveDir='save',log='log.log')
    # config = argparse.Namespace(img='../FailShip/*.jpg', video='G:/ShipDataSet/shi/*.MOV', modelname='detModel/YOLOS.model',
    #                             flagShow=True, saveDir='save', log='log.log')

    main(config)
    


