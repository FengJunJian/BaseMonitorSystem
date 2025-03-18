import argparse
from runDetection import main
from runOCR import mainOCR
import os
if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    #Common
    parser.add_argument('--flagCamera', action='store_true', help='whether to use camera,eg:--flagCamera',)
    parser.add_argument('--img', type=str, default='', help='image file, eg:--img PATH/demo.jpg')
    parser.add_argument('--saveDir', type=str, default='',
                        help='if want to save the result of image or video, Please place the Path,eg:--saveDir Path')
    #Ship OCR
    parser.add_argument('--flagOCR', action='store_true', help='whether to carrier the OCR,eg:--flagOCR', )
    parser.add_argument('--modelpath',type=str,default='',help='path to the ship OCR model,eg:PATH')

    #Ship Detction
    parser.add_argument('--flagShow', action='store_true', help='whether to show the results,eg:--flagShow', )
    parser.add_argument('--flagResize', action='store_true', help='whether to resize the image,eg:--flagResize', )
    parser.add_argument('--modelname', type=str, default='',
                        help='file of detection model, eg:PATH/demo.model')  
    parser.add_argument('--video',  type=str, default='', help='video file, eg:--video PATH/demo.mp4')
    parser.add_argument('--log', type=str, default='Detlog.txt',
                        help='Using default name if None')

    config = parser.parse_args()
    print(os.getcwd())
#--flagOCR --img ../../ShipDataSet/ShipText/%.jpg --modelpath . --saveDir save
#--img ../../ShipDataSet/BXShipDataset/JPEGImages/%.jpg --modelname detModel/YOLOS.model --flagShow --saveDir save
#config=argparse.Namespace(flagCamera=True, flagOCR=False, flagResize=True, flagShow=True, modelname='detModel/YOLOM.model', img='TestData/%.jpg', video='TestData/%.MOV',saveDir='save',log='Detlog.txt',)#程序参数

    if config.flagOCR:
        mainOCR(config)
    else:
        main(config)

