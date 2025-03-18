import numpy as np
from tkinter import *
from tkinter import ttk
import tkinter as tk
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import cv2
import sys
from collections import Counter


from runDetection import ShipDetection #ship detection main
import argparse

Ship_classNames=['passenger ship',#1
                'ore carrier',#2
                'general cargo ship',#3
                'fishing boat',#4
                'Sail boat',#5
                'Kayak',#6
                'flying bird',#flying bird/plane #7
                'vessel',#vessel/ship #8
                'Buoy',#9
                'Ferry',#10
                'container ship',#11
                'Other',#12
                'Boat',#13
                'Speed boat',#14
                'bulk cargo carrier',#15
                ]

class Window:
    def __init__(self, win, ww, wh):
        self.win = win
        self.ww = ww
        self.wh = wh
        self.win.geometry("%dx%d+%d+%d" % (ww, wh, 200, 50))  # 界面启动时的初始位置
        self.win.title("目标智能监测与航行安全预警预报系统")
        print('正在启动中,请稍等...')
        self.flag1 = False  # 船舶检测标志位
        self.img_src_path = None
        message = tk.Message(self.win, text="目标智能监测与航行安全预警预报系统", font=("微软雅黑", 15,"bold"), width=400)
        message.pack(pady=20)
        # 创建分隔线
        separator = ttk.Separator(self.win, orient="horizontal")
        separator.place(x=30, y=65, relwidth=0.95)

        self.label_src = Label(self.win, text='原始文件', font=('微软雅黑', 12)).place(x=50, y=80)#
        self.label_lic1 = Label(self.win, text='检测结果', font=('微软雅黑', 12)).place(x=550, y=80)
        self.can_src = Canvas(self.win, width=400, height=350, bg='white', relief='solid', borderwidth=1)  # 原图画布
        self.can_src.place(x=50, y=120)
        self.can_lic1 = Canvas(self.win, width=400, height=350, bg='white', relief='solid', borderwidth=1)  # 文字结果画布
        self.can_lic1.place(x=550, y=120)
        # 在底部创建按钮
        button_names = [
            "文件选择",  "船舶检测", "桥梁检测", "沉船检测",
            "抓图", "录像", "停止检测", "退出系统"
        ]

        buttons = []
        for i, name in enumerate(button_names):
            button = tk.Button(self.win, text=name,width=10, height=1,bg='#D3D3D3')
            button.place(x=50 + (i % 4) * 108, y=500 + (i // 4) * 40)
            buttons.append(button)

        #按钮功能实现
        buttons[0].config(command=self.load_show_img)
        buttons[1].config(command=self.function1)#船舶检测
        buttons[2].config(command=self.function2)
        buttons[3].config(command=self.function3)
        buttons[4].config(command=self.function4)
        buttons[7].config(command=self.exit)
        #继续实现其它按钮功能......
    
        # 在右侧框架内创建检测帧率、目标数量、目标类别和信息输出文本框
        tk.Label(self.win,text="检测帧率:").place(x=550, y=500)
        self.frame_rate_entry = tk.Entry(self.win)
        self.frame_rate_entry.place(x=610, y=500, height=25,width=70)

        tk.Label(self.win, text="目标数量:").place(x=690, y=500)
        self.target_count_entry = tk.Entry(self.win)
        self.target_count_entry.place(x=750, y=500, height=25,width=70)

        tk.Label(self.win, text="目标类别:").place(x=830, y=500)
        self.target_type_entry = tk.Entry(self.win)
        self.target_type_entry.place(x=890, y=500, height=25,width=70)

        # 信息输出文本框
        tk.Label(self.win, text="信息输出:").place(x=550, y=540)
        self.info_output_text = tk.Entry(self.win)
        self.info_output_text.place(x=610, y=540,height=40,width=350)
        # self.info_output_text.insert(0, "识别成功")
        # self.info_output_text.insert(0, "识别成功\t\n")
        self.AllInit()  # 所有初始化
        print("已启动,开始识别吧！")
    def function4(self):
        s=self.frame_rate_entry.get()
        self.frame_rate_entry.delete(0,len(s))
    def __del__(self):
        self.AllRelease()
    def show_function1(self, srcImg,dstImg,text="识别成功"):
        '''
        针对功能1
        :param image: cv2 image
        :param text: string
        :return: None
        '''
        self.can_src.delete('all')
        self.can_lic1.delete('all')
        srcimage = Image.fromarray(srcImg[:, :, ::-1])
        dstimage = Image.fromarray(dstImg[:, :, ::-1])
        dstimage = dstimage.resize((self.can_lic1.winfo_width(), self.can_lic1.winfo_height()), Image.ANTIALIAS)
        srcimage = srcimage.resize((self.can_src.winfo_width(), self.can_src.winfo_height()), Image.ANTIALIAS)
        self.img_Tk = ImageTk.PhotoImage(srcimage)
        self.img_Tk1 = ImageTk.PhotoImage(dstimage)
        self.can_src.create_image(0, 0, image=self.img_Tk, anchor='nw')#原图
        self.can_lic1.create_image(0, 0, image=self.img_Tk1, anchor='nw')#结果图
        # self.can_lic1.create_text(35, 15, text=text, anchor='nw', font=('黑体', 12))
        self.info_output_text.insert(0, text+'\t\n')
    # def show_function11(self, srcImg, can,text="识别成功"):
    #     '''
    #     针对功能1
    #     :param image: cv2 image
    #     :param text: string
    #     :return: None
    #     '''
    #     self.can_src.delete('all')
    #     self.can_lic1.delete('all')
    #     srcimage = Image.fromarray(srcImg[:, :, ::-1])
    #     dstimage = Image.fromarray(dstImg[:, :, ::-1])
    #     dstimage = dstimage.resize((can.winfo_width(), can.winfo_height()), Image.ANTIALIAS)#self.can_lic1
    #     #self.can_src
    #     self.img_Tk1 = ImageTk.PhotoImage(dstimage)
    #     can.create_image(0, 0, image=self.img_Tk, anchor='nw')#原图
    #     # self.can_lic1.create_text(35, 15, text=text, anchor='nw', font=('黑体', 12))
    #     self.info_output_text.insert(0, text+'\t\n')

    def _ProcessVideo(self):
        if self.videoC.isOpened():
            Times=2
            ret, frame = self.videoC.read()
            # self.show_function1(frame, "")
            if frame is not None and self.flag1:
                outImg,det=self.shipdetection.mainImg(frame)# det:xyxy, conf, cls
                self.show_function1(frame,outImg,"正在检测")
                self.win.after(Times, self._ProcessVideo)#设置计时器等待Times秒

                #目标数量
                target_str = self.target_count_entry.get()
                self.target_count_entry.delete(0, len(target_str))
                self.target_count_entry.insert(0, str(len(det)))
                # 目标类别
                target_type_list = []
                target_type_str = self.target_type_entry.get()
                self.target_type_entry.delete(0, len(target_type_str))
                target_type_str = ''
                for i in range(len(det)):
                    target_type_list.append(Ship_classNames[int(det[i][5])])
                for k, v in Counter(target_type_list).items():
                    target_type_str += f'{k}:{v}+\t'
                self.target_type_entry.insert(0, str(target_type_str))
            else:
                self.flag1=False
            #帧率
            stemp = self.frame_rate_entry.get()
            self.frame_rate_entry.delete(0, len(stemp))
            self.frame_rate_entry.insert(0,str(1/Times*10))

    def function1(self):
        # 目标检测功能实现，返回图片和文字
        if not self.flag1:
            _, ext = self.src_path.rsplit('.')
            self.flag1 = True
            if ext == 'MOV' or ext == 'mov' or ext == 'avi' or ext == '.mp4':  # video
                self.videoC = cv2.VideoCapture(self.src_path)
                self.win.after(2, self._ProcessVideo)

            else:  # default image
                img = cv2.imread(self.src_path)
                outImg = self.shipdetection.mainImg(img)
                self.show_function1(outImg, "识别结束")
        else:
            self.flag1 = False
        return


    def function2(self):
        #缺陷检测功能实现
        #self.show_function(image,text)
        pass

    def function3(self):
        #沉船检测功能实现
        #self.show_function(image,text)
        pass

        
    def load_show_img(self):
        self.clear()
        sv = StringVar()
        sv.set(askopenfilename())
        self.src_path = Entry(self.win, state='readonly', text=sv).get()  # 返回打开视频的路径


    def clear(self):
        self.can_src.delete('all')
        self.can_lic1.delete('all')
        self.img_src_path = None

    def exit(self): 
        self.win.quit()


    def AllInit(self):#所有算法初始化
        #1 船舶检测
        self.DetectionConfig = argparse.Namespace(flagOCR=False, flagResize=True, flagShow=False,
                                    modelname='detModel/expMAllM.model', img='', video='',
                                    saveDir='save', log='Detlog.txt')  # 程序参数
        self.shipdetection=ShipDetection(self.DetectionConfig)
        #2 缺陷检测


        #3 沉船检测

    def AllRelease(self):#所有算法析构
        #1 船舶检测
        del self.shipdetection

        #2 缺陷检测


        #3 沉船

if __name__ == '__main__':
    win = Tk()
    win.title("目标智能监测与航行安全预警预报系统")
    ww = 1000  # 窗口宽设定1000
    wh = 600  # 窗口高设定600
    Window(win, ww, wh)
    win.protocol("WM_DELETE_WINDOW")
    win.mainloop()
