# BaseMonitorSystem
构建监控系统软件

#安装步骤

1.安装Anaconda：https://repo.anaconda.com/archive/
  或Miniconda：https://docs.conda.io/en/latest/miniconda.html  #对版本没有要求
2. conda create -n Env python=3.7 -y  #创建虚拟环境python版本推荐3.7（已测试）
3. conda activate Env #进入虚拟环境
4. conda install geos -y#安装模块
5. pip install -r requirements.txt #安装依赖库 或者requirements37.txt
6. python setup.py build_ext --inplace #程序编译打包，win系统或Ubuntu系统编译，生成packagesShipCode（打包执行文件）
7. 船舶检测与船舷号识别，分别包含两种运行方式：单文件执行和批量文件执行


#1船舶检测##############################
1.1 python run.py --flagShow --modelname detModel/YOLOS.model --saveDir save --img TestData/demoShipDet.jpg --video TestData/DSC_6186.MOV#单文件模式 file.avi file.mov均可
或
1.2 python run.py --img %.jpg --modelname detModel/YOLOS.model --saveDir save --video TestData/%.MOV#批量文件模式 %.avi\%.mov均可
1.3 python run.py --modelname detModel/YOLOS.model --saveDir save --log Detlog.log --video TestData/DSC_6186.MOV #单视频

#************************************************************#
#2船舶检测+船舷号识别############################
2.1 python run.py --flagOCR --modelname detModel/YOLOS.model --modelpath . --flagShow --img TestData/%.jpg --video TestData/%.avi --saveDir saveT#单文件模式：文件为绝对路径，批量模式：路径/%.avi(%代替所有任意名字，视频后缀avi/mov均可)


#************************************************************#

8.结果文件输出 --saveDir路径下
#船舶检测##############################
输出json文件格式：[xmin,ymin,xmax,ymax,置信度,类别

#船舷号识别############################
输出json文件格式：[(x1,y1),(x2,y2),(x3,y3),(x4,y4),置信度,文本]#矩形的四个顶点



#Todo
系统模块优化：
1. 数据输入与预处理
2. 目标检测
3. 目标跟踪
4. 后处理：自动标注、可视化
5. 日志
