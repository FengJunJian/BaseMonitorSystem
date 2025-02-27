def horizon_detect(img):

    if img is None:
        return 0,0,0
    if isinstance(img,Image.Image):
        #img=np.array(img)
        imgray=np.array(img.convert('L'))
    else:#default:cv2
        if len(img.shape)==3:
            imgray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        else:
            imgray=img
    #imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgray = cv2.GaussianBlur(imgray, (3, 3), 0, 0)
    thesd = 0.0
    Mkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thesd, imggtem = cv2.threshold(imgray, thesd, 255, cv2.THRESH_OTSU)
    dilatedgray = cv2.dilate(imggtem, kernel=Mkernel)

    # imgsrc = img.copy()
    # cv2.imshow('dt', dilatedgray)
    # cv2.imshow('gt', imggtem)
    # cv2.waitKey(1)

    imgmask = cv2.reduce(dilatedgray, 1, cv2.REDUCE_AVG, cv2.CV_16S)  # //reduce to single column average
    row, col = imgray.shape
    # row = imgray.rows, col = imgray.cols;
    kuan_hight = round(row / 20)

    horizon_top = 0
    horizon_bottom = row - 1  # ;//区域上下界
    thesd, imgtem = cv2.threshold(imgmask, thesd, 255, cv2.THRESH_OTSU)
    imgtemd = np.abs(cv2.filter2D(imgtem, cv2.CV_16S, np.array([[-1], [0], [1]])))

    bottom_temp = row - 1  # ;//区域下界
    flagContinue = False
    for i in range(kuan_hight, row - 1):  # (int i = kuan_hight; i < row; i++)//获得海天线上下界
        ppre = imgtemd[i, 0]
        # paft=imgtem[i+1,0]

        if ppre == 0:  # 寻找跳变，先验认为天空255,当没有河流则可能被认为255，增加0->255的判断,抓住第一个跳变
            continue
        top_temp = i - 1.5 * kuan_hight  # //海天线上界
        horizon_top = 0 if top_temp < 0 else top_temp
        bottom_temp = i + 1.5 * kuan_hight
        horizon_bottom = row - 1 if bottom_temp >= row else bottom_temp  # 海天线下界
        break

    # horizonLine=round((horizon_bottom + horizon_top) / 2)
    # imgsrc=cv2.line(imgsrc,(0,horizonLine),(img.shape[1]-1,horizonLine),(0,0,255),2)
    # cv2.imshow('line',imgsrc)
    # cv2.waitKey(1)

    return horizon_top,horizon_bottom,(horizon_bottom + horizon_top) / 2#返回上界、下界、水平线