import cv2
import numpy as np
import keyboard
import gbvision as gbv
import settings

matrix = np.array([[661.92113903,   0.        , 307.48134487],
       [  0.        , 662.86886862, 231.33037259],
       [  0.        ,   0.        ,   1.        ]])
distortion = np.array([[ 1.63258818e-01, -1.29538370e+00, -2.96528789e-03,
         1.11106656e-03,  2.31431665e+00]])

cam = gbv.USBCamera(settings.CAMERA_PORT, gbv.LIFECAM_3000)
win = gbv.FeedWindow("window")
cam.set_exposure(settings.EXPOSURE)
while not keyboard.is_pressed('s'):
    
    ok, image = cam.read()
    h,  w = image.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(matrix,distortion,(w,h),1,(w,h))
    # undistort
    mapx,mapy = cv2.initUndistortRectifyMap(matrix,distortion,None,newcameramtx,(w,h),5)
    dst = cv2.remap(image,mapx,mapy,cv2.INTER_LINEAR)

    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    win.show_frame(dst)
