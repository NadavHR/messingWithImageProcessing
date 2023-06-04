# Import required modules
from time import sleep
import cv2
import numpy as np
import keyboard
import gbvision as gbv
import settings

# Define the dimensions of checkerboard
CHECKERBOARD = (8, 6)


# stop the iteration when specified
# accuracy, epsilon, is reached or
# specified number of iterations are completed.
criteria = (cv2.TERM_CRITERIA_EPS +
			cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# Vector for 3D points
threedpoints = []

# Vector for 2D points
twodpoints = []


# 3D points real world coordinates
objectp3d = np.zeros((1, CHECKERBOARD[0]
					* CHECKERBOARD[1],
					3), np.float32)
objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
							0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None



cam = gbv.USBCamera(settings.CAMERA_PORT, gbv.LIFECAM_3000)
win = gbv.FeedWindow("window")
cam.set_exposure(settings.EXPOSURE)
while not keyboard.is_pressed("f"):
    # print("hi")
    ok, image = cam.read()
    
    # cv2.imshow('img', image)
    if (image is not None):
        grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # win.show_frame(grayColor)

        # Find the chess board corners
        # If desired number of corners are
        # found in the image then ret = true

        ret, corners = cv2.findChessboardCorners(
                        grayColor, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        # If desired number of corners can be detected then,
        # refine the pixel coordinates and display
        # them on the images of checker board
        if ret == True:
            threedpoints.append(objectp3d)

            # Refining pixel coordinates
            # for given 2d points.
            corners2 = cv2.cornerSubPix(
                grayColor, corners, (11, 11), (-1, -1), criteria)

            twodpoints.append(corners2)

            # Draw and display the corners
            image = cv2.drawChessboardCorners(image,
                                            CHECKERBOARD,
                                            corners2, ret)

        # cv2.imshow('img', image)
        win.show_frame(image)
        sleep(0.5)

h, w = image.shape[:2]


# Perform camera calibration by
# passing the value of above found out 3D points (threedpoints)
# and its corresponding pixel coordinates of the
# detected corners (twodpoints)
ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
	threedpoints, twodpoints, grayColor.shape[::-1], None, None)


# Displaying required output
print(" Camera matrix:")
print(matrix)

print("\n Distortion coefficient:")
print(distortion)

# print("\n Rotation Vectors:")
# print(r_vecs)

# print("\n Translation Vectors:")
# print(t_vecs)

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
