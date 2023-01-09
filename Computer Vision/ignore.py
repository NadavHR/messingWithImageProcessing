import struct
import gbvision as gbv
import numpy as np
import settings as settings
import socket


threshold = settings.DEFAULT_TARGET_THRESHOLD + gbv.MedianBlur(3) + gbv.Dilate(12, 9
        )  + gbv.Erode(12, 9) + gbv.DistanceTransformThreshold(0.3)

pipe = threshold + gbv.find_contours + gbv.FilterContours(
    100) + gbv.contours_to_rotated_rects_sorted + gbv.filter_inner_rotated_rects

TARGET = settings.TARGET1

sock = socket.socket

def main():
    cam = gbv.USBCamera(settings.CAMERA_PORT, gbv.LIFECAM_3000) 
    cam.set_exposure(settings.EXPOSURE)
    
    ok, frame = cam.read()
    win = gbv.FeedWindow("window")
    thr = gbv.FeedWindow("threshold")
    raw = gbv.FeedWindow("raw")
    while win.show_frame(frame):
        ok, frame = cam.read()
        thr.show_frame(threshold(frame))
        raw.show_frame(settings.DEFAULT_TARGET_THRESHOLD(frame))
        cnts = pipe(frame)
        frame = gbv.draw_rotated_rects(frame, cnts, (255, 0, 0), thickness=5)
        if len(cnts) > 0:
            root = gbv.BaseRotatedRect.shape_root_area(cnts[0])
            center = gbv.BaseRotatedRect.shape_center(cnts[0])
            locals = TARGET.location_by_params(cam, root, center)
            print("distance:" + str(TARGET.distance_by_params(cam, root)))
            print("location:" + str(locals))
            print("angle:" + str(np.arcsin(locals[0] / locals[2]) * 180 / np.pi))
            
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                sock.sendto(struct.pack('fff', locals[0], locals[1], locals[2]),
                    ("255.255.255.255", 5162))
                
                


if __name__ == '__main__':
    main()
