from copy import copy
import math
import socket
import struct
import gbvision as gbv
import cv2 
import numpy as np
import settings as settings

class track_object:
    def __init__(self,  cam: gbv.usb_camera,
                hue, sat, val, pid_vals, range, target: gbv.GameObject, angle_offset = (0, 0, 0)):   
        self.__motion_derivative = [0, 0, 0] # motion derivative
        self.__locals = [0, 0, 0]
        self.__angle = [0, 0, 0]
        self.angle_offset = angle_offset
        self.__angle_derivative = [0, 0, 0]
        self.cam = cam
        self.__distance_derivative = 0 
        self.distance = 0
        self.__circs = [] # the circs
        self.rects = []
        self.__bbox = None # the bounding box from which we update the thr
        self.target_thr = gbv.ColorThreshold([[pid_vals[0] - range[0], 
                                           pid_vals[0] + range[0]], 
                                          [pid_vals[1] - range[1],
                                           pid_vals[1] + range[1]],
                                          [pid_vals[2] - range[2],
                                           pid_vals[2] + range[2]]],
                                         'HSV')
        self.__thr = copy(self.target_thr)
        self.__final_thr = copy(self.target_thr)
        self.__vals = pid_vals
        self.range = range.copy()
        ok, self.__frame = self.cam.read()
        self.__target = target
        # pid values
        self.__hue_pid = hue.copy()
        self.__sat_pid = sat.copy()
        self.__val_pid = val.copy()
        # previous errors and integrals
        self.__hue_integral = 0
        self.__hue_last_e = 0
        self.__sat_integral = 0
        self.__sat_last_e = 0 
        self.__val_last_e = 0
        self.__val_integral = 0
        
        
        
    def track_cycle(self, frame = None, mode = 0):
        ok = self.update_frame()
        if (frame is not None):
            self.__frame = frame
        # error
        hue_error = self.__calc_pid_error(0)
        sat_error = self.__calc_pid_error(1)
        val_error = self.__calc_pid_error(2)
        # integral
        self.__hue_integral += hue_error
        self.__sat_integral += sat_error
        self.__val_integral += val_error
        # derivative
        hue_d = self.__hue_last_e - hue_error
        sat_d = self.__sat_last_e - sat_error
        val_d = self.__val_last_e - val_error
        # hue calc
        hue = self.__vals[0]
        hue += (hue_error * self.__hue_pid[0]
                ) + (self.__hue_integral * self.__hue_pid[1]
                     ) - (hue_d * self.__hue_pid[2])
        # sat calc
        sat = self.__vals[1]
        sat += (sat_error * self.__sat_pid[0]
                ) + (self.__sat_integral * self.__sat_pid[1]
                     ) - (sat_d * self.__sat_pid[2])
        # val calc
        val = self.__vals[2]
        val += (val_error * self.__val_pid[0]
                ) + (self.__val_integral * self.__val_pid[1]
                     ) - (val_d * self.__val_pid[2])
        hue = int(hue)
        sat = int(sat)
        val = int(val)
        
        self.__final_thr = gbv.ColorThreshold([[hue - self.range[0], hue + self.range[0]],
                                        [sat - self.range[1], sat + self.range[1]],
                                        [val - self.range[2], val + self.range[2]]],
                                       'HSV') or self.__thr or self.target_thr
        if mode == 1:
            cnts = self.__rect()
        else:
            cnts = self.__circ()
        
        if len(cnts) > 0:
            # find locals
            if mode == 1:
                root = gbv.BaseRotatedRect.shape_root_area(cnts[0])
                center = gbv.BaseRotatedRect.shape_center(cnts[0])
            else:
                root = gbv.BaseCircle.shape_root_area(cnts[0])
                center = gbv.BaseCircle.shape_center(cnts[0])
            locals = self.__target.location_by_params(self.cam, root, center)
            for i in range(3):
                self.__motion_derivative[i] = self.__locals[i] - locals[i]
            self.__locals = locals
            self.__hue_last_e = hue_error
            self.__sat_last_e = sat_error
            self.__val_last_e = val_error
            if ok:
                self.update_thr()
            distance = self.__target.distance_by_params(self.cam, root)
            self.__distance_derivative = self.distance - distance
            self.distance = distance
        else:
            self.__thr = self.target_thr
            self.distance += self.__distance_derivative
            for i in range(3):
                self.__locals[i] += self.__motion_derivative[i]

    def get_threshold_pipe(self):
        return self.__final_thr + gbv.MedianBlur(3) + gbv.Dilate(5, 2
                                                               ) + gbv.Erode(5, 2) + gbv.DistanceTransformThreshold(0.1)
    
    def __update_bbox(self):
        try:
            bbox_pipe = self.get_threshold_pipe() + gbv.DistanceTransformThreshold(0.99
                                                                       ) + gbv.find_contours + gbv.contours_to_rects_sorted + gbv.filter_inner_rects
            # the box on the frame from which we choose the next thr
            self.__bbox = bbox_pipe(self.__frame)[0]
        except:
            pass
    
    def get_bbox(self):
        return self.__bbox
    
    def get_angle(self):
        angle = self.__angle
        try:
            bbox = self.__bbox
            rect = self.__circs[0]
            angle[0] = rect[2]
            bbox_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            rect_center = rect[0]
            dis = (rect_center[0] - bbox_center[0], rect_center[1] - bbox_center[1])
            angle[1] = math.atan(dis[0] / rect[1][0]) * 57.2957795
            angle[2] = math.atan(dis[1] / rect[1][1]) * 57.2957795
            # updates angle derivative
            self.__angle_derivative = (-self.__angle[0] + angle[0],
                                       -self.__angle[1] + angle[1],
                                       -self.__angle[2] + angle[2])
        except:
            angle = [self.__angle[0] + self.__angle_derivative[0],
                     self.__angle[1] + self.__angle_derivative[1],
                     self.__angle[2] + self.__angle_derivative[2]]
        self.__angle = angle
        return angle

    def update_thr(self):
        try:
            self.__update_bbox()
            # makes sure we only choose the next thr if the frame exists
            thr = gbv.median_threshold(self.__frame, [0, 0, 0], self.__bbox, 'HSV')
            self.__thr = thr
        except:
            pass
        
    
    def __circ(self):
        # circss pipeline
        pipe = self.get_threshold_pipe() + gbv.find_contours + gbv.FilterContours(
            100) + gbv.contours_to_circles_sorted + gbv.filter_inner_circles
        self.__circs = pipe(self.__frame)
        return self.__circs
    
    def __rect(self):
        pipe = self.get_threshold_pipe() + gbv.find_contours + gbv.FilterContours(
            100) + gbv.contours_to_rotated_rects + gbv.filter_inner_rotated_rects
        self.__rects = pipe(self.__frame)
        return self.__rects
    
    def get_circs(self):
        return self.__circs
    
    def get_rects(args):
        return self.__rects
    
    def __calc_pid_error(self, item):
        obj_error = (self.__vals[item] - self.__thr.params[item][0])
        return obj_error
    
    def get_locals(self):
        return self.__locals
    
    def getX(self):
        return self.__locals[0]
    
    def getY(self):
        return self.__locals[1]
    
    def getZ(self):
        return self.__locals[2]
    
    def get_motion_derivative(self):
        return self.__motion_derivative
    
    def get_raw_frame(self):
        return self.__frame
    
    def update_frame(self):
        ok, self.__frame = self.cam.read()
        return ok
    
    def get_final_thr(self):
        return copy(self.__final_thr)
    
    def get_angle_derivative(self):
        return self.__angle_derivative
        








def main():
    cam = gbv.USBCamera(settings.CAMERA_PORT, gbv.LIFECAM_3000)
    cam.set_exposure(settings.EXPOSURE)
    obj = track_object(cam=cam, pid_vals=settings.DEFAULT_VALS, hue=[settings.HUE_KP, settings.HUE_KI,
                            settings.HUE_KD], sat=[settings.SAT_KP, settings.SAT_KI,
                            settings.SAT_KD], val=[settings.VAL_KP, settings.VAL_KI, settings.VAL_KD],
                            range=settings.DEFAULT_RANGE, target=settings.TARGET)
    # main window (shows the base frame with outlines)
    win = gbv.FeedWindow("window")
    # threshold window (shows the base frame after the entire threshold pipeline)
    thr = gbv.FeedWindow("threshold")
    # raw threshold window (shows only what was detected by the base threshold)
    raw = gbv.FeedWindow("raw")
    while True:
            obj.track_cycle()
            # shows the red square shoqing the place from which we choose our next thr
            raw.show_frame(obj.get_final_thr()(obj.get_raw_frame()))
            # draws the blue squares showing the objects detected
            frame = gbv.draw_rotated_rects(
                obj.get_raw_frame(), obj.get_circs(), (255, 0, 0), thickness=5)
            try:
                frame2 = gbv.draw_circles(frame, [obj.get_bbox()], (0, 0, 255), thickness=5)
                frame = frame2
            except:
                pass
            thr.show_frame(obj.get_threshold_pipe()(obj.get_raw_frame()))
            win.show_frame(frame)
            locals = obj.get_locals()
            angle = obj.get_angle()
            print(locals)
            print(angle)
            print(obj.distance)
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                sock.sendto(struct.pack('ffffff', locals[0], locals[1], locals[2], 
                                        angle[0], angle[1], angle[2]),
                            ("255.255.255.255", 7112))
            

if __name__ == '__main__':
    main()