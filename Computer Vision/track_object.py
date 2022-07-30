from cmath import rect
from copy import copy
import gbvision as gbv
import numpy as np
import settings as settings

from main import main

class track_object:
    def __init__(self,  cam: gbv.usb_camera,
                hue, sat, val, vals, range, target: gbv.GameObject):   
        self.__derivative = [0, 0, 0]# motion derivative
        self.__locals = [0, 0, 0]
        self.__cam = cam
        self.__distance_derivative = 0
        self.distance = 0
        self.__target_thr = gbv.ColorThreshold([[vals[0] - range[0], 
                                           vals[0] + range[0]], 
                                          [vals[1] - range[1],
                                           vals[1] + range[1]],
                                          [vals[2] - range[2],
                                           vals[2] + range[2]]],
                                         'HSV')
        self.__thr = copy(self.__target_thr)
        self.__final_thr = copy(self.__target_thr)
        self.__vals = vals
        self.__range = range.copy()
        ok, self.__raw_frame = self.__cam.read()
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
        
        
        
    def track_cycle(self):
        ok = self.update_frame()
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
        
        self.__final_thr = gbv.ColorThreshold([[hue - self.__range[0], hue + self.__range[0]],
                                        [sat - self.__range[1], sat + self.__range[1]],
                                        [val - self.__range[2], val + self.__range[2]]],
                                       'HSV') or self.__thr or self.__target_thr
        
        rects = self.rect()
        
        if len(rects) > 0:
            # find locals
            root = gbv.BaseRotatedRect.shape_root_area(rects[0])
            center = gbv.BaseRotatedRect.shape_center(rects[0])
            locals = self.__target.location_by_params(self.__cam, root, center)
            for i in range(3):
                self.__derivative[i] = self.__locals[i] - locals[i]
            self.__locals = locals
            if ok:
                self.bbox()
            distance = self.__target.distance_by_params(self.__cam, root)
            self.__distance_derivative = self.distance - distance
            self.distance = distance
        else:
            self.__thr = self.__target_thr
            self.distance += self.__distance_derivative
            for i in range(3):
                self.__locals[i] += self.__derivative[i]

    def __threshold(self):
        return self.__final_thr + gbv.MedianBlur(5) + gbv.Dilate(15, 3
                                                               ) + gbv.Erode(10, 2) + gbv.DistanceTransformThreshold(0.2)
    
    def get__bbox(self):
        try:
            bbox_pipe = self.__threshold() + gbv.DistanceTransformThreshold(0.99
                                                                       ) + gbv.find_contours + gbv.contours_to_rects_sorted + gbv.filter_inner_rects
            # the box on the frame from which we choose the next thr
            return bbox_pipe(self.__raw_frame)[0]
        except:
            pass
    
    def bbox(self):
        try:
            bbox = self.get__bbox
            # makes sure we only choose the next thr if the frame exists
            thr = gbv.median_threshold(self.__raw_frame, [0, 0, 0], bbox, 'HSV')
            self.__thr = thr
        except:
            pass
        
    def thr(self):
        return self.__threshold()(self.__raw_frame)

    def rect(self):
        # rects pipeline
        pipe = self.__threshold() + gbv.find_contours + gbv.FilterContours(
            100) + gbv.contours_to_rotated_rects_sorted + gbv.filter_inner_rotated_rects
        return pipe(self.__raw_frame)
    
    
    def __calc_pid_error(self, item):
        obj_error = (self.__vals[item] - self.__vals[item])
        return obj_error
    
    def set_thr(self, thr: gbv.ColorThreshold):
        self.__target_thr = copy(thr)
    
    def set_cam(self, cam: gbv.usb_camera):
        self.__cam = cam
    
    def get_locals(self):
        return self.__locals
    
    def get_derivative(self):
        return self.__derivative
    
    def get_thr(self):
        return copy(self.__target_thr)
    
    def get_cam(self):
        return self.__cam
    
    def get_raw_frame(self):
        return self.__raw_frame
    
    def update_frame(self):
        ok, self.__raw_frame = self.__cam.read()
        return ok
    def get_final_thr(self):
        return copy(self.__final_thr)
        








def main():
    cam = gbv.USBCamera(settings.CAMERA_PORT, gbv.LIFECAM_3000)
    cam.set_exposure(settings.EXPOSURE)
    obj = track_object(cam=cam, vals=settings.DEFAULT_VALS, hue=[settings.HUE_KP, settings.HUE_KI,
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
            # draws the blue squares showing the objects detected
            frame = gbv.draw_rotated_rects(
                obj.get_raw_frame(), obj.rect(), (255, 0, 0), thickness=5)
            # shows the red square shoqing the place from which we choose our next thr
            try:
                frame2 = gbv.draw_rects(frame, [obj.get__bbox()], (0, 0, 255), thickness=5)
                frame = frame2
            except:
                pass
                
            thr.show_frame(obj.thr())
            raw.show_frame(obj.get_final_thr()(frame))
            win.show_frame(frame)
            print(obj.get_locals())
            print(obj.distance)
            

if __name__ == '__main__':
    main()