import gbvision as gbv
import numpy as np
import settings as settings
import socket


default_vals = settings.DEFAULT_VALS
default_range = settings.DEFAULT_RANGE
TARGET = settings.TARGET
sock = socket.socket

def main():
    # camera
    cam = gbv.USBCamera(settings.CAMERA_PORT, gbv.LIFECAM_3000)
    cam.set_exposure(settings.EXPOSURE)

    # frame
    ok, frame = cam.read()

    
    
    # main window (shows the base frame with outlines)
    win = gbv.FeedWindow("window")
    # threshold window (shows the base frame after the entire threshold pipeline)
    thr = gbv.FeedWindow("threshold")
    # raw threshold window (shows only what was detected by the base threshold)
    raw = gbv.FeedWindow("raw")




    # the current threshold detected by the camera (used for error calculations)
    cur_thr = settings.DEFAULT_TARGET_THRESHOLD
    # the final threshold used for the next frame
    final_thr = settings.DEFAULT_TARGET_THRESHOLD



    # default values
    hue = default_vals[0]
    sat = default_vals[1]
    val = default_vals[2]
    range_hue = default_range[0]
    range_sat = default_range[1]
    range_val = default_range[2]


    # previous errors and integrals
    last_exposure_e = 0
    exposure_integral = 0
    hue_integral = 0
    hue_last_e = 0
    sat_integral = 0
    sat_last_e = 0


    while True:
        # hue PID
        hue_error = (hue - cur_thr.__getitem__(0)[0])
        hue_integral += hue_error
        hue_derivative = hue_error - hue_last_e
        hue += (hue_error * settings.HUE_KP
                ) + (hue_integral * settings.HUE_KI
                     ) + (hue_derivative * settings.HUE_KD)
        hue_last_e = hue_error
        
        
        # sat PID
        sat_error = -(sat - cur_thr.__getitem__(1)[0])
        sat_integral += sat_error
        sat_derivative = sat_last_e - sat_error
        sat_last_e = sat_error
        sat += (sat_error * settings.SAT_KP
                ) + (sat_integral * settings.SAT_KI
                     ) - (sat_derivative * settings.SAT_KD)


        # val PID
        exposure_error = -(val - cur_thr.__getitem__(2)[0])
        exposure_integral += exposure_error
        exposure_derivative = exposure_error - last_exposure_e
        val += (exposure_error * settings.VAL_KP
                ) + (exposure_integral * settings.VAL_KI
                     ) - (exposure_derivative * settings.VAL_KD)
        last_exposure_e = exposure_error



        # final threshold
        final_thr = gbv.ColorThreshold([[hue - range_hue, hue + range_hue],
                                        [sat - range_sat, sat + range_sat],
                                        [val - range_val, val + range_val]],
                                       'HSV') or settings.DEFAULT_TARGET_THRESHOLD or final_thr



        # threshold pipeline
        threshold = final_thr + gbv.MedianBlur(5) + gbv.Dilate(15, 3
                                                               ) + gbv.Erode(10, 2) + gbv.DistanceTransformThreshold(0.2)
        # rects pipeline
        pipe = threshold + gbv.find_contours + gbv.FilterContours(
            100) + gbv.contours_to_rotated_rects_sorted + gbv.filter_inner_rotated_rects




        # gets rects
        cnts = pipe(frame)



        # checks if something was found
        if len(cnts) > 0:

            # finds locals
            root = gbv.BaseRotatedRect.shape_root_area(cnts[0])
            center = gbv.BaseRotatedRect.shape_center(cnts[0])
            locals = TARGET.location_by_params(cam, root, center)

            # prints locals
            print("distance:" + str(TARGET.distance_by_params(cam, root)))
            print("location:" + str(locals))
            print("angle:" + str(np.arcsin(locals[0] / locals[2]) * 180 / np.pi))



            # how we choose the next thr, we do a massive distance transform to get only a very small and accurate part of the color we want
            # the red square shows the part we choose our next thr from
            bbox_pipe = threshold + gbv.DistanceTransformThreshold(0.99
                                                                   ) + gbv.find_contours + gbv.contours_to_rects_sorted + gbv.filter_inner_rects



            # the box on the frame from which we choose the next thr
            bbox = bbox_pipe(frame)[0]

            # makes sure we only choose the next thr if the frame exists
            if (ok):
                cur_thr = gbv.median_threshold(frame, [0, 0, 0], bbox, 'HSV')



            # draws the blue squares showing the objects detected
            frame = gbv.draw_rotated_rects(
                frame, cnts, (255, 0, 0), thickness=5)
            
            
            
            # shows the red square shoqing the place from which we choose our next thr
            frame = gbv.draw_rects(frame, [bbox], (0, 0, 255), thickness=5)



            # NOT RELEVENT: broadcats locally the locals found, only usefull for robotics
            # with socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP) as sock:
            #     sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            #     sock.sendto(struct.pack('ddd', locals[0], locals[1], locals[2]),
            #                 ("255.255.255.255", 5162))
        else:
            # in case no object was found, the thr will start aiming towards the defult thr
            cur_thr = settings.DEFAULT_TARGET_THRESHOLD



        # this part shows everything on the different windows
        thr.show_frame(threshold(frame))
        raw.show_frame(final_thr(frame))
        win.show_frame(frame)




        # gets the next frame
        ok, frame = cam.read()


if __name__ == '__main__':
    main()
