import struct
import gbvision as gbv
import numpy as np
import settings as settings
import socket


cones_thr = gbv.ColorThreshold([[12, 22], [207, 255], [47, 147]], 'HSV') + gbv.MedianBlur(3) + gbv.Dilate(12, 3
        )  + gbv.Erode(12, 3) + gbv.DistanceTransformThreshold(0.1)
cubes_thr = gbv.ColorThreshold([[115, 125], [114, 174], [0, 94]], 'HSV') + gbv.MedianBlur(3) + gbv.Dilate(12, 3
        )  + gbv.Erode(12, 3) + gbv.DistanceTransformThreshold(0.1)
cones_pipe = cones_thr + gbv.find_contours + gbv.FilterContours(
    100) + gbv.contours_to_rotated_rects_sorted + gbv.filter_inner_rotated_rects
cubes_pipe = cubes_thr + gbv.find_contours + gbv.FilterContours(
    100) + gbv.contours_to_rotated_rects_sorted + gbv.filter_inner_rotated_rects
# TARGET = settings.TARGET1
CONES = gbv.GameObject(0.2449489742783178)
CUBES = gbv.GameObject(0.21)
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
        thr.show_frame(cones_thr(frame) + cubes_thr(frame))
        raw.show_frame(settings.DEFAULT_TARGET_THRESHOLD(frame))
        cones_cnts = cones_pipe(frame)
        cubes_cnts = cubes_pipe(frame)
        frame = gbv.draw_rotated_rects(frame, cones_cnts, (0, 150, 255), thickness=5)
        frame = gbv.draw_rotated_rects(frame, cubes_cnts, (255, 0, 30), thickness=5)
        # if len(cones_cnts) > 0:
        #     root = gbv.BaseRotatedRect.shape_root_area(cones_cnts[0])
        #     center = gbv.BaseRotatedRect.shape_center(cones_cnts[0])
        #     locals = CONES.location_by_params(cam, root, center)
        #     print("distance:" + str(CONES.distance_by_params(cam, root)))
        #     print("location:" + str(locals))
        #     print("angle:" + str(np.arcsin(locals[0] / locals[2]) * 180 / np.pi))
            
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            # pack = struct.pack('fff', locals[0], locals[1], locals[2])
            pack = pack_game_pieces([CONES.location_by_params(cam, gbv.BaseRotatedRect.shape_root_area(a),
                                                                gbv.BaseRotatedRect.shape_center(a)) for a in cones_cnts],
                                    [CUBES.location_by_params(cam, gbv.BaseRotatedRect.shape_root_area(a),
                                                                gbv.BaseRotatedRect.shape_center(a)) for a in cubes_cnts])
            sock.sendto(pack, ("255.255.255.255", 4590))
                
                
def pack_game_pieces(cones, cubes):
    cones_ammount = len(cones)
    cubes_ammount = len(cubes)
    
    worked_array = tuple(cones) + tuple(cubes)
    bytes_pack = struct.pack('ii', cones_ammount, cubes_ammount)
    for v in worked_array:
        bytes_pack += struct.pack('f', v[0])
        bytes_pack += struct.pack('f', v[1])
        bytes_pack += struct.pack('f', v[2])
    
    return bytes_pack

if __name__ == '__main__':
    main()
