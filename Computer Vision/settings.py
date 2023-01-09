import gbvision as gbv

# settings file, here you have the settings, stuff that change based on context

CAMERA_PORT = 1
EXPOSURE = -7

limelight_cam = gbv.CameraData(23.65066003307307,1.0402162342 , 0.86742863824, name="limelight") 

with open('thr.txt') as f:
  DEFAULT_VALS = [int(f.readline()),
                  int(f.readline()),
                  int(f.readline())]
DEFAULT_RANGE = [5, 60, 80]

DEFAULT_TARGET_THRESHOLD = gbv.ColorThreshold([[DEFAULT_VALS[0] - DEFAULT_RANGE[0], 
                                           DEFAULT_VALS[0] + DEFAULT_RANGE[0]], 
                                          [DEFAULT_VALS[1] - DEFAULT_RANGE[1],
                                           DEFAULT_VALS[1] + DEFAULT_RANGE[1]],
                                          [DEFAULT_VALS[2] - DEFAULT_RANGE[2],
                                           DEFAULT_VALS[2] + DEFAULT_RANGE[2]]],
                                         'HSV')

APRIL_TAG_THRESHOLD = gbv.ColorThreshold([[0, 255], [0, 255], [27, 255]], 'HSV')

# this is the square root of your targets area, in this case its a rubber duck
TARGET =  gbv.GameObject(0.039633272976)
TARGET1 = gbv.GameObject(0.109544511501)

HUE_KP = 0.0000003
HUE_KI = 0.000001
HUE_KD = 0.000004

SAT_KP = 0.05
SAT_KI = 0.0
SAT_KD = 0.0013

VAL_KP = 0.15
VAL_KI = 0.0 
VAL_KD = 0.018



