import gbvision as gbv

# settings file, here you have the settings, stuff that change based on context

CAMERA_PORT = 1
EXPOSURE = -9

with open('thr.txt') as f:
  DEFAULT_VALS = [int(f.readline()),
                  int(f.readline()),
                  int(f.readline())]
DEFAULT_RANGE = [5, 30, 40]

DEFAULT_TARGET_THRESHOLD = gbv.ColorThreshold([[DEFAULT_VALS[0] - DEFAULT_RANGE[0], 
                                           DEFAULT_VALS[0] + DEFAULT_RANGE[0]], 
                                          [DEFAULT_VALS[1] - DEFAULT_RANGE[1],
                                           DEFAULT_VALS[1] + DEFAULT_RANGE[1]],
                                          [DEFAULT_VALS[2] - DEFAULT_RANGE[2],
                                           DEFAULT_VALS[2] + DEFAULT_RANGE[2]]],
                                         'HSV')

# this is the square root of your targets area, in this case its a rubber duck
TARGET =  gbv.GameObject(0.039633272976)
TARGET1 = gbv.GameObject(0.109544511501)

HUE_KP = 0.0000003
HUE_KI = 0.00000000001
HUE_KD = 0.000004

SAT_KP = 0.014
SAT_KI = 0.0000001
SAT_KD = 0.007 

VAL_KP = 0.0091
VAL_KI = 0.0000021 
VAL_KD = 0.0878



