import cv2
import numpy as np
import gbvision as gbv
import settings as settings
stdv = np.array([5, 30, 50])


# basically this whole piece of code is here to write the HSV values to thr.txt


def main():
    camera = gbv.USBCamera(settings.CAMERA_PORT)
    camera.set_exposure(settings.EXPOSURE)
    window = gbv.CameraWindow('feed', camera)
    window.open()
    
    while True:
        frame = window.show_and_get_frame()
        k = window.last_key_pressed
        if k == 'r':
            bbox = cv2.selectROI('feed', frame)
            print(bbox)
            thr = gbv.median_threshold(frame, stdv, bbox, 'HSV')
            print(thr)
            break
    cv2.destroyAllWindows()
    
    with open('thr.txt', 'w') as f:\
        f.write('{}\n{}\n{}'.format(thr.__getitem__(0)[0],
                                    thr.__getitem__(1)[0],
                                    thr.__getitem__(2)[0]))

    original = gbv.FeedWindow(window_name='original')
    after_proc = gbv.FeedWindow(window_name='after threshold', drawing_pipeline=thr)

    original.open()
    after_proc.open()
    while True:
        ok, frame = camera.read()
        if not original.show_frame(frame):
            break
        if not after_proc.show_frame(frame):
            break

    exit()


if __name__ == '__main__':
    main()