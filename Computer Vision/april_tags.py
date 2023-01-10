import copy
import math
import time
import gbvision as gbv
import settings
import cv2 as cv
from pupil_apriltags import Detector

def main():

    elapsed_time = 0
    F_LENGTH = 697.0395744431028
    cam = gbv.USBCamera(settings.CAMERA_PORT, gbv.LIFECAM_3000)
    cam.set_exposure(settings.EXPOSURE)
    at_detector = Detector(families="tag16h5", quad_sigma=0.8, decode_sharpening=0.4)
    win = gbv.FeedWindow("window")
    thr = gbv.FeedWindow("threshold")
    pipe = settings.APRIL_TAG_THRESHOLD + gbv.Erode(4, 1) + gbv.Dilate(4, 1)
    # raw = gbv.FeedWindow("raw")
    ghost_tags = []
    while True:
        start_time = time.time()

        ret, image = cam.read()
        if not ret:
            pass
        debug_image = copy.deepcopy(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        
        tags = at_detector.detect(
            pipe(debug_image),
            estimate_tag_pose=False,
            camera_params=None,
            tag_size=None,
        )
        piped = pipe(debug_image)
        thr.show_frame(piped)

        debug_image = draw_tags(debug_image, tags, elapsed_time)
        locations = get_tags_locations(tags, 15.5**2, F_LENGTH, cam.get_width(), cam.get_height())
        debug_image = draw_tags_locations(debug_image, tags, locations)
        
        
        win.show_frame(debug_image)
        elapsed_time = time.time() - start_time


    cam.release()
    cv.destroyAllWindows()
def ghosts_by_cur_and_last(current_tags, last_tags):
    ghosts = []
    for tag in current_tags:
        for old in last_tags:
            if tag.tag_id == old.tag_id:
                ghost = copy.deepcopy(tag)
                ghost.center += tag.center - old.center
                for i in len(ghost.corners):
                    ghost.corners[i] += tag.corners[i] - old.corners[i]
                ghosts.append(ghost)
    return ghosts
                 
def get_tags_locations(tags, size_area, focal_length, frame_width, frame_height):
    ''' :param focal_length:  the focal length of the camera at it's default state, in units of pixels
    can be described as the square root of the amount of pixels an object takes on a frame, multiplied by it's distance from the camera and divided by the square root of it's surface

    FOCAL_LENGTH = :math:' sqrt(P) * D / sqrt(S)'

    where P is the amount of pixels in the frame representing the object, D is the real life distance between the object and the camera S is the real life surface area (in 2d projection) of the object note that this is a constant, whatever object you choose to use, this formula will yield the same result
    
        :param tags: tags array
        :param size_area: area of target 2d quad that encapsulates object
        '''
    locations = []
    for tag in tags:
        xyz = [0, 0, 0]
        center = tag.center
        corners = tag.corners
        line1 = distance(corners[0], corners[1])
        line2 = distance(corners[1], corners[2])
        line3 = distance(corners[2], corners[3])
        line4 = distance(corners[3], corners[0])
        diagonal = distance(corners[0], corners[2])
        pixels_area = triangle_area(line1, line2, diagonal) + triangle_area(line3, line4, diagonal)
        xyz[2] = (size_area**0.5) * focal_length / (pixels_area**0.5)
        xyz[0] = (((size_area**0.5)/(pixels_area**0.5))) * (center[0] - (frame_width/2))
        xyz[1] = (((size_area**0.5)/(pixels_area**0.5))) * (center[1] - (frame_height/2))
        locations.append(xyz)
    return locations

def distance(v1, v2):
    magnitude = 0
    for i in range(len(v1)):
        magnitude += (v1[i] - v2[i])**2
    return magnitude**0.5

def triangle_area(a, b, c):
    # c**2 = a**2 + b**2 - 2ab*cos(angle(a,b))
    # (-(c - a - b))/(2 * sqrt(a*b)) = cos(angle(a,b))
    return (math.sin(math.acos((-(c**2 - a**2 - b**2))/(2 * a * b))) * a * b)/2

def draw_tags(image, tags, elapsed_time):
    for tag in tags:
        # tag_family = tag.tag_family
        tag_id = tag.tag_id
        center = tag.center
        corners = tag.corners

        center = (int(center[0]), int(center[1]))
        corner_01 = (int(corners[0][0]), int(corners[0][1]))
        corner_02 = (int(corners[1][0]), int(corners[1][1]))
        corner_03 = (int(corners[2][0]), int(corners[2][1]))
        corner_04 = (int(corners[3][0]), int(corners[3][1]))
        cv.circle(image, (center[0], center[1]), 5, (0, 0, 255), 2)
        cv.line(image, (corner_01[0], corner_01[1]),
                (corner_02[0], corner_02[1]), (255, 0, 0), 2)
        cv.line(image, (corner_02[0], corner_02[1]),
                (corner_03[0], corner_03[1]), (255, 0, 0), 2)
        cv.line(image, (corner_03[0], corner_03[1]),
                (corner_04[0], corner_04[1]), (0, 255, 0), 2)
        cv.line(image, (corner_04[0], corner_04[1]),
                (corner_01[0], corner_01[1]), (0, 255, 0), 2)

        cv.putText(image, str(tag_id), (center[0] - 10, center[1] - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv.LINE_AA)

    cv.putText(image,
               "Elapsed Time:" + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
               cv.LINE_AA)

    return image


def draw_tags_locations(image, tags, locations):
    for i in range(len(tags)):
        tag = tags[i]
        # tag_family = tag.tag_family
        center = tag.center

        loc =  [round(a * 100)/100 for a in locations[i]]
        center = (int(center[0]), int(center[1]))
        cv.putText(image, str(loc), (center[0] + 10, center[1] + 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2, cv.LINE_AA)



    return image


if __name__ == '__main__':
    main()