import copy
import math
import socket
import struct
import time
import gbvision as gbv
import settings
import cv2 as cv
from pupil_apriltags import Detector

def main():

    elapsed_time = 0
    F_LENGTH = (697.0395744431028 / 38) * 34
    cam = gbv.USBCamera(settings.CAMERA_PORT, gbv.LIFECAM_3000)
    cam.set_exposure(settings.EXPOSURE)
    at_detector = Detector(families="tag16h5", quad_sigma=0.8, decode_sharpening=0.4)
    win = gbv.FeedWindow("window")
    thr = gbv.FeedWindow("threshold")
    pipe = settings.APRIL_TAG_THRESHOLD + gbv.Erode(1, 4) + gbv.Dilate(1, 4) 
    # raw = gbv.FeedWindow("raw")
    ghost_tags = []
    last_tags = []
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
        ghost_tags = ghosts_by_cur_and_last(tags, last_tags)
        last_tags = tags
        tags = sort_ghosts_from_cur(ghost_tags, tags)
        piped = pipe(debug_image)
        thr.show_frame(piped)

        debug_image = draw_tags(debug_image, tags, elapsed_time)
        locations = wall_angle_and_locations(tags, 15.3, F_LENGTH, cam.get_width(), cam.get_height())
        debug_image = draw_tags_locations_and_wall_angle(debug_image, tags, locations)
        
        
        win.show_frame(debug_image)
        elapsed_time = time.time() - start_time
        
        # test data send
        # try:
        #     with socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP) as sock:
        #             sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        #             sock.sendto(struct.pack('fffff', locations[0][0][0], locations[0][0][1], locations[0][0][2], # xyz
        #                                     locations[0][1][0], locations[0][1][1]), # angle
        #                 ("255.255.255.255", 7112))
        # except:
        #     pass


    cam.release()
    cv.destroyAllWindows()
def ghosts_by_cur_and_last(current_tags, last_tags):
    ghosts = []
    for tag in current_tags:
        for old in last_tags:
            if tag.tag_id == old.tag_id:
                ghost = copy.deepcopy(tag)
                ghost.center += tag.center - old.center
                for i in range(len(ghost.corners)):
                    ghost.corners[i] += tag.corners[i] - old.corners[i]
                ghosts.append(ghost)
    return ghosts

def sort_ghosts_from_cur(ghosts, tags):
    for ghost in ghosts:
        for tag in tags:
            if ghost.tag_id == tag.tag_id:
                break;
        else:
            tags.append(ghost)
    return tags

def wall_angle_and_locations(tags, side_length, focal_length, frame_width, frame_height):
    ''' :param focal_length:  the focal length of the camera at it's default state, in units of pixels
    can be described as the square root of the amount of pixels an object takes on a frame, multiplied by it's distance from the camera and divided by the square root of it's surface

    FOCAL_LENGTH = :math:' sqrt(P) * D / sqrt(S)'

    where P is the amount of pixels in the frame representing the object, D is the real life distance between the object and the camera S is the real life surface area (in 2d projection) of the object note that this is a constant, whatever object you choose to use, this formula will yield the same result
    
        :param tags: tags array
        :param side_length: area of one side of the quad that defines the object (irl)
        '''
    locations = []
    for tag in tags:
        xyz = [0, 0, 0]
        side1_xyz = [0, 0, 0]
        side2_xyz = [0, 0, 0]
        center = tag.center
        corners = tag.corners
        
        line1 = distance(corners[0], corners[1])
        line2 = distance(corners[1], corners[2])
        # line3 = distance(corners[2], corners[3])
        # line4 = distance(corners[3], corners[0])
        # diagonal = distance(corners[0], corners[2]) 
        # pixels_area = triangle_area(line1, line2, diagonal) + triangle_area(line3, line4, diagonal)
        pixels_area = get_true_square_area(corners[0], corners[1], corners[2], corners[3])
        xyz[2] = (side_length) * focal_length / (pixels_area**0.5)
        xyz[0] = (((side_length)/(pixels_area**0.5))) * (center[0] - (frame_width/2))
        xyz[1] = (((side_length)/(pixels_area**0.5))) * (center[1] - (frame_height/2))
        
        line1_center = midpoint(corners[0], corners[1])
        line2_center = midpoint(corners[1], corners[2])
        
        side1_xyz[2] = side_length * focal_length / (line1)
        side1_xyz[0] = (((side_length)/(line1))) * (line1_center[0] - (frame_width/2))
        side1_xyz[1] = (((side_length)/(line1))) * (line1_center[1] - (frame_height/2))
        
        side2_xyz[2] = side_length * focal_length / (line2)
        side2_xyz[0] = (((side_length)/(line2))) * (line2_center[0] - (frame_width/2))
        side2_xyz[1] = (((side_length)/(line2))) * (line2_center[1] - (frame_height/2))
        
        # side1_xyz = subtract(side1_xyz, xyz)
        # side2_xyz = subtract(side2_xyz, xyz)
        # # base: {side1_xyz, side2_xyz}
        # d = distance(side1_xyz, side2_xyz)
        # s1 = magnitude(side1_xyz)
        # s2 = magnitude(side2_xyz)
        # # d**2 = s1**2 + s2**2 - 2*s1**s2*cos(a)
        # # -(d**2 - s1**2 - s2**2)/(2*s1**s2*) = cos(a)
        # angle = math.acos(-(d**2 - s1**2 - s2**2) / (2*s1*s2))
        # uses side1_xyz and side2_xyz bc they are further apart from each other than they are from xyz 

        angle_to_wall_x = angle_to_wall(side1_xyz[0], side1_xyz[2], side2_xyz[0], side2_xyz[2])
        angle_to_wall_y = angle_to_wall(side1_xyz[1], side1_xyz[2], side2_xyz[1], side2_xyz[2])
        # angle_to_wall_y = 0
        # try:
        #     # hypotenuse1 = magnitude(side1_xyz)
        #     # hypotenuse2 = magnitude(side2_xyz)
        #     side = side_length/2
        #     front_to_angle1 = distance(side1_xyz, xyz)
        #     front_to_angle2 = distance(side2_xyz, xyz)
        #     next_to_angle1 = abs(xyz[2] - side1_xyz[2])
        #     next_to_angle2 = xyz[2] - side2_xyz[2]
        #     # f**2 = h**2 + n**2 - 2nh*cos(a)
        #     # -(f**2 - h**2 - n**2)/(2hn) = cos(a)
        #     # angle_to_wall_x = math.acos(-(front_to_angle1**2 - hypotenuse1**2 - next_to_angle1**2)/
        #     #     (2 * hypotenuse1 * next_to_angle1))
        #     # angle_to_wall_y = math.acos(-(front_to_angle2**2 - hypotenuse2**2 - next_to_angle2**2)/
        #     #     (2 * hypotenuse2 * next_to_angle2))
            
        #     #angle_to_wall_x = math.acos(-(front_to_angle1**2 - side**2 - next_to_angle1**2) / (2 * next_to_angle1 * side))
        #     angle_to_wall_x = math.asin(next_to_angle1 / front_to_angle1)
        #     angle_to_wall_y = math.acos(-(front_to_angle2**2 - side**2 - next_to_angle2**2) / (2 * next_to_angle2 * side))
            
        #     #f
        # except:
        #     pass
        # # angle_to_wall = math.asin(next_to_angle/hypotenuse)
        # angle_to_wall = math.atan2(front_to_angle, next_to_angle) 

        
        locations.append((xyz, (angle_to_wall_x, angle_to_wall_y)))
    return locations
        
def midpoint(point1, point2):
    mid = [0, 0]
    mid[0] = (point1[0] + point2[0]) / 2
    mid[1] = (point1[1] + point2[1]) / 2
    return mid
     
def magnitude(vec):
    mag = 0
    for a in vec:   
        mag += a**2
    return mag**0.5
 
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
        # line1 = distance(corners[0], corners[1])
        # line2 = distance(corners[1], corners[2])
        # line3 = distance(corners[2], corners[3])
        # line4 = distance(corners[3], corners[0])
        # diagonal = distance(corners[0], corners[2]) 
        # pixels_area = triangle_area(line1, line2, diagonal) + triangle_area(line3, line4, diagonal)
        pixels_area = get_true_square_area(corners[0], corners[1], corners[2], corners[3])
        xyz[2] = (size_area**0.5) * focal_length / (pixels_area**0.5)
        xyz[0] = (((size_area**0.5)/(pixels_area**0.5))) * (center[0] - (frame_width/2))
        xyz[1] = (((size_area**0.5)/(pixels_area**0.5))) * (center[1] - (frame_height/2))
        locations.append(xyz)
    return locations

def angle_to_wall(x1, z1, x2, z2):
    angle_to_z_side1 = angle_two_vectors(x1, z1)
    angle_to_z_side2 = angle_two_vectors(x2, z2)
    angle_1_to_2 = angle_to_z_side1 - angle_to_z_side2
    side1_projection_of_side2 = math.cos(angle_1_to_2) * magnitude([x2, z2])
    adjacent_side_1 = (magnitude([x2, z2])**2 - side1_projection_of_side2**2)**0.5
    side_1_difference = magnitude([x1, z1]) - side1_projection_of_side2
    return 0.5*math.pi - math.atan2(adjacent_side_1, side_1_difference)
    
def distance(v1, v2):
    magnitude = 0
    for i in range(len(v1)):
        magnitude += (v1[i] - v2[i])**2
    return magnitude**0.5

def triangle_area(a, b, c):
    # c**2 = a**2 + b**2 - 2ab*cos(angle(a,b))
    # (-(c - a - b))/(2 * sqrt(a*b)) = cos(angle(a,b))
    return (math.sin(math.acos((-(c**2 - a**2 - b**2))/(2 * a * b))) * a * b)/2

def subtract(v1, v2):
    v1 = copy.deepcopy(v1)
    for i in range(len(v1)):
        v1[i] -= v2[i]
    return v1

def divide_by_v(v1, v2):
    v1 = copy.deepcopy(v1)
    for i in range(len(v1)):
        v1[i] /= v2[i]
    return v1

def divide_by_s(v, s):
    v = copy.deepcopy(v)
    for i in range(len(v)):
        v[i] /= s
    return v

def multiply_by_s(v, s):
    v = copy.deepcopy(v)
    for i in range(len(v)):
        v[i] *= s
    return v

def angle_two_vectors(v1, v2):
    return math.atan2(v1, v2)

def add_vectors(v1, v2):
    v1 = copy.deepcopy(v1)
    for i in range(len(v1)):
        v1[i] += v2[i]
    return v1

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
        line1_center = midpoint(corner_02, corner_01)
        line1_center = (int(line1_center[0]), int(line1_center[1]))
        
        line2_center = midpoint(corner_02, corner_03)
        line2_center = (int(line2_center[0]), int(line2_center[1]))
        cv.circle(image, (center[0], center[1]), 5, (0, 0, 255), 2)
        cv.circle(image, (line2_center[0], line2_center[1]), 5, (0, 255, 0), 2)
        cv.circle(image, (line1_center[0], line1_center[1]), 5, (0, 0, 255), 2)
        
        cv.line(image, (corner_01[0], corner_01[1]),
                (corner_02[0], corner_02[1]), (0, 0, 255), 2)
        cv.line(image, (corner_02[0], corner_02[1]),
                (corner_03[0], corner_03[1]), (0, 0, 255), 2)
        cv.line(image, (corner_03[0], corner_03[1]),
                (corner_04[0], corner_04[1]), (0, 255, 0), 2)
        cv.line(image, (corner_04[0], corner_04[1]),
                (corner_01[0], corner_01[1]), (0, 255, 0), 2)
        # choose half 01, 02, 03
        len_line_1 = distance(corner_01, corner_02)
        len_line_2 = distance(corner_02, corner_03)
        cv.putText(image, str(tag_id), (center[0] - 10, center[1] - 10),
            cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv.LINE_AA)
        corner_01 = list(corner_01)
        corner_02 = list(corner_02)
        corner_03 = list(corner_03)
        corner_04 = list(corner_04)
        center = midpoint(midpoint(corner_01, corner_02), midpoint(corner_03, corner_04))
        if len_line_1 >= len_line_2: 
            # choose line 2 and center (02, 03)
            true_half_side_length = get_half_side_length(center, corner_02, corner_03)
            axl_intersection = get_axl_intersection(corner_02, corner_03, center)
            opposed_axl_intersection = get_axl_intersection(corner_01, corner_04, center)
        else:
            # choose line 1 and center (02, 01)
            true_half_side_length = get_half_side_length(center, corner_02, corner_01)
            axl_intersection = get_axl_intersection(corner_02, corner_01, center)
            opposed_axl_intersection = get_axl_intersection(corner_03, corner_04, center)
        
        corners_facing_cam_01 = add_vectors(center, [true_half_side_length, true_half_side_length]) 
        corners_facing_cam_02 = add_vectors(center, [-true_half_side_length, true_half_side_length]) 
        corners_facing_cam_03 = add_vectors(center, [-true_half_side_length, -true_half_side_length]) 
        corners_facing_cam_04 = add_vectors(center, [true_half_side_length, -true_half_side_length]) 
        
        
        corners_facing_cam_01 = [int(corners_facing_cam_01[0]), int(corners_facing_cam_01[1])]
        corners_facing_cam_02 = [int(corners_facing_cam_02[0]), int(corners_facing_cam_02[1])]
        corners_facing_cam_03 = [int(corners_facing_cam_03[0]), int(corners_facing_cam_03[1])]
        corners_facing_cam_04 = [int(corners_facing_cam_04[0]), int(corners_facing_cam_04[1])]
        
        cv.line(image, (int(axl_intersection[0]), int(axl_intersection[1])),
                (int(opposed_axl_intersection[0]), int(opposed_axl_intersection[1])), (0, 255,0), 2)
        cv.circle(image, (int(axl_intersection[0]), int(axl_intersection[1])), 5, (0, 255, 0), 2)
        
        cv.line(image, (corners_facing_cam_01[0], corners_facing_cam_01[1]),
                (corners_facing_cam_02[0], corners_facing_cam_02[1]), (0, 0, 255), 2)
        cv.line(image, (corners_facing_cam_02[0], corners_facing_cam_02[1]),
                (corners_facing_cam_03[0], corners_facing_cam_03[1]), (0, 0, 255), 2)
        cv.line(image, (corners_facing_cam_03[0], corners_facing_cam_03[1]),
                (corners_facing_cam_04[0], corners_facing_cam_04[1]), (0, 255, 0), 2)
        cv.line(image, (corners_facing_cam_04[0], corners_facing_cam_04[1]),
                (corners_facing_cam_01[0], corners_facing_cam_01[1]), (0, 255, 0), 2)


    cv.putText(image,
               "Elapsed Time:" + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
               cv.LINE_AA)

    return image


def get_true_square_area(corner_01, corner_02, corner_03, corner_04):
    center = midpoint(midpoint(corner_01, corner_02), midpoint(corner_03, corner_04))
    len_line_1 = distance(corner_01, corner_02)
    len_line_2 = distance(corner_02, corner_03)
    if len_line_1 >= len_line_2: 
        # choose line 2 and center (02, 03)
        true_half_side_length = get_half_side_length(center, corner_02, corner_03)
    else:
        # choose line 1 and center (02, 01)
        true_half_side_length = get_half_side_length(center, corner_02, corner_01)
    return (true_half_side_length * 2)**2

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

def get_half_side_length(center, corner_01, corner_02):
    try:
        intersect_axl_with_side = get_axl_intersection(corner_01, corner_02, center)
        axl_to_center = distance(intersect_axl_with_side, center)
        mid_side_opposed = midpoint(corner_01, corner_02)
        axl_to_mid_side = distance(intersect_axl_with_side, mid_side_opposed)
        half_side_len = distance(mid_side_opposed, corner_01)
        axl_by_half_side = (axl_to_mid_side/half_side_len)
        # true_half_side_len**2 + (true_half_side_len * (axl_to_mid_side / half_side_len))**2 = axl_to_center**2
        # (true_half_side_len**2)*(1 + (axl_by_half_side**2)) = axl_to_center**2
        true_half_side_length = (((axl_to_center**2) / (1 + (axl_by_half_side**2)))**0.5)
        return true_half_side_length
    except:
        return 0

def get_axl_intersection(corner_01, corner_02, center):
    len_line_1 = distance(center, corner_01)
    len_line_2 = distance(center, corner_02)
    ratio_line_1 = len_line_1 / (len_line_1  + len_line_2)
    return add_vectors(multiply_by_s(subtract(corner_01, corner_02), ratio_line_1), corner_02)


def draw_tags_locations_and_wall_angle(image, tags, locations):
    for i in range(len(tags)):
        tag = tags[i]
        # tag_family = tag.tag_family
        center = tag.center

        loc =  [round(a * 100)/100 for a in locations[i][0]]
        ang =  [round(math.degrees(a) * 100 - 90)/100 for a in locations[i][1]]
        center = (int(center[0]), int(center[1]))
        cv.putText(image, str(loc), (center[0] + 10, center[1] + 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2, cv.LINE_AA)
        cv.putText(image, str(ang), (center[0] - 20, center[1] + 35),
                   cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2, cv.LINE_AA)

    return image

if __name__ == '__main__':
    main()