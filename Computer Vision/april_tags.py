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
    F_LENGTH = (697.0395744431028 / 38) #* 34
    SIDE_LENGTH = 0.153
    cam = gbv.USBCamera(settings.CAMERA_PORT, gbv.LIFECAM_3000)
    cam.set_exposure(settings.EXPOSURE)
    at_detector = Detector(families="tag16h5", quad_sigma=0.8, decode_sharpening=0.4)
    win = gbv.FeedWindow("window")
    thr = gbv.FeedWindow("threshold")
    pipe = settings.APRIL_TAG_THRESHOLD + gbv.Erode(1, 4) + gbv.Dilate(1, 4) 
    raw = gbv.FeedWindow("raw")
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
        locations, angles = wall_angle_and_locations(tags, SIDE_LENGTH, F_LENGTH, cam.get_width(), cam.get_height())
        # TEMP!!!   
        black = copy.deepcopy(debug_image)
        black = gbv.ColorThreshold([[0,0], [0,0], [0,0]], 'HSV')(black)
        black = cv.cvtColor(black, cv.COLOR_BAYER_GR2BGR)
        black = draw_tags(black, tags, 0)
        raw.show_frame(black)
        # end temp
        debug_image = draw_tags_locations_and_wall_angle(debug_image, tags, locations, angles)
        debug_image = draw_tags_facing_cam(debug_image, tags, locations, F_LENGTH, SIDE_LENGTH)
        
        
        win.show_frame(debug_image)
        elapsed_time = time.time() - start_time
        
        # test data send
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP) as sock:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                    sock.sendto(struct.pack('fffff', locations[0][0][0], locations[0][0][1], locations[0][0][2], # xyz
                                            locations[0][1][0], locations[0][1][1]), # angle
                        ("255.255.255.255", 7112))
        except:
            pass


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
    angles = []
    for tag in tags:
        corners = tag.corners
        sides_coords = get_true_coords_sides(corners, frame_width, frame_height, focal_length, side_length)
        location = get_true_coords_center(sides_coords[0], sides_coords[1], sides_coords[2], sides_coords[3], focal_length)
        locations.append(location)
        angle = get_square_angle(sides_coords)
        angles.append(angle)
        
        
    return (locations, angles)
        
def wall_angle_and_locations_outdated(tags, side_length, focal_length, frame_width, frame_height):
    ''' :param focal_length:  the focal length of the camera at it's default state, in units of pixels
    can be described as the square root of the amount of pixels an object takes on a frame, multiplied by it's distance from the camera and divided by the square root of it's surface

    FOCAL_LENGTH = :math:' sqrt(P) * D / sqrt(S)'

    where P is the amount of pixels in the frame representing the object, D is the real life distance between the object and the camera S is the real life surface area (in 2d projection) of the object note that this is a constant, whatever object you choose to use, this formula will yield the same result
    
        :param tags: tags array
        :param side_length: area of one side of the quad that defines the object (irl)
        '''
    locations = []
    angles = []
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

        # angle_to_wall_x = angle_to_wall(side1_xyz[0], side1_xyz[2], side2_xyz[0], side2_xyz[2])
        # angle_to_wall_y = angle_to_wall(side1_xyz[1], side1_xyz[2], side2_xyz[1], side2_xyz[2])
        if line1 >= line2: 
            # choose line 2 and center (02, 03)
            true_diagonal_length = get_half_side_length(center, corners[0], corners[1]) * 2
        else:
            # choose line 1 and center (02, 01)
            true_diagonal_length = get_half_side_length(center, corners[1], corners[2]) * 2
        true_diagonal_length *= (2**0.5)

        try:
            # angle_to_wall_x = get_line_angle_to_plane(true_diagonal_length, distance(corners[0], corners[2]))
            # angle_to_wall_y = get_line_angle_to_plane(true_diagonal_length, distance(corners[1], corners[3]))
            angle_to_wall = get_square_angle(xyz)
        except:
            angle_to_wall = [0,0]
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
        # if angle_to_wall_x ==0:
        #     print(str(true_diagonal_length))
        locations.append(xyz)
        angles.append(angle_to_wall)
    return (locations, angles)
        

def wall_angle_and_locations_v2(tags, side_length, focal_length, frame_width, frame_height):
    ''' :param focal_length:  the focal length of the camera at it's default state, in units of pixels
    can be described as the square root of the amount of pixels an object takes on a frame, multiplied by it's distance from the camera and divided by the square root of it's surface

    FOCAL_LENGTH = :math:' sqrt(P) * D / sqrt(S)'

    where P is the amount of pixels in the frame representing the object, D is the real life distance between the object and the camera S is the real life surface area (in 2d projection) of the object note that this is a constant, whatever object you choose to use, this formula will yield the same result
    
        :param tags: tags array
        :param side_length: area of one side of the quad that defines the object (irl)
        '''
    location_estimates, angles_estimate = wall_angle_and_locations(tags, side_length, focal_length, frame_width, frame_height)
    locations = []
    angles = []
    for i in range(len(tags)):
        tag = tags[i]
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

        # angle_to_wall_x = angle_to_wall(side1_xyz[0], side1_xyz[2], side2_xyz[0], side2_xyz[2])
        # angle_to_wall_y = angle_to_wall(side1_xyz[1], side1_xyz[2], side2_xyz[1], side2_xyz[2])
        if line1 >= line2: 
            # choose line 2 and center (02, 03)
            true_diagonal_length = get_half_side_length(center, corners[0], corners[1]) * 2
        else:
            # choose line 1 and center (02, 01)
            true_diagonal_length = get_half_side_length(center, corners[1], corners[2]) * 2
        true_diagonal_length *= (2**0.5)

        xyz = midpoint(xyz, location_estimates[i])
        try:
            # angle_to_wall_x = get_line_angle_to_plane(true_diagonal_length, distance(corners[0], corners[2]))
            # angle_to_wall_y = get_line_angle_to_plane(true_diagonal_length, distance(corners[1], corners[3]))
            angle_to_wall = get_square_angle(xyz)
        except:
            angle_to_wall = [0,0]
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
        # if angle_to_wall_x ==0:
        #     print(str(true_diagonal_length))
        
        locations.append(xyz)
        angles.append(angle_to_wall)
    return (locations, angles)
        
def midpoint(point1, point2):
    mid = []
    for i in range(len(point1)):
        mid.append((point1[i] + point2[i]) / 2)
    return mid
     
def magnitude(vec):
    mag = 0
    for a in vec:   
        mag += a**2
    return mag**0.5

# outdated
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

# outdated
def get_line_angle_to_plane(true_line_length, frame_line_length):
    """
    calculates how much you need to move the line in the z axis to get real line distance, only works one lines rotated to be smaller (such as the diagonals who always appear the correct size or shorter)
    :param true_line_length: not the actual lines length IRL but the length in pixels it would have taken if the tag faced the camera
    :param frame_line_length: the distance between on edge of the line to the other in frame
    """
    return math.asin(frame_line_length / true_line_length)

# outdated
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

def get_length_in_frame(focal_length, line_distance, length_irl):
    ''' :param focal_length:  the focal length of the camera at it's default state, in units of pixels
    can be described as the length of a line on frame, multiplied by it's distance from the camera and divided by its length in real life

    FOCAL_LENGTH = :math:' P * D / S'

    where P is the amount of pixels in the frame representing the object, D is the real life distance between the object and the camera S is the real life length of the line
    :param line_distance: the distance of the line from the camera in real life
    :param length_irl: the length of the line in real life
    '''
    # let P be the ammount of pixels the line takes on frame, let D be the lines distance from the camera, let S be the lines length, let F be focal_length
    # P * D / S = F ----> F * S = P * D ---> P = F * S / D
    return focal_length * length_irl / line_distance
def get_tag_points_location_by_height(tag, tag_size, tag_height, frame_height, frame_width):
    corners = tag.corners
    
def get_true_coords_center(line1_coords, line2_coords, line3_coords, line4_coords, focal_length):
    '''
    let line coords be the real locations of each of the lines sides
    :returns: the real locatioon of the center of the square
    '''
    xyz = midpoint(midpoint(line1_coords, line3_coords), midpoint(line2_coords, line4_coords))
    # xyz[2] = min(midpoint(line1_coords, line3_coords)[2], midpoint(line2_coords, line4_coords)[2])
    size_est = (distance(line1_coords, line3_coords) + distance(line2_coords, line4_coords))/2
    xyz[2] = (size_est) * focal_length / get_length_in_frame(focal_length, xyz[2], size_est) 
    return xyz
    
def get_true_coords_sides(pixel_coords, width, height, focal_length, length_irl):
    ''' :param focal_length:  the focal length of the camera at it's default state, in units of pixels
    can be described as the square root of the amount of pixels an object takes on a frame, multiplied by it's distance from the camera and divided by the square root of it's surface

    FOCAL_LENGTH = :math:' sqrt(P) * D / sqrt(S)'

    where P is the amount of pixels in the frame representing the object, D is the real life distance between the object and the camera S is the real life surface area (in 2d projection) of the object note that this is a constant, whatever object you choose to use, this formula will yield the same result
    :returns: an array representing the true coordinations of the sides of the square
    '''
    # let lmbda be the ratio between the distance calculated with the line at an angle and the real distance
    # lmbda is also equal to the ratio between the size of the rotated object in frame to the size of the object if not rotated in frame
    # (let pixels be a 1d line of pixels) pixels*lmbda = the ammount of pixels the object should take if rotated to look at the camera directly
    # raw coords represent an unaccurate estimation of the lines location in real life
    line1_raw_coords = calc_line_coords(pixel_coords[0], pixel_coords[1], length_irl, focal_length, width, height)
    line2_raw_coords = calc_line_coords(pixel_coords[1], pixel_coords[2], length_irl, focal_length, width, height)
    line3_raw_coords = calc_line_coords(pixel_coords[2], pixel_coords[3], length_irl, focal_length, width, height)
    line4_raw_coords = calc_line_coords(pixel_coords[3], pixel_coords[0], length_irl, focal_length, width, height)
    # let lmbda1 be lmbda for the pair (line1, line3) and lmbda2 be the lmbda for the pair (line2, line4)
    lmbda1 = length_irl / distance(line1_raw_coords, line3_raw_coords) 
    lmbda2 = length_irl / distance(line2_raw_coords, line4_raw_coords) 
    lmbda = (lmbda1*lmbda2)**0.5
    print("lmbda1: " + str(lmbda1) + " lmbda2: " + str(lmbda2) + " lmbda: " + str(lmbda))
    
    # let line coords be the actual coords of the lines
    line1_coords = calc_line_coords(pixel_coords[0], pixel_coords[1], length_irl, focal_length, width, height, lmbda1)
    line2_coords = calc_line_coords(pixel_coords[1], pixel_coords[2], length_irl, focal_length, width, height, lmbda2)
    line3_coords = calc_line_coords(pixel_coords[2], pixel_coords[3], length_irl, focal_length, width, height, lmbda1)
    line4_coords = calc_line_coords(pixel_coords[3], pixel_coords[0], length_irl, focal_length, width, height, lmbda2)
    # return [divide_by_s(a, lmbda) for a in [line1_coords, line2_coords, line3_coords, line4_coords]]
    lines =  [line1_coords, line2_coords, line3_coords, line4_coords]
    for line in lines:
        line[2] *= lmbda
    return lines

def calc_line_coords(point1, point2, length_irl, focal_length, frame_width, frame_height, lmbda= 1):
    ''' :param focal_length:  the focal length of the camera at it's default state, in units of pixels
    can be described as the length of a line on frame, multiplied by it's distance from the camera and divided by its length in real life

    FOCAL_LENGTH = :math:' P * D / S'

    where P is the amount of pixels in the frame representing the object, D is the real life distance between the object and the camera S is the real life length of the line
    '''
    pixels = distance(point1, point2) * lmbda
    center = midpoint(point1, point2)
    x = (length_irl/pixels) * (center[0] - (frame_width/2))
    y = (length_irl/pixels) * (center[1] - (frame_height/2))
    z = (length_irl) * focal_length / (pixels)
    return [x, y, z]
def get_square_angle(sides_coords):
    '''
    :param sides_coords: the coords of the sides of the square in real life, a float array of length 3 array of length 4 
    '''
    xyz1 = subtract(sides_coords[0], sides_coords[2])
    xyz2 = subtract(sides_coords[1], sides_coords[3])
    return [math.atan2(magnitude([xyz1[0], xyz1[1]]), xyz1[2]), math.atan2(magnitude([xyz2[0], xyz2[1]]), xyz2[2])]
    
    # return (math.acos(xy1/line_length), math.acos(xy2/line_length))
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
        # print("a" + str(true_half_side_length*2*(2**0.5)))
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
        
        # cv.line(image, (corners_facing_cam_01[0], corners_facing_cam_01[1]),
        #         (corners_facing_cam_02[0], corners_facing_cam_02[1]), (0, 0, 255), 2)
        # cv.line(image, (corners_facing_cam_02[0], corners_facing_cam_02[1]),
        #         (corners_facing_cam_03[0], corners_facing_cam_03[1]), (0, 0, 255), 2)
        # cv.line(image, (corners_facing_cam_03[0], corners_facing_cam_03[1]),
        #         (corners_facing_cam_04[0], corners_facing_cam_04[1]), (0, 255, 0), 2)
        # cv.line(image, (corners_facing_cam_04[0], corners_facing_cam_04[1]),
        #         (corners_facing_cam_01[0], corners_facing_cam_01[1]), (0, 255, 0), 2)


    cv.putText(image,
               "Elapsed Time:" + '{:.1f}'.format(elapsed_time * 1000) + "ms",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
               cv.LINE_AA)

    return image

# outdated
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

def draw_tags_facing_cam(image, tags, locations, focal_length, length_irl):
    for i in range(len(tags)):
        tag = tags[i]
        center = [int(a) for a in tag.center]
        half_side_length = int(get_length_in_frame(focal_length, locations[i][2], length_irl)/2)
        corners_facing_cam_01 = add_vectors(center, [half_side_length, half_side_length]) 
        corners_facing_cam_02 = add_vectors(center, [-half_side_length, half_side_length]) 
        corners_facing_cam_03 = add_vectors(center, [-half_side_length, -half_side_length]) 
        corners_facing_cam_04 = add_vectors(center, [half_side_length, -half_side_length]) 
        
        
        corners_facing_cam_01 = (int(corners_facing_cam_01[0]), int(corners_facing_cam_01[1]))
        corners_facing_cam_02 = (int(corners_facing_cam_02[0]), int(corners_facing_cam_02[1]))
        corners_facing_cam_03 = (int(corners_facing_cam_03[0]), int(corners_facing_cam_03[1]))
        corners_facing_cam_04 = (int(corners_facing_cam_04[0]), int(corners_facing_cam_04[1]))
        cv.line(image, corners_facing_cam_01, corners_facing_cam_02, (0, 255,0), 2)
        cv.line(image, corners_facing_cam_03, corners_facing_cam_02, (0, 255,0), 2)
        cv.line(image, corners_facing_cam_03, corners_facing_cam_04, (0, 255,0), 2)
        cv.line(image, corners_facing_cam_01, corners_facing_cam_04, (0, 255,0), 2)
    return image
# outdated
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
        # this calculation is very much not true but its more accurate than returning an arbitrary number
        return ((distance(corner_01, corner_02) + distance(corner_01, center) + distance(corner_02, center)) / 2)*1.5
# outdated
def get_axl_intersection(corner_01, corner_02, center):
    len_line_1 = distance(center, corner_01)
    len_line_2 = distance(center, corner_02)
    ratio_line_1 = len_line_1 / (len_line_1  + len_line_2)
    return add_vectors(multiply_by_s(subtract(corner_01, corner_02), ratio_line_1), corner_02)

def draw_tags_locations_and_wall_angle(image, tags, locations, angles):
    for i in range(len(tags)):
        tag = tags[i]
        # tag_family = tag.tag_family
        center = tag.center

        loc =  [round(a * 100)/100 for a in locations[i]]
        ang =  [round((90 - math.degrees(a)) * 100)/100 for a in angles[i]]
        center = (int(center[0]), int(center[1]))
        cv.putText(image, str(loc), (center[0] + 10, center[1] + 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2, cv.LINE_AA)
        cv.putText(image, str(ang), (center[0] - 20, center[1] + 35),
                   cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2, cv.LINE_AA)

    return image

if __name__ == '__main__':
    main()