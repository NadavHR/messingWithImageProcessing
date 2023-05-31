import math
import copy
import numpy as np
import cv2
import gbvision as gbv
import settings


'''
perspective projection of point
focal_length should be given as: sqrt(tan(fov_x/2)*2*tan(fov_y/2)*2)
'''
def project_point(x, y, z, width, height, focal_length):
    return [width * (x/(focal_length*(width/height)*z) + 0.5), height * (y/(focal_length*(height/width)*z) + 0.5)]

'''
this method takes the corners of the projected rectangle and finds the point on frame that coresponds to the middle point of the original rectangle
'''
def find_projected_rect_middle(corners):
    # points go p1 -> p2 -> p3 -> p4 -> p1
    # diagonals are (p1, p3) and (p2, p4)
    # the diagonals equations are: 
    # d1(x) = x*((p3[1] - p1[1])/(p3[0] - p1[0])) + p1[1] - p1[0]*((p3[1] - p1[1])/(p3[0] - p1[0]))
    # d2(x) = x*((p4[1] - p2[1])/(p4[0] - p2[0])) + p2[1] - p2[0]*((p4[1] - p2[1])/(p4[0] - p2[0]))
    p1 = corners[0]
    p2 = corners[1]
    p3 = corners[2]
    p4 = corners[3]
    x = (p2[1] - (p2[0]*(p4[1] - p2[1])/(p4[0]-p2[0])) - p1[1]+(p1[0]*(p3[1]-p1[1])/(p3[0]-p1[0]))
         ) / (((p3[1]-p1[1])/(p3[0]-p1[0])) - ((p4[1]-p2[1])/(p4[0]-p2[0])))
    # print(x*((p3[1] - p1[1])/(p3[0] - p1[0])) + p1[1] - p1[0]*((p3[1] - p1[1])/(p3[0] - p1[0])) - x*((p4[1] - p2[1])/(p4[0] - p2[0])) - p2[1] + p2[0]*((p4[1] - p2[1])/(p4[0] - p2[0])))
    return [x, (x*(p3[1] - p1[1])/(p3[0] - p1[0])) + p1[1] - (p1[0]*(p3[1] - p1[1])/(p3[0] - p1[0]))]
    # return [x, ((x*(p3[1] - p1[1])/(p3[0] - p1[0])) + p1[1] - (p1[0]*(p3[1] - p1[1])/(p3[0] - p1[0]))+x*((p4[1] - p2[1])/(p4[0] - p2[0])) + p2[1] - p2[0]*((p4[1] - p2[1])/(p4[0] - p2[0])))/2]

'''
this function takes the 2d projected points on the frame and returns 3d normlized vectors that get projected to that point
focal_length should be given as: sqrt(tan(fov_x/2)*2*tan(fov_y/2)*2)
'''
def proj_points_to_norm_vectors_3d(corners, focal_length, width, height):
    norm_vecs = []
    for corner in corners:
        norm_vecs.append(proj_point_to_norm_vector_3d(corner, focal_length, width, height))
    return norm_vecs

'''
this function takes a single 2d projected point and returns the normalized 3d vector that gets projected to that point
focal_length should be given as: sqrt(tan(fov_x/2)*2*tan(fov_y/2)*2)
'''
def proj_point_to_norm_vector_3d(point, focal_length, width, height):
    # px = width * (x/(focal_length*(width/height)*z) + 0.5)
    # py = height * (y/(focal_length*(height/width)*z) + 0.5)
    # (px/width - 0.5) = x/(focal_length*(width/height)*z) ---> (px/width - 0.5)*focal_length*(width/height) = x/z
    # (py/height - 0.5)*focal_length*(height/width) = y/z
    # [x/z, y/z, 1] / M[x/z, y/z, 1]
    v = norm([(point[0]/width - 0.5)*focal_length*(width/height), (point[1]/height - 0.5)*focal_length*(height/width), 1])
    return np.array(v)
'''
this method returns the magnitude of the vector v
'''
def magnitude(v):
    sum = 0
    for a in v:
        sum += a**2
    return sum**0.5
'''
this method normalizes the vector v
'''
def norm(v):
    v2 = copy.deepcopy(v)
    m = magnitude(v)
    for i in range(len(v2)):
        v2[i] = v2[i] / m
    return v2
'''
this method takes a 3d vector and returns its rotation around the x axis and y axis in radians
'''
def vec_3d_x_y_rot(v):
    return [math.atan2(v[1], v[2]), math.atan2(v[0], v[2])]

'''
this method returns the magnitude of the distance between two vectors
'''
def distance(v1, v2):
    return magnitude(np.array(v1) - np.array(v2))

''' 
this method returns the xyz position of the rectangles corners in real life
corners: the corners of the rectangle
diag_length: the length of the diagonal of the rectangle in real life
focal_length should be given as: sqrt(tan(fov_x/2)*2*tan(fov_y/2)*2)
rect_width and rect_height: the width and the height of the rect irl
'''
def find_rect_corners_xyz(corners, rect_width, rect_height, focal_length, width, height):
    diag_length = (rect_width**2+rect_height**2)**0.5
    proj_center = find_projected_rect_middle(corners) # where the center is projected to
    norm_center = proj_point_to_norm_vector_3d(proj_center, focal_length, width, height) # the normalized vector that points to the direction of the center irl
    # angle_to_center = vec_3d_x_y_rot(norm_center) # the x and y rotation of the vector pointing to the center (in radians)
    norm_corners = proj_points_to_norm_vectors_3d(corners, focal_length, width, height) # the normalized vectors pointing to the corners
    norm_p1 = norm_corners[0]
    norm_p2 = norm_corners[1]
    norm_p3 = norm_corners[2]
    norm_p4 = norm_corners[3]
    p1, p3 = get_opposite_corners_xyz(norm_p1, norm_p3, norm_center, diag_length)
    p2, p4 = get_opposite_corners_xyz(norm_p2, norm_p4, norm_center, diag_length)
    # temp/
    scalar = magnitude(0.5*(p1 + p3)) / magnitude(0.5*(p2 + p4))
    if (scalar > 1):
        p2 *= scalar
        p4 *= scalar
    else:
        p1 *= (1/scalar)
        p3 *= (1/scalar)
    scalar = ((rect_width*rect_height)/(distance(p1, p2)*distance(p2,p3)))
    p1 = p1 * scalar
    p2 = p2 * scalar
    p3 = p3 * scalar
    p4 = p4 * scalar
    # /temp
    return [p1, p2, p3, p4]
    
'''
takes the normalized vectors pointing to two opposing corners and the center and also the length of the diagonal irl and returns the real life positions of the opposing sides
'''
def get_opposite_corners_xyz(norm_v1, norm_v2, norm_center, diag_length):
    theta_1 = math.acos(np.dot(norm_v1, norm_center))# the absolute angle between v1 and the center
    theta_2 = math.acos(np.dot(norm_v2, norm_center))# the absolute angle between v2 and the center
    # let a be the the angle between the lines (v1, center) and (camera, center), let camera be [0,0,0]
    # the area of the triangle (v1, center, camera) is equal to the area of the triangle (v2, center, camera) as the first one is:
    # S(v1, center, camera) = (diag_length/2) * distance(camera, center) * sin(a)/2
    # S(v2, center, camera) = (diag_length/2) * distance(camera, center) * sin(pi - a)/2 = (diag_length/2) * distance(camera, center) * sin(a)/2 = S(v1, center, camera)
    # which means that distance(camera, v1) * distance(camera, center) * sin(theta_1)/2 = distance(camera, v2) * distance(camera, center) * sin(theta_2)/2 --->
    # distance(camera, v1) * sin(theta_1) = distance(camera, v2) * sin(theta_2) --->
    # sin(theta_1)/sin(theta_2) = distance(camera, v2)/ distance(camera, v1)
    ratiod_v1 = np.array(norm_v1) * math.sin(theta_2) / math.sin(theta_1) # v1 scaled to have a the correct relative magnitue to norm_v2
    ratiod_v2 = np.array(norm_v2) # v2 scaled to have the correct relative magnitude to ratiod_v1
    ratiod_diag_length = distance(ratiod_v1, ratiod_v2) # the length of the diagonal if scaled to the relative magnitude of ratiod_v1 and ratiod_v2
    # ratiod_diag_length = ((math.sin(theta_2) / math.sin(theta_1))**2 + 1 - 2*(math.sin(theta_2) / math.sin(theta_1))*np.dot(norm_v1, norm_v2))**0.5 # the length of the diagonal if scaled to the relative magnitude of ratiod_v1 and ratiod_v2
    correct_scalar = diag_length / ratiod_diag_length # the scalar by whicch if we multiply ratiod_v1 and ratiod_v2 we get v1 and v2
    # correct_v1=ratiod_v1 * correct_scalar
    # correct_v2 =  ratiod_v2 * correct_scalar
    return [ratiod_v1 * correct_scalar, ratiod_v2 * correct_scalar]

    
'''
returns the point in 3d space that coresponds to the end of the line of the given length that is perpendicular to the two points given and facing the camera (you can ignore the right hand rule)
p1 and p2: 2 3d corners of the rectangle who are perpendicular to each other
center: 3d center point of the rectangle
line_length: the length of the axis line
'''
def get_cross_axis_no_flip(p1,p2, center, line_length = 1):
    changed_zero_p1 = p1 - center
    changed_zero_p2 = p2 - center
    adjusted_cross_product = norm(np.cross(changed_zero_p1, changed_zero_p2)) * line_length
    option1 = center - adjusted_cross_product
    option2 = center + adjusted_cross_product
    if (magnitude(option1) < magnitude(option2)):
        return option1
    return option2
 
def main():
    # np.set_printoptions(7)
    cam = gbv.USBCamera(settings.CAMERA_PORT, gbv.LIFECAM_3000)
    cam.set_exposure(settings.EXPOSURE)
    F_LENGTH = (math.tan(0.4435563306578366)*2*math.tan(0.3337068395920225)*2)**0.5 
    SIDE_LENGTH = 15.3
    
    win = gbv.FeedWindow("window")
    while True:
        ok, frame = cam.read()
        if frame is not None:
            # getting the 2d points on the frame
            processed_frame = copy.deepcopy(frame)
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
            cv2.aruco_dict = cv2.aruco.DICT_APRILTAG_16h5
            parameters = cv2.aruco.DetectorParameters()
            
            detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco_dict), parameters)
            proj_squares, ids, rejected_img_points = detector.detectMarkers(processed_frame)
            
            for corners in proj_squares:
                
                corners_3d = find_rect_corners_xyz(corners[0], SIDE_LENGTH, SIDE_LENGTH, F_LENGTH, cam.get_width(), cam.get_height())
                
                
                center_3d = 0.5*(corners_3d[0] + corners_3d[2])
                # print(math.acos(np.dot(norm(corners_3d[0] - center_3d), norm(corners_3d[1] - center_3d))) * 180/math.pi) #  this was to check if the angles really are 90 degrees as they should be
                proj_center =  project_point(center_3d[0], center_3d[1], center_3d[2], cam.get_width(), cam.get_height(), F_LENGTH)
                proj_cros_axis_point = get_cross_axis_no_flip(corners_3d[0], corners_3d[1], center_3d, SIDE_LENGTH)
                proj_cros_axis_point = project_point(proj_cros_axis_point[0], proj_cros_axis_point[1], proj_cros_axis_point[2], cam.get_width(), cam.get_height(), F_LENGTH)
                
                cv2.circle(frame, (int(proj_center[0]), int(proj_center[1])), 5, (255,0,0), 10)
                print(magnitude(center_3d))
                # print(distance(corners_3d[0], corners_3d[1]) )
                # print(max(max(max(corners_3d[0][2], corners_3d[1][2]), corners_3d[2][2]), corners_3d[3][2]) - min(min(min(corners_3d[0][2], corners_3d[1][2]), corners_3d[2][2]), corners_3d[3][2]))
                for i in range(len(corners_3d)):
                    corner = corners_3d[i]
                    proj_corner = project_point(corner[0], corner[1], corner[2], cam.get_width(), cam.get_height(), F_LENGTH)
                    # print(magnitude(np.array(proj_corner) - np.array(corners[0][i])))
                    
                    cv2.circle(frame, (int(proj_corner[0]), int(proj_corner[1])), int(0.01/corner[2]), (255,0,0), 10)
                    cv2.putText(frame, str((int(corner[2] * 1000))/1000), (int(proj_corner[0]) + 10, int(proj_corner[1]) + 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
                cv2.line(frame,(int(proj_center[0]), int(proj_center[1])), 
                         (int(proj_cros_axis_point[0]), int(proj_cros_axis_point[1])), (0,0,255), 5)
            win.show_frame(frame)
        
        
        
        
        

        
        


if __name__ == '__main__':
    main()
    