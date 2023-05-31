import gbvision as gbv
import copy
import cv2
import keyboard
import math

FOCAL_LENGTH_X = 1000
FOCAL_LENGTH_Y = 562.5
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

def project_point(x, y, z, frame):
    center = [int(FRAME_WIDTH * (x/(FOCAL_LENGTH_X*z) + 0.5)),
                       int(FRAME_HEIGHT * (y/(FOCAL_LENGTH_Y*z) + 0.5))]
    frame = cv2.circle(frame, center, 1, (255, 0, 0), 5)
    # return [FRAME_WIDTH * (x/(FOCAL_LENGTH_X*z) + 0.5), FRAME_HEIGHT * (y/(FOCAL_LENGTH_Y*z) + 0.5)]
    return frame


def project_sphere(x, y, z, radius, frame):
    center = [int(FRAME_WIDTH * (x/(FOCAL_LENGTH_X*z) + 0.5)),
                       int(FRAME_HEIGHT * (y/(FOCAL_LENGTH_Y*z) + 0.5))]
    frame = cv2.circle(frame, center,  int((radius*radius*FRAME_HEIGHT*FRAME_WIDTH/(z*z*FOCAL_LENGTH_X*FOCAL_LENGTH_Y))**0.5), (0, 255, 0),5)
    # return [FRAME_WIDTH * (x/(FOCAL_LENGTH_X*z) + 0.5), FRAME_HEIGHT * (y/(FOCAL_LENGTH_Y*z) + 0.5)]
    return frame

def main():
    z = 0.0
    y = 0.0
    x = 0.0
    speed = 0.05
    z_speed = 0.0005
    rotation_speed = 0.5
    rotation_x = 0
    rotation_y = 0
    rotation_z = 0
    epsilon = 0.0001
    # matrix1 = [[-1,1,0], [1,1,0], [1,-1,0], [-1,-1,0]]
    # matrix2 = [[-1,1,0.01], [1,1,0.01], [1,-1,0.01], [-1,-1,0.01]]
    matrix1 = [[[-1],[ 1],[0]],
               [[1 ],[ 1],[0]],
               [[1 ],[-1],[0]],
               [[-1],[-1],[0]]]
    # matrix2 = [[[-1],[1 ],[0.01]],
    #            [[1 ],[1 ],[0.01]],
    #            [[1 ],[-1],[0.01]],
    #            [[-1],[-1],[0.01]]]
    # F_LENGTH = 697.0395744431028 # (sqrt(P) * D) / sqrt(s)
    image = cv2.imread('black_screen.jpg')
    frame = copy.deepcopy(image)
    win = gbv.FeedWindow("win")
    while (True):
        frame = copy.deepcopy(image)
        if keyboard.is_pressed("s"):
            z -= z_speed
        if keyboard.is_pressed("w"):
            z += z_speed
        if keyboard.is_pressed("down"):
            y += speed
        if keyboard.is_pressed("up"):
            y -= speed
        if keyboard.is_pressed("right"):
            x += speed
        if keyboard.is_pressed("left"):
            x -= speed
        if keyboard.is_pressed("a"):
            rotation_y -= rotation_speed
        if keyboard.is_pressed("d"):
            rotation_y += rotation_speed
        if keyboard.is_pressed("z") and keyboard.is_pressed("+"):
            rotation_z += rotation_speed
        if keyboard.is_pressed("z") and keyboard.is_pressed("-"):
            rotation_z -= rotation_speed
        if keyboard.is_pressed("x") and keyboard.is_pressed("+"):
            rotation_x += rotation_speed
        if keyboard.is_pressed("x") and keyboard.is_pressed("-"):
            rotation_x -= rotation_speed
        z = max(z, epsilon)
        frame = project_sphere(x,y,z,math.sqrt(2),frame)
        # matrix1t = rotate_yaw_pitch_roll(copy.deepcopy(matrix1), rotation_x, rotation_y, rotation_z)
        matrix2t = copy.deepcopy(matrix1)
        for i in range(len(matrix1)):
            # frame = project_point(matrix1t[i][0][0] + x, matrix1t[i][1][0]+ y, matrix1t[i][2][0] + z, frame)
            frame = project_point(matrix2t[i][0][0] + x, matrix2t[i][1][0]+ y, matrix2t[i][2][0] + z, frame)
        win.show_frame(frame)
        
def multiply_matrices(matrix1, matrix2):
    # Check if the dimensions of the matrices are compatible for multiplication
    # if len(matrix1) != len(matrix2[0]) or len(matrix1[0]) != len(matrix2):
    #     return None  # Return None if the dimensions are not compatible
    
    # Create a result matrix with the appropriate dimensions
    result = [[[0]] * len(matrix2[0]) for _ in range(len(matrix1))]
    
    # Perform matrix multiplication
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                result[i][j][0] += matrix1[i][k][0] * matrix2[k][j][0]
    
    return result

def rotate_yaw_pitch_roll(matrix, yaw, pitch, roll):
    # Create individual rotation matrices for each of the yaw, pitch, and roll angles
    # The order of multiplication is roll-pitch-yaw (i.e., the yaw rotation is applied first)
    yaw_matrix = [[[math.cos(yaw)], [-math.sin(yaw)], [0]],
                  [[math.sin(yaw)], [math.cos(yaw)], [0]],
                  [[0], [0], [1]]]
    
    pitch_matrix = [[[math.cos(pitch)], [0], [math.sin(pitch)]],
                    [[0], [1], [0]],
                    [[-math.sin(pitch)], [0], [math.cos(pitch)]]]
    
    roll_matrix = [[[1], [0], [0]],
                   [[0], [math.cos(roll)], [-math.sin(roll)]],
                   [[0], [math.sin(roll)], [math.cos(roll)]]]
    
    # Multiply the rotation matrices in the appropriate order
    rotation_matrix = multiply_matrices(roll_matrix, pitch_matrix)
    rotation_matrix = multiply_matrices(rotation_matrix, yaw_matrix)
    
    # Apply the rotation to the input matrix by matrix multiplication
    rotated_matrix = multiply_matrices(rotation_matrix, transpose(matrix))
    
    return rotated_matrix

def transpose(matrix):
    # Create a result matrix with the appropriate dimensions
    result = [[[0]] * len(matrix) for _ in range(len(matrix[0][0]))]
    
    # Perform matrix transposition
    for i in range(len(matrix)):
        for j in range(len(matrix[0][0])):
            result[j][i][0] = matrix[i][0][j]
    
    return result

if __name__ == '__main__':
    main()
    