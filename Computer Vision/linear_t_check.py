import math
import cv2
import keyboard
import copy
import gbvision as gbv
import numpy as np
WIDTH = 1280
HEIGHT = 720
F_LENGTH_INVERSE = 1/1 # = sqrt(s) / (sqrt(P) * D)
def main():
    z = 0.0
    y = 0.0
    x = 0.0
    speed = 1
    z_speed = 1
    rotation_speed = 1
    rotation_x = 0
    rotation_y = 0
    rotation_z = 0
    epsilon = 0.01
    matrix1 = [[-1,1,0], [1,1,0], [1,-1,0], [-1,-1,0]]
    matrix2 = [[-1,1,0.01], [1,1,0.01], [1,-1,0.01], [-1,-1,0.01]]
    # matrix1 = [[[-1],[ 1],[0]],
    #            [[1 ],[ 1],[0]],
    #            [[1 ],[-1],[0]],
    #            [[-1],[-1],[0]]]
    # matrix2 = [[[-1],[1 ],[0.01]],
    #            [[1 ],[1 ],[0.01]],
    #            [[1 ],[-1],[0.01]],
    #            [[-1],[-1],[0.01]]]
    # F_LENGTH = 697.0395744431028 # (sqrt(P) * D) / sqrt(s)
    image = cv2.imread('black_screen.jpg')
    frame = copy.deepcopy(image)
    win = gbv.FeedWindow("win")
    while (True):
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
        # matrix = copy.deepcopy(matrix2)
        matrix = [[1, 0, 0], [0, 1, 0] , [0, 0, 1]]
        matrix = rotate(matrix, rotation_x, rotation_y, rotation_z)
        colors = [(255,0,0),(0,255,0), (0,0,255)]
        matrix_f = [add_vectors(a, [x,y,z]) for a in matrix]
        
        i = 0
        while i < len(matrix_f) - 1: # sort by distance
            if (magnitude(matrix_f[i]) < magnitude(matrix_f[i+1])):
                temp = copy.deepcopy(matrix_f[i])
                matrix_f[i] = matrix_f[i+1]
                matrix_f[i+1] = temp
                temp = copy.deepcopy(colors[i])
                colors[i] = colors[i+1]
                colors[i+1] = temp
                i = max(i-1, 0)
            else:
                i += 1
        
        for i in range(len(matrix_f)):
            v1 = matrix_f[i]
            # v2 = matrix_f[(i+1)%len(matrix_f)]
            p1 = T(v1[0], v1[1], v1[2])
            p2 = T(x, y, z)
            # p2 = T(v2[0], v2[1], v2[2])
            try:
                frame = cv2.line(frame, p1, p2, colors[i], 5)
                frame = cv2.circle(frame, T((x+v1[0])/2, (y+v1[1])/2, (z+v1[2])/2), 1, colors[(i+1)%len(colors)], 10)
            except:
                pass
        
        print(str((rotation_x, rotation_y, rotation_z)))
        # print(str([x,y,z]))
        # matrix_f = [add_vectors(a, [x,y,z]) for a in matrix1]
        # for i in range(len(matrix_f)):
        #     v1 = matrix_f[i]
        #     v2 = matrix_f[(i+1)%len(matrix_f)]
        #     p1 = T(v1[0], v1[1], v1[2])
        #     p2 = T(v2[0], v2[1], v2[2])
        #     try:
        #         frame = cv2.line(frame, p1, p2, (0,0,255), 5)
        #     except:
        #         pass
        
        
        # matrix_f = matrix2
        # for i in range(len(matrix_f)):
        #     v1 = matrix_f[i]
        #     v2 = matrix_f[(i+1)%len(matrix_f)]
        #     p1 = tuple(add_vectors(list(T(v1[0], v1[1], v1[2])), add_vectors(list(T(x, y, z)), [int(-WIDTH/2), int(-HEIGHT/2)])))
        #     p2 = tuple(add_vectors(list(T(v2[0], v2[1], v2[2])), add_vectors(list(T(x, y, z)), [int(-WIDTH/2), int(-HEIGHT/2)])))
        #     frame = cv2.line(frame, p1, p2, (0,255,0), 5)
        
        
        win.show_frame(frame)
        frame = copy.deepcopy(image)
        

def T(x, y, z):
    return (int(WIDTH*(F_LENGTH_INVERSE * x/(z)) + WIDTH/2), int(HEIGHT*(F_LENGTH_INVERSE *y/(z * HEIGHT/WIDTH))+HEIGHT/2))

def rotate(matrix, x, y, z):
    r_matrix = ((np.array([[1,0,0],
                                 [0, math.cos(math.radians(x)), -math.sin(math.radians(x))],
                                 [0, math.sin(math.radians(x)), math.cos(math.radians(x))]])  @ 
               np.array([[math.cos(math.radians(y)), 0, math.sin(math.radians(y))],
                                 [0,1,0],
                                 [-math.sin(math.radians(y)), 0, math.cos(math.radians(y))]])) @
              [[math.cos(math.radians(z)), -math.sin(math.radians(z)), 0],
                                 [math.sin(math.radians(z)), math.cos(math.radians(z)), 0], 
                                 [0, 0, 1]]) 
    matrix = r_matrix @ matrix
    return matrix
    # matrix = matrix_mult([[1,0,0],
    #                              [0, math.cos(math.radians(x)), -math.sin(math.radians(x))],
    #                              [0, math.sin(math.radians(x)), math.cos(math.radians(x))]],
    #                              matrix)
    # matrix = matrix_mult([[math.cos(math.radians(y)), 0, math.sin(math.radians(y))],
    #                              [0,1,0],
    #                              [-math.sin(math.radians(y)), 0, math.cos(math.radians(y))]],
    #                              matrix)
    # matrix = matrix_mult([[math.cos(math.radians(z)), -math.sin(math.radians(z)), 0],
    #                              [math.sin(math.radians(z)), math.cos(math.radians(z)), 0], 
    #                              [0, 0, 1]], 
    #                             matrix)
    # for i in range(len(square)):
        # square[i] = matrix_mult([[1,0,0],
        #                          [0, math.cos(math.radians(x)), -math.sin(math.radians(x))],
        #                          [0, math.sin(math.radians(x)), math.cos(math.radians(x))]],
        #                          matrix_transpose([square[i]]))
        # square[i] = matrix_mult([[math.cos(math.radians(y)), 0, math.sin(math.radians(y))],
        #                          [0,1,0],
        #                          [-math.sin(math.radians(y)), 0, math.cos(math.radians(y))]],
        #                          matrix_transpose([square[i]]))
        # square[i] = matrix_mult([[math.cos(math.radians(z)), -math.sin(math.radians(z)), 0],
        #                          [math.sin(math.radians(z)), math.cos(math.radians(z)), 0], 
        #                          [0, 0, 1]], 
        #                         matrix_transpose([square[i]]))
  
def add_vectors(v1, v2):
    v1 = copy.deepcopy(v1)
    for i in range(len(v1)):
        v1[i] += v2[i]
    return v1  

def flatten_list(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list

def matrix_mult(B, A):
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])

    if cols_A != rows_B:
        raise Exception("Cannot multiply the two matrices. Incorrect dimensions.")

    # Create the result matrix C with dimensions rows_A x cols_B
    C = [[0 for row in range(cols_B)] for col in range(rows_A)]

    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                try:
                    C[i][j] += A[i][k] * B[k][j]
                except:
                    C[i][j] += A[i][k] * B[k][j][0]
    return C

def matrix_transpose(A):
    rows = len(A)
    cols = len(A[0])

    # Create the transposed matrix B with dimensions cols x rows
    B = [[0 for row in range(rows)] for col in range(cols)]

    for i in range(rows):
        for j in range(cols):
            B[j][i] = A[i][j]

    return B

def magnitude(v):
    sum = 0
    for s in v:
        sum += s**2
    return sum**0.5

if __name__ == '__main__':
    main()
    