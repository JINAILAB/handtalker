import json
import cv2
import numpy as np
import math

# jason file 불러오기

def plot_keypoints(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)

    keypoints = np.array(data['people']['pose_keypoints_2d']).reshape(-1, 3)
    return keypoints

jason_data = plot_keypoints('./NIA_SL_WORD1511_REAL01_F_000000000037_keypoints.json')
x, y, c = jason_data[0]
print(x, y, c)

def draw_bodypose_from_json(json_data, canvas):
    
    stickwidth = 5
    limbSeq = [[2, 3], [2, 6], # 어깨
               [3, 4], [4, 5], # 왼팔
               [6, 7], [7, 8], # 오른팔
            #    [2, 9], [9, 10], [10, 11], # 왼다리
            #    [2, 12], [12, 13], [13, 14], # 오른다리
               [2, 1],  # 목
               [1, 16], [16, 18], # 왼 얼굴
               [1, 17], [17, 19], # 오른 얼굴
            #    [3, 17], [6, 18] # 뭐지
               ] # 19
    colors = [[153, 0, 0], [153, 51, 0], 
              [153, 102, 0], [153, 153, 0],
              [102, 153, 0], [51, 153, 0],
              [0, 0, 153],
              [51, 0, 153], [102, 0, 153],
              [153, 0, 153], [153, 0, 102],
              ] # 11

    joint_colors = [[255, 0, 0], [255, 85, 0], 
                    [155, 170, 0], [255, 255, 0], [170, 255, 0],
                    [85, 255, 0], [0, 255, 0], [0, 255, 85],
                    [0, 0, 0], [0, 0, 0], [0, 0, 0],
                    [0, 0, 0], [0, 0, 0], [0, 0, 0],
                    [170, 0, 255], [255, 0, 255],
                    [255, 0, 170], [255, 0, 85], [255, 0, 85]
                    ]

    # Draw keypoints on the canvas
    num = list(range(8)) + list(range(15,20))
    for i in range(19):
        if i in num:
            x, y, c = jason_data[i]
            if c != 0:
                cv2.circle(canvas, (int(x), int(y)), 4, joint_colors[i], thickness= 10)
        
    # Draw limbs on the canvas
    for i in range(len(limbSeq)):
        index = limbSeq[i]
        x1, y1, c1 = jason_data[index[0]-1]
        x2, y2, c2 = jason_data[index[1]-1]
        if c1 != 0 and c2 != 0:
            start_point = (int(x1), int(y1))  # ensure coordinates are int
            end_point = (int(x2), int(y2))    # ensure coordinates are int
            color = colors[i]
            thickness = 10
            cv2.line(canvas, start_point, end_point, colors[i], thickness)  # draw line

    # cv2.imshow('Canvas', canvas)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return canvas

import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
# Load json data
with open('./NIA_SL_WORD1511_REAL01_F_000000000037_keypoints.json', 'r') as f:
    data = json.load(f)

# Create a blank canvas
canvas = np.zeros((1280, 2000, 3), dtype=np.uint8)

# Draw the body pose on the canvas
canvas = draw_bodypose_from_json(data, canvas)

# Display the canvas
plt.imshow(canvas)
plt.show()
plt.savefig('dkdk.png')
