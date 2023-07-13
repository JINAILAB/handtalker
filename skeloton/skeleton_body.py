import json
import cv2
import numpy as np
import math

# jason file 불러오기


def plot_keypoints(json_file):
    with open(json_file, "r") as file:
        data = json.load(file)

    keypoints = np.array(data["people"]["pose_keypoints_2d"]).reshape(-1, 3)
    return keypoints


# jason_data = plot_keypoints('./NIA_SL_WORD1511_REAL01_F_000000000037_keypoints.json')
# x, y, c = jason_data[0]
# print(x, y, c)


def draw_bodypose_from_json(json_data, canvas):
    stickwidth = 5
    limbSeq = [
        [2, 3],
        [2, 6],  # 어깨
        [3, 4],
        [4, 5],  # 오른팔
        [6, 7],
        [7, 8],  # 왼팔
        [2, 10],  # 오른 엉덩이 # [9, 10], [10, 11], # 왼다리
        [2, 13],  # 왼 엉덩이  # [12, 13], [13, 14], # 오른다리
        [2, 1],  # 목
        [1, 16],
        [16, 18],  # 오른 얼굴
        [1, 17],
        [17, 19],  # 왼 얼굴
        #    [3, 17], [6, 18] # 뭐지
    ]  # 17
    # colors = [[153, 0, 0], [153, 51, 0],
    #           [153, 102, 0], [153, 153, 0],
    #           [102, 153, 0], [51, 153, 0],
    #           [0, 0, 153],
    #           [51, 0, 153], [102, 0, 153],
    #           [153, 0, 153], [153, 0, 102],
    #           ] # 11
    colors = [
        [153, 0, 1],
        [153, 51, 1],
        [152, 102, 1],
        [153, 153, 0],
        [102, 152, 0],
        [55, 152, 9],
        [4, 154, 1],
        [28, 148, 145],
        [0, 0, 152],
        [59, 1, 151],
        [99, 1, 153],
        [193, 5, 188],
        [160, 0, 104],
    ]

    joint_colors = [
        [255, 0, 0],
        [255, 85, 0],
        [155, 170, 0],
        [255, 255, 0],
        [170, 255, 0],
        [85, 255, 0],
        [0, 255, 0],
        [0, 255, 85],
        [0, 0, 0],
        [4, 154, 1],
        [0, 0, 0],
        [0, 0, 0],
        [28, 148, 145],
        [0, 0, 0],
        [170, 0, 255],
        [255, 0, 255],
        [255, 0, 170],
        [255, 0, 85],
        [255, 0, 85],
    ]

    # Draw keypoints on the canvas
    num = list(range(8)) + list(range(15, 20)) + [9, 12]
    for i in range(19):
        if i in num:
            x, y, c = json_data[i]
            if c != 0:
                cv2.circle(canvas, (int(x), int(y)), 4, joint_colors[i], thickness=10)

    # Draw limbs on the canvas
    for i in range(len(limbSeq)):
        index = limbSeq[i]
        x1, y1, c1 = json_data[index[0] - 1]
        x2, y2, c2 = json_data[index[1] - 1]
        if c1 != 0 and c2 != 0:
            start_point = (int(x1), int(y1))  # ensure coordinates are int
            end_point = (int(x2), int(y2))  # ensure coordinates are int
            color = colors[i]
            thickness = 10
            cv2.line(canvas, start_point, end_point, colors[i], thickness)  # draw line

    # cv2.imshow('Canvas', canvas)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return canvas


import matplotlib.pyplot as plt

# Load json data
filename = []
for i in range(10, 71, 10):
    filename.append(
        f"./keypoint_json/NIA_SL_WORD1507_REAL01_F_0000000000{i}_keypoints.json"
    )
print(filename)

for file in filename:
    # with open(file, 'r') as f:
    #     data = json.load(f)
    data = plot_keypoints(file)
    # Create a blank canvas
    canvas = np.zeros((1280, 2000, 3), dtype=np.uint8)

    # Draw the body pose on the canvas
    canvas = draw_bodypose_from_json(data, canvas)
    # Display the canvas
    # plt.imshow(canvas)
    # plt.savefig(f'pose_body_{file[-18:-15]}.png')
    # plt.show()
    # plt.close()
    # cv2.imwrite(f'pose_body_{file[-18:-15]}.png', canvas)
    # RGB 형식으로 변환하여 저장
    rgb_image = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"./res_body_pose_image/pose_body_hip{file[-18:-15]}.png", rgb_image)
