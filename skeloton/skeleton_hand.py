import numpy as np
import json
import math
import cv2
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


# jason file 불러오기
# file_name = '/content/drive/MyDrive/Colab Notebooks/수어_영상_생성_서비스_프로젝트/dataset/NIA_SL_WORD1507_REAL01_F_000000000050_keypoints.json'
def json2keypoints(json_file, hand_dir):
    with open(json_file, "r") as file:
        data = json.load(file)
    # data_json = data['people']['hand_left_keypoints_2d']
    keypoints = np.array(data["people"][f"hand_{hand_dir}_keypoints_2d"]).reshape(-1, 3)
    return keypoints


# ## matplot으로 그려서 확인 하기
# keypoints = json2keypoints(file_name, 'right')
# plt.scatter(keypoints[:, 0], keypoints[:, 1])
# # 키포인트 인덱스를 추가하여 어떤 키포인트가 어디에 있는지 확인합니다.
# for i, (x, y) in enumerate(zip(keypoints[:, 0], keypoints[:, 1])):
#     plt.text(x, y, str(i), fontsize=10, ha='right')

# plt.gca().invert_yaxis()

# 색상 값확인
import cv2

rgb_colors = ["BE280E", "A1C900", "00C73C", "22519E", "B300CD"]
bgr_colors = []

for rgb_color in rgb_colors:
    rgb_tuple = tuple(int(rgb_color[i : i + 2], 16) for i in (0, 2, 4))
    bgr_color = [rgb_tuple[2], rgb_tuple[1], rgb_tuple[0]]
    bgr_colors.append(bgr_color)

# print(bgr_colors)
# [[14, 40, 190], [0, 201, 161], [60, 199, 0], [158, 81, 34], [205, 0, 179]]


def draw_hand(json_data, canvas):
    limbSeq = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],  # 엄지
        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8],  # 검지
        [0, 9],
        [9, 10],
        [10, 11],
        [11, 12],  # 중지
        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16],  # 약지
        [0, 17],
        [17, 18],
        [18, 19],
        [19, 20],
    ]
    # joint_color = [
    #     [14, 40, 190],
    #     [0, 201, 161],
    #     [60, 199, 0],
    #     [158, 81, 34],
    #     [205, 0, 179],
    # ]
    limSeq_color = [
        [14, 40, 190],
        [14, 40, 190],
        [14, 40, 190],
        [14, 40, 190],
        [0, 201, 161],
        [0, 201, 161],
        [0, 201, 161],
        [0, 201, 161],
        [60, 199, 0],
        [60, 199, 0],
        [60, 199, 0],
        [60, 199, 0],
        [158, 81, 34],
        [158, 81, 34],
        [158, 81, 34],
        [158, 81, 34],
        [205, 0, 179],
        [205, 0, 179],
        [205, 0, 179],
        [205, 0, 179],
    ]

    # limbSeq
    for i in range(len(limbSeq)):
        idx = limbSeq[i]
        x1, y1, c1 = json_data[idx[0]]
        x2, y2, c2 = json_data[idx[1]]
        if c1 != 0 and c2 != 0:
            start_point = (int(x1), int(y1))  # ensure coordinates are int
            end_point = (int(x2), int(y2))  # ensure coordinates are int
            color = limSeq_color[i]
            thickness = 4
            cv2.line(canvas, start_point, end_point, color, thickness)  # draw line
    # joint
    for i in range(len(json_data)):
        x, y, c = json_data[i]

        cv2.circle(canvas, (int(x), int(y)), 5, [255, 0, 0], thickness=-1)

    return canvas


filename = []
for i in range(10, 71, 10):
    filename.append(
        f"./keypoint_json/NIA_SL_WORD1507_REAL01_F_0000000000{i}_keypoints.json"
    )
# print(filename)

for name in filename:
    data_right = json2keypoints(name, "right")
    data_left = json2keypoints(name, "left")

    # canvas 설정
    canvas_right = np.zeros((1280, 2000, 3), dtype=np.uint8)
    canvas_left = np.zeros((1280, 2000, 3), dtype=np.uint8)

    # Draw the face pose on the canvas
    canvas_right = draw_hand(data_right, canvas_right)
    canvas_left = draw_hand(data_left, canvas_left)

    cv2.imwrite(f"./res_hand_pose_image/pose_right{name[-18:-15]}.png", canvas_right)
    cv2.imwrite(f"./res_hand_pose_image/pose_left{name[-18:-15]}.png", canvas_left)
