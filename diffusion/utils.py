import cv2
import numpy as np
import os

# folder dirs에서 image 갖고오기 # 갖고오면 resize도 같이 해줌.
def get_images_resize(folder_dir, resize_hw):
    img_dirs = os.listdir(folder_dir)
        
    imgs = []
    for img in sorted(img_dirs):
        if img.endswith(".png") or img.endswith(".jpg"):
            img = cv2.imread(img, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (resize_hw, resize_hw))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (2, 0, 1))
            imgs.append(img)
            
    return imgs

# batchsize로 나눠주고 남은 image는 삭제
# ex) 18장의 image를 받고 batchsize가 4이면 뒤에 마지막 두장을 삭제하고 [[4장],[4장],[4장],[4장]] 다음과 같은 형식으로 리턴해줌.
def split_by_batch_size(imgs, batch_size):
    imgs_cnt = int((len(imgs) // batch_size) * batch_size)
    imgs = imgs[:imgs_cnt]
    imgs = np.array(imgs)
    imgs = imgs.reshape(len(imgs) // batch_size, batch_size)
    
    return imgs


    
    




        
    