import cv2
import numpy as np
import os

# folder dirs에서 image 갖고오기
def get_images(folder_dir):
    img_dirs = os.listdir(folder_dir)
        
    imgs = []
    for img in sorted(img_dirs):
        if img.endswith(".png") or img.endswith(".jpg"):
            img = cv2.imread(img, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (2, 0, 1))
            imgs.append(img)
            
    return imgs

# batchsize로 나눠주고 남은 image는 삭제
def split_by_batch_size(imgs, batch_size):
    imgs_cnt = int((len(imgs) // batch_size) * batch_size)
    imgs = imgs[:imgs_cnt]
    imgs = np.array(imgs)
    imgs = imgs.reshape(len(imgs) // batch_size, batch_size)
    
    return imgs
    
    




        
    