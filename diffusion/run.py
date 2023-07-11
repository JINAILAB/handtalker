from utils import *
from model import *
import argparse
import numpy as np
import torch
import cv2
from PIL import Image

parser = argparse.ArgumentParser(description='make_movie_with_controlnet')
parser.add_argument('--folder_name', type=str,
                    default='./open_pose_images',
                    help='open_pose_image가 들어있는 folder의 이름')

parser.add_argument('--batch_size', type=int,
                    default=2,
                    help='gpu에 맞는 batchsize를 사용할 것')

parser.add_argument('--img_size', type=tuple,
                    default=(560, 480),
                    help='gpu에 맞는 batchsize를 사용할 것')





args = parser.parse_args()


def main(args=args):
    
    image = get_an_image_resize('./IU.png', (560, 480))
    pose_images = get_images_resize(args.folder_name, hw=(560, 480))
    pose_images = split_by_batch_size(pose_images, args.batch_size)
    for i in range(len(pose_images)):
        output_images = run_control_net(prompt=['best quality, a cute girl with white shirts, white background']*args.batch_size,
                    person_image=[image]*4,
                    control_net_image=pose_images[i],
                    negative_prompt=None,
                    num_inference_steps=35,
                    pipe=available_control_net())
        for j, img in enumerate(output_images):
            output_images.images[j].save(f'./final_images/{i}_{j}.png', 'png')
            
            
        
        
        
            
            
            
    
    
    prompt = "best quality, extremely detailed"
    
    
    
    
    
    
    
if __name__ == "__main__":
    main(args)