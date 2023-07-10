from utils import get_images
from model import run_control_net
import argparse
import numpy as np
import torch

parser = argparse.ArgumentParser(description='make_movie_with_controlnet')
parser.add_argument('--folder_name', type=str,
                    default='./open_pose_images',
                    help='open_pose_image가 들어있는 folder의 이름')

parser.add_argument('--batch_size', type=int,
                    default='./open_pose_images',
                    help='gpu에 맞는 batchsize를 사용할 것')

parser.add_argument('--batch_size', type=int,
                    default='./open_pose_images',
                    help='gpu에 맞는 batchsize를 사용할 것')





args = parser.parse_args()


def main(args=args):
    images = get_images(args.folder_name)
    
    prompt = "best quality, extremely detailed"
    
    generator = [torch.Generator(device="cuda").manual_seed(102) for i in range(batch_size)]
    
    
    
    return pass
    
    
    
    
    
    
    
if __name__ == "__main__":
    main(args)