from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from diffusers.utils import load_image
import cv2
import numpy as np





def available_control_net():
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()
    
    return pipe


def run_control_net(prompt, images, negative_prompt, generator, num_inference_steps):
    output = available_control_net()(
        prompt,
        images,
        negative_prompt,
        generator = generator,
        num_inference_steps = num_inference_steps
    )
    return output
    




prompt = ", best quality, extremely detailed"
prompt = [t + prompt for t in ["Sandra Oh", "Kim Kardashian", "rihanna", "taylor swift"]]
generator = [torch.Generator(device="cuda").manual_seed(102) for i in range(len(prompt))]

output = pipe(
    prompt,
    canny_image,
    negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * len(prompt),
    generator=generator,
    num_inference_steps=20,
)

print(output.images[0])

for i in range(4):
    output.images[i].save(f'./images/{i+2}.png', 'png')
