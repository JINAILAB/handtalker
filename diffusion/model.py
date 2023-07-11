from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
import torch


def available_control_net():
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16)
    # pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    #     "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    # )
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "stablediffusionapi/anything-v5", controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()

    return pipe


# prompt 개수에 따라서 생기는 image의 개수가 바뀜.(prompt의 개수는 곧 batchsize이다.)
# gudiance_scale Prompt를 얼마나 잘 따라서 만들 것인가
# controlnet_conditioning_scale: controlnet를 얼마나 잘 살릴 것인가
# strength: controlnet를 얼마나 잘 살릴 것인가
# num_inference_steps: 생성된 이미지 퀄리티, Diffusion denoising steps
def run_control_net(prompt,
                    person_image,
                    control_net_image,
                    negative_prompt=None,
                    num_inference_steps=30,
                    pipe=available_control_net()):
    if negative_prompt is None:
        negative_prompt = ["nsfw, stuff, monochrome, lowres, bad anatomy, worst quality, low quality"]
    diffusion_images = pipe(
        prompt,
        negative_prompt=negative_prompt * len(prompt),
        image=person_image,
        control_image=control_net_image,
        guidance_scale=25, # 20, 7.5
        controlnet_conditioning_scale=1.5, # 1.5. 1.3
        strength=0.9, # 0.9
        generator=[torch.Generator(device="cuda").manual_seed(102) for _ in range(len(prompt))],
        num_inference_steps=num_inference_steps
    )
    return diffusion_images

