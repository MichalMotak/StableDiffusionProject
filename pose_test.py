








import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXVideoToVideoPipeline, CogVideoXTransformer3DModel, CogVideoXDPMScheduler, PoseCogVideoXTransformer3DModel
from diffusers.utils import export_to_video, load_video
from transformers import T5EncoderModel




import os
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"













if __name__ == "__main__":
    print("a")  


    model_id = "THUDM/CogVideoX-5b"

    transformer = PoseCogVideoXTransformer3DModel.from_pretrained("camenduru/cogvideox-5b-float16", subfolder="transformer", torch_dtype=torch.float16)
    text_encoder = T5EncoderModel.from_pretrained("camenduru/cogvideox-5b-float16", subfolder="text_encoder", torch_dtype=torch.float16)
    vae = AutoencoderKLCogVideoX.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)
    print("0")

    pipe = CogVideoXVideoToVideoPipeline.from_pretrained(
        model_id,
        text_encoder=text_encoder,
        transformer=transformer,
        vae=vae,
        torch_dtype=torch.float16,
    )

    print("1")

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)

    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_tiling()


    print("2")

    input_video = load_video(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/hiker.mp4"
    )
    prompt = (
        "An astronaut stands triumphantly at the peak of a towering mountain. Panorama of rugged peaks and "
        "valleys. Very futuristic vibe and animated aesthetic. Highlights of purple and golden colors in "
        "the scene. The sky is looks like an animated/cartoonish dream of galaxies, nebulae, stars, planets, "
        "moons, but the remainder of the scene is mostly realistic."
    )

    print("input_video ", len(input_video))



    video = pipe(video=input_video, prompt=prompt, strength=0.7, guidance_scale=6, use_dynamic_cfg=True, num_inference_steps=50).frames[0]

    print("3")

    export_to_video(video, "output.mp4", fps=8)
    export_to_video(input_video, "input.mp4", fps=8)

    # from IPython.display import display, Video
    # display(Video("input.mp4", embed=True))
    # display(Video("output.mp4", embed=True))