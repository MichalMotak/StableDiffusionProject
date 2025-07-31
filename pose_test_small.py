








import torch
from diffusers import AutoencoderKLCogVideoX,PoseCogVideoXTransformer3DModel, CogVideoXVideoToVideoPipelineSmall, CogVideoXVideoToVideoPipeline, CogVideoXTransformer3DModel, CogVideoXDPMScheduler, PoseCogVideoXTransformer3DModelSmall
from diffusers.utils import export_to_video, load_video
from transformers import T5EncoderModel




import os
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"













if __name__ == "__main__":
    # print("a")  


    # model_id = "THUDM/CogVideoX-5b"

    # transformer = PoseCogVideoXTransformer3DModelSmall() #.from_pretrained("camenduru/cogvideox-5b-float16", subfolder="transformer", torch_dtype=torch.float16)
    # vae = AutoencoderKLCogVideoX.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)
    # print("0")

    # pipe = CogVideoXVideoToVideoPipelineSmall(
    #                 transformer=transformer,
    #                 vae=vae,
    #                 torch_dtype=torch.float16)
    
    # # pipe = CogVideoXVideoToVideoPipelineSmall.from_pretrained(
    # #     model_id,
    # #     text_encoder=text_encoder,
    # #     transformer=transformer,
    # #     vae=vae,
    # #     torch_dtype=torch.float16,
    # # )

    # print("1")

    # pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)

    # pipe.enable_sequential_cpu_offload()
    # pipe.vae.enable_tiling()


    # print("2")

    # input_video = load_video(
    # "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/hiker.mp4"
    # )
    # prompt = (
    #     "An astronaut stands triumphantly at the peak of a towering mountain. Panorama of rugged peaks and "
    #     "valleys. Very futuristic vibe and animated aesthetic. Highlights of purple and golden colors in "
    #     "the scene. The sky is looks like an animated/cartoonish dream of galaxies, nebulae, stars, planets, "
    #     "moons, but the remainder of the scene is mostly realistic."
    # )

    # print("input_video ", len(input_video))

    # video = pipe(video=input_video, prompt=prompt, strength=0.7, guidance_scale=6, use_dynamic_cfg=True, num_inference_steps=50).frames[0]

    # print("3")

    # export_to_video(video, "output.mp4", fps=8)
    # export_to_video(input_video, "input.mp4", fps=8)

    # # from IPython.display import display, Video
    # # display(Video("input.mp4", embed=True))
    # # display(Video("output.mp4", embed=True))









    #  TEST 
    

    model_id = "THUDM/CogVideoX-5b"


    # transformer = PoseCogVideoXTransformer3DModelSmall.from_pretrained("camenduru/cogvideox-5b-float16", subfolder="transformer", torch_dtype=torch.float16)

    # model_kwargs = {"torch_dtype ": torch.float16}
    transformer = PoseCogVideoXTransformer3DModelSmall().to(dtype=torch.float16)
    vae = AutoencoderKLCogVideoX.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)
    print("0")
    # vae = AutoencoderKLCogVideoX()
    # vae = AutoencoderKLCogVideoX().to(dtype=torch.float16)


    text_encoder = T5EncoderModel.from_pretrained("camenduru/cogvideox-5b-float16", subfolder="text_encoder", torch_dtype=torch.float16)

    pipe = CogVideoXVideoToVideoPipelineSmall.from_pretrained(
        model_id,
        # text_encoder=text_encoder,
        transformer=transformer,
        vae=vae,
        torch_dtype=torch.float16,
    )

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
    input_video = torch.ones([1, 49, 3, 224, 224])

    print("input_video ", len(input_video))

    video = pipe(video=input_video, prompt=prompt, strength=0.7, guidance_scale=6, use_dynamic_cfg=True, num_inference_steps=50).frames[0]

    print("3")

    export_to_video(video, "output.mp4", fps=8)
    export_to_video(input_video, "input.mp4", fps=8)

    # from IPython.display import display, Video
    # display(Video("input.mp4", embed=True))
    # display(Video("output.mp4", embed=True))



        # num_attention_heads: int = 30,
        # attention_head_dim: int = 64,
        # in_channels: int = 16,
        # out_channels: Optional[int] = 16,
        # flip_sin_to_cos: bool = True,
        # freq_shift: int = 0,
        # time_embed_dim: int = 512,
        # ofs_embed_dim: Optional[int] = None,
        # text_embed_dim: int = 4096,
        # num_layers: int = 30,
        # dropout: float = 0.0,
        # attention_bias: bool = True,
        # sample_width: int = 90,
        # sample_height: int = 60,
        # sample_frames: int = 49,
        # patch_size: int = 2,
        # patch_size_t: Optional[int] = None,
        # temporal_compression_ratio: int = 4,
        # max_text_seq_length: int = 226,
        # activation_fn: str = "gelu-approximate",
        # timestep_activation_fn: str = "silu",
        # norm_elementwise_affine: bool = True,
        # norm_eps: float = 1e-5,
        # spatial_interpolation_scale: float = 1.875,
        # temporal_interpolation_scale: float = 1.0,
        # use_rotary_positional_embeddings: bool = False,
        # use_learned_positional_embeddings: bool = False,
        # patch_bias: bool = True,