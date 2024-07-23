from diffsynth import ModelManager, SDVideoPipeline, ControlNetConfigUnit, VideoData, save_video, save_frames
from diffsynth.extensions.RIFE import RIFESmoother
import torch


# Download models
# `models/stable_diffusion/flat2DAnimerge_v45Sharp.safetensors`: [link](https://civitai.com/api/download/models/266360?type=Model&format=SafeTensor&size=pruned&fp=fp16)
# `models/AnimateDiff/mm_sd_v15_v2.ckpt`: [link](https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt)
# `models/ControlNet/control_v11p_sd15_lineart.pth`: [link](https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart.pth)
# `models/ControlNet/control_v11f1e_sd15_tile.pth`: [link](https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1e_sd15_tile.pth)
# `models/Annotators/sk_model.pth`: [link](https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model.pth)
# `models/Annotators/sk_model2.pth`: [link](https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model2.pth)
# `models/textual_inversion/verybadimagenegative_v1.3.pt`: [link](https://civitai.com/api/download/models/25820?type=Model&format=PickleTensor&size=full&fp=fp16)
# `models/RIFE/flownet.pkl`: [link](https://drive.google.com/file/d/1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7_/view?usp=sharing)


# Load models
model_manager = ModelManager(torch_dtype=torch.float16, device="cuda")
model_manager.load_textual_inversions("models/textual_inversion")
model_manager.load_models([
    "models/stable_diffusion/flat2DAnimerge_v45Sharp.safetensors",
    "models/AnimateDiff/mm_sd_v15_v2.ckpt",
    "models/ControlNet/control_v11p_sd15_lineart.pth",
    "models/ControlNet/control_v11f1e_sd15_tile.pth",
    "models/RIFE/flownet.pkl"
])
pipe = SDVideoPipeline.from_model_manager(
    model_manager,
    [
        ControlNetConfigUnit(
            processor_id="lineart",
            model_path="models/ControlNet/control_v11p_sd15_lineart.pth",
            scale=0.5
        ),
        ControlNetConfigUnit(
            processor_id="tile",
            model_path="models/ControlNet/control_v11f1e_sd15_tile.pth",
            scale=0.5
        )
    ]
)
smoother = RIFESmoother.from_model_manager(model_manager)

# Load video (we only use 60 frames for quick testing)
# The original video is here: https://www.bilibili.com/video/BV19w411A7YJ/
video = VideoData(
    video_file="data/bilibili_videos/៸៸᳐_⩊_៸៸᳐ 66 微笑调查队🌻/៸៸᳐_⩊_៸៸᳐ 66 微笑调查队🌻 - 1.66 微笑调查队🌻(Av278681824,P1).mp4",
    height=1024, width=1024)
input_video = [video[i] for i in range(40*60, 41*60)]

# Toon shading (20G VRAM)
torch.manual_seed(0)
output_video = pipe(
    prompt="best quality, perfect anime illustration, light, a girl is dancing, smile, solo",
    negative_prompt="verybadimagenegative_v1.3",
    cfg_scale=3, clip_skip=2,
    controlnet_frames=input_video, num_frames=len(input_video),
    num_inference_steps=10, height=1024, width=1024,
    animatediff_batch_size=32, animatediff_stride=16,
    vram_limit_level=0,
)
output_video = smoother(output_video)

# Save video
save_video(output_video, "output_video.mp4", fps=60)
