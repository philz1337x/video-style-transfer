import sys
sys.path.insert(0, 'DiffSynth-Studio')

import cv2

from diffsynth import ModelManager, SDVideoPipeline, ControlNetConfigUnit, VideoData, save_video
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

# Load video
def count_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

def get_framerate(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return fps

def load_video(video_file, input_framerate, start_frame=None, end_frame=None, target_fps=None):
    video = VideoData(
        video_file=video_file,
        height=1024, width=1024
    )
    
    # Calculate frame range
    start_frame = start_frame or 0
    end_frame = end_frame or len(video)
    target_fps = target_fps or input_framerate  # Use video's FPS if not specified

    frame_rate = get_framerate(video_file)
    # Select frames based on start_frame and end_frame
    selected_frames = []
    for i in range(start_frame, min(end_frame, len(video))):
        if i % (frame_rate // target_fps) == 0:
            selected_frames.append(video[i])
    
    if not selected_frames:
        raise ValueError("No frames selected. Check start_frame, end_frame, and target_fps settings.")
    
    return selected_frames

video_path = 'a.mp4'  # Hier den Pfad zu Ihrer MP4-Datei angeben
frame_count = count_frames(video_path)
input_framerate = get_framerate(video_path)
print(f"The video has {frame_count} frames.")

# Load video with optional frame range and target FPS
input_video = load_video(video_path, input_framerate, start_frame=1, end_frame=60)

if input_video is None or len(input_video) == 0:
    raise ValueError("Input video data is empty or not initialized.")

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
save_video(output_video, "output_video.mp4", fps=input_framerate) 
