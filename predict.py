import sys
import torch
import cv2
import time
import uuid

sys.path.insert(0, 'DiffSynth-Studio')
from diffsynth import ModelManager, SDVideoPipeline, ControlNetConfigUnit, VideoData, save_video
from diffsynth.extensions.RIFE import RIFESmoother
from moviepy.editor import VideoFileClip, AudioFileClip

from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        setup_time = time.time()

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

        self.pipe = SDVideoPipeline.from_model_manager(
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

        self.smoother = RIFESmoother.from_model_manager(model_manager)

        print(f"Setup took {round(time.time() - setup_time,2)} seconds")

    # Load video
    def count_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total_frames

    def get_framerate(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        return fps

    def load_video(self, video_file, input_framerate, input_width, input_height, start_frame=None, end_frame=None, target_fps=None):
        video = VideoData(
            video_file=video_file,
            height=input_height, width=input_width
        )
        
        # Calculate frame range
        start_frame = start_frame or 0
        end_frame = end_frame or len(video)
        target_fps = target_fps or input_framerate  # Use video's FPS if not specified

        frame_rate = self.get_framerate(video_file)
        # Select frames based on start_frame and end_frame
        selected_frames = []
        for i in range(start_frame, min(end_frame, len(video))):
            if i % (frame_rate // target_fps) == 0:
                selected_frames.append(video[i])
        
        if not selected_frames:
            raise ValueError("No frames selected. Check start_frame, end_frame, and target_fps settings.")
        
        return selected_frames

    def extract_audio(self, video_path: Path, audio_output_path: str):
        video_path = str(video_path)
        video = VideoFileClip(video_path)
        
        if video.audio is not None:
            audio = video.audio
            audio.write_audiofile(audio_output_path)
        else:
            print("The video does not contain an audio track. Continuing without audio extraction.")

    def combine_audio_video(self, video_path, audio_path, final_output_path):
        video_path = str(video_path)
        audio_path = str(audio_path)
        final_output_path = str(final_output_path)
        
        video = VideoFileClip(video_path)
        
        if Path(audio_path).is_file():
            audio = AudioFileClip(audio_path)
            final_video = video.set_audio(audio)
        else:
            print("No audio file found. Continuing with video only.")
            final_video = video
        
        final_video.write_videofile(final_output_path, codec='libx264', audio_codec='aac')
    
    def predict(
        self,
        video: Path = Input(description="input video"),
        prompt: str = Input(
            description="prompt",
            default="best quality, perfect anime illustration, light, ",
        ),
        negative_prompt: str = Input(
            description="negative prompt",
            default="verybadimagenegative_v1.3",
        ),
        end_frame: int = Input(
            description="frame where to end",
            default=0,
        ),
        target_fps: int = Input(
            description="target fps",
            default=0,
        ),
        input_width: int = Input(
            description="input width",
            default=1024,
        ),
        input_height: int = Input(
            description="input height",
            default=1024,
        ),
        output_width: int = Input(
            description="output width",
            default=1024,
        ),
        output_height: int = Input(
            description="output height",
            default=1024,
        ),
        cfg_scale: float = Input(
            description="cfg scale",
            default=3,
        ),
        clip_skip: int = Input(
            description="clip skip",
            default=2,
        ),
        num_inference_steps: int = Input(
            description="num inference steps",
            default=10,
        ),
        animatediff_batch_size: int = Input(
            description="animatediff batch size",
            default=24,
        ),
        animatediff_stride: int = Input(
            description="animatediff stride",
            default=16,
        ),

    ) -> list[Path]:
        """Run a single prediction on the model"""
        print("Running prediction")
        start_time = time.time()
        frame_count = self.count_frames(video)
        input_framerate = self.get_framerate(video)
        print(f"The video has {frame_count} frames.")

        audio_output_path = Path(f"extracted_audio_{uuid.uuid1()}.mp3")
        self.extract_audio(video, audio_output_path)
        print("Audio extracted")

        if end_frame > 0:
            actual_end_frame = min(end_frame, frame_count)
        else:
            actual_end_frame = frame_count

        # Load video with optional frame range and target FPS
        input_video = self.load_video(video, input_framerate, input_width, input_height, start_frame=1, end_frame=actual_end_frame, target_fps=target_fps)

        if input_video is None or len(input_video) == 0:
            raise ValueError("Input video data is empty or not initialized.")

        seed=0
        # Toon shading (20G VRAM)
        torch.manual_seed(seed)
        
        output_video = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            cfg_scale=cfg_scale, clip_skip=clip_skip,
            controlnet_frames=input_video, num_frames=len(input_video),
            num_inference_steps=num_inference_steps, height=output_height, width=output_width,
            animatediff_batch_size=animatediff_batch_size, animatediff_stride=animatediff_stride,
            vram_limit_level=0,
        )
        output_video = self.smoother(output_video)

        if target_fps:
            input_framerate = target_fps
            
        outputs = []   
        final_output_path = Path(f"{seed}-{uuid.uuid1()}-final.mp4")
        temp_path = f"{seed}-{uuid.uuid1()}-temp.mp4"
        save_video(output_video, temp_path, fps=input_framerate)
        self.combine_audio_video(temp_path, audio_output_path, final_output_path)
        outputs.append(final_output_path)

        print(f"Prediction took {round(time.time() - start_time,2)} seconds")
        return outputs