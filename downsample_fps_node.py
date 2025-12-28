import torch
import cv2
import numpy as np
import os
import warnings
from tqdm import tqdm

# Import downsampling method implementations
from .downsample_methods.frame_drop import frame_drop
from .downsample_methods.frame_blend import frame_blend
from .downsample_methods.optical_flow import optical_flow
from .downsample_methods.motion_compensated import motion_compensated

# Suppress noisy torchvision warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")


class DownsampleFPSNode:
    """
    A node that downsamples a sequence of video frames from a higher FPS
    to a lower FPS using several selectable CPU‑based methods.

    Supported methods:
        - Frame dropping
        - Frame blending
        - Optical flow interpolation
        - Motion‑compensated retiming (scene‑aware, time‑accurate)
    """

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the UI‑visible input fields for this node.
        Allows selecting FPS values, downsampling method, and CPU thread count.
        """
        max_cores = os.cpu_count()

        return {
            "required": {
                "frames": ("IMAGE",),  # Input tensor of frames (float32 0–1)
                "input_fps": ("INT", {"default": 48, "min": 1, "max": 240}),
                "target_fps": ("INT", {"default": 24, "min": 1, "max": 240}),
                "method": ([
                    "Frame dropping",
                    "Frame blending (CPU)",
                    "Optical flow (CPU)",
                    "Motion‑compensated (CPU)"
                ],),
                "cpu_threads": (
                    ["auto"] + [str(i) for i in range(1, max_cores + 1)],
                    {"default": "auto"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("downsampled_frames",)
    FUNCTION = "downsample"
    CATEGORY = "Video"

    def __init__(self):
        """
        Prints basic initialization info when the node is created.
        """
        print("\n=== DownsampleFPSNode STATUS ===")
        print("Using CPU-only methods (no OpenCV CUDA).")
        print(f"CPU cores detected: {os.cpu_count()}")
        print("Methods: Frame dropping, CPU blending, CPU optical flow, CPU motion compensation.\n")

    def downsample(self, frames, input_fps, target_fps, method, cpu_threads):
        """
        Main entry point for downsampling.

        Parameters:
            frames      - Input tensor of frames (float32, 0–1)
            input_fps   - Original frame rate
            target_fps  - Desired output frame rate
            method      - Downsampling method to use
            cpu_threads - Number of CPU threads ("auto" = all cores)

        Returns:
            A tensor of downsampled frames (float32, 0–1)
        """

        # If target FPS is not lower, skip processing
        if target_fps >= input_fps:
            print("Target FPS is equal or higher than input FPS. No downsampling applied.")
            return (frames,)

        # Convert tensor → uint8 numpy array for OpenCV processing
        np_frames = (frames.cpu().numpy() * 255).astype(np.uint8)

        # ------------------------------------------------------------
        # Integer‑based index generation (used by simple methods)
        # ------------------------------------------------------------
        ratio = input_fps / target_fps
        frame_count = int(len(np_frames) / ratio)
        indices_int = [int(i * ratio) for i in range(frame_count)]

        # ------------------------------------------------------------
        # Time‑accurate float index generation (used by motion‑compensated)
        # ------------------------------------------------------------
        def generate_time_indices(num_frames, src_fps, dst_fps):
            """
            Generates floating‑point indices representing exact time positions
            for true retiming (e.g., 60 → 24 fps).
            """
            duration = num_frames / float(src_fps)
            target_frame_count = int(np.round(duration * float(dst_fps)))
            times = np.linspace(0.0, duration, target_frame_count, endpoint=False)
            return (times * float(src_fps)).tolist()

        print(f"[DownsampleFPSNode] Method: {method}, "
              f"input_fps={input_fps}, target_fps={target_fps}, "
              f"in_frames={len(np_frames)}")

        # Resolve CPU thread count
        threads = os.cpu_count() if cpu_threads == "auto" else int(cpu_threads)

        # ------------------------------------------------------------
        # Method dispatch
        # ------------------------------------------------------------
        if method == "Frame dropping":
            indices = indices_int
            selected = frame_drop(np_frames, indices)

        elif method == "Frame blending (CPU)":
            indices = indices_int
            selected = frame_blend(np_frames, indices, threads)

        elif method == "Optical flow (CPU)":
            indices = indices_int
            selected = optical_flow(np_frames, indices, threads)

        elif method == "Motion‑compensated (CPU)":
            # Use time‑accurate float indices for high‑quality retiming
            indices = generate_time_indices(len(np_frames), input_fps, target_fps)
            print(f"[DownsampleFPSNode] Motion‑compensated: "
                  f"in_frames={len(np_frames)}, out_frames={len(indices)} (time‑accurate)")

            selected = motion_compensated(np_frames, indices, threads)

        # ------------------------------------------------------------
        # Convert back to float32 tensor (0–1 range)
        # ------------------------------------------------------------
        selected = np.stack(selected).astype(np.float32) / 255.0
        out_tensor = torch.from_numpy(selected)

        return (out_tensor,)
