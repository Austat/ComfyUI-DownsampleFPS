# file: ComfyUI/custom_nodes/DownsampleFPSNode/downsample_fps_node.py

import torch
import cv2
import numpy as np
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Silencing warnings like a stern librarian shushing noisy visitors.
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")


class DownsampleFPSNode:
    @classmethod
    def INPUT_TYPES(cls):
        # Detecting the number of CPU cores available — the digital equivalent
        # of counting how many horses you have before deciding how big a wagon to pull.
        max_cores = os.cpu_count()

        return {
            "required": {
                # The raw frames — the precious lifeblood of any video-processing ritual.
                "frames": ("IMAGE",),

                # The original FPS, the frantic heartbeat of the incoming footage.
                "input_fps": ("INT", {"default": 48, "min": 1, "max": 240}),

                # The target FPS, a calmer, more composed tempo we aspire to achieve.
                "target_fps": ("INT", {"default": 24, "min": 1, "max": 240}),

                # The sacred method of temporal reduction — each with its own philosophy.
                "method": ([
                    "Frame dropping",        # Brutal efficiency: simply discard the excess.
                    "Frame blending (CPU)",  # A gentle merging of moments.
                    "Optical flow (CPU)",    # A sophisticated dance of motion estimation.
                ],),

                # CPU thread selection — choose your army size.
                "cpu_threads": (
                    ["auto"] + [str(i) for i in range(1, max_cores + 1)],
                    {"default": "auto"},
                ),
            }
        }

    # The node promises to return a single artifact: the transformed frames.
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("downsampled_frames",)
    FUNCTION = "downsample"
    CATEGORY = "Video"

    def __init__(self):
        # A dramatic proclamation of the node’s existence and capabilities.
        print("\n=== DownsampleFPSNode STATUS ===")
        print("Using CPU-only methods (no OpenCV CUDA).")
        print(f"CPU cores detected: {os.cpu_count()}")
        print("Methods: Frame dropping, CPU blending, CPU optical flow.\n")

    # ---------------------------------------------------------
    # CPU OPTICAL FLOW (Farneback)
    # ---------------------------------------------------------
    @staticmethod
    def _cpu_optical_flow_pair(args):
        # A pair of frames enters — a single interpolated frame leaves.
        frame1, frame2 = args

        # Convert frames to grayscale, stripping them of their colorful identities
        # to reveal their raw structural essence.
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Summoning the Farneback algorithm — a venerable wizard of motion estimation.
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2,
            None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        # Extracting the dimensions of the grayscale frame — the canvas of motion.
        h, w = gray1.shape

        # Halving the flow, as if gently easing the motion into existence.
        flow_half = flow * 0.5

        # Constructing a grid of pixel coordinates — the map of our tiny universe.
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

        # Warping the first frame toward the second using the flow field.
        map_x = (grid_x + flow_half[..., 0]).astype(np.float32)
        map_y = (grid_y + flow_half[..., 1]).astype(np.float32)

        # Remapping the frame — bending space and time to our will.
        return cv2.remap(frame1, map_x, map_y, cv2.INTER_LINEAR)

    # ---------------------------------------------------------
    # CPU FRAME BLENDING (MULTITHREADED)
    # ---------------------------------------------------------
    @staticmethod
    def _cpu_blend_pair(args):
        # Two frames enter — one harmonious fusion emerges.
        frame1, frame2 = args
        return cv2.addWeighted(frame1, 0.5, frame2, 0.5, 0)

    # ---------------------------------------------------------
    # MAIN DOWNSAMPLE FUNCTION
    # ---------------------------------------------------------
    def downsample(self, frames, input_fps, target_fps, method, cpu_threads):
        # If the target FPS is not lower, we gracefully bow out.
        if target_fps >= input_fps:
            print("Target FPS is equal or higher than input FPS. No downsampling applied.")
            return (frames,)

        # Convert torch tensor frames into NumPy arrays — a metamorphosis
        # from PyTorch’s structured world into NumPy’s freeform wilderness.
        np_frames = (frames.cpu().numpy() * 255).astype(np.uint8)

        # Calculating the sacred ratio — the mathematical essence of downsampling.
        ratio = input_fps / target_fps

        # Determining how many frames shall survive the ritual.
        frame_count = int(len(np_frames) / ratio)

        # Selecting indices with the precision of a cosmic metronome.
        indices = [int(i * ratio) for i in range(frame_count)]

        print(f"[DownsampleFPSNode] Method: {method}, "
              f"input_fps={input_fps}, target_fps={target_fps}, "
              f"in_frames={len(np_frames)}, out_frames={len(indices)}")

        # Choosing the number of CPU threads — assembling our computational battalion.
        if cpu_threads == "auto":
            threads = os.cpu_count()
        else:
            threads = int(cpu_threads)

        # --------------------------------
        # METHOD: Frame Dropping (CPU)
        # --------------------------------
        if method == "Frame dropping":
            print("[Frame Dropping] Processing…")
            selected = []
            # Marching through the chosen indices with the solemnity of a funeral procession.
            for i in tqdm(indices, desc="Dropping", unit="frame", colour="blue"):
                if i < len(np_frames):
                    selected.append(np_frames[i])

        # --------------------------------
        # METHOD: Frame Blending (CPU, MULTITHREAD)
        # --------------------------------
        elif method == "Frame blending (CPU)":
            print(f"[Frame Blending CPU] Using {threads} threads")

            # Preparing pairs of frames to be fused into temporal smoothies.
            tasks = []
            for idx in indices:
                if idx + 1 < len(np_frames):
                    tasks.append((np_frames[idx], np_frames[idx + 1]))
                else:
                    # If we reach the end, we blend the frame with itself —
                    # a lonely but necessary act.
                    tasks.append((np_frames[idx], np_frames[idx]))

            selected = []
            # Unleashing a multithreaded storm of blending operations.
            with ThreadPoolExecutor(max_workers=threads) as executor:
                for result in tqdm(executor.map(self._cpu_blend_pair, tasks),
                                   total=len(tasks),
                                   desc="Blending",
                                   colour="blue",
                                   unit="frame"):
                    selected.append(result)

        # --------------------------------
        # METHOD: Optical Flow (CPU, MULTITHREAD)
        # --------------------------------
        elif method == "Optical flow (CPU)":
            print(f"[Optical Flow CPU] Using {threads} threads")

            # Preparing frame pairs for motion sorcery.
            tasks = []
            for idx in indices:
                if idx + 1 < len(np_frames):
                    tasks.append((np_frames[idx], np_frames[idx + 1]))
                else:
                    # Again, the final frame must face its destiny alone.
                    tasks.append((np_frames[idx], np_frames[idx]))

            selected = []
            # Summoning multiple threads to compute optical flow in parallel —
            # a computational ballet of vectors and warping.
            with ThreadPoolExecutor(max_workers=threads) as executor:
                for result in tqdm(executor.map(self._cpu_optical_flow_pair, tasks),
                                   total=len(tasks),
                                   desc="Optical Flow",
                                   colour="blue",
                                   unit="frame"):
                    selected.append(result)

        # Converting the selected frames back into a torch tensor —
        # restoring them to their original homeland.
        selected = np.stack(selected).astype(np.float32) / 255.0
        out_tensor = torch.from_numpy(selected)

        # Returning the final artifact — the reborn sequence of frames.
        return (out_tensor,)
