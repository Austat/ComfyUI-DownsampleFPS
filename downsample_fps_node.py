# file: ComfyUI/custom_nodes/DownsampleFPSNode/downsample_fps_node.py

import torch
import cv2
import numpy as np
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")


def create_blue_progress(task_name, total):
    return Progress(
        TextColumn(f"[bold cyan]{task_name}[/bold cyan]"),
        BarColumn(bar_width=None, style="bright_blue", complete_style="cyan"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=True,
    ), total


class DownsampleFPSNode:
    @classmethod
    def INPUT_TYPES(cls):
        max_cores = os.cpu_count()

        return {
            "required": {
                "frames": ("IMAGE",),
                "input_fps": ("INT", {"default": 48, "min": 1, "max": 240}),
                "target_fps": ("INT", {"default": 24, "min": 1, "max": 240}),
                "method": ([
                    "Frame dropping",
                    "Frame blending (CPU)",
                    "Optical flow (CPU)",
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
        print("\n=== DownsampleFPSNode STATUS ===")
        print("Using CPU-only methods (no OpenCV CUDA).")
        print(f"CPU cores detected: {os.cpu_count()}")
        print("Methods: Frame dropping, CPU blending, CPU optical flow.\n")

    # ---------------------------------------------------------
    # CPU OPTICAL FLOW (Farneback)
    # ---------------------------------------------------------
    @staticmethod
    def _cpu_optical_flow_pair(args):
        frame1, frame2 = args

        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2,
            None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        h, w = gray1.shape
        flow_half = flow * 0.5
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow_half[..., 0]).astype(np.float32)
        map_y = (grid_y + flow_half[..., 1]).astype(np.float32)

        return cv2.remap(frame1, map_x, map_y, cv2.INTER_LINEAR)

    # ---------------------------------------------------------
    # CPU FRAME BLENDING (MULTITHREADED)
    # ---------------------------------------------------------
    @staticmethod
    def _cpu_blend_pair(args):
        frame1, frame2 = args
        return cv2.addWeighted(frame1, 0.5, frame2, 0.5, 0)

    # ---------------------------------------------------------
    # MAIN DOWNSAMPLE FUNCTION
    # ---------------------------------------------------------
    def downsample(self, frames, input_fps, target_fps, method, cpu_threads):
        if target_fps >= input_fps:
            print("Target FPS is equal or higher than input FPS. No downsampling applied.")
            return (frames,)

        np_frames = (frames.cpu().numpy() * 255).astype(np.uint8)

        ratio = input_fps / target_fps
        frame_count = int(len(np_frames) / ratio)
        indices = [int(i * ratio) for i in range(frame_count)]

        print(f"[DownsampleFPSNode] Method: {method}, "
              f"input_fps={input_fps}, target_fps={target_fps}, "
              f"in_frames={len(np_frames)}, out_frames={len(indices)}")

        # Determine thread count
        if cpu_threads == "auto":
            threads = os.cpu_count()
        else:
            threads = int(cpu_threads)

        # --------------------------------
        # METHOD: Frame Dropping (CPU)
        # --------------------------------
        if method == "Frame dropping":
            progress, total = create_blue_progress("Dropping", len(indices))
            selected = []

            with progress:
                task = progress.add_task("drop", total=total)
                for i in indices:
                    if i < len(np_frames):
                        selected.append(np_frames[i])
                    progress.update(task, advance=1)

        # --------------------------------
        # METHOD: Frame Blending (CPU, MULTITHREAD)
        # --------------------------------
        elif method == "Frame blending (CPU)":
            print(f"[Frame Blending CPU] Using {threads} threads")

            tasks = []
            for idx in indices:
                if idx + 1 < len(np_frames):
                    tasks.append((np_frames[idx], np_frames[idx + 1]))
                else:
                    tasks.append((np_frames[idx], np_frames[idx]))

            progress, total = create_blue_progress("Blending", len(tasks))
            selected = []

            with progress:
                task = progress.add_task("blend", total=total)
                with ThreadPoolExecutor(max_workers=threads) as executor:
                    for result in executor.map(self._cpu_blend_pair, tasks):
                        selected.append(result)
                        progress.update(task, advance=1)

        # --------------------------------
        # METHOD: Optical Flow (CPU, MULTITHREAD)
        # --------------------------------
        elif method == "Optical flow (CPU)":
            print(f"[Optical Flow CPU] Using {threads} threads")

            tasks = []
            for idx in indices:
                if idx + 1 < len(np_frames):
                    tasks.append((np_frames[idx], np_frames[idx + 1]))
                else:
                    tasks.append((np_frames[idx], np_frames[idx]))

            progress, total = create_blue_progress("Optical Flow", len(tasks))
            selected = []

            with progress:
                task = progress.add_task("flow", total=total)
                with ThreadPoolExecutor(max_workers=threads) as executor:
                    for result in executor.map(self._cpu_optical_flow_pair, tasks):
                        selected.append(result)
                        progress.update(task, advance=1)

        # Convert back to torch tensor
        selected = np.stack(selected).astype(np.float32) / 255.0
        out_tensor = torch.from_numpy(selected)

        return (out_tensor,)
