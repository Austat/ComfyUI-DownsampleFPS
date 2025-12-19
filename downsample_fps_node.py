import torch

class DownsampleFPSNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "input_fps": ("INT", {"default": 32, "min": 1, "max": 240}),
                "target_fps": ("INT", {"default": 24, "min": 1, "max": 240}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("downsampled_frames",)
    FUNCTION = "downsample"
    CATEGORY = "Video"

    def downsample(self, frames, input_fps, target_fps):
        if target_fps >= input_fps:
            print("Target FPS is equal or higher than input FPS. No downsampling applied.")
            return (frames,)

        ratio = input_fps / target_fps
        frame_count = int(len(frames) / ratio)
        indices = [int(i * ratio) for i in range(frame_count)]
        selected_frames = [frames[i] for i in indices if i < len(frames)]

        # Muunna takaisin batchiksi
        downsampled = torch.cat(selected_frames, dim=0).unsqueeze(0) if len(selected_frames) == 1 else torch.stack(selected_frames)
        return (downsampled,)