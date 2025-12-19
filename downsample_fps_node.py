import torch

class DownsampleFPSNode:
    @classmethod
    def INPUT_TYPES(cls):
        # INPUT_TYPES defines what inputs this node expects.
        # "frames" = images/frames (type IMAGE)
        # "input_fps" = original video frame rate (frames per second)
        # "target_fps" = desired frame rate after downsampling
        return {
            "required": {
                "frames": ("IMAGE",),
                "input_fps": ("INT", {"default": 32, "min": 1, "max": 240}),
                "target_fps": ("INT", {"default": 24, "min": 1, "max": 240}),
            }
        }

    # RETURN_TYPES specifies what this node outputs (here: IMAGE).
    RETURN_TYPES = ("IMAGE",)
    # RETURN_NAMES gives a name to the returned data (here: "downsampled_frames").
    RETURN_NAMES = ("downsampled_frames",)
    # FUNCTION tells which method will be executed when the node runs.
    FUNCTION = "downsample"
    # CATEGORY defines where this node appears in the UI.
    CATEGORY = "Video"

    def downsample(self, frames, input_fps, target_fps):
        # If the target FPS is greater than or equal to the input FPS,
        # no downsampling is needed. Return the original frames.
        if target_fps >= input_fps:
            print("Target FPS is equal or higher than input FPS. No downsampling applied.")
            return (frames,)

        # Calculate the ratio: how many original frames correspond to one target frame.
        ratio = input_fps / target_fps

        # Determine how many frames will be selected for the downsampled video.
        frame_count = int(len(frames) / ratio)

        # Create a list of indices for the frames to be selected.
        # For example, if ratio = 1.33, pick every 1.33rd frame.
        indices = [int(i * ratio) for i in range(frame_count)]

        # Select frames at the calculated indices, making sure not to go out of bounds.
        selected_frames = [frames[i] for i in indices if i < len(frames)]

        # Convert the selected frames back into a batch tensor:
        # - If only one frame was selected, use torch.cat and add an extra dimension.
        # - If multiple frames were selected, use torch.stack to combine them.
        downsampled = (
            torch.cat(selected_frames, dim=0).unsqueeze(0)
            if len(selected_frames) == 1
            else torch.stack(selected_frames)
        )

        # Return the downsampled frames as a tuple, since RETURN_TYPES = ("IMAGE",).
        return (downsampled,)
