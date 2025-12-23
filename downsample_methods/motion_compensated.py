import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def _motion_comp_pair(args):
    frame1, frame2 = args

    # Convert to grayscale for flow
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Optical flow
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2,
        None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )

    h, w = gray1.shape
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    # Warp both frames toward the midpoint
    half_flow = flow * 0.5

    map_x_f1 = (grid_x + half_flow[..., 0]).astype(np.float32)
    map_y_f1 = (grid_y + half_flow[..., 1]).astype(np.float32)

    map_x_f2 = (grid_x - half_flow[..., 0]).astype(np.float32)
    map_y_f2 = (grid_y - half_flow[..., 1]).astype(np.float32)

    warped1 = cv2.remap(frame1, map_x_f1, map_y_f1, cv2.INTER_LINEAR)
    warped2 = cv2.remap(frame2, map_x_f2, map_y_f2, cv2.INTER_LINEAR)

    # Blend the warped frames
    return cv2.addWeighted(warped1, 0.5, warped2, 0.5, 0)


def motion_compensated(np_frames, indices, threads):
    tasks = []
    for idx in indices:
        if idx + 1 < len(np_frames):
            tasks.append((np_frames[idx], np_frames[idx + 1]))
        else:
            tasks.append((np_frames[idx], np_frames[idx]))

    selected = []
    with ThreadPoolExecutor(max_workers=threads) as executor:
        for result in tqdm(executor.map(_motion_comp_pair, tasks),
                           total=len(tasks),
                           desc="Motion-Compensated",
                           colour="blue",
                           unit="frame"):
            selected.append(result)

    return selected