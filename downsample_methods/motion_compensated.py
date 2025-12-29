import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


# ---------------------------------------------------------
# Optical flow (Farnebäck)
# ---------------------------------------------------------
def _compute_flow(frame_a, frame_b):
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        gray_a, gray_b,
        None,
        pyr_scale=0.5,
        levels=5,
        winsize=21,
        iterations=7,
        poly_n=7,
        poly_sigma=1.5,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    )
    return flow


def _warp_frame(frame, flow, t):
    h, w = flow.shape[:2]
    flow_scaled = flow * t

    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + flow_scaled[..., 0]).astype(np.float32)
    map_y = (grid_y + flow_scaled[..., 1]).astype(np.float32)

    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)


def _interpolate_frame_cached(frame_a, frame_b, t, flow_fwd, flow_bwd):
    warp_a = _warp_frame(frame_a, flow_fwd, t)
    warp_b = _warp_frame(frame_b, flow_bwd, 1.0 - t)

    mag_fwd = np.linalg.norm(flow_fwd, axis=2)
    mag_bwd = np.linalg.norm(flow_bwd, axis=2)

    weight_a = np.exp(-mag_fwd)
    weight_b = np.exp(-mag_bwd)

    weights = (weight_a + weight_b + 1e-6)
    weight_a /= weights
    weight_b /= weights

    blended = warp_a * weight_a[..., None] + warp_b * weight_b[..., None]
    return blended.astype(np.uint8)


# ---------------------------------------------------------
# Scene‑cut detection
# ---------------------------------------------------------
def _detect_scene_cuts(frames, threshold=0.5):
    n = len(frames)
    if n < 2:
        return np.zeros(0, dtype=bool)

    cuts = np.zeros(n - 1, dtype=bool)

    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    prev_hist = cv2.calcHist([prev_gray], [0], None, [64], [0, 256])
    prev_hist = cv2.normalize(prev_hist, None).flatten()

    for i in range(1, n):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        hist = cv2.normalize(hist, None).flatten()

        score = cv2.compareHist(prev_hist.astype(np.float32),
                                hist.astype(np.float32),
                                cv2.HISTCMP_CORREL)

        cuts[i - 1] = (score < threshold)

        prev_gray = gray
        prev_hist = hist

    return cuts


# ---------------------------------------------------------
# Main API with MULTITHREADED flow‑cache precomputation
# ---------------------------------------------------------
def motion_compensated(np_frames, indices, threads=8):
    frames = list(np_frames)
    num_frames = len(frames)
    if num_frames < 2:
        return frames

    # Detect scene cuts
    cuts = _detect_scene_cuts(frames, threshold=0.5)

    # -----------------------------------------------------
    # MULTITHREADED FLOW‑CACHE PRECOMPUTATION
    # -----------------------------------------------------
    print(f"[Motion‑compensated] Precomputing optical flow cache using {threads} threads...")

    flow_fwd_cache = {}
    flow_bwd_cache = {}

    # Build list of flow tasks (skip scene cuts)
    flow_tasks = [
        i for i in range(num_frames - 1)
        if not cuts[i]
    ]

    def _compute_pair(i):
        frame_a = frames[i]
        frame_b = frames[i + 1]
        flow_fwd = _compute_flow(frame_a, frame_b)
        flow_bwd = _compute_flow(frame_b, frame_a)
        return i, flow_fwd, flow_bwd

    # Run flow computations in parallel
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for i, flow_fwd, flow_bwd in tqdm(
            pool.map(_compute_pair, flow_tasks),
            total=len(flow_tasks),
            desc="Flow cache",
            colour="yellow",
            unit="pair"
        ):
            flow_fwd_cache[i] = flow_fwd
            flow_bwd_cache[i] = flow_bwd

    # -----------------------------------------------------
    # Build interpolation tasks
    # -----------------------------------------------------
    tasks = []
    for idx in indices:
        if idx <= 0.0:
            tasks.append((0, 0.0, False))
            continue
        if idx >= num_frames - 1:
            tasks.append((num_frames - 2, 1.0, False))
            continue

        base = int(np.floor(idx))
        t = float(idx - base)
        is_cut = cuts[base] if 0 <= base < len(cuts) else False

        tasks.append((base, t, is_cut))

    # -----------------------------------------------------
    # Interpolation using cached flows (also threaded)
    # -----------------------------------------------------
    def _process(task):
        base, t, is_cut = task
        frame_a = frames[base]
        frame_b = frames[base + 1]

        if is_cut:
            return frame_a.copy() if t < 0.5 else frame_b.copy()

        if t <= 1e-6:
            return frame_a.copy()
        if t >= 1.0 - 1e-6:
            return frame_b.copy()

        flow_fwd = flow_fwd_cache.get(base)
        flow_bwd = flow_bwd_cache.get(base)

        if flow_fwd is None or flow_bwd is None:
            return frame_a.copy() if t < 0.5 else frame_b.copy()

        return _interpolate_frame_cached(frame_a, frame_b, t, flow_fwd, flow_bwd)

    selected = []
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for out in tqdm(pool.map(_process, tasks),
                        total=len(tasks),
                        desc="Motion‑compensated (CPU, cached)",
                        colour="blue",
                        unit="frame"):
            selected.append(out)

    return selected
