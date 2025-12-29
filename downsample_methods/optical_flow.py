import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


# ============================================================
# Scene‑cut detection (histogram correlation)
# ============================================================
def detect_scene_cuts(frames, threshold=0.5):
    """
    Detects hard scene cuts using grayscale histogram correlation.
    Returns a boolean array where True means a cut between frame i and i+1.
    """
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

# ============================================================
# RLOF Optical Flow (requires opencv-contrib-python)
# ============================================================
def compute_rlof(gray1, gray2):
    """
    Computes RLOF optical flow between two grayscale frames.
    """
    rlof = cv2.optflow.createOptFlow_DenseRLOF()
    flow = rlof.calc(gray1, gray2, None)
    return flow

# ============================================================
# Warp frame using flow * 0.5 (mid‑frame interpolation)
# ============================================================
def warp_halfway(frame, flow, grid_x, grid_y):
    flow_half = flow * 0.5
    map_x = (grid_x + flow_half[..., 0]).astype(np.float32)
    map_y = (grid_y + flow_half[..., 1]).astype(np.float32)
    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)

# ============================================================
# Main optical flow downsampler (RLOF + scene‑cut + cache)
# ============================================================
def optical_flow(np_frames, indices, threads):
    """
    High‑quality optical‑flow downsampling using:
      - RLOF optical flow
      - Scene‑cut detection
      - Flow‑cache (each pair computed once)
      - Multithreaded processing
    """
    h, w = np_frames[0].shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    # Detect scene cuts
    cuts = detect_scene_cuts(np_frames, threshold=0.5)

    # Flow cache: store flow for each frame pair
    flow_cache = {}

    def compute_flow_for_pair(i):
        """Compute RLOF flow for frame i → i+1 (unless cut)."""
        if cuts[i]:
            return i, None  # no flow across scene cuts

        gray1 = cv2.cvtColor(np_frames[i], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(np_frames[i + 1], cv2.COLOR_BGR2GRAY)
        flow = compute_rlof(gray1, gray2)
        return i, flow

    # Precompute flows in parallel
    flow_pairs = list(range(len(np_frames) - 1))

    with ThreadPoolExecutor(max_workers=threads) as ex:
        for i, flow in tqdm(
            ex.map(compute_flow_for_pair, flow_pairs),
            total=len(flow_pairs),
            desc="RLOF Flow Cache",
            colour="yellow",
            unit="pair"
        ):
            flow_cache[i] = flow

    # Interpolation function
    def process(idx):
        i1 = idx
        i2 = min(idx + 1, len(np_frames) - 1)

        # Scene cut → no interpolation
        if cuts[i1]:
            return np_frames[i1]

        flow = flow_cache.get(i1)
        if flow is None:
            return np_frames[i1]

        return warp_halfway(np_frames[i1], flow, grid_x, grid_y)

    # Run interpolation in parallel
    with ThreadPoolExecutor(max_workers=threads) as ex:
        results = list(tqdm(
            ex.map(process, indices),
            total=len(indices),
            desc="Optical Flow (RLOF)",
            colour="blue",
            unit="frame"
        ))

    return results
