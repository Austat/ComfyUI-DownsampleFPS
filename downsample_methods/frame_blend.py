import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


# ============================================================
# Scene‑cut detection (histogram correlation)
# ============================================================
def detect_scene_cuts(frames, threshold=0.5):
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
# Motion magnitude estimation (simple but effective)
# ============================================================
def estimate_motion(frame_a, frame_b):
    """
    Computes a simple motion magnitude estimate using absolute difference.
    Returns a scalar motion value in [0, 1].
    """
    diff = cv2.absdiff(frame_a, frame_b)
    motion = diff.mean() / 255.0
    return motion


# ============================================================
# Gamma‑correct, per‑channel weighted blending
# ============================================================
def blend_linear_per_channel(frames, weights_bgr):
    """
    frames: list of uint8 BGR frames
    weights_bgr: list of (wB, wG, wR) tuples, one per frame
    """
    # Convert to linear light
    lin = [(f.astype(np.float32) / 255.0) ** 2.2 for f in frames]

    # Accumulate per channel
    acc = np.zeros_like(lin[0])
    for f, (wb, wg, wr) in zip(lin, weights_bgr):
        acc[..., 0] += f[..., 0] * wb
        acc[..., 1] += f[..., 1] * wg
        acc[..., 2] += f[..., 2] * wr

    # Convert back to sRGB
    out = np.clip(acc, 0, 1) ** (1.0 / 2.2)
    return (out * 255).astype(np.uint8)


# ============================================================
# Main advanced blending function
# ============================================================
def frame_blend(np_frames, indices, threads, base_radius=3):
    """
    Advanced frame blending with:
      - Gamma‑correct blending
      - Scene‑cut awareness
      - Adaptive radius based on motion magnitude
      - Per‑channel weighting
      - Multithreading

    base_radius = maximum radius (e.g., 3 → up to 7‑frame window)
    """

    cuts = detect_scene_cuts(np_frames, threshold=0.5)

    def is_cut_between(a, b):
        if a == b:
            return False
        lo, hi = sorted((a, b))
        return np.any(cuts[lo:hi])

    # ------------------------------------------------------------
    # Worker function
    # ------------------------------------------------------------
    def process(idx):
        center = idx
        n = len(np_frames)

        # --------------------------------------------------------
        # Adaptive radius based on motion magnitude
        # --------------------------------------------------------
        if center < n - 1:
            motion = estimate_motion(np_frames[center], np_frames[center + 1])
        else:
            motion = 0.0

        # Low motion → larger radius, High motion → smaller radius
        # motion in [0,1]
        adaptive_radius = int(base_radius * (1.0 - motion))
        adaptive_radius = max(1, min(adaptive_radius, base_radius))

        # --------------------------------------------------------
        # Build blending window
        # --------------------------------------------------------
        start = max(0, center - adaptive_radius)
        end = min(n - 1, center + adaptive_radius)

        frames_to_blend = []
        weights_bgr = []

        for i in range(start, end + 1):
            if is_cut_between(center, i):
                continue

            dist = abs(i - center)

            # Gaussian falloff
            w = np.exp(-(dist ** 2) / (2 * (adaptive_radius / 2) ** 2))

            # Per‑channel weighting:
            # - Give more weight to green (luminance)
            # - Slightly less to blue (noise‑prone)
            wb = w * 0.9
            wg = w * 1.0
            wr = w * 0.95

            frames_to_blend.append(np_frames[i])
            weights_bgr.append((wb, wg, wr))

        # Normalize per channel
        sum_b = sum(w[0] for w in weights_bgr)
        sum_g = sum(w[1] for w in weights_bgr)
        sum_r = sum(w[2] for w in weights_bgr)

        weights_bgr = [
            (w[0] / sum_b, w[1] / sum_g, w[2] / sum_r)
            for w in weights_bgr
        ]

        # --------------------------------------------------------
        # Gamma‑correct, per‑channel blending
        # --------------------------------------------------------
        return blend_linear_per_channel(frames_to_blend, weights_bgr)

    # ------------------------------------------------------------
    # Multithreaded execution
    # ------------------------------------------------------------
    with ThreadPoolExecutor(max_workers=threads) as ex:
        results = list(tqdm(
            ex.map(process, indices),
            total=len(indices),
            desc=f"Blending (adaptive radius={base_radius})",
            colour="blue",
            unit="frame"
        ))

    return results
