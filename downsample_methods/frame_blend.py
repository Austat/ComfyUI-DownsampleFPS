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
# Gamma‑correct (linear‑light) blending
# ============================================================
def blend_linear(frames, weights):
    """
    Blends frames in linear light (gamma‑correct).
    frames: list of uint8 BGR frames
    weights: list of float weights (same length)
    """
    # Convert to float32 linear light
    lin = [(f.astype(np.float32) / 255.0) ** 2.2 for f in frames]

    # Weighted sum
    acc = np.zeros_like(lin[0])
    for f, w in zip(lin, weights):
        acc += f * w

    # Convert back to sRGB
    out = np.clip(acc, 0, 1) ** (1.0 / 2.2)
    return (out * 255).astype(np.uint8)


# ============================================================
# Main advanced blending function
# ============================================================
def frame_blend(np_frames, indices, threads, radius=3):
    """
    Advanced frame blending with:
      - Gamma‑correct blending
      - Scene‑cut awareness
      - Blend radius (3–5 frames recommended)
      - Multithreading

    radius = number of frames on each side (e.g., 3 → 7‑frame window)
    """

    # Clamp radius
    radius = max(1, min(radius, 5))

    # Detect scene cuts
    cuts = detect_scene_cuts(np_frames, threshold=0.5)

    # Precompute scene boundaries
    def is_cut_between(a, b):
        """Returns True if any cut exists between frames a and b."""
        if a == b:
            return False
        lo, hi = sorted((a, b))
        return np.any(cuts[lo:hi])

    # Worker function
    def process(idx):
        center = idx
        n = len(np_frames)

        # Build blending window
        start = max(0, center - radius)
        end = min(n - 1, center + radius)

        frames_to_blend = []
        weights = []

        for i in range(start, end + 1):
            # Skip frames across scene cuts
            if is_cut_between(center, i):
                continue

            # Weight: Gaussian‑like falloff
            dist = abs(i - center)
            w = np.exp(-(dist ** 2) / (2 * (radius / 2) ** 2))
            frames_to_blend.append(np_frames[i])
            weights.append(w)

        # Normalize weights
        weights = np.array(weights, dtype=np.float32)
        weights /= weights.sum()

        # Gamma‑correct blending
        return blend_linear(frames_to_blend, weights)

    # Multithreaded execution
    with ThreadPoolExecutor(max_workers=threads) as ex:
        results = list(tqdm(
            ex.map(process, indices),
            total=len(indices),
            desc=f"Blending (radius={radius})",
            colour="blue",
            unit="frame"
        ))

    return results
