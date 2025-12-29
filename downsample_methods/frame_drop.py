import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def frame_drop(np_frames, indices, threads=0):
    """
    Unified frame dropping function.

    Modes:
        threads = 0  → Fastest vectorized NumPy mode
        threads > 0 → Threaded mode (consistent with other methods)

    Parameters:
        np_frames : numpy array of frames (N, H, W, 3)
        indices   : list of frame indices to select
        threads   : 0 = vectorized mode, >0 = threaded mode

    Returns:
        list of selected frames
    """

    # ------------------------------------------------------------
    # MODE 1: FASTEST (VECTORIZED)
    # ------------------------------------------------------------
    if threads == 0:
        idx = np.asarray(indices, dtype=np.int32)
        idx = np.clip(idx, 0, len(np_frames) - 1)

        selected = np_frames[idx]

        # Cosmetic progress bar only
        for _ in tqdm(range(len(idx)), desc="Dropping (fast)", unit="frame", colour="blue"):
            pass

        return list(selected)

    # ------------------------------------------------------------
    # MODE 2: THREADED (CONSISTENT WITH OTHER METHODS)
    # ------------------------------------------------------------
    def pick(i):
        if i < len(np_frames):
            return np_frames[i]
        return np_frames[-1]

    with ThreadPoolExecutor(max_workers=threads) as ex:
        results = list(tqdm(
            ex.map(pick, indices),
            total=len(indices),
            desc=f"Dropping (threaded x{threads})",
            colour="blue",
            unit="frame"
        ))

    return results
