import os
import cv2
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


# ============================================================
# Validation and preprocessing
# ============================================================
def validate_frames(frames):
    """
    Validates that all frames are uint8 BGR images of equal size.
    """
    if frames is None or len(frames) == 0:
        raise ValueError("np_frames must contain at least one frame.")

    first_shape = frames[0].shape

    if len(first_shape) != 3 or first_shape[2] != 3:
        raise ValueError(
            "Frames must be BGR images with shape (height, width, 3)."
        )

    for i, frame in enumerate(frames):
        if frame is None:
            raise ValueError(f"Frame {i} is None.")

        if frame.shape != first_shape:
            raise ValueError(
                f"Frame {i} has shape {frame.shape}, "
                f"but expected {first_shape}."
            )

        if frame.dtype != np.uint8:
            raise ValueError(
                f"Frame {i} has dtype {frame.dtype}, "
                "but uint8 is required."
            )


def resize_for_analysis(image, max_width=480):
    """
    Downscales an image for scene-cut and motion analysis.

    The original image is returned if it is already small enough.
    """
    height, width = image.shape[:2]

    if width <= max_width:
        return image

    scale = max_width / width
    new_height = max(1, int(round(height * scale)))

    return cv2.resize(
        image,
        (max_width, new_height),
        interpolation=cv2.INTER_AREA
    )


def preprocess_grayscale(frames, analysis_width=480):
    """
    Converts frames to grayscale and optionally downsizes them for
    scene-cut and optical-flow analysis.
    """
    grayscale_frames = []

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = resize_for_analysis(gray, max_width=analysis_width)
        grayscale_frames.append(gray)

    return grayscale_frames


# ============================================================
# Exact sRGB transfer functions
# ============================================================
def srgb_to_linear(frame):
    """
    Converts a uint8 sRGB BGR frame to linear-light float32 BGR.

    Output range is [0, 1].
    """
    srgb = frame.astype(np.float32) / 255.0

    return np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4
    ).astype(np.float32)


def linear_to_srgb(frame):
    """
    Converts a linear-light float BGR frame to uint8 sRGB BGR.
    """
    linear = np.clip(frame, 0.0, 1.0)

    srgb = np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055
    )

    return np.clip(
        np.rint(srgb * 255.0),
        0,
        255
    ).astype(np.uint8)


# ============================================================
# Structural similarity estimation
# ============================================================
def structural_similarity(gray_a, gray_b):
    """
    Computes a lightweight SSIM-style structural similarity score.

    Returns a value that is normally close to:
      1.0 = highly similar
      0.0 = highly dissimilar

    This implementation only depends on NumPy and OpenCV.
    """
    if gray_a.shape != gray_b.shape:
        raise ValueError("SSIM input images must have equal dimensions.")

    image_a = gray_a.astype(np.float32)
    image_b = gray_b.astype(np.float32)

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    mu_a = cv2.GaussianBlur(image_a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(image_b, (11, 11), 1.5)

    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a_sq = (
        cv2.GaussianBlur(image_a * image_a, (11, 11), 1.5)
        - mu_a_sq
    )
    sigma_b_sq = (
        cv2.GaussianBlur(image_b * image_b, (11, 11), 1.5)
        - mu_b_sq
    )
    sigma_ab = (
        cv2.GaussianBlur(image_a * image_b, (11, 11), 1.5)
        - mu_ab
    )

    numerator = (
        (2.0 * mu_ab + c1)
        * (2.0 * sigma_ab + c2)
    )

    denominator = (
        (mu_a_sq + mu_b_sq + c1)
        * (sigma_a_sq + sigma_b_sq + c2)
    )

    score_map = numerator / np.maximum(denominator, 1e-12)

    return float(np.clip(score_map.mean(), -1.0, 1.0))


# ============================================================
# Scene-cut detection
# ============================================================
def calculate_histogram(gray):
    """
    Calculates a normalized grayscale histogram.
    """
    histogram = cv2.calcHist(
        [gray],
        [0],
        None,
        [64],
        [0, 256]
    )

    histogram = cv2.normalize(
        histogram,
        None,
        alpha=1.0,
        norm_type=cv2.NORM_L1
    )

    return histogram.astype(np.float32).flatten()


def detect_scene_cuts(
    gray_frames,
    hard_hist_threshold=0.35,
    soft_hist_threshold=0.70,
    ssim_threshold=0.45
):
    """
    Detects scene cuts using histogram correlation and structural similarity.

    A cut is detected when either:

      1. Histogram correlation is extremely low, or
      2. Histogram correlation is moderately low and structural
         similarity is also low.

    Returns:
        Boolean array of length len(gray_frames) - 1.

        cuts[i] is True when there is a scene cut between
        frames i and i + 1.
    """
    frame_count = len(gray_frames)

    if frame_count < 2:
        return np.zeros(0, dtype=bool)

    histograms = [
        calculate_histogram(gray)
        for gray in gray_frames
    ]

    cuts = np.zeros(frame_count - 1, dtype=bool)

    for i in range(1, frame_count):
        histogram_score = cv2.compareHist(
            histograms[i - 1],
            histograms[i],
            cv2.HISTCMP_CORREL
        )

        ssim_score = structural_similarity(
            gray_frames[i - 1],
            gray_frames[i]
        )

        hard_histogram_cut = (
            histogram_score < hard_hist_threshold
        )

        combined_cut = (
            histogram_score < soft_hist_threshold
            and ssim_score < ssim_threshold
        )

        cuts[i - 1] = hard_histogram_cut or combined_cut

    return cuts


# ============================================================
# Motion estimation using optical flow
# ============================================================
def estimate_motion_optical_flow(
    gray_a,
    gray_b,
    motion_reference=8.0
):
    """
    Estimates actual image motion using Farneback optical flow.

    motion_reference controls how many analysis-image pixels of motion
    correspond approximately to normalized motion value 1.0.

    Returns:
        Normalized motion value in [0, 1].
    """
    flow = cv2.calcOpticalFlowFarneback(
        gray_a,
        gray_b,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    magnitude = cv2.magnitude(
        flow[..., 0],
        flow[..., 1]
    )

    finite_magnitude = magnitude[np.isfinite(magnitude)]

    if finite_magnitude.size == 0:
        return 0.0

    # Median is robust against isolated optical-flow errors.
    median_motion = float(np.median(finite_magnitude))

    normalized_motion = median_motion / max(
        float(motion_reference),
        1e-6
    )

    return float(np.clip(normalized_motion, 0.0, 1.0))


def precompute_motion(
    gray_frames,
    cuts,
    motion_reference=8.0,
    show_progress=True
):
    """
    Precomputes motion values between consecutive frames.

    Motion is set to 1.0 across scene cuts so that temporal blending
    collapses to the center frame near the cut.
    """
    frame_count = len(gray_frames)

    if frame_count < 2:
        return np.zeros(0, dtype=np.float32)

    motion_values = np.zeros(
        frame_count - 1,
        dtype=np.float32
    )

    iterator = range(frame_count - 1)

    if show_progress:
        iterator = tqdm(
            iterator,
            total=frame_count - 1,
            desc="Motion analysis",
            colour="green",
            unit="pair"
        )

    for i in iterator:
        if cuts[i]:
            motion_values[i] = 1.0
            continue
        motion_values[i] = estimate_motion_optical_flow(
            gray_frames[i],
            gray_frames[i + 1],
            motion_reference=motion_reference
        )

    return motion_values


# ============================================================
# Gamma-correct weighted blending
# ============================================================
def blend_linear_frames(linear_frames, weights):
    """
    Blends precomputed linear-light frames using scalar weights.

    Args:
        linear_frames:
            Sequence of float32 linear-light BGR frames.

        weights:
            One scalar temporal weight per frame.
    """
    if len(linear_frames) == 0:
        raise ValueError("No frames were supplied for blending.")

    weights = np.asarray(weights, dtype=np.float32)

    if len(weights) != len(linear_frames):
        raise ValueError(
            "The number of weights must match the number of frames."
        )

    weight_sum = float(weights.sum())

    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        raise ValueError("Invalid blending weights.")

    weights = weights / weight_sum

    accumulator = np.zeros_like(
        linear_frames[0],
        dtype=np.float32
    )

    for frame, weight in zip(linear_frames, weights):
        accumulator += frame * weight

    return linear_to_srgb(accumulator)


# ============================================================
# Main advanced frame-blending function
# ============================================================
def frame_blend(
    np_frames,
    indices,
    threads=None,
    base_radius=3,
    min_radius=0,
    analysis_width=480,
    motion_reference=8.0,
    hard_hist_threshold=0.35,
    soft_hist_threshold=0.70,
    ssim_threshold=0.45,
    show_progress=True
):
    """
    Advanced temporal frame blending with:

      - Exact sRGB gamma-correct blending
      - Combined histogram and SSIM scene-cut detection
      - Farneback optical-flow motion estimation
      - Motion-adaptive blending radius
      - Stable Gaussian temporal weighting
      - Precomputed linear-light frames
      - Precomputed motion values
      - Multithreaded output-frame processing

    Args:
        np_frames:
            List or sequence of uint8 BGR frames.

        indices:
            Indices of output frames to process. Results are returned
            in the same order.

        threads:
            Number of worker threads. None selects a conservative
            automatic value.

        base_radius:
            Maximum number of neighboring frames on each side.

            base_radius=3 means a maximum seven-frame window:
            center - 3 ... center + 3.

        min_radius:
            Minimum adaptive radius.

            Use 0 to disable neighboring-frame blending during
            sufficiently high motion.

            Use 1 if at least one neighboring frame must always
            be considered.

        analysis_width:
            Width used for scene-cut and optical-flow analysis.
            Smaller values increase speed.

        motion_reference:
            Approximate optical-flow magnitude that maps to normalized
            motion value 1.0.

            Lower value:
                More conservative blending.

            Higher value:
                More temporal smoothing.

        hard_hist_threshold:
            Histogram score below which a transition is treated as
            a scene cut regardless of SSIM.

        soft_hist_threshold:
            Histogram threshold used together with SSIM.

        ssim_threshold:
            SSIM threshold used together with soft_hist_threshold.

        show_progress:
            Enables tqdm progress bars.

    Returns:
        List of blended uint8 BGR frames.
    """
    validate_frames(np_frames)

    frame_count = len(np_frames)
    indices = list(indices)

    if base_radius < 0:
        raise ValueError("base_radius must be at least 0.")

    if min_radius < 0:
        raise ValueError("min_radius must be at least 0.")

    if min_radius > base_radius:
        raise ValueError(
            "min_radius cannot be larger than base_radius."
        )

    if analysis_width < 16:
        raise ValueError("analysis_width must be at least 16.")

    for idx in indices:
        if not isinstance(idx, (int, np.integer)):
            raise TypeError(
                f"Frame index {idx!r} is not an integer."
            )

        if idx < 0 or idx >= frame_count:
            raise IndexError(
                f"Frame index {idx} is outside valid range "
                f"0...{frame_count - 1}."
            )

    if len(indices) == 0:
        return []

    if threads is None:
        threads = min(8, max(1, os.cpu_count() or 1))

    if threads < 1:
        raise ValueError("threads must be at least 1.")

    # Avoid nested OpenCV and Python thread pools oversubscribing CPUs.
    cv2.setNumThreads(1)

    # ------------------------------------------------------------
    # Shared preprocessing
    # ------------------------------------------------------------
    gray_frames = preprocess_grayscale(
        np_frames,
        analysis_width=analysis_width
    )

    cuts = detect_scene_cuts(
        gray_frames,
        hard_hist_threshold=hard_hist_threshold,
        soft_hist_threshold=soft_hist_threshold,
        ssim_threshold=ssim_threshold
    )

    motion_values = precompute_motion(
        gray_frames,
        cuts,
        motion_reference=motion_reference,
        show_progress=show_progress
    )

    # Convert every source frame to linear light exactly once.
    linear_frames = [
        srgb_to_linear(frame)
        for frame in tqdm(
            np_frames,
            desc="Linear-light conversion",
            colour="yellow",
            unit="frame",
            disable=not show_progress
        )
    ]

    # Prefix sum allows a constant-time scene-cut check.
    #
    # cut_prefix[k] contains the number of cuts before frame k.
    cut_prefix = np.zeros(frame_count, dtype=np.int32)

    if cuts.size > 0:
        cut_prefix[1:] = np.cumsum(
            cuts,
            dtype=np.int32
        )

    def is_cut_between(frame_a, frame_b):
        """
        Returns True if one or more scene cuts exist between frames.
        """
        if frame_a == frame_b:
            return False

        low, high = sorted((frame_a, frame_b))

        return bool(
            cut_prefix[high] - cut_prefix[low] > 0
        )

    def local_motion(center):
        """
        Uses motion on both sides of the center frame.

        Taking the maximum makes the blender conservative when either
        adjacent transition contains substantial motion.
        """
        neighboring_motion = []

        if center > 0:
            neighboring_motion.append(
                float(motion_values[center - 1])
            )

        if center < frame_count - 1:
            neighboring_motion.append(
                float(motion_values[center])
            )

        if not neighboring_motion:
            return 0.0

        return max(neighboring_motion)

    def process(center):
        motion = local_motion(center)

        # Smooth nonlinear mapping:
        #   motion=0   -> base_radius
        #   motion=1   -> min_radius
        #
        # Smoothstep avoids abrupt radius changes between frames.
        smooth_motion = motion * motion * (3.0 - 2.0 * motion)

        radius_float = (
            base_radius
            - (base_radius - min_radius) * smooth_motion
        )

        adaptive_radius = int(round(radius_float))
        adaptive_radius = int(np.clip(
            adaptive_radius,
            min_radius,
            base_radius
        ))

        # Radius zero intentionally returns an independent copy.
        if adaptive_radius == 0:
            return np_frames[center].copy()

        start = max(0, center - adaptive_radius)
        end = min(
            frame_count - 1,
            center + adaptive_radius
        )

        selected_frames = []
        selected_weights = []

        # sigma >= 1 prevents an unnecessarily sharp Gaussian when
        # adaptive_radius is small.
        sigma = max(1.0,adaptive_radius / 2.0)

        denominator = 2.0 * sigma * sigma

        for frame_index in range(start, end + 1):
            if is_cut_between(center, frame_index):
                continue

            distance = abs(frame_index - center)

            temporal_weight = np.exp(
                -(distance * distance) / denominator
            )

            selected_frames.append(
                linear_frames[frame_index]
            )
            selected_weights.append(
                temporal_weight
            )

        # The center frame should always be available, but this
        # defensive fallback prevents an unexpected empty-window crash.
        if not selected_frames:
            return np_frames[center].copy()

        return blend_linear_frames(
            selected_frames,
            selected_weights
        )

    # ------------------------------------------------------------
    # Parallel output-frame processing
    # ------------------------------------------------------------
    with ThreadPoolExecutor(max_workers=threads) as executor:
        blended_iterator = executor.map(
            process,
            indices
        )

        results = list(tqdm(
            blended_iterator,
            total=len(indices),
            desc=f"Blending (maximum radius={base_radius})",
            colour="blue",
            unit="frame",
            disable=not show_progress
        ))

    return results
