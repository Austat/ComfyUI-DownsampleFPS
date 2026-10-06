import os
import cv2
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


# ============================================================
# Input validation
# ============================================================
def _validate_frames(frames):
    """
    Validates that frames are equal-sized uint8 BGR images.
    """
    if frames is None or len(frames) == 0:
        raise ValueError("np_frames must contain at least one frame.")

    expected_shape = frames[0].shape

    if (
        len(expected_shape) != 3
        or expected_shape[2] != 3
    ):
        raise ValueError(
            "Frames must have shape (height, width, 3)."
        )

    for i, frame in enumerate(frames):
        if frame is None:
            raise ValueError(f"Frame {i} is None.")

        if frame.shape != expected_shape:
            raise ValueError(
                f"Frame {i} has shape {frame.shape}, "
                f"but expected {expected_shape}."
            )

        if frame.dtype != np.uint8:
            raise ValueError(
                f"Frame {i} has dtype {frame.dtype}, "
                "but uint8 is required."
            )


# ============================================================
# Exact sRGB transfer functions
# ============================================================
def _srgb_to_linear(frame):
    """
    Converts a uint8 sRGB BGR frame to float32 linear light.

    Output range: [0, 1].
    """
    srgb = frame.astype(np.float32) / 255.0

    linear = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4
    )

    return linear.astype(np.float32)


def _linear_to_srgb(frame):
    """
    Converts a linear-light BGR frame to uint8 sRGB.
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
# Optical flow, deliberately using the original heavy settings
# ============================================================
def _compute_flow(frame_a, frame_b):
    """
    Calculates dense Farneback optical flow from frame_a to frame_b.

    The computationally heavy original settings are intentionally kept.
    """
    gray_a = cv2.cvtColor(
        frame_a,
        cv2.COLOR_BGR2GRAY
    )

    gray_b = cv2.cvtColor(
        frame_b,
        cv2.COLOR_BGR2GRAY
    )

    flow = cv2.calcOpticalFlowFarneback(
        gray_a,
        gray_b,
        None,
        pyr_scale=0.5,
        levels=5,
        winsize=21,
        iterations=7,
        poly_n=7,
        poly_sigma=1.5,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    )

    return flow.astype(np.float32)


# ============================================================
# Remapping helpers
# ============================================================
def _create_coordinate_grid(height, width):
    """
    Creates reusable float32 pixel-coordinate maps.
    """
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32)
    )

    return grid_x, grid_y


def _sample_field(field, map_x, map_y):
    """
    Samples a scalar, image, or vector field using backward remapping.
    """
    return cv2.remap(
        field,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )


def _warp_with_flow(
    image,
    flow,
    scale,
    grid_x,
    grid_y,
    inverse_iterations=2,
    border_mode=cv2.BORDER_REFLECT101
):
    """
    Warps an image or field using a forward optical-flow field.

    OpenCV remap requires destination-to-source coordinates. A forward
    flow describes source-to-destination displacement, so the inverse
    mapping is approximated iteratively.

    Starting estimate:
        source = destination - scale * flow(destination)

    Refinement:
        source = destination - scale * flow(source)

    This is more accurate than directly using either:

        grid + scale * flow

    or a single fixed:

        grid - scale * flow
    """
    scale = float(scale)

    if abs(scale) <= 1e-12:
        return image.copy()

    map_x = grid_x - scale * flow[..., 0]
    map_y = grid_y - scale * flow[..., 1]

    for _ in range(max(0, int(inverse_iterations))):
        sampled_flow_x = cv2.remap(
            flow[..., 0],
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        sampled_flow_y = cv2.remap(
            flow[..., 1],
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

        map_x = grid_x - scale * sampled_flow_x
        map_y = grid_y - scale * sampled_flow_y

    warped = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=border_mode,
        borderValue=0
    )

    return warped


def _warp_mask_with_flow(
    mask,
    flow,
    scale,
    grid_x,
    grid_y,
    inverse_iterations=2
):
    """
    Warps a scalar mask and clamps it to [0, 1].
    """
    warped = _warp_with_flow(
        mask.astype(np.float32),
        flow,
        scale,
        grid_x,
        grid_y,
        inverse_iterations=inverse_iterations,
        border_mode=cv2.BORDER_CONSTANT
    )

    return np.clip(warped, 0.0, 1.0)


# ============================================================
# Forward-backward flow consistency and occlusion confidence
# ============================================================
def _compute_consistency_confidence(
    flow_fwd,
    flow_bwd,
    grid_x,
    grid_y,
    consistency_scale=1.5,
    motion_scale=0.05
):
    """
    Computes confidence maps for both flow directions using
    forward-backward consistency.

    For forward flow at source location x:

        error_fwd =
            ||F(x) + B(x + F(x))||

    If the flows are mutually consistent, the sum should be close to
    zero. A similar calculation is performed for backward flow.

    The threshold grows slightly with flow magnitude, because large
    displacements naturally produce somewhat larger numerical errors.

    Returns:
        confidence_fwd, confidence_bwd

    Values are in [0, 1]:
        1 = high confidence
        0 = likely occluded, invalid, or inconsistent
    """
    height, width = flow_fwd.shape[:2]

    fwd_target_x = grid_x + flow_fwd[..., 0]
    fwd_target_y = grid_y + flow_fwd[..., 1]

    bwd_target_x = grid_x + flow_bwd[..., 0]
    bwd_target_y = grid_y + flow_bwd[..., 1]

    sampled_bwd = _sample_field(
        flow_bwd,
        fwd_target_x,
        fwd_target_y
    )

    sampled_fwd = _sample_field(
        flow_fwd,
        bwd_target_x,
        bwd_target_y
    )

    error_fwd = np.linalg.norm(
        flow_fwd + sampled_bwd,
        axis=2
    )

    error_bwd = np.linalg.norm(
        flow_bwd + sampled_fwd,
        axis=2
    )

    magnitude_fwd = np.linalg.norm(
        flow_fwd,
        axis=2
    )

    magnitude_bwd = np.linalg.norm(
        flow_bwd,
        axis=2
    )

    magnitude_sampled_bwd = np.linalg.norm(
        sampled_bwd,
        axis=2
    )

    magnitude_sampled_fwd = np.linalg.norm(
        sampled_fwd,
        axis=2
    )

    threshold_fwd = (
        consistency_scale
        + motion_scale
        * (magnitude_fwd + magnitude_sampled_bwd)
    )

    threshold_bwd = (
        consistency_scale
        + motion_scale
        * (magnitude_bwd + magnitude_sampled_fwd)
    )

    threshold_fwd = np.maximum(
        threshold_fwd,
        1e-6
    )

    threshold_bwd = np.maximum(
        threshold_bwd,
        1e-6
    )

    # Smooth confidence instead of a hard binary cutoff.
    confidence_fwd = np.exp(
        -np.square(error_fwd / threshold_fwd)
    )

    confidence_bwd = np.exp(
        -np.square(error_bwd / threshold_bwd)
    )

    # Coordinates outside the other image cannot be validated.
    valid_fwd = (
        (fwd_target_x >= 0.0)
        & (fwd_target_x <= width - 1.0)
        & (fwd_target_y >= 0.0)
        & (fwd_target_y <= height - 1.0)
    )

    valid_bwd = (
        (bwd_target_x >= 0.0)
        & (bwd_target_x <= width - 1.0)
        & (bwd_target_y >= 0.0)
        & (bwd_target_y <= height - 1.0)
    )

    confidence_fwd *= valid_fwd.astype(np.float32)
    confidence_bwd *= valid_bwd.astype(np.float32)

    # Light smoothing prevents noisy single-pixel mask changes.
    confidence_fwd = cv2.GaussianBlur(
        confidence_fwd.astype(np.float32),
        (5, 5),
        0
    )

    confidence_bwd = cv2.GaussianBlur(
        confidence_bwd.astype(np.float32),
        (5, 5),
        0
    )

    return (
        np.clip(confidence_fwd, 0.0, 1.0),
        np.clip(confidence_bwd, 0.0, 1.0)
    )


# ============================================================
# Structural similarity for scene-cut detection
# ============================================================
def _structural_similarity(gray_a, gray_b):
    """
    Computes an SSIM-style structural similarity score.

    Normally:
        1.0 = highly similar
        0.0 = highly dissimilar
    """
    image_a = gray_a.astype(np.float32)
    image_b = gray_b.astype(np.float32)

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    mu_a = cv2.GaussianBlur(
        image_a,
        (11, 11),
        1.5
    )

    mu_b = cv2.GaussianBlur(
        image_b,
        (11, 11),
        1.5
    )

    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a_sq = (
        cv2.GaussianBlur(
            image_a * image_a,
            (11, 11),
            1.5
        )
        - mu_a_sq
    )

    sigma_b_sq = (
        cv2.GaussianBlur(
            image_b * image_b,
            (11, 11),
            1.5
        )
        - mu_b_sq
    )

    sigma_ab = (
        cv2.GaussianBlur(
            image_a * image_b,
            (11, 11),
            1.5
        )
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

    score_map = numerator / np.maximum(
        denominator,
        1e-12
    )

    return float(np.clip(
        score_map.mean(),
        -1.0,
        1.0
    ))


def _calculate_histogram(gray):
    """
    Calculates an L1-normalized grayscale histogram.
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


def _resize_for_analysis(image, max_width=640):
    """
    Downscales an image for scene-cut analysis.
    """
    height, width = image.shape[:2]

    if width <= max_width:
        return image

    scale = max_width / float(width)
    new_height = max(
        1,
        int(round(height * scale))
    )

    return cv2.resize(
        image,
        (max_width, new_height),
        interpolation=cv2.INTER_AREA
    )


def _detect_scene_cuts(
    frames,
    hard_hist_threshold=0.35,
    soft_hist_threshold=0.70,
    ssim_threshold=0.45,
    analysis_width=640
):
    """
    Detects scene cuts using both histogram correlation and structural
    similarity.

    A cut is detected if:

      1. Histogram correlation is extremely low, or
      2. Histogram correlation is moderately low and SSIM is also low.
    """
    frame_count = len(frames)

    if frame_count < 2:
        return np.zeros(0, dtype=bool)

    gray_frames = []
    histograms = []

    for frame in frames:
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        gray = _resize_for_analysis(
            gray,
            max_width=analysis_width
        )

        gray_frames.append(gray)
        histograms.append(
            _calculate_histogram(gray)
        )

    cuts = np.zeros(
        frame_count - 1,
        dtype=bool
    )

    for i in range(frame_count - 1):
        histogram_score = cv2.compareHist(
            histograms[i],
            histograms[i + 1],
            cv2.HISTCMP_CORREL
        )

        ssim_score = _structural_similarity(
            gray_frames[i],
            gray_frames[i + 1]
        )

        hard_cut = (
            histogram_score < hard_hist_threshold
        )

        combined_cut = (
            histogram_score < soft_hist_threshold
            and ssim_score < ssim_threshold
        )

        cuts[i] = hard_cut or combined_cut

    return cuts


# ============================================================
# Occlusion-aware interpolation
# ============================================================
def _interpolate_frame_cached(
    frame_a_linear,
    frame_b_linear,
    t,
    flow_fwd,
    flow_bwd,
    confidence_fwd,
    confidence_bwd,
    grid_x,
    grid_y,
    inverse_iterations=2,
    minimum_confidence=1e-4
):
    """
    Creates an intermediate frame using bidirectional optical flow,
    inverse-mapping refinement, consistency confidence, and
    occlusion-aware gamma-correct blending.

    frame_a_linear and frame_b_linear must already be in linear light.
    """
    t = float(np.clip(t, 0.0, 1.0))

    if t <= 1e-8:
        return _linear_to_srgb(
            frame_a_linear
        )

    if t >= 1.0 - 1e-8:
        return _linear_to_srgb(
            frame_b_linear
        )

    # Move frame A forward toward the intermediate instant.
    warp_a = _warp_with_flow(
        frame_a_linear,
        flow_fwd,
        t,
        grid_x,
        grid_y,
        inverse_iterations=inverse_iterations
    )

    # Move frame B backward toward the intermediate instant.
    warp_b = _warp_with_flow(
        frame_b_linear,
        flow_bwd,
        1.0 - t,
        grid_x,
        grid_y,
        inverse_iterations=inverse_iterations
    )

    # Move source-coordinate confidence maps into the same
    # intermediate coordinate system.
    confidence_a = _warp_mask_with_flow(
        confidence_fwd,
        flow_fwd,
        t,
        grid_x,
        grid_y,
        inverse_iterations=inverse_iterations
    )

    confidence_b = _warp_mask_with_flow(
        confidence_bwd,
        flow_bwd,
        1.0 - t,
        grid_x,
        grid_y,
        inverse_iterations=inverse_iterations
    )

    # Temporal contribution:
    # near A, A should dominate;
    # near B, B should dominate.
    weight_a = (1.0 - t) * confidence_a
    weight_b = t * confidence_b

    weight_sum = weight_a + weight_b
    valid = weight_sum > minimum_confidence

    normalized_a = np.zeros_like(
        weight_a,
        dtype=np.float32
    )

    normalized_b = np.zeros_like(
        weight_b,
        dtype=np.float32
    )

    normalized_a[valid] = (
        weight_a[valid] / weight_sum[valid]
    )

    normalized_b[valid] = (
        weight_b[valid] / weight_sum[valid]
    )

    blended = (
        warp_a * normalized_a[..., None]
        + warp_b * normalized_b[..., None]
    )

    # If both directions are unreliable, use a temporal nearest-frame
    # fallback. This avoids dividing by near-zero confidence and tends
    # to produce fewer transparent-looking ghosts.
    if not np.all(valid):
        fallback = warp_a if t < 0.5 else warp_b
        blended[~valid] = fallback[~valid]

    return _linear_to_srgb(blended)


# ============================================================
# Main API
# ============================================================
def motion_compensated(
    np_frames,
    indices,
    threads=8,
    hard_hist_threshold=0.35,
    soft_hist_threshold=0.70,
    ssim_threshold=0.45,
    scene_analysis_width=640,
    consistency_scale=1.5,
    consistency_motion_scale=0.05,
    inverse_iterations=2,
    show_progress=True
):
    """
    CPU-based motion-compensated frame interpolation.

    Preserved intentionally:
      - Heavy Farneback parameters
      - Full forward-flow cache
      - Full backward-flow cache
      - Multithreaded flow precomputation
      - Multithreaded interpolation

    Improvements:
      - Correct backward remapping
      - Iterative inverse-map refinement
      - Forward-backward consistency confidence
      - Occlusion-aware weighting
      - Gamma-correct blending
      - Histogram + SSIM scene-cut detection
      - Rounded and clipped uint8 output
      - Input and index validation

    Args:
        np_frames:
            Sequence of uint8 BGR source frames.

        indices:
            Fractional source-frame positions.

            Examples:
                0.0  -> first source frame
                0.5  -> halfway between frames 0 and 1
                3.25 -> quarter-way from frame 3 to frame 4

        threads:
            Worker-thread count.

        hard_hist_threshold:
            Histogram correlation below which the boundary is always
            treated as a scene cut.

        soft_hist_threshold:
            Histogram threshold used together with SSIM.

        ssim_threshold:
            Structural-similarity threshold used with the soft
            histogram threshold.

        scene_analysis_width:
            Scene-cut analysis width.

        consistency_scale:
            Base tolerance for forward-backward consistency.

        consistency_motion_scale:
            Adds tolerance proportionally to motion magnitude.

        inverse_iterations:
            Number of inverse-warp fixed-point refinement iterations.

        show_progress:
            Enables tqdm progress bars.

    Returns:
        List of interpolated uint8 BGR frames in the same order as
        indices.
    """
    frames = list(np_frames)
    requested_indices = list(indices)

    _validate_frames(frames)

    if threads is None:
        threads = min(
            8,
            max(1, os.cpu_count() or 1)
        )

    if threads < 1:
        raise ValueError("threads must be at least 1.")

    if inverse_iterations < 0:
        raise ValueError(
            "inverse_iterations cannot be negative."
        )

    for idx in requested_indices:
        if not isinstance(
            idx,
            (int, float, np.integer, np.floating)
        ):
            raise TypeError(
                f"Interpolation index {idx!r} is not numeric."
            )

        if not np.isfinite(idx):
            raise ValueError(
                f"Interpolation index {idx!r} is not finite."
            )

    if len(requested_indices) == 0:
        return []

    num_frames = len(frames)

    if num_frames == 1:
        return [
            frames[0].copy()
            for _ in requested_indices
        ]

    # Prevent nested OpenCV and Python thread pools from creating
    # excessive CPU oversubscription.
    cv2.setNumThreads(1)

    height, width = frames[0].shape[:2]
    grid_x, grid_y = _create_coordinate_grid(
        height,
        width
    )

    # ------------------------------------------------------------
    # Convert source frames to linear light once.
    # ------------------------------------------------------------
    linear_frames = [
        _srgb_to_linear(frame)
        for frame in tqdm(
            frames,
            total=num_frames,
            desc="Linear-light conversion",
            colour="green",
            unit="frame",
            disable=not show_progress
        )
    ]

    # ------------------------------------------------------------
    # Scene-cut detection
    # ------------------------------------------------------------
    cuts = _detect_scene_cuts(
        frames,
        hard_hist_threshold=hard_hist_threshold,
        soft_hist_threshold=soft_hist_threshold,
        ssim_threshold=ssim_threshold,
        analysis_width=scene_analysis_width
    )

    # ------------------------------------------------------------
    # Full-video bidirectional flow-cache precomputation
    # ------------------------------------------------------------
    print(
        "[Motion-compensated] "
        f"Precomputing full optical-flow cache "
        f"using {threads} threads..."
    )

    flow_fwd_cache = {}
    flow_bwd_cache = {}

    confidence_fwd_cache = {}
    confidence_bwd_cache = {}

    flow_tasks = [
        i
        for i in range(num_frames - 1)
        if not cuts[i]
    ]

    def _compute_pair(i):
        frame_a = frames[i]
        frame_b = frames[i + 1]

        flow_fwd = _compute_flow(
            frame_a,
            frame_b
        )

        flow_bwd = _compute_flow(
            frame_b,
            frame_a
        )

        confidence_fwd, confidence_bwd = (
            _compute_consistency_confidence(
                flow_fwd,
                flow_bwd,
                grid_x,
                grid_y,
                consistency_scale=consistency_scale,
                motion_scale=consistency_motion_scale
            )
        )

        return (
            i,
            flow_fwd,
            flow_bwd,
            confidence_fwd,
            confidence_bwd
        )

    with ThreadPoolExecutor(
        max_workers=threads
    ) as pool:
        pair_iterator = pool.map(
            _compute_pair,
            flow_tasks
        )

        for result in tqdm(
            pair_iterator,
            total=len(flow_tasks),
            desc="Flow and confidence cache",
            colour="yellow",
            unit="pair",
            disable=not show_progress
        ):
            (
                i,
                flow_fwd,
                flow_bwd,
                confidence_fwd,
                confidence_bwd
            ) = result

            flow_fwd_cache[i] = flow_fwd
            flow_bwd_cache[i] = flow_bwd

            confidence_fwd_cache[i] = confidence_fwd
            confidence_bwd_cache[i] = confidence_bwd

    # ------------------------------------------------------------
    # Build interpolation tasks
    # ------------------------------------------------------------
    tasks = []

    for requested_index in requested_indices:
        position = float(requested_index)

        if position <= 0.0:
            tasks.append({
                "kind": "exact",
                "frame_index": 0
            })
            continue

        if position >= num_frames - 1:
            tasks.append({
                "kind": "exact",
                "frame_index": num_frames - 1
            })
            continue

        base = int(np.floor(position))
        t = float(position - base)

        if t <= 1e-8:
            tasks.append({
                "kind": "exact",
                "frame_index": base
            })
            continue

        if t >= 1.0 - 1e-8:
            tasks.append({
                "kind": "exact",
                "frame_index": base + 1
            })
            continue

        tasks.append({
            "kind": "interpolate",
            "base": base,
            "t": t,
            "is_cut": bool(cuts[base])
        })

    # ------------------------------------------------------------
    # Interpolation using cached bidirectional flows
    # ------------------------------------------------------------
    def _process(task):
        if task["kind"] == "exact":
            return frames[
                task["frame_index"]
            ].copy()

        base = task["base"]
        t = task["t"]

        frame_a = frames[base]
        frame_b = frames[base + 1]

        # Never interpolate across a detected cut.
        if task["is_cut"]:
            return (
                frame_a.copy()
                if t < 0.5
                else frame_b.copy()
            )

        flow_fwd = flow_fwd_cache.get(base)
        flow_bwd = flow_bwd_cache.get(base)

        confidence_fwd = confidence_fwd_cache.get(base)
        confidence_bwd = confidence_bwd_cache.get(base)

        if (
            flow_fwd is None
            or flow_bwd is None
            or confidence_fwd is None
            or confidence_bwd is None
        ):
            return (
                frame_a.copy()
                if t < 0.5
                else frame_b.copy()
            )

        return _interpolate_frame_cached(
            linear_frames[base],
            linear_frames[base + 1],
            t,
            flow_fwd,
            flow_bwd,
            confidence_fwd,
            confidence_bwd,
            grid_x,
            grid_y,
            inverse_iterations=inverse_iterations
        )

    selected = []

    with ThreadPoolExecutor(
        max_workers=threads
    ) as pool:
        output_iterator = pool.map(
            _process,
            tasks
        )

        for output_frame in tqdm(
            output_iterator,
            total=len(tasks),
            desc="Motion-compensated interpolation",
            colour="blue",
            unit="frame",
            disable=not show_progress
        ):
            selected.append(output_frame)

    return selected