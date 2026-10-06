import os
import cv2
import threading
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


# ============================================================
# Validation
# ============================================================
def _validate_frames(frames):
    """
    Validates that all input frames are equal-sized uint8 BGR images.
    """
    if frames is None or len(frames) == 0:
        raise ValueError("np_frames must contain at least one frame.")

    expected_shape = frames[0].shape

    if len(expected_shape) != 3 or expected_shape[2] != 3:
        raise ValueError(
            "Frames must be BGR images with shape "
            "(height, width, 3)."
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


def _validate_rlof_support():
    """
    Checks that OpenCV contrib and Dense RLOF are available.
    """
    if not hasattr(cv2, "optflow"):
        raise RuntimeError(
            "cv2.optflow is unavailable. Install a compatible "
            "opencv-contrib-python build."
        )

    if not hasattr(cv2.optflow, "createOptFlow_DenseRLOF"):
        raise RuntimeError(
            "Dense RLOF is unavailable in this OpenCV build."
        )


# ============================================================
# Exact sRGB transfer functions
# ============================================================
def _srgb_to_linear(frame):
    """
    Converts a uint8 sRGB BGR frame to float32 linear light.
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
    Converts a linear-light BGR image to uint8 sRGB.
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
# Analysis helpers
# ============================================================
def _resize_for_analysis(image, max_width):
    """
    Downscales an image for scene-cut analysis.
    """
    height, width = image.shape[:2]

    if max_width is None or width <= max_width:
        return image

    scale = max_width / float(width)
    new_height = max(1, int(round(height * scale)))

    return cv2.resize(
        image,
        (max_width, new_height),
        interpolation=cv2.INTER_AREA
    )


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

    return histogram.astype(np.float32).reshape(-1)


def _structural_similarity(gray_a, gray_b):
    """
    Computes a lightweight SSIM-style similarity score.
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
        np.mean(score_map),
        -1.0,
        1.0
    ))


# ============================================================
# Scene-cut detection
# ============================================================
def detect_scene_cuts(
    frames,
    hard_hist_threshold=0.30,
    soft_hist_threshold=0.70,
    ssim_threshold=0.42,
    analysis_width=640
):
    """
    Detects hard scene cuts using histogram correlation and SSIM.

    A cut is detected when either:

      1. Histogram correlation is extremely low.
      2. Histogram correlation is moderately low and SSIM is low.
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
            analysis_width
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
# RLOF optical flow
# ============================================================
_thread_local = threading.local()


def _get_rlof_instance():
    """
    Returns one Dense RLOF instance per worker thread.

    Reusing the instance avoids constructing it separately for every
    frame pair while also avoiding sharing one instance across threads.
    """
    if not hasattr(_thread_local, "rlof"):
        _thread_local.rlof = (
            cv2.optflow.createOptFlow_DenseRLOF()
        )

    return _thread_local.rlof


def compute_rlof(frame_a, frame_b):
    """
    Computes dense RLOF optical flow from frame_a to frame_b.

    BGR input is intentionally preserved for compatibility with RLOF
    support-region configurations that use color information.
    """
    rlof = _get_rlof_instance()

    flow = rlof.calc(
        frame_a,
        frame_b,
        None
    )

    if flow is None:
        raise RuntimeError(
            "Dense RLOF returned no optical-flow field."
        )

    if (
        flow.ndim != 3
        or flow.shape[2] != 2
    ):
        raise RuntimeError(
            f"Unexpected RLOF flow shape: {flow.shape}."
        )

    return flow.astype(
        np.float32,
        copy=False
    )


# ============================================================
# Coordinate and remapping helpers
# ============================================================
def _create_coordinate_grid(height, width):
    """
    Creates reusable float32 pixel-coordinate grids.
    """
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32)
    )

    return grid_x, grid_y


def _sample_field(
    field,
    map_x,
    map_y,
    border_mode=cv2.BORDER_CONSTANT
):
    """
    Samples a scalar or vector field at floating-point coordinates.
    """
    return cv2.remap(
        field,
        map_x.astype(np.float32, copy=False),
        map_y.astype(np.float32, copy=False),
        interpolation=cv2.INTER_LINEAR,
        borderMode=border_mode,
        borderValue=0
    )


def _build_inverse_map(
    flow,
    scale,
    grid_x,
    grid_y,
    inverse_iterations=3
):
    """
    Builds an approximate destination-to-source map from a forward
    optical-flow field.

    RLOF flow expresses source-to-destination displacement, while
    cv2.remap requires destination-to-source coordinates. The inverse
    mapping is refined with fixed-point iterations.
    """
    scale = float(scale)

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

    return (
        map_x.astype(np.float32, copy=False),
        map_y.astype(np.float32, copy=False)
    )


def _warp_with_map(
    image,
    map_x,
    map_y,
    border_mode=cv2.BORDER_REFLECT101
):
    """
    Warps an image or field using a precomputed inverse map.
    """
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=border_mode,
        borderValue=0
    )


def _validity_from_map(map_x, map_y, width, height):
    """
    Returns image-bound validity for a remapping field.

    A small margin is used because bilinear interpolation needs valid
    neighboring pixels.
    """
    return (
        (map_x >= 0.0)
        & (map_x < width - 1.0)
        & (map_y >= 0.0)
        & (map_y < height - 1.0)
    ).astype(np.float32)


# ============================================================
# Flow confidence and occlusion estimation
# ============================================================
def _compute_consistency_confidence(
    flow_fwd,
    flow_bwd,
    grid_x,
    grid_y,
    base_tolerance=1.5,
    motion_tolerance=0.05
):
    """
    Computes bidirectional consistency confidence.

    For forward flow F and backward flow B:

        error_fwd(x) = ||F(x) + B(x + F(x))||

    A consistent forward and backward flow pair should sum to
    approximately zero.
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

    sampled_bwd_magnitude = np.linalg.norm(
        sampled_bwd,
        axis=2
    )

    sampled_fwd_magnitude = np.linalg.norm(
        sampled_fwd,
        axis=2
    )

    threshold_fwd = (
        base_tolerance
        + motion_tolerance
        * (magnitude_fwd + sampled_bwd_magnitude)
    )

    threshold_bwd = (
        base_tolerance
        + motion_tolerance
        * (magnitude_bwd + sampled_fwd_magnitude)
    )

    threshold_fwd = np.maximum(
        threshold_fwd,
        1e-6
    )

    threshold_bwd = np.maximum(
        threshold_bwd,
        1e-6
    )

    confidence_fwd = np.exp(
        -np.square(error_fwd / threshold_fwd)
    )

    confidence_bwd = np.exp(
        -np.square(error_bwd / threshold_bwd)
    )

    valid_fwd = (
        (fwd_target_x >= 0.0)
        & (fwd_target_x < width - 1.0)
        & (fwd_target_y >= 0.0)
        & (fwd_target_y < height - 1.0)
    )

    valid_bwd = (
        (bwd_target_x >= 0.0)
        & (bwd_target_x < width - 1.0)
        & (bwd_target_y >= 0.0)
        & (bwd_target_y < height - 1.0)
    )

    confidence_fwd *= valid_fwd.astype(np.float32)
    confidence_bwd *= valid_bwd.astype(np.float32)

    # Mild smoothing prevents isolated single-pixel confidence holes.
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


def _compute_photometric_confidence(
    frame_a_linear,
    frame_b_linear,
    flow_fwd,
    flow_bwd,
    grid_x,
    grid_y,
    photometric_scale=0.08
):
    """
    Estimates flow confidence from brightness/color agreement.

    This is complementary to forward-backward consistency. A flow can
    be geometrically consistent but still point to a photometrically
    implausible location.
    """
    fwd_target_x = grid_x + flow_fwd[..., 0]
    fwd_target_y = grid_y + flow_fwd[..., 1]

    bwd_target_x = grid_x + flow_bwd[..., 0]
    bwd_target_y = grid_y + flow_bwd[..., 1]

    sampled_b = _sample_field(
        frame_b_linear,
        fwd_target_x,
        fwd_target_y,
        border_mode=cv2.BORDER_REFLECT101
    )

    sampled_a = _sample_field(
        frame_a_linear,
        bwd_target_x,
        bwd_target_y,
        border_mode=cv2.BORDER_REFLECT101
    )

    error_fwd = np.mean(
        np.abs(frame_a_linear - sampled_b),
        axis=2
    )

    error_bwd = np.mean(
        np.abs(frame_b_linear - sampled_a),
        axis=2
    )

    scale = max(float(photometric_scale), 1e-6)

    confidence_fwd = np.exp(
        -np.square(error_fwd / scale)
    )

    confidence_bwd = np.exp(
        -np.square(error_bwd / scale)
    )

    return (
        np.clip(
            confidence_fwd.astype(np.float32),
            0.0,
            1.0
        ),
        np.clip(
            confidence_bwd.astype(np.float32),
            0.0,
            1.0
        )
    )


# ============================================================
# High-quality bidirectional interpolation
# ============================================================
def _interpolate_cached(
    frame_a_linear,
    frame_b_linear,
    t,
    flow_fwd,
    flow_bwd,
    confidence_fwd,
    confidence_bwd,
    grid_x,
    grid_y,
    inverse_iterations=3,
    minimum_weight=1e-5
):
    """
    Produces an intermediate frame using:

      - bidirectional RLOF
      - refined inverse remapping
      - forward-backward confidence
      - photometric confidence
      - occlusion-aware blending
      - exact linear-light blending
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

    height, width = flow_fwd.shape[:2]

    # Frame A moves forward by t.
    map_a_x, map_a_y = _build_inverse_map(
        flow_fwd,
        t,
        grid_x,
        grid_y,
        inverse_iterations=inverse_iterations
    )

    # Frame B moves backward by 1-t.
    map_b_x, map_b_y = _build_inverse_map(
        flow_bwd,
        1.0 - t,
        grid_x,
        grid_y,
        inverse_iterations=inverse_iterations
    )

    warp_a = _warp_with_map(
        frame_a_linear,
        map_a_x,
        map_a_y
    )

    warp_b = _warp_with_map(
        frame_b_linear,
        map_b_x,
        map_b_y
    )

    confidence_a = _warp_with_map(
        confidence_fwd,
        map_a_x,
        map_a_y,
        border_mode=cv2.BORDER_CONSTANT
    )

    confidence_b = _warp_with_map(
        confidence_bwd,
        map_b_x,
        map_b_y,
        border_mode=cv2.BORDER_CONSTANT
    )

    valid_a = _validity_from_map(
        map_a_x,
        map_a_y,
        width,
        height
    )

    valid_b = _validity_from_map(
        map_b_x,
        map_b_y,
        width,
        height
    )

    confidence_a = np.clip(
        confidence_a * valid_a,
        0.0,
        1.0
    )

    confidence_b = np.clip(
        confidence_b * valid_b,
        0.0,
        1.0
    )

    # Temporal weighting ensures the result approaches A near t=0
    # and B near t=1.
    weight_a = (
        (1.0 - t)
        * confidence_a
    )

    weight_b = (
        t
        * confidence_b
    )

    weight_sum = weight_a + weight_b

    valid_blend = (
        weight_sum > minimum_weight
    )

    normalized_a = np.zeros_like(
        weight_a,
        dtype=np.float32
    )

    normalized_b = np.zeros_like(
        weight_b,
        dtype=np.float32
    )

    normalized_a[valid_blend] = (
        weight_a[valid_blend]
        / weight_sum[valid_blend]
    )

    normalized_b[valid_blend] = (
        weight_b[valid_blend]
        / weight_sum[valid_blend]
    )

    blended = (
        warp_a * normalized_a[..., None]
        + warp_b * normalized_b[..., None]
    )

    # Both confidence maps may be weak in a newly revealed region.
    # Choose the geometrically valid side before falling back to the
    # temporally nearer warped frame.
    invalid = ~valid_blend

    if np.any(invalid):
        only_a = (
            invalid
            & (valid_a > 0.5)
            & (valid_b <= 0.5)
        )

        only_b = (
            invalid
            & (valid_b > 0.5)
            & (valid_a <= 0.5)
        )

        both_or_neither = (
            invalid
            & ~only_a
            & ~only_b
        )

        blended[only_a] = warp_a[only_a]
        blended[only_b] = warp_b[only_b]

        if t < 0.5:
            blended[both_or_neither] = (
                warp_a[both_or_neither]
            )
        else:
            blended[both_or_neither] = (
                warp_b[both_or_neither]
            )

    return _linear_to_srgb(blended)


# ============================================================
# Main API
# ============================================================
def optical_flow(
    np_frames,
    indices,
    threads=8,
    interpolation_t=0.5,
    hard_hist_threshold=0.30,
    soft_hist_threshold=0.70,
    ssim_threshold=0.42,
    scene_analysis_width=640,
    consistency_tolerance=1.5,
    consistency_motion_tolerance=0.05,
    photometric_scale=0.08,
    inverse_iterations=3,
    show_progress=True
):
    """
    High-quality RLOF interpolation.

    Compatibility with the original function:

        indices=[0, 1, 2]

    with interpolation_t=0.5 produces intermediate frames between:

        frame 0 and 1
        frame 1 and 2
        frame 2 and 3

    Main features:

      - Bidirectional Dense RLOF
      - Full flow and confidence cache
      - Thread-local RLOF instances
      - Histogram + SSIM scene-cut detection
      - Refined inverse warping
      - Forward-backward consistency confidence
      - Photometric confidence
      - Occlusion-aware blending
      - Linear-light interpolation
      - Multithreaded flow and output processing

    Args:
        np_frames:
            Sequence of uint8 BGR frames.

        indices:
            Integer indices identifying source frame pairs.

        threads:
            Number of worker threads.

        interpolation_t:
            Interpolation position inside each pair.

            0.0 = frame A
            0.5 = halfway frame
            1.0 = frame B

        inverse_iterations:
            Number of fixed-point inverse-map refinements. Values 2-4
            are generally reasonable when quality is prioritized.

    Returns:
        List of interpolated uint8 BGR frames.
    """
    frames = list(np_frames)
    requested_indices = list(indices)

    _validate_frames(frames)
    _validate_rlof_support()

    if threads is None:
        threads = min(
            8,
            max(1, os.cpu_count() or 1)
        )

    if threads < 1:
        raise ValueError(
            "threads must be at least 1."
        )

    if not 0.0 <= interpolation_t <= 1.0:
        raise ValueError(
            "interpolation_t must be in range [0, 1]."
        )

    if inverse_iterations < 0:
        raise ValueError(
            "inverse_iterations cannot be negative."
        )

    if len(requested_indices) == 0:
        return []

    frame_count = len(frames)

    if frame_count == 1:
        return [
            frames[0].copy()
            for _ in requested_indices
        ]

    normalized_indices = []

    for idx in requested_indices:
        if not isinstance(idx, (int, np.integer)):
            raise TypeError(
                f"Pair index {idx!r} is not an integer."
            )

        idx = int(idx)

        if idx < 0 or idx >= frame_count:
            raise IndexError(
                f"Pair index {idx} is outside range "
                f"0...{frame_count - 1}."
            )

        normalized_indices.append(idx)

    # RLOF is already computationally parallel internally in some
    # OpenCV builds. Limiting OpenCV's own pool avoids uncontrolled
    # nested thread oversubscription.
    cv2.setNumThreads(1)

    height, width = frames[0].shape[:2]

    grid_x, grid_y = _create_coordinate_grid(
        height,
        width
    )

    # ------------------------------------------------------------
    # Source preprocessing
    # ------------------------------------------------------------
    linear_frames = [
        _srgb_to_linear(frame)
        for frame in tqdm(
            frames,
            total=frame_count,
            desc="Linear-light conversion",
            colour="green",
            unit="frame",
            disable=not show_progress
        )
    ]

    cuts = detect_scene_cuts(
        frames,
        hard_hist_threshold=hard_hist_threshold,
        soft_hist_threshold=soft_hist_threshold,
        ssim_threshold=ssim_threshold,
        analysis_width=scene_analysis_width
    )

    # ------------------------------------------------------------
    # Determine which pairs are actually required
    # ------------------------------------------------------------
    required_pairs = sorted({
        idx
        for idx in normalized_indices
        if idx < frame_count - 1
        and not cuts[idx]
    })

    # ------------------------------------------------------------
    # Full bidirectional caches for requested pairs
    # ------------------------------------------------------------
    flow_fwd_cache = {}
    flow_bwd_cache = {}

    confidence_fwd_cache = {}
    confidence_bwd_cache = {}

    def _compute_pair(pair_index):
        frame_a = frames[pair_index]
        frame_b = frames[pair_index + 1]

        flow_fwd = compute_rlof(
            frame_a,
            frame_b
        )

        flow_bwd = compute_rlof(
            frame_b,
            frame_a
        )

        (
            consistency_fwd,
            consistency_bwd
        ) = _compute_consistency_confidence(
            flow_fwd,
            flow_bwd,
            grid_x,
            grid_y,
            base_tolerance=consistency_tolerance,
            motion_tolerance=(
                consistency_motion_tolerance
            )
        )

        (
            photometric_fwd,
            photometric_bwd
        ) = _compute_photometric_confidence(
            linear_frames[pair_index],
            linear_frames[pair_index + 1],
            flow_fwd,
            flow_bwd,
            grid_x,
            grid_y,
            photometric_scale=photometric_scale
        )

        # Geometric and photometric confidences complement one
        # another. Square root keeps the product from becoming
        # needlessly aggressive.
        confidence_fwd = np.sqrt(
            consistency_fwd * photometric_fwd
        ).astype(np.float32)

        confidence_bwd = np.sqrt(
            consistency_bwd * photometric_bwd
        ).astype(np.float32)

        confidence_fwd = cv2.GaussianBlur(
            confidence_fwd,
            (5, 5),
            0
        )

        confidence_bwd = cv2.GaussianBlur(
            confidence_bwd,
            (5, 5),
            0
        )

        return (
            pair_index,
            flow_fwd,
            flow_bwd,
            np.clip(confidence_fwd, 0.0, 1.0),
            np.clip(confidence_bwd, 0.0, 1.0)
        )

    with ThreadPoolExecutor(
        max_workers=threads
    ) as executor:
        pair_iterator = executor.map(
            _compute_pair,
            required_pairs
        )

        for result in tqdm(
            pair_iterator,
            total=len(required_pairs),
            desc="Bidirectional RLOF cache",
            colour="yellow",
            unit="pair",
            disable=not show_progress
        ):
            (
                pair_index,
                flow_fwd,
                flow_bwd,
                confidence_fwd,
                confidence_bwd
            ) = result

            flow_fwd_cache[pair_index] = flow_fwd
            flow_bwd_cache[pair_index] = flow_bwd

            confidence_fwd_cache[pair_index] = (
                confidence_fwd
            )

            confidence_bwd_cache[pair_index] = (
                confidence_bwd
            )

    # ------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------
    def _process(pair_index):
        # Last frame has no following pair.
        if pair_index >= frame_count - 1:
            return frames[-1].copy()

        frame_a = frames[pair_index]
        frame_b = frames[pair_index + 1]

        # Never morph two unrelated scenes.
        if cuts[pair_index]:
            return (
                frame_a.copy()
                if interpolation_t < 0.5
                else frame_b.copy()
            )

        flow_fwd = flow_fwd_cache.get(pair_index)
        flow_bwd = flow_bwd_cache.get(pair_index)

        confidence_fwd = confidence_fwd_cache.get(
            pair_index
        )

        confidence_bwd = confidence_bwd_cache.get(
            pair_index
        )

        if (
            flow_fwd is None
            or flow_bwd is None
            or confidence_fwd is None
            or confidence_bwd is None
        ):
            return (
                frame_a.copy()
                if interpolation_t < 0.5
                else frame_b.copy()
            )

        return _interpolate_cached(
            linear_frames[pair_index],
            linear_frames[pair_index + 1],
            interpolation_t,
            flow_fwd,
            flow_bwd,
            confidence_fwd,
            confidence_bwd,
            grid_x,
            grid_y,
            inverse_iterations=inverse_iterations
        )

    with ThreadPoolExecutor(
        max_workers=threads
    ) as executor:
        output_iterator = executor.map(
            _process,
            normalized_indices
        )

        results = list(tqdm(
            output_iterator,
            total=len(normalized_indices),
            desc="RLOF interpolation",
            colour="blue",
            unit="frame",
            disable=not show_progress
        ))

    return results
