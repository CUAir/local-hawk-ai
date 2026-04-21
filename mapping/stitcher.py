import cv2
import numpy as np
import gc
import time
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import lsqr
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

MAX_CANVAS_DIM = 10000
NEIGHBOR_RADIUS_M = 150.0
MAX_ANCHORS = 8
RATIO_TEST = 0.75         # Lowe's recommended threshold; tighter reduces false matches in repetitive terrain
RANSAC_REPROJ = 6.0
SIFT_FEATURES = 8000
SIFT_CONTRAST = 0.005
GPS_DIR_COS_MIN = 0.35    # Require SIFT-predicted direction within ~70° of GPS direction
GPS_MAG_MIN = 0.1
GPS_MAG_MAX = 2.5         # Reject if SIFT says image is >2.5x farther than GPS suggests
GPS_MIN_DIST_M = 2.0
GPS_ABS_MAX_M = 22.0              # GPS absolute position tolerance; beyond this, try to GPS-correct rather than reject
GPS_ABS_CORRECT_MIN_INLIERS = 30  # If inliers >= this and GPS abs fails, shift chain to GPS position instead of rejecting
MIN_PLACEMENT_INLIERS = 40        # Minimum RANSAC inliers for passes 2+ (stricter keeps weak chains out of the pose solve)
MIN_PLACEMENT_INLIERS_PASS1 = 50  # Stricter floor for pass 1 (sparse anchor graph = higher bad-seed risk)

PREVIEW_EVERY = 25
REFINE_MAX_ITERS = 200
REFINE_STEP = 0.3         # Fallback iterative step if LSQR solve is unavailable
REFINE_EARLY_STOP = 1e-5
LAPLACIAN_LEVELS = 5
BLEND_MEM_BUDGET = 2.0 * 1024 ** 3   # 2 GB cap for float32 Laplacian accumulator
NUM_WORKERS = 4           # Parallel workers for feature extraction and warping
GAIN_COMPENSATION = True  # Normalize per-image brightness before blending
GAIN_CLIP = (0.7, 1.4)    # Clamp gain to avoid over-correction on outlier exposures
VORONOI_FEATHER_SIGMA = 2   # Seam feather radius; graph-cut seams are already content-clean so 2 is enough
SEAM_SCALE = 0.25           # Downsample factor for graph-cut seam finding
GAIN_SCALE = 0.25           # Downsample factor for gain compensator feed
CLAHE_CLIP = 2.0    # clipLimit for grayscale CLAHE before SIFT detection
CLAHE_TILE = 8      # tile grid size
AKAZE_THRESH = 0.005  # AKAZE detector sensitivity (higher = fewer, more reliable keypoints)


def _map_radians(deg):
    return deg * np.pi / 180.0


def _get_gps_pixel_delta(coord1, coord2, ref_lat, PPM):
    """Returns (dx, dy) in pixels from coord1 to coord2."""
    R = 6378137.0
    lat1, lon1 = _map_radians(coord1[0]), _map_radians(coord1[1])
    lat2, lon2 = _map_radians(coord2[0]), _map_radians(coord2[1])
    dy_m = (lat2 - lat1) * R
    dx_m = (lon2 - lon1) * R * np.cos(_map_radians(ref_lat))
    return dx_m * PPM, -dy_m * PPM


def _homography_area_ratio(H, w, h):
    """Estimate area scale change by projecting image corners through H."""
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(corners, H)
    return abs(cv2.contourArea(dst)) / (w * h + 1e-6)


def _nearest_placed(i, placed_indices, positions):
    best_d, nearest = np.inf, None
    for j in placed_indices:
        d = np.hypot(positions[i][0] - positions[j][0], positions[i][1] - positions[j][1])
        if d < best_d:
            best_d, nearest = d, j
    return nearest, best_d


def _valid_H(H):
    """Return True if H is finite and non-degenerate."""
    return (np.all(np.isfinite(H)) and
            abs(np.linalg.det(H[:2, :2])) > 1e-5)


def _cell_key(xy, cell_size):
    return (int(np.floor(xy[0] / cell_size)), int(np.floor(xy[1] / cell_size)))


def _grid_add(grid, idx, pos, cell_size):
    grid.setdefault(_cell_key(pos, cell_size), []).append(idx)


def _grid_candidates(grid, pos, cell_size):
    cx, cy = _cell_key(pos, cell_size)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            yield from grid.get((cx + dx, cy + dy), [])


def _extract_one(args):
    """Extract SIFT + AKAZE features for a single image (runs in thread pool)."""
    i, img = args
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_TILE, CLAHE_TILE))
    enhanced = clahe.apply(gray)
    sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES, contrastThreshold=SIFT_CONTRAST)
    akaze = cv2.AKAZE_create(threshold=AKAZE_THRESH)
    kp_s, des_s = sift.detectAndCompute(enhanced, None)
    kp_a, des_a = akaze.detectAndCompute(enhanced, None)
    return i, (kp_s, des_s, kp_a, des_a)


_bf_l2  = cv2.BFMatcher(cv2.NORM_L2)
_bf_ham = cv2.BFMatcher(cv2.NORM_HAMMING)


def _match_pair(feat_new, feat_ref):
    """
    Match SIFT (L2) and AKAZE (HAMMING) descriptors, apply Lowe ratio test on each,
    merge correspondences. Returns (src_pts, dst_pts) float32 (-1,1,2) or (None, None).
    """
    kp_sn, des_sn, kp_an, des_an = feat_new
    kp_sr, des_sr, kp_ar, des_ar = feat_ref
    pairs_src, pairs_dst = [], []

    if des_sn is not None and des_sr is not None and len(des_sn) >= 2 and len(des_sr) >= 2:
        for m, n in _bf_l2.knnMatch(des_sn, des_sr, k=2):
            if m.distance < RATIO_TEST * n.distance:
                pairs_src.append(kp_sn[m.queryIdx].pt)
                pairs_dst.append(kp_sr[m.trainIdx].pt)

    if des_an is not None and des_ar is not None and len(des_an) >= 2 and len(des_ar) >= 2:
        for m, n in _bf_ham.knnMatch(des_an, des_ar, k=2):
            if m.distance < 0.9 * n.distance:
                pairs_src.append(kp_an[m.queryIdx].pt)
                pairs_dst.append(kp_ar[m.trainIdx].pt)

    if len(pairs_src) < 4:
        return None, None
    return (np.float32(pairs_src).reshape(-1, 1, 2),
            np.float32(pairs_dst).reshape(-1, 1, 2))


# ---------------------------------------------------------------------------
# Direct similarity-pose solve (replaces iterative averaging refinement)
# ---------------------------------------------------------------------------

def _solve_poses_lsqr(placed_info: Dict,
                       placement_links: List[Tuple[int, int, np.ndarray]],
                       anchor_idx: int) -> Dict[int, np.ndarray]:
    """
    Global linear least-squares solve for similarity poses.

    Each similarity pose H_i is parametrized as (tx, ty, a, b) where the 2x2
    block is [[a, -b], [b, a]]. For every placement link (i, j, H_ij), we add
    the 4 linear equations encoding H_i = H_j @ H_ij. Anchor image is pinned
    to identity with a high weight. Solved with sparse LSQR.

    Returns: {image_idx: 3x3 similarity H} for every image that appears in any
    placement link. Images unreachable from placement links are omitted so the
    caller can keep their original GPS-fallback pose.
    """
    if not _SCIPY_AVAILABLE:
        raise RuntimeError("scipy not available")

    # Only solve for images participating in at least one link (+ anchor)
    link_images = set()
    for i, j, _ in placement_links:
        link_images.add(i)
        link_images.add(j)
    link_images.add(anchor_idx)
    indices = sorted(link_images)
    idx_to_var = {idx: k for k, idx in enumerate(indices)}
    n = len(indices)
    if n == 0:
        return {}

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    rhs: List[float] = []
    row = 0
    ANCHOR_WEIGHT = 1e3

    # Anchor: pose at anchor_idx pinned to identity (tx=0, ty=0, a=1, b=0)
    if anchor_idx in idx_to_var:
        ka = idx_to_var[anchor_idx]
        for j, target in ((0, 0.0), (1, 0.0), (2, 1.0), (3, 0.0)):
            rows.append(row)
            cols.append(4 * ka + j)
            vals.append(ANCHOR_WEIGHT)
            rhs.append(ANCHOR_WEIGHT * target)
            row += 1

    # Each placement link: H_i = H_j @ H_ij (similarity composition)
    # Let (ai, bi, txi, tyi) be pose_i and (aij, bij, txij, tyij) come from H_ij.
    # Composition yields:
    #   a_i  = aij·a_j - bij·b_j
    #   b_i  = bij·a_j + aij·b_j
    #   tx_i = tx_j + txij·a_j - tyij·b_j
    #   ty_i = ty_j + tyij·a_j + txij·b_j
    for i, j, H_ij in placement_links:
        if i not in idx_to_var or j not in idx_to_var:
            continue
        ki = idx_to_var[i]
        kj = idx_to_var[j]
        aij = float(H_ij[0, 0])
        bij = float(H_ij[1, 0])
        txij = float(H_ij[0, 2])
        tyij = float(H_ij[1, 2])

        # Eq 1 (a): a_i - aij*a_j + bij*b_j = 0
        rows.extend([row, row, row])
        cols.extend([4 * ki + 2, 4 * kj + 2, 4 * kj + 3])
        vals.extend([1.0, -aij, bij])
        rhs.append(0.0)
        row += 1

        # Eq 2 (b): b_i - bij*a_j - aij*b_j = 0
        rows.extend([row, row, row])
        cols.extend([4 * ki + 3, 4 * kj + 2, 4 * kj + 3])
        vals.extend([1.0, -bij, -aij])
        rhs.append(0.0)
        row += 1

        # Eq 3 (tx): tx_i - tx_j - txij*a_j + tyij*b_j = 0
        rows.extend([row, row, row, row])
        cols.extend([4 * ki + 0, 4 * kj + 0, 4 * kj + 2, 4 * kj + 3])
        vals.extend([1.0, -1.0, -txij, tyij])
        rhs.append(0.0)
        row += 1

        # Eq 4 (ty): ty_i - ty_j - tyij*a_j - txij*b_j = 0
        rows.extend([row, row, row, row])
        cols.extend([4 * ki + 1, 4 * kj + 1, 4 * kj + 2, 4 * kj + 3])
        vals.extend([1.0, -1.0, -tyij, -txij])
        rhs.append(0.0)
        row += 1

    if row == 0:
        return {}

    A = csr_matrix((vals, (rows, cols)), shape=(row, 4 * n))
    b_vec = np.asarray(rhs, dtype=np.float64)

    sol = lsqr(A, b_vec, atol=1e-10, btol=1e-10, iter_lim=4000)
    x = sol[0]

    new_poses: Dict[int, np.ndarray] = {}
    for idx, k in idx_to_var.items():
        tx, ty, a, b = x[4 * k], x[4 * k + 1], x[4 * k + 2], x[4 * k + 3]
        H = np.array([[a, -b, tx],
                      [b,  a, ty],
                      [0,  0, 1.0]], dtype=np.float64)
        new_poses[idx] = H
    return new_poses


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def stitch_geolocated_images(images: List[np.ndarray],
                              coordinates: List[Tuple[float, float]],
                              match_threshold: int = 10,
                              skip_assembly: bool = False):
    """
    GPS-guided SIFT stitcher using full 8-DOF homography, Voronoi seam
    finding, and Laplacian pyramid blending.

    Returns (canvas, placed_indices, placed_info).
    """
    if not images:
        return None, [], {}

    num_images = len(images)
    PPM = 30.0
    ref_lat = coordinates[0][0]

    # Convert GPS coords to metric offsets from first image
    R = 6378137.0
    lat0, lon0 = _map_radians(coordinates[0][0]), _map_radians(coordinates[0][1])
    positions = []
    for lat, lon in coordinates:
        lr, lor = _map_radians(lat), _map_radians(lon)
        positions.append(((lor - lon0) * R * np.cos(lat0), (lr - lat0) * R))

    placed_info: Dict = {0: {"H": np.eye(3, dtype=np.float64), "pos": positions[0]}}
    placed_indices = [0]
    cell_size = NEIGHBOR_RADIUS_M
    placed_grid: Dict[Tuple[int, int], List[int]] = {}
    _grid_add(placed_grid, 0, positions[0], cell_size)

    # --- PARALLEL feature extraction ---
    print(f"Pre-computing SIFT features ({NUM_WORKERS} workers)...")
    img_features = [None] * num_images
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for i, feat in executor.map(_extract_one, enumerate(images)):
            img_features[i] = feat
            if (i + 1) % 50 == 0:
                print(f"  Extracted {i+1}/{num_images}")

    unplaced = set(range(1, num_images))
    pass_num = 1
    # Collect (i, anchor, H_i_to_anchor) from successful placements for refinement
    placement_links: List[Tuple[int, int, np.ndarray]] = []

    while unplaced:
        print(f"--- Pass {pass_num} (unplaced: {len(unplaced)}) ---")
        placed_in_pass = 0

        for i in sorted(unplaced):
            img_to_add = images[i]
            kp_s, des_s, kp_a, des_a = img_features[i]
            h_n, w_n = img_to_add.shape[:2]

            # GPS fallback for feature-poor images
            if des_s is None or len(kp_s) < max(4, match_threshold // 2):
                print(f"  Image {i}: poor features, GPS fallback.")
                nearest, _ = _nearest_placed(i, placed_indices, positions)
                if nearest is not None:
                    exp_dx, exp_dy = _get_gps_pixel_delta(coordinates[nearest], coordinates[i], ref_lat, PPM)
                    T = np.array([[1, 0, exp_dx], [0, 1, exp_dy], [0, 0, 1]], dtype=np.float64)
                    placed_H = placed_info[nearest]["H"].dot(T)
                else:
                    placed_H = np.eye(3, dtype=np.float64)
                placed_indices.append(i)
                placed_info[i] = {"H": placed_H, "pos": positions[i], "gps_fallback": True}
                unplaced.remove(i)
                placed_in_pass += 1
                _grid_add(placed_grid, i, positions[i], cell_size)
                continue

            # Find GPS-close placed images as anchor candidates
            potential_anchors = []
            for idx in _grid_candidates(placed_grid, positions[i], cell_size):
                dist_m = np.hypot(positions[i][0] - positions[idx][0],
                                  positions[i][1] - positions[idx][1])
                if dist_m < NEIGHBOR_RADIUS_M:
                    potential_anchors.append((idx, dist_m))
            if not potential_anchors:
                for adj in (i - 1, i + 1):
                    if adj in placed_info:
                        dist_m = np.hypot(positions[i][0] - positions[adj][0],
                                          positions[i][1] - positions[adj][1])
                        if dist_m < NEIGHBOR_RADIUS_M * 1.5:
                            potential_anchors.append((adj, dist_m))
            potential_anchors.sort(key=lambda x: x[1])

            best_H_to_map = None
            best_anchor = None
            best_H_rel = None
            max_inliers = -1

            # Stricter inlier floor in pass 1 — sparse anchor graph means one bad
            # seed can propagate to many images; loosen in later passes for harder datasets.
            pass_min = MIN_PLACEMENT_INLIERS_PASS1 if pass_num == 1 else MIN_PLACEMENT_INLIERS

            for idx, _ in potential_anchors[:MAX_ANCHORS]:
                if img_features[idx][1] is None:  # des_s of anchor
                    continue

                src_pts, dst_pts = _match_pair(img_features[i], img_features[idx])
                if src_pts is None or len(src_pts) < max(pass_min, match_threshold // 2):
                    continue

                # --- Similarity transform (rotation + uniform scale + translation) ---
                # estimateAffinePartial2D prevents shear/non-uniform-scale accumulation
                # in long chains — the correct model for nadir aerial imagery.
                M_sim, mask = cv2.estimateAffinePartial2D(
                    src_pts, dst_pts, method=cv2.RANSAC,
                    ransacReprojThreshold=RANSAC_REPROJ)
                if M_sim is None or mask is None:
                    continue
                inliers = int(mask.sum())
                if inliers < max(pass_min, match_threshold // 2):
                    continue
                H_rel = np.vstack([M_sim, [0, 0, 1]])  # 2×3 → 3×3, no projective row

                # Area-ratio sanity check
                area_ratio = _homography_area_ratio(H_rel, w_n, h_n)
                if area_ratio < 0.3 or area_ratio > 3.0:
                    continue

                # Rotation sanity from upper-left 2×2
                try:
                    rot_deg = abs(np.degrees(np.arctan2(H_rel[1, 0], H_rel[0, 0])))
                except Exception:
                    rot_deg = 0.0
                if rot_deg > 20.0:
                    continue

                # GPS direction / magnitude sanity check
                center_new = np.float32([[[w_n / 2, h_n / 2]]])
                proj_center = cv2.perspectiveTransform(center_new, H_rel)[0][0]
                h_ref, w_ref = images[idx].shape[:2]
                vec_sift = np.array([proj_center[0] - w_ref / 2,
                                     proj_center[1] - h_ref / 2])
                exp_dx, exp_dy = _get_gps_pixel_delta(coordinates[idx], coordinates[i],
                                                       ref_lat, PPM)
                vec_gps = np.array([exp_dx, exp_dy])
                mag_sift = np.linalg.norm(vec_sift)
                mag_gps = np.linalg.norm(vec_gps)
                if mag_gps > GPS_MIN_DIST_M * PPM:
                    cos_sim = np.dot(vec_sift, vec_gps) / (mag_sift * mag_gps + 1e-10)
                    if cos_sim < GPS_DIR_COS_MIN:
                        continue
                    if mag_sift < GPS_MAG_MIN * mag_gps or mag_sift > GPS_MAG_MAX * mag_gps:
                        continue

                # GPS absolute-position check — hard reject if SIFT placement
                # deviates more than GPS_ABS_MAX_M from GPS truth.
                # No re-anchor: bad rotation/scale from chain drift is not recoverable.
                H_cand = placed_info[idx]["H"].dot(H_rel)
                H_cand_norm = H_cand.copy()
                H_cand_norm[2, 0] = 0.0; H_cand_norm[2, 1] = 0.0
                if abs(H_cand_norm[2, 2]) > 1e-10:
                    H_cand_norm /= H_cand_norm[2, 2]
                actual_c = cv2.perspectiveTransform(
                    np.float32([[[w_n / 2, h_n / 2]]]), H_cand_norm)[0][0]
                h0, w0 = images[0].shape[:2]
                abs_dx, abs_dy = _get_gps_pixel_delta(
                    coordinates[0], coordinates[i], ref_lat, PPM)
                gps_c = np.array([w0 / 2 + abs_dx, h0 / 2 + abs_dy])
                gps_abs_dev_m = np.linalg.norm(actual_c - gps_c) / PPM
                if gps_abs_dev_m > GPS_ABS_MAX_M:
                    if inliers < GPS_ABS_CORRECT_MIN_INLIERS:
                        continue  # Too few inliers to trust rotation/scale — reject
                    # Translation-only correction: trust SIFT rotation/scale, snap position to GPS.
                    # Only applied when inliers >= 30 — enough to trust the local pair geometry.
                    corr = gps_c - actual_c
                    T_corr = np.eye(3, dtype=np.float64)
                    T_corr[0, 2] = corr[0]
                    T_corr[1, 2] = corr[1]
                    H_cand_norm = T_corr.dot(H_cand_norm)

                if inliers > max_inliers:
                    max_inliers = inliers
                    # H_cand_norm already has projective stripped and GPS correction applied if needed
                    best_H_to_map = H_cand_norm
                    best_anchor = idx
                    best_H_rel = H_rel

            if best_H_to_map is not None:
                print(f"  Image {i} placed (anchor={best_anchor}, inliers={max_inliers})")
                placed_indices.append(i)
                placed_info[i] = {"H": best_H_to_map}
                unplaced.remove(i)
                placed_in_pass += 1
                _grid_add(placed_grid, i, positions[i], cell_size)
                # Record for refinement: H_rel maps i → anchor
                placement_links.append((i, best_anchor, best_H_rel))
                if i < 10 or placed_in_pass % PREVIEW_EVERY == 0:
                    _save_preview(images, placed_indices, placed_info)

        if placed_in_pass == 0:
            print("No more images could be linked. Terminating passes.")
            break
        print(f"Finished Pass {pass_num}: placed {placed_in_pass} images.")
        pass_num += 1

    # GPS fallback for remaining unplaced
    if unplaced:
        print(f"GPS fallback for {len(unplaced)} remaining images...")
        for i in sorted(unplaced):
            nearest, _ = _nearest_placed(i, placed_indices, positions)
            if nearest is not None:
                exp_dx, exp_dy = _get_gps_pixel_delta(coordinates[nearest], coordinates[i],
                                                        ref_lat, PPM)
                T = np.array([[1, 0, exp_dx], [0, 1, exp_dy], [0, 0, 1]], dtype=np.float64)
                placed_H = placed_info[nearest]["H"].dot(T)
            else:
                placed_H = np.eye(3, dtype=np.float64)
            placed_indices.append(i)
            placed_info[i] = {"H": placed_H, "pos": positions[i], "gps_fallback": True}
            print(f"  Image {i} placed by GPS fallback.")

    # --- GLOBAL REFINEMENT ---
    # Direct linear least-squares solve over similarity poses.
    # Falls back to the legacy iterative averaging if scipy is unavailable
    # or the solve fails for any reason.
    solved_via_lsqr = False
    if _SCIPY_AVAILABLE and placement_links and placed_indices:
        anchor_idx = placed_indices[0]
        print(f"Solving {len(placement_links)} similarity constraints via LSQR (anchor={anchor_idx})...")
        t_solve = time.time()
        try:
            new_poses = _solve_poses_lsqr(placed_info, placement_links, anchor_idx=anchor_idx)
            updated = 0
            for idx, H_new in new_poses.items():
                if idx in placed_info and _valid_H(H_new):
                    placed_info[idx]["H"] = H_new
                    updated += 1
            print(f"  LSQR solved {updated}/{len(new_poses)} poses in {time.time() - t_solve:.2f}s")
            solved_via_lsqr = True
        except Exception as e:
            print(f"  LSQR pose solve failed ({e}); falling back to iterative refinement.")

    if not solved_via_lsqr:
        print(f"Refining poses with {len(placement_links)} placement constraints (iterative)...")
        for iteration in range(REFINE_MAX_ITERS):
            total_delta, count_delta = 0.0, 0
            for i, j, H_ij in placement_links:
                if i not in placed_info or j not in placed_info:
                    continue
                H_i = placed_info[i]["H"]
                H_j = placed_info[j]["H"]

                try:
                    target_i = H_j.dot(H_ij)
                    new_i = (1 - REFINE_STEP) * H_i + REFINE_STEP * target_i
                    if abs(new_i[2, 2]) > 1e-10:
                        new_i /= new_i[2, 2]
                    if _valid_H(new_i):
                        total_delta += float(np.mean(np.abs(new_i - H_i)))
                        count_delta += 1
                        placed_info[i]["H"] = new_i

                    H_ji = np.linalg.inv(H_ij)
                    target_j = placed_info[i]["H"].dot(H_ji)
                    new_j = (1 - REFINE_STEP) * H_j + REFINE_STEP * target_j
                    if abs(new_j[2, 2]) > 1e-10:
                        new_j /= new_j[2, 2]
                    if _valid_H(new_j):
                        total_delta += float(np.mean(np.abs(new_j - H_j)))
                        count_delta += 1
                        placed_info[j]["H"] = new_j
                except (np.linalg.LinAlgError, Exception):
                    continue

            avg_delta = (total_delta / count_delta) if count_delta > 0 else 0.0
            if (iteration + 1) % 10 == 0:
                print(f"  Refinement iter {iteration+1}/{REFINE_MAX_ITERS} (avg Δ={avg_delta:.6f})")
            if avg_delta < REFINE_EARLY_STOP:
                print(f"  Early stop at iter {iteration+1} (avg Δ={avg_delta:.6f})")
                break

    # --- GPS FALLBACK RE-ALIGNMENT ---
    print("Attempting re-alignment of GPS fallback images...")
    gps_fallbacks = [idx for idx in placed_indices
                     if placed_info.get(idx, {}).get("gps_fallback", False)]
    MIN_LOCAL_INLIERS = 12
    realignment_edges = []  # (idx, j) from successful GPS re-alignment
    for idx in gps_fallbacks:
        neighs = sorted(
            [(j, np.hypot(positions[idx][0] - positions[j][0],
                          positions[idx][1] - positions[j][1]))
             for j in placed_indices if j != idx],
            key=lambda x: x[1]
        )[:6]
        for j, _ in neighs:
            src_pts, dst_pts = _match_pair(img_features[idx], img_features[j])
            if src_pts is None or len(src_pts) < MIN_LOCAL_INLIERS:
                continue
            M_sim, mask = cv2.estimateAffinePartial2D(
                src_pts, dst_pts, method=cv2.RANSAC,
                ransacReprojThreshold=RANSAC_REPROJ)
            if M_sim is None or mask is None or mask.sum() < MIN_LOCAL_INLIERS:
                continue
            H_rel = np.vstack([M_sim, [0, 0, 1]])  # 2×3 → 3×3, no projective row
            h_idx, w_idx = images[idx].shape[:2]
            if not (0.5 < _homography_area_ratio(H_rel, w_idx, h_idx) < 2.0):
                continue
            H_composed = placed_info[j]["H"].dot(H_rel)
            # Strip accumulated projective terms from composition — GPS fallback images
            # have no real perspective distortion; H[2,0]/H[2,1] are numerical artifacts
            # from H_neighbor's projective row being multiplied through the dot product.
            H_composed[2, 0] = 0.0
            H_composed[2, 1] = 0.0
            if abs(H_composed[2, 2]) > 1e-10:
                H_composed /= H_composed[2, 2]
            placed_info[idx]["H"] = H_composed
            placed_info[idx].pop("gps_fallback", None)
            print(f"  Re-aligned GPS fallback {idx} to neighbor {j} "
                  f"(inliers={int(mask.sum())})")
            realignment_edges.append((idx, j))
            break

    # --- BRIDGE MATCHING ---
    # Build full connectivity graph: SIFT links + GPS re-alignment links
    all_edges = [(i, j) for (i, j, _) in placement_links] + realignment_edges
    adj = {i: set() for i in placed_indices}
    for a, b in all_edges:
        if a in adj and b in adj:
            adj[a].add(b)
            adj[b].add(a)

    # BFS from image 0 to find main component
    main_comp = set()
    queue = [placed_indices[0]]
    while queue:
        node = queue.pop()
        if node in main_comp:
            continue
        main_comp.add(node)
        for nb in adj.get(node, set()):
            if nb not in main_comp:
                queue.append(nb)

    secondary_all = [i for i in placed_indices if i not in main_comp]
    if secondary_all:
        print(f"Bridge matching: {len(secondary_all)} images outside main component...")
        # Group secondary images into sub-components via BFS
        sec_visited = set()
        sec_components = []
        for start in secondary_all:
            if start in sec_visited:
                continue
            comp = []
            q = [start]
            sec_visited.add(start)
            while q:
                node = q.pop()
                comp.append(node)
                for nb in adj.get(node, set()):
                    if nb not in sec_visited:
                        sec_visited.add(nb)
                        q.append(nb)
            sec_components.append(comp)

        BRIDGE_RADIUS_PX = NEIGHBOR_RADIUS_M * PPM * 3
        BRIDGE_MIN_INLIERS = 20
        BRIDGE_MAX_SCALE_C = 3.0   # reject bridges requiring >3x scale correction

        for comp_idx, sec_comp in enumerate(sec_components):
            sec_set = set(sec_comp)
            bridge_candidates = []
            for b in sec_comp:
                for a in main_comp:
                    dist = np.hypot(positions[b][0] - positions[a][0],
                                    positions[b][1] - positions[a][1])
                    if dist < BRIDGE_RADIUS_PX:
                        bridge_candidates.append((a, b, dist))
            bridge_candidates.sort(key=lambda x: x[2])

            bridge_found = False
            for a, b, _ in bridge_candidates[:30]:
                src_pts, dst_pts = _match_pair(img_features[b], img_features[a])
                if src_pts is None or len(src_pts) < BRIDGE_MIN_INLIERS:
                    continue
                M_sim, mask = cv2.estimateAffinePartial2D(
                    src_pts, dst_pts, method=cv2.RANSAC,
                    ransacReprojThreshold=RANSAC_REPROJ)
                if M_sim is None or mask is None or mask.sum() < BRIDGE_MIN_INLIERS:
                    continue
                H_rel = np.vstack([M_sim, [0, 0, 1]])
                h_b, w_b = images[b].shape[:2]
                # Wider area ratio — bridge may span large scale difference
                if not (0.1 < _homography_area_ratio(H_rel, w_b, h_b) < 10.0):
                    continue
                # C maps secondary component into main:
                # new_H_b = H_a · H_rel ; C = new_H_b · inv(old_H_b)
                new_H_b = placed_info[a]["H"].dot(H_rel)
                new_H_b[2, 0] = 0.0
                new_H_b[2, 1] = 0.0
                if abs(new_H_b[2, 2]) > 1e-10:
                    new_H_b /= new_H_b[2, 2]
                old_H_b = placed_info[b]["H"]
                try:
                    C = new_H_b.dot(np.linalg.inv(old_H_b))
                except np.linalg.LinAlgError:
                    continue
                C[2, 0] = 0.0
                C[2, 1] = 0.0
                if abs(C[2, 2]) > 1e-10:
                    C /= C[2, 2]
                scale_C = np.linalg.norm(C[:2, 0])
                # Reject bridges requiring an implausible scale correction —
                # scale_C >> 1 means the images don't actually overlap.
                if not (1.0 / BRIDGE_MAX_SCALE_C < scale_C < BRIDGE_MAX_SCALE_C):
                    continue
                print(f"  Bridge (comp {comp_idx}): img {b} -> img {a}, "
                      f"inliers={int(mask.sum())}, scale_C={scale_C:.3f}")
                main_comp.update(sec_set)
                for i in sec_comp:
                    H_new = C.dot(placed_info[i]["H"])
                    H_new[2, 0] = 0.0
                    H_new[2, 1] = 0.0
                    if abs(H_new[2, 2]) > 1e-10:
                        H_new /= H_new[2, 2]
                    placed_info[i]["H"] = H_new
                bridge_found = True
                break

            if not bridge_found:
                print(f"  No bridge for secondary component {comp_idx} "
                      f"({len(sec_comp)} images)")

    # Remove oblique GPS fallbacks
    ROT_DELETE_DEG = 12.0
    for idx in list(placed_indices):
        info = placed_info.get(idx, {})
        if not info.get("gps_fallback", False):
            continue
        try:
            lin = info["H"][:2, :2]
            rot_deg = abs(np.degrees(np.arctan2(lin[1, 0], lin[0, 0])))
            avg_scale = (np.linalg.norm(lin[:, 0]) + np.linalg.norm(lin[:, 1])) / 2.0
            ortho_err = np.linalg.norm(lin.T.dot(lin) - np.eye(2) * avg_scale ** 2)
            if rot_deg > ROT_DELETE_DEG or ortho_err > 0.6:
                print(f"  Removing oblique GPS fallback {idx} (rot={rot_deg:.1f}°)")
                placed_indices.remove(idx)
                placed_info.pop(idx, None)
        except Exception:
            continue

    if skip_assembly:
        return None, placed_indices, placed_info

    # --- SSIM seam consistency diagnostic ---
    compute_seam_ssim(images, placed_indices, placed_info, positions)

    print("Assembling final map...")
    canvas = _assemble_blended_map(images, placed_indices, placed_info)
    return canvas, placed_indices, placed_info


# ---------------------------------------------------------------------------
# SSIM seam consistency diagnostic
# ---------------------------------------------------------------------------

def compute_seam_ssim(images, placed_indices, placed_info, positions, sample_pairs=50):
    """
    Diagnostic: compute SSIM over overlapping image pairs and print a summary.
    Does not affect stitching output.
    """
    SEAM_MAX_DIM = 800
    results = []
    pairs_checked = 0

    for ii in range(len(placed_indices)):
        if pairs_checked >= sample_pairs:
            break
        i = placed_indices[ii]
        for jj in range(ii + 1, len(placed_indices)):
            if pairs_checked >= sample_pairs:
                break
            j = placed_indices[jj]
            dist = np.hypot(positions[i][0] - positions[j][0],
                            positions[i][1] - positions[j][1])
            if dist > NEIGHBOR_RADIUS_M:
                continue
            pairs_checked += 1

            img1, img2 = images[i], images[j]
            try:
                H_ij = np.linalg.inv(placed_info[i]["H"]).dot(placed_info[j]["H"])
            except np.linalg.LinAlgError:
                continue

            h, w = img1.shape[:2]
            scale = min(1.0, SEAM_MAX_DIM / max(h, w))
            if scale < 1.0:
                img1s = cv2.resize(img1, None, fx=scale, fy=scale)
                img2s = cv2.resize(img2, None, fx=scale, fy=scale)
                S = np.diag([scale, scale, 1.0])
                H_s = S.dot(H_ij).dot(np.linalg.inv(S))
                h, w = img1s.shape[:2]
            else:
                img1s, img2s, H_s = img1, img2, H_ij

            try:
                warped2 = cv2.warpPerspective(img2s, np.linalg.inv(H_s), (w, h))
            except Exception:
                continue

            g1 = cv2.cvtColor(img1s, cv2.COLOR_BGR2GRAY).astype(np.float32)
            g2 = cv2.cvtColor(warped2, cv2.COLOR_BGR2GRAY).astype(np.float32)
            mask = (g1 > 5) & (g2 > 5)
            if mask.sum() < 100:
                continue

            g1[~mask] = 0
            g2[~mask] = 0
            ksize, sigma = (11, 11), 1.5
            mu1 = cv2.GaussianBlur(g1, ksize, sigma)
            mu2 = cv2.GaussianBlur(g2, ksize, sigma)
            s1sq = cv2.GaussianBlur(g1 * g1, ksize, sigma) - mu1 * mu1
            s2sq = cv2.GaussianBlur(g2 * g2, ksize, sigma) - mu2 * mu2
            s12  = cv2.GaussianBlur(g1 * g2, ksize, sigma) - mu1 * mu2
            C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
            ssim_map = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / (
                (mu1 ** 2 + mu2 ** 2 + C1) * (s1sq + s2sq + C2) + 1e-12)
            ssim_val = float(np.mean(ssim_map[mask]))
            results.append((ssim_val, i, j))

    if results:
        results.sort()
        mean_ssim = sum(r[0] for r in results) / len(results)
        print(f"[SSIM] Checked {len(results)} pairs — mean={mean_ssim:.3f}")
        print(f"[SSIM] Worst 5 pairs:")
        for ssim_val, i, j in results[:5]:
            print(f"  imgs {i}\u2194{j}: SSIM={ssim_val:.3f}")


# ---------------------------------------------------------------------------
# Assembly: content-aware seam finding + Laplacian pyramid blending
# ---------------------------------------------------------------------------

def _warp_and_dist(args):
    """Warp one image and compute its coverage distance map (runs in thread pool)."""
    idx, img, H_canvas, canvas_w, canvas_h = args
    warped = cv2.warpPerspective(img, H_canvas, (canvas_w, canvas_h),
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = np.any(warped > 10, axis=2).astype(np.uint8)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    return dist  # return dist only; warped is re-computed in phase 2


def _warp_at_scale(args):
    """Warp an image + mask at a downsampled canvas scale, return cropped patch + corner."""
    img, H_canvas, out_w, out_h, scale = args
    S = np.diag([scale, scale, 1.0])
    H_scaled = S.dot(H_canvas)
    warped = cv2.warpPerspective(img, H_scaled, (out_w, out_h),
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = np.any(warped > 5, axis=2).astype(np.uint8) * 255
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None, None, None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return warped[y1:y2, x1:x2].copy(), mask[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def _graphcut_seam_masks(images, placed_indices, H_canvases,
                         canvas_w, canvas_h, seam_scale=SEAM_SCALE):
    """
    Content-aware per-image seam masks using cv2.detail.GraphCutSeamFinder.

    Runs at a downsampled canvas scale for memory/speed, then upsamples the
    resulting binary masks back to full canvas resolution.

    Returns: list of uint8 masks (canvas_h x canvas_w), one per placed image,
             255 where that image "wins" the seam. None on failure so caller
             can fall back to the Voronoi label map.
    """
    try:
        # Availability probe
        _ = cv2.detail.GraphCutSeamFinder("COST_COLOR_GRAD")
    except (AttributeError, cv2.error) as e:
        print(f"  GraphCutSeamFinder unavailable ({e}); falling back to Voronoi.")
        return None

    seam_w = max(1, int(canvas_w * seam_scale))
    seam_h = max(1, int(canvas_h * seam_scale))

    warp_args = [(images[idx], H_canvases[i], seam_w, seam_h, seam_scale)
                 for i, idx in enumerate(placed_indices)]

    warps, masks, regions = [], [], []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for w, m, r in executor.map(_warp_at_scale, warp_args):
            warps.append(w)
            masks.append(m)
            regions.append(r)

    valid_ks = [k for k, w in enumerate(warps) if w is not None]
    if len(valid_ks) < 2:
        return None

    src_list = [warps[k].astype(np.float32) for k in valid_ks]
    mask_list = [masks[k].copy() for k in valid_ks]
    corners = [(int(regions[k][0]), int(regions[k][1])) for k in valid_ks]

    try:
        finder = cv2.detail.GraphCutSeamFinder("COST_COLOR_GRAD")
        umat_src = [cv2.UMat(s) for s in src_list]
        umat_masks = [cv2.UMat(m) for m in mask_list]
        finder.find(umat_src, corners, umat_masks)
        out_masks = [u.get() for u in umat_masks]
    except (cv2.error, Exception) as e:
        print(f"  GraphCutSeamFinder.find failed ({e}); falling back to Voronoi.")
        return None

    # Place cropped masks back into their seam-scale positions, then upsample
    full_masks: List[np.ndarray] = [np.zeros((canvas_h, canvas_w), dtype=np.uint8)
                                     for _ in placed_indices]
    for local_k, orig_k in enumerate(valid_ks):
        region = regions[orig_k]
        if region is None:
            continue
        x1, y1, x2, y2 = region
        small_canvas = np.zeros((seam_h, seam_w), dtype=np.uint8)
        small_canvas[y1:y2, x1:x2] = out_masks[local_k]
        full_masks[orig_k] = cv2.resize(small_canvas, (canvas_w, canvas_h),
                                         interpolation=cv2.INTER_NEAREST)
    return full_masks


def _compute_overlap_gains(images, placed_indices, H_canvases,
                           canvas_w, canvas_h, gain_scale=GAIN_SCALE):
    """
    Per-image exposure gains from cv2.detail.GainCompensator on downsampled warps.

    Returns np.ndarray of length len(placed_indices) with one scalar gain per
    image, clipped to GAIN_CLIP. Falls back to legacy mean-brightness gains on
    any failure.
    """
    n = len(placed_indices)
    fallback_gains = np.ones(n, dtype=np.float32)

    try:
        gain_w = max(1, int(canvas_w * gain_scale))
        gain_h = max(1, int(canvas_h * gain_scale))
        warp_args = [(images[idx], H_canvases[i], gain_w, gain_h, gain_scale)
                     for i, idx in enumerate(placed_indices)]
        warps, masks, regions = [], [], []
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            for w, m, r in executor.map(_warp_at_scale, warp_args):
                warps.append(w)
                masks.append(m)
                regions.append(r)

        valid_ks = [k for k, w in enumerate(warps) if w is not None]
        if len(valid_ks) < 2:
            return fallback_gains

        src_list = [warps[k].astype(np.uint8) for k in valid_ks]
        mask_list = [masks[k] for k in valid_ks]
        corners = [(int(regions[k][0]), int(regions[k][1])) for k in valid_ks]

        try:
            compensator = cv2.detail.GainCompensator(1)
        except (AttributeError, TypeError):
            compensator = cv2.detail_GainCompensator(1)

        umat_src = [cv2.UMat(s) for s in src_list]
        umat_masks = [cv2.UMat(m) for m in mask_list]
        compensator.feed(corners, umat_src, umat_masks)

        raw_gains = None
        try:
            raw_gains = compensator.gains()
            raw_gains = np.asarray(raw_gains, dtype=np.float32).flatten()
        except (AttributeError, cv2.error, TypeError):
            raw_gains = None

        if raw_gains is None or len(raw_gains) != len(valid_ks):
            # Probe with a uniform image: apply scales pixels by the gain
            raw_gains = np.ones(len(valid_ks), dtype=np.float32)
            probe_base = 128
            for local_k, orig_k in enumerate(valid_ks):
                probe = np.full((4, 4, 3), probe_base, dtype=np.uint8)
                probe_mask = np.full((4, 4), 255, dtype=np.uint8)
                try:
                    compensator.apply(local_k, corners[local_k], probe, probe_mask)
                    raw_gains[local_k] = float(np.mean(probe[:, :, 0])) / float(probe_base)
                except (cv2.error, Exception):
                    raw_gains[local_k] = 1.0

        gains = np.ones(n, dtype=np.float32)
        for local_k, orig_k in enumerate(valid_ks):
            g = float(raw_gains[local_k])
            if not np.isfinite(g) or g <= 0:
                g = 1.0
            gains[orig_k] = float(np.clip(g, GAIN_CLIP[0], GAIN_CLIP[1]))
        return gains
    except Exception as e:
        print(f"  GainCompensator failed ({e}); falling back to mean-brightness gains.")
        return fallback_gains


def _assemble_blended_map(images, placed_indices, placed_info):
    """
    Assemble orthomosaic using:
    1. Voronoi seam assignment (distance-transform, parallel)
    2. Laplacian pyramid blending (smooth seam transitions, streaming)
    """
    # Compute canvas bounds
    all_pts = []
    for idx in placed_indices:
        h, w = images[idx].shape[:2]
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        all_pts.append(cv2.perspectiveTransform(pts, placed_info[idx]["H"]))
    all_pts = np.concatenate(all_pts)
    # Filter NaN/Inf from any degenerate homographies
    all_pts = all_pts[np.isfinite(all_pts).all(axis=2).squeeze()]
    if len(all_pts) == 0:
        return images[placed_indices[0]]
    xmin, ymin = np.int32(np.clip(all_pts.min(axis=0).ravel() - 0.5, -1e7, 1e7))
    xmax, ymax = np.int32(np.clip(all_pts.max(axis=0).ravel() + 0.5, -1e7, 1e7))

    canvas_w = int(xmax - xmin)
    canvas_h = int(ymax - ymin)
    if canvas_w <= 0 or canvas_h <= 0:
        return images[placed_indices[0]]

    scale = min(1.0, MAX_CANVAS_DIM / canvas_w, MAX_CANVAS_DIM / canvas_h)
    # Memory-adaptive scale: keep float32 Laplacian accumulator under BLEND_MEM_BUDGET
    raw_w = int((xmax - xmin) * scale)
    raw_h = int((ymax - ymin) * scale)
    # accum: LEVELS × H × W × 3 float32; weight_accum: LEVELS × H × W × 1 float32
    accum_bytes = raw_w * raw_h * LAPLACIAN_LEVELS * 4 * 4  # 3ch + 1ch weight
    if accum_bytes > BLEND_MEM_BUDGET:
        mem_scale = (BLEND_MEM_BUDGET / accum_bytes) ** 0.5
        scale *= mem_scale
    canvas_w = max(1, int((xmax - xmin) * scale))
    canvas_h = max(1, int((ymax - ymin) * scale))
    print(f"Canvas: {canvas_w}x{canvas_h} (scale={scale:.3f})")

    T = np.array([[scale, 0, -xmin * scale],
                  [0, scale, -ymin * scale],
                  [0, 0, 1]], dtype=np.float64)

    # Pre-compute canvas homographies for all images
    H_canvases = [T.dot(placed_info[idx]["H"]) for idx in placed_indices]

    # --- Phase 1: Content-aware seams (graph-cut; Voronoi fallback) ---
    print(f"Computing graph-cut seam masks (scale={SEAM_SCALE})...")
    t_seam = time.time()
    seam_masks = _graphcut_seam_masks(images, placed_indices, H_canvases,
                                       canvas_w, canvas_h, SEAM_SCALE)

    label_map = None
    if seam_masks is None:
        print(f"  Falling back to Voronoi seam labels ({NUM_WORKERS} workers)...")
        label_map = np.full((canvas_h, canvas_w), -1, dtype=np.int32)
        dist_map = np.zeros((canvas_h, canvas_w), dtype=np.float32)

        warp_args = [(idx, images[idx], H_canvases[i], canvas_w, canvas_h)
                     for i, idx in enumerate(placed_indices)]

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            for i, dist in enumerate(executor.map(_warp_and_dist, warp_args)):
                update = dist > dist_map
                dist_map[update] = dist[update]
                label_map[update] = i
        del dist_map
    else:
        print(f"  Graph-cut seams done in {time.time() - t_seam:.2f}s")

    # --- Gain compensation: overlap-based (falls back to mean-brightness) ---
    gains = np.ones(len(placed_indices), dtype=np.float32)
    if GAIN_COMPENSATION:
        print(f"Computing overlap-based gain compensation (scale={GAIN_SCALE})...")
        t_gain = time.time()
        gains = _compute_overlap_gains(images, placed_indices, H_canvases,
                                        canvas_w, canvas_h, GAIN_SCALE)
        print(f"  Gains computed in {time.time() - t_gain:.2f}s "
              f"(range {float(gains.min()):.3f}..{float(gains.max()):.3f})")

    # --- Phase 2: Laplacian pyramid blend (streaming accumulator) ---
    print(f"Building {LAPLACIAN_LEVELS}-level Laplacian pyramid accumulator...")

    # Determine level shapes by pyrDown-ing a dummy
    dummy = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    level_shapes = []
    cur = dummy
    for _ in range(LAPLACIAN_LEVELS):
        level_shapes.append(cur.shape)
        cur = cv2.pyrDown(cur)
    del dummy, cur

    # Allocate float32 accumulators for each pyramid level (3-channel)
    # and single-channel weight accumulators for normalization
    accum = [np.zeros((*sh, 3), dtype=np.float32) for sh in level_shapes]
    weight_accum = [np.zeros(sh, dtype=np.float32) for sh in level_shapes]

    for i, idx in enumerate(placed_indices):
        warped = cv2.warpPerspective(images[idx], H_canvases[i], (canvas_w, canvas_h),
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0
                                     ).astype(np.float32)

        # Apply per-image gain correction
        warped *= gains[i]

        # Feathered seam weight: Gaussian-blur the binary win-mask to soften edges
        if seam_masks is not None:
            seam_w = (seam_masks[i] > 0).astype(np.float32)
        else:
            seam_w = (label_map == i).astype(np.float32)
        if VORONOI_FEATHER_SIGMA > 0:
            seam_w = cv2.GaussianBlur(seam_w, (0, 0), float(VORONOI_FEATHER_SIGMA))

        # Build pyramids
        lp = _build_laplacian_pyramid(warped, LAPLACIAN_LEVELS, level_shapes)
        del warped
        gp_w = _build_gaussian_pyramid(seam_w, LAPLACIAN_LEVELS, level_shapes)
        del seam_w

        # Accumulate weighted Laplacian and weight sum for normalization
        for lvl in range(LAPLACIAN_LEVELS):
            lh, lw = level_shapes[lvl]
            w_2d = gp_w[lvl][:lh, :lw]
            accum[lvl] += lp[lvl][:lh, :lw] * w_2d[:, :, np.newaxis]
            weight_accum[lvl] += w_2d

        del lp, gp_w
        gc.collect()

        if (i + 1) % 50 == 0:
            print(f"  Blended {i+1}/{len(placed_indices)} images...")

    if label_map is not None:
        del label_map
    if seam_masks is not None:
        del seam_masks
    gc.collect()

    # Reconstruct from normalized Laplacian pyramid
    # Normalize each level: weighted_laplacian / weight_sum before reconstruction
    print("Reconstructing from pyramid...")
    result = accum[-1] / np.maximum(weight_accum[-1][:, :, np.newaxis], 1e-6)
    for lvl in range(LAPLACIAN_LEVELS - 2, -1, -1):
        th, tw = level_shapes[lvl]
        result = cv2.pyrUp(result, dstsize=(tw, th))
        norm = np.maximum(weight_accum[lvl][:th, :tw, np.newaxis], 1e-6)
        result += accum[lvl][:th, :tw] / norm

    filled = np.clip(result, 0, 255).astype(np.uint8)

    # Fill hairline black seam gaps left by warpPerspective edge interpolation
    coverage = np.any(filled > 10, axis=2).astype(np.uint8)
    kernel = np.ones((11, 11), np.uint8)
    dilated = cv2.dilate(coverage, kernel)
    gap_mask = ((dilated > 0) & (coverage == 0)).astype(np.uint8)
    if gap_mask.any():
        filled = cv2.inpaint(filled, gap_mask * 255, 5, cv2.INPAINT_TELEA)

    return filled


def _build_laplacian_pyramid(img, levels, level_shapes):
    """Build Laplacian pyramid, forcing each level to match level_shapes."""
    gp = [img]
    for _ in range(1, levels):
        gp.append(cv2.pyrDown(gp[-1]))
    lp = []
    for lvl in range(levels - 1):
        th, tw = level_shapes[lvl]
        up = cv2.pyrUp(gp[lvl + 1], dstsize=(tw, th))
        lp.append(gp[lvl][:th, :tw] - up[:th, :tw])
    lp.append(gp[-1])
    return lp


def _build_gaussian_pyramid(img, levels, level_shapes):
    """Build Gaussian pyramid, forcing each level to match level_shapes."""
    gp = [img]
    for _ in range(1, levels):
        gp.append(cv2.pyrDown(gp[-1]))
    return [g[:level_shapes[lvl][0], :level_shapes[lvl][1]] for lvl, g in enumerate(gp)]


# ---------------------------------------------------------------------------
# Preview helper
# ---------------------------------------------------------------------------

def _save_preview(images, placed_indices, placed_info):
    try:
        all_pts = []
        for idx in placed_indices:
            h, w = images[idx].shape[:2]
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            all_pts.append(cv2.perspectiveTransform(pts, placed_info[idx]["H"]))
        all_pts = np.concatenate(all_pts)
        xmin, ymin = np.int32(all_pts.min(axis=0).ravel() - 0.5)
        xmax, ymax = np.int32(all_pts.max(axis=0).ravel() + 0.5)
        pw, ph = xmax - xmin, ymax - ymin
        p_scale = 1200.0 / max(pw, ph) if max(pw, ph) > 1200 else 1.0
        tw, th = max(1, int(pw * p_scale)), max(1, int(ph * p_scale))
        H_pre = np.array([[p_scale, 0, -xmin * p_scale],
                          [0, p_scale, -ymin * p_scale],
                          [0, 0, 1]])
        preview = np.zeros((th, tw, 3), dtype=np.uint8)
        for idx in placed_indices:
            H_c = H_pre.dot(placed_info[idx]["H"])
            warped = cv2.warpPerspective(images[idx], H_c, (tw, th))
            preview[np.any(warped > 10, axis=2)] = warped[np.any(warped > 10, axis=2)]
        cv2.imwrite("stitch_progress.jpg", preview)
    except Exception:
        pass
