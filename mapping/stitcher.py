import cv2
import numpy as np
import gc
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor

MAX_CANVAS_DIM = 10000
NEIGHBOR_RADIUS_M = 150.0
MAX_ANCHORS = 8
RATIO_TEST = 0.75         # Lowe's recommended threshold; tighter reduces false matches in repetitive terrain
RANSAC_REPROJ = 6.0
SIFT_FEATURES = 8000
SIFT_CONTRAST = 0.005
CLAHE_CLIP = 2.0
CLAHE_TILE = 8
AKAZE_THRESH = 0.005
AKAZE_RATIO_TEST = 0.90
GPS_DIR_COS_MIN = 0.35    # Require SIFT-predicted direction within ~70° of GPS direction
GPS_MAG_MIN = 0.1
GPS_MAG_MAX = 2.5         # Reject if SIFT says image is >2.5x farther than GPS suggests
GPS_MIN_DIST_M = 2.0
GPS_ABS_MAX_M = 22.0              # GPS absolute position tolerance; beyond this, try to GPS-correct rather than reject
GPS_ABS_CORRECT_MIN_INLIERS = 30  # If inliers >= this and GPS abs fails, shift chain to GPS position instead of rejecting
MIN_PLACEMENT_INLIERS = 25        # Minimum RANSAC inliers for passes 2+ (lenient for harder datasets)
MIN_PLACEMENT_INLIERS_PASS1 = 50  # Stricter floor for pass 1 (sparse anchor graph = higher bad-seed risk)

PREVIEW_EVERY = 25
REFINE_MAX_ITERS = 200
REFINE_STEP = 0.3         # Larger step works now that initial poses are similarity transforms
REFINE_EARLY_STOP = 1e-5
LAPLACIAN_LEVELS = 7
BLEND_MEM_BUDGET = 2.0 * 1024 ** 3   # 2 GB cap for float32 Laplacian accumulator
NUM_WORKERS = 4           # Parallel workers for feature extraction and warping
GAIN_COMPENSATION = True  # Normalize per-image brightness before blending
GAIN_CLIP = (0.5, 2.0)    # Clamp gain to avoid over-correction on outlier exposures
VORONOI_FEATHER_SIGMA = 10  # 0 = hard Voronoi (binary weights); >0 = Gaussian feather radius in output pixels


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
    """Extract SIFT(primary) + AKAZE(fallback) features for one image."""
    i, img = args
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_TILE, CLAHE_TILE))
    enhanced = clahe.apply(gray)
    sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES, contrastThreshold=SIFT_CONTRAST)
    akaze = cv2.AKAZE_create(threshold=AKAZE_THRESH)
    kp_s, des_s = sift.detectAndCompute(enhanced, None)
    kp_a, des_a = akaze.detectAndCompute(enhanced, None)
    return i, (kp_s, des_s, kp_a, des_a)


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
    print(f"Pre-computing features (SIFT primary + AKAZE fallback, {NUM_WORKERS} workers)...")
    img_features = [None] * num_images
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for i, feat in executor.map(_extract_one, enumerate(images)):
            img_features[i] = feat
            if (i + 1) % 50 == 0:
                print(f"  Extracted {i+1}/{num_images}")

    bf_l2 = cv2.BFMatcher(cv2.NORM_L2)
    bf_ham = cv2.BFMatcher(cv2.NORM_HAMMING)
    unplaced = set(range(1, num_images))
    pass_num = 1
    # Collect (i, anchor, H_i_to_anchor) from successful placements for refinement
    placement_links: List[Tuple[int, int, np.ndarray]] = []

    def _try_link_candidate(
        i: int,
        idx: int,
        h_n: int,
        w_n: int,
        kp_new,
        des_new,
        kp_ref,
        des_ref,
        matcher,
        ratio_test: float,
        pass_min: int,
    ):
        if des_new is None or des_ref is None:
            return None
        if len(kp_new) < max(pass_min, match_threshold // 2):
            return None
        if len(kp_ref) < max(pass_min, match_threshold // 2):
            return None

        matches = matcher.knnMatch(des_new, des_ref, k=2)
        good = [m for m, n in matches if m.distance < ratio_test * n.distance]
        if len(good) < max(pass_min, match_threshold // 2):
            return None

        src_pts = np.float32([kp_new[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Similarity transform is still the primary model.
        M_sim, mask = cv2.estimateAffinePartial2D(
            src_pts, dst_pts, method=cv2.RANSAC,
            ransacReprojThreshold=RANSAC_REPROJ)
        if M_sim is None or mask is None:
            return None
        inliers = int(mask.sum())
        if inliers < max(pass_min, match_threshold // 2):
            return None
        H_rel = np.vstack([M_sim, [0, 0, 1]])  # 2×3 → 3×3, no projective row

        area_ratio = _homography_area_ratio(H_rel, w_n, h_n)
        if area_ratio < 0.3 or area_ratio > 3.0:
            return None

        try:
            rot_deg = abs(np.degrees(np.arctan2(H_rel[1, 0], H_rel[0, 0])))
        except Exception:
            rot_deg = 0.0
        if rot_deg > 20.0:
            return None

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
                return None
            if mag_sift < GPS_MAG_MIN * mag_gps or mag_sift > GPS_MAG_MAX * mag_gps:
                return None

        H_cand = placed_info[idx]["H"].dot(H_rel)
        H_cand_norm = H_cand.copy()
        H_cand_norm[2, 0] = 0.0
        H_cand_norm[2, 1] = 0.0
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
                return None  # Too few inliers to trust rotation/scale — reject
            corr = gps_c - actual_c
            T_corr = np.eye(3, dtype=np.float64)
            T_corr[0, 2] = corr[0]
            T_corr[1, 2] = corr[1]
            H_cand_norm = T_corr.dot(H_cand_norm)

        return H_cand_norm, H_rel, inliers

    while unplaced:
        print(f"--- Pass {pass_num} (unplaced: {len(unplaced)}) ---")
        placed_in_pass = 0

        for i in sorted(unplaced):
            img_to_add = images[i]
            kp_s_new, des_s_new, kp_a_new, des_a_new = img_features[i]
            h_n, w_n = img_to_add.shape[:2]

            # GPS fallback for feature-poor images
            sift_poor = (des_s_new is None or len(kp_s_new) < max(4, match_threshold // 2))
            akaze_poor = (des_a_new is None or len(kp_a_new) < max(4, match_threshold // 2))
            if sift_poor and akaze_poor:
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
                kp_s_ref, des_s_ref, kp_a_ref, des_a_ref = img_features[idx]

                # SIFT remains primary matching path.
                candidate = _try_link_candidate(
                    i, idx, h_n, w_n,
                    kp_s_new, des_s_new, kp_s_ref, des_s_ref,
                    bf_l2, RATIO_TEST, pass_min,
                )
                # AKAZE is only a conditional fallback if SIFT can't link this anchor.
                if candidate is None:
                    candidate = _try_link_candidate(
                        i, idx, h_n, w_n,
                        kp_a_new, des_a_new, kp_a_ref, des_a_ref,
                        bf_ham, AKAZE_RATIO_TEST, pass_min,
                    )
                if candidate is None:
                    continue

                H_cand_norm, H_rel, inliers = candidate
                if inliers > max_inliers:
                    max_inliers = inliers
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
    # Reuse already-computed placement H_rels (no extra SIFT matching needed).
    print(f"Refining poses with {len(placement_links)} placement constraints...")
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
            kp_s_new, des_s_new, kp_a_new, des_a_new = img_features[idx]
            kp_s_ref, des_s_ref, kp_a_ref, des_a_ref = img_features[j]

            # SIFT primary in re-alignment; AKAZE only if SIFT has no viable matches.
            good = []
            use_akaze = False
            if des_s_new is not None and des_s_ref is not None:
                matches = bf_l2.knnMatch(des_s_new, des_s_ref, k=2)
                good = [m for m, n in matches if m.distance < RATIO_TEST * n.distance]
            if len(good) < MIN_LOCAL_INLIERS and des_a_new is not None and des_a_ref is not None:
                matches = bf_ham.knnMatch(des_a_new, des_a_ref, k=2)
                good = [m for m, n in matches if m.distance < AKAZE_RATIO_TEST * n.distance]
                use_akaze = True
            if len(good) < MIN_LOCAL_INLIERS:
                continue

            if use_akaze:
                src_pts = np.float32([kp_a_new[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_a_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            else:
                src_pts = np.float32([kp_s_new[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_s_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
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
                kp_s_a, des_s_a, kp_a_a, des_a_a = img_features[a]
                kp_s_b, des_s_b, kp_a_b, des_a_b = img_features[b]

                # Bridge match uses SIFT first; AKAZE only when SIFT is insufficient.
                good = []
                use_akaze = False
                if des_s_a is not None and des_s_b is not None:
                    matches = bf_l2.knnMatch(des_s_b, des_s_a, k=2)
                    good = [m for m, n in matches if m.distance < RATIO_TEST * n.distance]
                if len(good) < BRIDGE_MIN_INLIERS and des_a_a is not None and des_a_b is not None:
                    matches = bf_ham.knnMatch(des_a_b, des_a_a, k=2)
                    good = [m for m, n in matches if m.distance < AKAZE_RATIO_TEST * n.distance]
                    use_akaze = True
                if len(good) < BRIDGE_MIN_INLIERS:
                    continue

                if use_akaze:
                    src_pts = np.float32([kp_a_b[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_a_a[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                else:
                    src_pts = np.float32([kp_s_b[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_s_a[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
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
            print(f"  imgs {i}↔{j}: SSIM={ssim_val:.3f}")


# ---------------------------------------------------------------------------
# Assembly: Voronoi seam finding + Laplacian pyramid blending
# ---------------------------------------------------------------------------

def _warp_and_dist(args):
    """Warp one image and compute its coverage distance map (runs in thread pool)."""
    idx, img, H_canvas, canvas_w, canvas_h = args
    warped = cv2.warpPerspective(img, H_canvas, (canvas_w, canvas_h),
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = np.any(warped > 10, axis=2).astype(np.uint8)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    return dist  # return dist only; warped is re-computed in phase 2


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

    # --- Phase 1: Build Voronoi label map (parallel warp+dist) ---
    print(f"Computing Voronoi seam labels ({NUM_WORKERS} workers)...")
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

    # --- Gain compensation: normalize per-image brightness to global median ---
    gains = np.ones(len(placed_indices), dtype=np.float32)
    if GAIN_COMPENSATION:
        brightnesses = []
        for idx in placed_indices:
            b = float(np.mean(images[idx].astype(np.float32)))
            brightnesses.append(b)
        target_brightness = float(np.median(brightnesses))
        for k, b in enumerate(brightnesses):
            if b > 1.0:
                gains[k] = float(np.clip(target_brightness / b,
                                         GAIN_CLIP[0], GAIN_CLIP[1]))

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

        # Feathered Voronoi weight: Gaussian-blur the binary mask to soften seam edges
        voronoi_w = (label_map == i).astype(np.float32)
        if VORONOI_FEATHER_SIGMA > 0:
            voronoi_w = cv2.GaussianBlur(voronoi_w, (0, 0), float(VORONOI_FEATHER_SIGMA))

        # Build pyramids
        lp = _build_laplacian_pyramid(warped, LAPLACIAN_LEVELS, level_shapes)
        del warped
        gp_w = _build_gaussian_pyramid(voronoi_w, LAPLACIAN_LEVELS, level_shapes)
        del voronoi_w

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

    del label_map

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
