"""Data augmentation: multi-stride + random stride sampling (Idea 1, enhanced).

GoogLeNet-style: fixed strides [1,2,3,5,7] for consistent multi-scale features.
Random stride: each window gets a random stride from U(1, 10) for diversity.
Window size variation: random jitter around target window size.

Key invariants:
  - All windows from the same case ALWAYS stay in the same CV fold (case_ids tracked)
  - Active region sampling avoids the equilibrium tail
  - Minority classes in task3/task4 get more windows per case
"""

import numpy as np


def _pick_region(active_region, stride):
    """Convert active_region (single or list) to scaled (start, end)."""
    if isinstance(active_region, list):
        region = active_region[np.random.randint(0, len(active_region))]
    else:
        region = active_region
    return max(0, region[0] // stride), min(9999, region[1] // stride)


def sample_windows_from_series(series, stride, window_size, active_region, n_windows):
    """Sample n_windows random crops from a single (downsampled) time series.

    active_region can be a single (start,end) tuple or a list of tuples
    for multi-region sampling (strategy 3: domain shift mitigation).
    """
    T, C = series.shape
    ar_start, ar_end = _pick_region(active_region, stride)
    ar_end = min(T, ar_end)
    valid_end = max(ar_start + window_size, ar_end)

    windows = []
    for _ in range(n_windows):
        # Re-pick region for diversity if multi-region
        if isinstance(active_region, list):
            ar_start, ar_end = _pick_region(active_region, stride)
            ar_end = min(T, ar_end)

    windows = []
    for _ in range(n_windows):
        t_max = max(ar_start, min(valid_end, T) - window_size)
        t0 = np.random.randint(ar_start, max(ar_start + 1, t_max)) if t_max > ar_start else 0
        t0 = min(t0, max(0, T - window_size))
        w = series[t0:t0 + window_size].copy()

        # Pad if needed
        if w.shape[0] < window_size:
            pad = np.zeros((window_size - w.shape[0], C), dtype=np.float32)
            w = np.concatenate([w, pad], axis=0)

        # Random window size jitter: resize then pad back to uniform
        if np.random.random() < 0.3 and w.shape[0] == window_size:
            jitter = int(window_size * np.random.uniform(-0.1, 0.1))
            new_ws = max(window_size // 2, window_size + jitter)
            if new_ws > window_size:
                w = w[:window_size]  # crop to target
            elif new_ws < window_size:
                offset = np.random.randint(0, window_size - new_ws)
                w = w[offset:offset + new_ws]
                pad_len = window_size - new_ws
                w = np.concatenate([w, np.zeros((pad_len, C), dtype=np.float32)], axis=0)

        windows.append((w, stride))
    return windows


def enhanced_sampling(X, y, fixed_strides, window_size, active_region,
                      n_fixed_per_stride, n_random_total, random_stride_range=(1, 10)):
    """Enhanced multi-stride + random stride + multi-region sampling.

    active_region can be:
      - (start, end) tuple → single region
      - list of (start, end) tuples → randomly pick one per window (strategy 3)
    """
    N, T, C = X.shape

    # Compute per-case window counts (more for minority)
    minority_mask = np.zeros(N, dtype=bool)
    for task_key in ["task3", "task4"]:
        labels = y[task_key]
        for cls in np.unique(labels):
            if cls != 0:
                minority_mask |= (labels == cls)

    majority_idx = np.where(~minority_mask)[0]
    minority_idx = np.where(minority_mask)[0]
    n_majority = len(majority_idx)
    n_minority = len(minority_idx)

    # Minority gets 2.5x more windows per source
    minority_fixed_mult = 3
    minority_random_mult = 3

    total_windows = (
        n_majority * (len(fixed_strides) * n_fixed_per_stride + n_random_total) +
        n_minority * (len(fixed_strides) * n_fixed_per_stride * minority_fixed_mult +
                      n_random_total * minority_random_mult)
    )

    print(f"  Enhanced sampling: {n_majority} majority cases × "
          f"({len(fixed_strides)}×{n_fixed_per_stride}+{n_random_total}) + "
          f"{n_minority} minority × "
          f"({len(fixed_strides)}×{n_fixed_per_stride*minority_fixed_mult}+{n_random_total*minority_random_mult})"
          f" = ~{total_windows} windows")

    X_list, case_ids, strides_used = [], [], []
    y_list = {k: [] for k in y}

    for i in range(N):
        is_minority = minority_mask[i]

        # 1. Fixed-stride sampling
        n_fixed = n_fixed_per_stride * (minority_fixed_mult if is_minority else 1)
        for stride in fixed_strides:
            xs = X[i, ::stride, :]
            windows = sample_windows_from_series(xs, stride, window_size,
                                                  active_region, n_fixed)
            for w, s in windows:
                X_list.append(w)
                strides_used.append(s)
                case_ids.append(i)

        # 2. Random-stride sampling
        n_rand = n_random_total * (minority_random_mult if is_minority else 1)
        for _ in range(n_rand):
            stride = np.random.randint(random_stride_range[0], random_stride_range[1] + 1)
            xs = X[i, ::stride, :]
            windows = sample_windows_from_series(xs, stride, window_size,
                                                  active_region, 1)
            for w, s in windows:
                X_list.append(w)
                strides_used.append(-s)  # negative = random stride
                case_ids.append(i)

        # Extend labels
        total_win = (len(fixed_strides) * n_fixed + n_rand)
        for k, v in y.items():
            y_list[k].extend([v[i]] * total_win)

    X_w = np.stack(X_list, axis=0).astype(np.float32)
    y_w = {k: np.array(v) for k, v in y_list.items()}
    case_ids = np.array(case_ids, dtype=np.int32)
    strides_used = np.array(strides_used, dtype=np.int32)

    return X_w, y_w, case_ids, strides_used


def create_training_dataset(X_train_norm, y_train, cfg):
    """One-call interface for training data construction."""

    # Check which version to use
    use_enhanced = cfg["data"].get("use_enhanced", True)

    if use_enhanced:
        fixed_strides = cfg["data"].get("fixed_strides", cfg["data"].get("strides", [1, 2, 3, 5, 7]))
        window_size = cfg["data"]["window_size"]
        ar = cfg["data"]["active_region"]
        active_region = [tuple(r) for r in ar] if (isinstance(ar, list) and isinstance(ar[0], list)) else tuple(ar)
        n_fixed = cfg["data"]["windows_per_case"]["majority"]
        n_random = cfg["data"].get("random_windows_per_case", 4)
        random_range = tuple(cfg["data"].get("random_stride_range", [1, 10]))

        return enhanced_sampling(
            X_train_norm, y_train,
            fixed_strides=fixed_strides,
            window_size=window_size,
            active_region=active_region,
            n_fixed_per_stride=n_fixed,
            n_random_total=n_random,
            random_stride_range=random_range,
        )
    else:
        # Legacy: fixed strides only, no random
        strides = cfg["data"]["strides"]
        window_size = cfg["data"]["window_size"]
        active_region = tuple(cfg["data"]["active_region"])
        n_windows_map = cfg["data"]["windows_per_case"]
        # Call enhanced_sampling with random=0
        return enhanced_sampling(
            X_train_norm, y_train,
            fixed_strides=strides,
            window_size=window_size,
            active_region=active_region,
            n_fixed_per_stride=n_windows_map["majority"],
            n_random_total=0,  # no random windows
            random_stride_range=(1, 1),
        )
