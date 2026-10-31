"""Run a model over all test windows; per-window, per-channel MSE scores.

The score uses only ``model.evaluation_output(batch, latent_mode=...)``
(see models/base.py), so it works for reconstruction and forecasting models
alike: score[window, channel] = mean over output steps of (output - target)^2,
in the stored (normalized, possibly log10) units of the dataset.

Reconstruction models return all T samples (time_index 0 ... T - 1, the
default). Forecasting models return their H predicted samples, the true
values at those samples and time_index = the window positions they predict
(T - H ... T - 1 for models.ForecastingModel).
"""

import numpy as np
import torch

from train.progress import BatchProgress

from .data import iter_values


def check_time_index(time_index, n_steps: int, window_size: int) -> np.ndarray:
    """The window positions of the output steps: 1-D integers, one per output
    step, strictly increasing, inside [0, T). None means 0 ... n_steps - 1."""
    if time_index is None:
        if n_steps != window_size:
            raise ValueError(f"evaluation_output returned {n_steps} of {window_size} samples without a time_index; "
                             "models that output fewer samples than the window must say which ones")
        return np.arange(n_steps)
    index = np.asarray(time_index.detach().cpu() if torch.is_tensor(time_index) else time_index)
    if index.ndim != 1 or index.dtype.kind not in "iu":
        raise ValueError(f"time_index must be a 1-D integer array, got shape {index.shape} dtype {index.dtype}")
    if index.size != n_steps:
        raise ValueError(f"time_index has {index.size} entries but the output has {n_steps} steps")
    if index.size and (index[0] < 0 or index[-1] >= window_size or np.any(np.diff(index) <= 0)):
        raise ValueError(f"time_index must be strictly increasing window positions in [0, {window_size}), "
                         f"got {index.tolist()}")
    return index.astype(np.int64)


def score_windows(model, test, *, batch_size, device, latent_mode, keep_rows=(), verbose=0):
    """Return (scores float64 (N, C), kept {row: {"input", "output", "target", "time_index"}}, time_index).

    time_index (int64 (T',)) is the same for every window: a model whose
    output positions change between batches is rejected, because its scores
    would not be comparable.

    "input" is the full (C, T) window the model received, so plots can show
    the context in front of a forecast.
    """
    keep_rows = set(int(row) for row in keep_rows)
    scores = np.full((test.n_windows, len(test.channel_ids)), np.nan)
    kept = {}
    common_index = None
    model.eval()
    live = BatchProgress("Scoring test windows", -(-test.n_windows // batch_size)) if verbose else None
    with torch.no_grad():
        for number, (first, values) in enumerate(iter_values(test.path, batch_size), start=1):
            rows = np.arange(first, first + len(values))
            batch = {"values": torch.from_numpy(values).to(device),
                     "index": torch.from_numpy(rows).to(device)}
            try:
                result = model.evaluation_output(batch, latent_mode=latent_mode)
            except (ValueError, RuntimeError) as error:
                raise RuntimeError(f"The model could not process test windows of shape {values.shape[1:]}: {error}") from error
            output, target = result["output"], result["target"]
            if output.shape != target.shape or output.ndim != 3 or output.shape[:2] != values.shape[:2]:
                raise ValueError(f"evaluation_output returned output {tuple(output.shape)} and target "
                                 f"{tuple(target.shape)}; expected matching (B, C, T') shapes")
            time_index = check_time_index(result.get("time_index"), output.shape[2], values.shape[2])
            if common_index is None:
                common_index = time_index
            elif not np.array_equal(common_index, time_index):
                raise ValueError(f"time_index changed between batches ({common_index.tolist()} -> "
                                 f"{time_index.tolist()}); every window must be scored on the same samples")
            # Move to the CPU before converting to float64: MPS has no float64.
            batch_scores = ((output - target) ** 2).mean(dim=2).detach().cpu().double().numpy()
            if not np.isfinite(batch_scores).all():
                raise FloatingPointError(f"Non-finite scores in rows {first}..{first + len(values) - 1}")
            scores[rows] = batch_scores
            for position, row in enumerate(rows):
                if row in keep_rows:
                    kept[int(row)] = {"input": values[position].astype(np.float64),
                                      "output": output[position].detach().cpu().double().numpy(),
                                      "target": target[position].detach().cpu().double().numpy(),
                                      "time_index": time_index}
            if live is not None:
                live.update(number, first + len(values), float(np.mean(batch_scores)))
    if live is not None:
        live.close()
    return scores, kept, common_index
