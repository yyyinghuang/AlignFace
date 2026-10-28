"""Fixed reference percentiles that preserve score ordering and ties."""
import numpy as np


class PercentileCalibration:
    def __init__(self, scores, percentiles):
        self.scores = np.asarray(scores, dtype=np.float64)
        self.percentiles = np.asarray(percentiles, dtype=np.float64)
        if (self.scores.ndim != 1 or self.percentiles.shape != self.scores.shape
                or self.scores.size < 2 or not np.isfinite(self.scores).all()
                or not np.isfinite(self.percentiles).all()
                or not np.all(np.diff(self.scores) > 0)
                or not np.all(np.diff(self.percentiles) > 0)
                or self.percentiles[0] <= 0 or self.percentiles[-1] >= 1):
            raise ValueError("Calibration requires finite, strictly increasing scores and percentiles in (0, 1).")

    @classmethod
    def fit(cls, scores):
        values = np.asarray(scores, dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("Reference scores must be a finite one-dimensional array.")
        unique, counts = np.unique(values, return_counts=True)
        ranks = (np.cumsum(counts)-counts/2)/max(values.size, 1)
        return cls(unique, ranks)

    def transform(self, scores):
        values = np.asarray(scores, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Scores must be finite.")
        x, y = self.scores, self.percentiles
        result = np.interp(values, x, y)
        left_scale = y[0]*(x[1]-x[0])/(y[1]-y[0])
        right_scale = (1-y[-1])*(x[-1]-x[-2])/(y[-1]-y[-2])
        left = values < x[0]
        right = values > x[-1]
        result = np.asarray(result)
        result[left] = y[0]*np.exp((values[left]-x[0])/left_scale)
        result[right] = 1-(1-y[-1])*np.exp(-(values[right]-x[-1])/right_scale)
        return result

    def to_dict(self):
        return dict(method="midrank_linear_with_exponential_tails", scores=self.scores.tolist(), percentiles=self.percentiles.tolist())

    @classmethod
    def from_dict(cls, data):
        if data["method"] != "midrank_linear_with_exponential_tails":
            raise ValueError("Unsupported percentile calibration method.")
        return cls(data["scores"], data["percentiles"])

    def transform_tensor(self, scores):
        """Torch equivalent retaining piecewise input gradients."""
        import torch
        if not torch.is_floating_point(scores):
            raise TypeError("Scores must be floating-point tensors.")
        x = torch.as_tensor(self.scores, dtype=scores.dtype, device=scores.device)
        y = torch.as_tensor(self.percentiles, dtype=scores.dtype, device=scores.device)
        right = torch.searchsorted(x, scores.contiguous()).clamp(1, x.numel()-1)
        left = right-1
        result = y[left]+(scores-x[left])/(x[right]-x[left])*(y[right]-y[left])
        left_scale = y[0]*(x[1]-x[0])/(y[1]-y[0])
        right_scale = (1-y[-1])*(x[-1]-x[-2])/(y[-1]-y[-2])
        lower = y[0]*torch.exp(((scores-x[0])/left_scale).clamp(max=0))
        upper = 1-(1-y[-1])*torch.exp((-(scores-x[-1])/right_scale).clamp(max=0))
        return torch.where(scores<x[0], lower, torch.where(scores>x[-1], upper, result))
