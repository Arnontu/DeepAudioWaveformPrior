import torch
import numpy as np
from pesq import pesq as pesq_function
from torch import Tensor


def _check_same_shape(preds: Tensor, target: Tensor) -> None:
    """Check that predictions and target have the same shape, else raise error."""
    if preds.shape != target.shape:
        raise RuntimeError("Predictions and targets are expected to have the same shape")


def snr(target: Tensor, preds: Tensor, zero_mean: bool = False) -> Tensor:
    _check_same_shape(preds, target)
    eps = torch.finfo(preds.dtype).eps
    if zero_mean:
        target = target - torch.mean(target, dim=-1, keepdim=True)
        preds = preds - torch.mean(preds, dim=-1, keepdim=True)
    noise = target - preds
    snr_value = (torch.sum(target ** 2, dim=-1) + eps) / (torch.sum(noise ** 2, dim=-1) + eps)
    snr_value = 10 * torch.log10(snr_value)

    return snr_value


def pesq(target, preds):
    try:
        return pesq_function(16000, target.detach().cpu().numpy().squeeze(), preds.detach().cpu().numpy().squeeze(),
                             mode="wb")
    except:
        return 0.0


def si_snr(target, preds):
    target = target.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()
    normalized_s = ((target * preds).sum() / (np.linalg.norm(target) ** 2)) * target
    error = preds - normalized_s
    return 10 * np.log10((np.linalg.norm(normalized_s) ** 2) / (np.linalg.norm(error) ** 2))


def psnr(target, preds):
    '''20*log_10(MaxI) -10 log_10(MSE)
    '''
    mse = ((target - preds) ** 2).sum() / target.size(-1)
    return (20 * np.log10(2) - 10 * torch.log10(mse)).item()


OPTIONAL_METRICS = ["SI-SNR", "PSNR", "PESQ"]
_func_map = {"SI-SNR": si_snr, "PSNR": psnr, "PESQ": pesq}


class _Metric:
    def __init__(self, name, clean, noisy, ylim=None):
        self.clean_vals = []
        self.noisy_vals = []
        self.best_clean = -np.inf
        self.best_clean_epoch = 0
        self.best_clean_wav = None
        self.name = name
        self._f = _func_map[name]
        self.ylim = ylim
        self._clean_audio = clean
        self._noisy_audio = noisy
        self._ref = self(clean, noisy)

    def __call__(self, target, preds):
        _check_same_shape(target, preds)
        return self._f(target, preds)

    def __str__(self):
        return self.name

    def __len__(self):
        return len(self.clean_vals)

    def __gt__(self, other):
        return self.best_clean > other.best_clean

    def get_ref(self):
        return self._ref

    def clean(self, preds):
        value = self(self._clean_audio, preds)
        if value > self.best_clean:
            self.best_clean = value
            self.best_clean_wav = preds.detach().cpu().squeeze().numpy()
            self.best_clean_epoch = len(self._clean_audio)
        self.clean_vals.append(value)

    def noisy(self, preds):
        self.noisy_vals.append(self(self._noisy_audio, preds))

    def get_clean(self):
        return np.array(self.clean_vals)

    def get_noisy(self):
        return np.array(self.noisy_vals)

    def get_wav(self):
        return self.best_clean_wav


class MetricsTracker:
    def __init__(self, clean, noisy):
        self.metrics = [
            _Metric("SI-SNR", clean, noisy, -20),
            _Metric("PSNR", clean, noisy, 28),
            _Metric("PESQ", clean, noisy)
        ]

    def __iter__(self):
        return iter(self.metrics)
