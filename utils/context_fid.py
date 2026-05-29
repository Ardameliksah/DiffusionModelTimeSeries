import scipy
import numpy as np
from pathlib import Path

from utils.ts2vec.ts2vec import TS2Vec

# Cached weights produced by pretrain_ts2vec.py — same directory as this file
_DEFAULT_WEIGHTS = Path(__file__).parent / "ts2vec_weights_stocks.pt"


def _device_str(device) -> str:
    """Normalise device to a string — torch.load map_location must be str, not int."""
    if isinstance(device, int):
        return f"cuda:{device}"
    return str(device)


def calculate_fid(act1, act2):
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)


def Context_FID(ori_data, generated_data, device="cuda"):
    """
    Compute Context-FID between real and generated time series.

    TS2Vec is loaded from cached weights if available (run pretrain_ts2vec.py once
    to generate them). Falls back to training from scratch if not found.

    Args:
        ori_data:        (N, seq_len, features) numpy array — real data
        generated_data:  (N, seq_len, features) numpy array — generated data
        device:          "cuda", "cpu", or "cuda:N" — NOT an integer (torch.load
                         requires a string map_location, not an int)

    Returns:
        Scalar FID score (lower = better)
    """
    device = _device_str(device)   # ensure string, never int

    model = TS2Vec(
        input_dims=ori_data.shape[-1],
        device=device,
        batch_size=8,
        lr=0.001,
        output_dims=320,
        max_train_length=3000,
    )

    if _DEFAULT_WEIGHTS.exists():
        model.load(str(_DEFAULT_WEIGHTS))
        print(f"   TS2Vec: loaded cached weights ({_DEFAULT_WEIGHTS.name})")
    else:
        print("   TS2Vec: no cached weights — training from scratch (~30-60s). "
              "Run pretrain_ts2vec.py once to cache.")
        model.fit(ori_data, verbose=False)

    ori_repr = model.encode(ori_data, encoding_window='full_series')
    gen_repr = model.encode(generated_data, encoding_window='full_series')
    idx = np.random.permutation(ori_data.shape[0])
    return calculate_fid(ori_repr[idx], gen_repr[idx])
