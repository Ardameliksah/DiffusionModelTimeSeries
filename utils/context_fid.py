import scipy
import numpy as np

from utils.ts2vec.ts2vec import TS2Vec


def calculate_fid(act1, act2):
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)


def Context_FID(ori_data, generated_data, device=0):
    """
    Compute Context-FID between real and generated time series.

    Args:
        ori_data:        (N, seq_len, features) numpy array — real data
        generated_data:  (N, seq_len, features) numpy array — generated data
        device:          GPU device index (int) or 'cpu'

    Returns:
        Scalar FID score (lower = better)
    """
    model = TS2Vec(
        input_dims=ori_data.shape[-1],
        device=device,
        batch_size=8,
        lr=0.001,
        output_dims=320,
        max_train_length=3000,
    )
    model.fit(ori_data, verbose=False)
    ori_repr = model.encode(ori_data, encoding_window='full_series')
    gen_repr = model.encode(generated_data, encoding_window='full_series')
    idx = np.random.permutation(ori_data.shape[0])
    return calculate_fid(ori_repr[idx], gen_repr[idx])
