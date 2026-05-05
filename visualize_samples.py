"""
Visualize normalized generated samples vs real normalized windows.

Usage:
  python MyCode/visualize_samples.py --generated path/to/generated_samples_normalized.npy

By default it loads `output/generated_samples_normalized.npy` and the dataset
defined in `config/diffusion_config.py` to pull real normalized windows.
"""
import argparse
from pathlib import Path
import numpy as np
import torch

from config.stocks_config import Config
from utils.data_utils import StockDataset, build_scaler
from utils.visualization import plot_real_vs_generated, plot_statistics_comparison, compute_statistics


def load_generated(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Generated samples not found: {path}")
    arr = np.load(path)
    # Expected shape: (N, num_features, seq_len)
    return torch.from_numpy(arr).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated', type=str, default=None,
                        help='Path to generated normalized .npy (overrides config)')
    parser.add_argument('--csv', type=str, default=None, help='Optional CSV path to override config')
    parser.add_argument('--which', type=str, choices=['train', 'test'], default='test')
    parser.add_argument('--num-samples', type=int, default=4)
    parser.add_argument('--output-dir', type=str, default='output/visualizations')
    parser.add_argument('--feature-names', type=str, nargs='*', default=None)
    args = parser.parse_args()

    cfg = Config()
    csv_path = args.csv if args.csv is not None else cfg.data.data_path

    # Load dataset
    scaler = build_scaler(cfg.data.neg_one_to_one)
    dataset = StockDataset(
        csv_path=csv_path,
        scaler=scaler,
        window_length=cfg.model.sequence_length,
    )

    # Select real windows
    if args.which == 'test':
        real_windows = dataset.get_test_data()
    else:
        real_windows = dataset.get_train_data()

    # Resolve generated file path (use config sampling dir by default)
    candidates = []
    if args.generated:
        candidates.append(Path(args.generated))
    # primary default from config
    candidates.append(Path(cfg.sampling.output_dir) / 'generated_samples_normalized.npy')
    candidates.append(Path(cfg.sampling.output_dir) / 'generated_samples.npy')
    # legacy/common paths
    candidates.append(Path('output') / 'generated_samples_normalized.npy')
    candidates.append(Path('output') / 'generated_samples.npy')

    gen_path = None
    for p in candidates:
        if p.exists():
            gen_path = p
            break

    if gen_path is None:
        checked = '\n'.join([str(p) for p in candidates])
        raise FileNotFoundError(
            f"Generated samples not found. Checked these paths:\n{checked}\nRun `python MyCode/sample.py` to produce samples or pass --generated PATH"
        )

    generated = load_generated(gen_path)

    # Align sample counts
    n_real = real_windows.shape[0]
    n_gen = generated.shape[0]
    n = min(n_real, n_gen, args.num_samples)
    if n == 0:
        raise RuntimeError('No samples available to visualize')

    # Take first n samples for visualization
    real_vis = real_windows[:n]
    gen_vis = generated[:n]

    # Ensure tensors (batch, features, seq_len)
    if isinstance(real_vis, np.ndarray):
        real_vis = torch.from_numpy(real_vis)
    if isinstance(gen_vis, np.ndarray):
        gen_vis = torch.from_numpy(gen_vis)

    # Plot time series side-by-side
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_names = args.feature_names
    fig1 = plot_real_vs_generated(real_vis, gen_vis, num_samples=n, feature_names=feature_names,
                                  save_path=str(out_dir / 'real_vs_generated.png'))

    # Compute and plot statistics (normalized values)
    real_stats = compute_statistics(real_vis, feature_names)
    gen_stats = compute_statistics(gen_vis, feature_names)
    fig2 = plot_statistics_comparison(real_stats, gen_stats, save_path=str(out_dir / 'statistics_comparison.png'))

    print(f"Saved visualizations to {out_dir}")


if __name__ == '__main__':
    main()
