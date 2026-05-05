"""
Image transformation module - matches ImagenTime implementation exactly.
Converts time series ↔ image representations for U-Net diffusion models.

Supports:
- DelayEmbedding: Time-delay embedding (creates 2D images from 1D signals)
- STFTEmbedding: Short-Time Fourier Transform (frequency domain representation)
"""

from abc import ABC, abstractmethod
import torch
import torchaudio.transforms as T
from sklearn.preprocessing import MinMaxScaler


def MinMaxArgs(data, min_val, max_val):
    """
    MinMax normalization helper - maps data from [min_val, max_val] to [0, 1].
    
    Args:
        data: Tensor to normalize
        min_val: Minimum value in data
        max_val: Maximum value in data
    
    Returns:
        Normalized tensor in [0, 1] range
    """
    return (data - min_val) / (max_val - min_val + 1e-8)


class TsImgEmbedder(ABC):
    """Abstract base class for time series ↔ image transformations."""

    def __init__(self, device, seq_len):
        self.device = device
        self.seq_len = seq_len

    @abstractmethod
    def ts_to_img(self, signal):
        """
        Convert time series to image.
        
        Args:
            signal: (batch, seq_len, channels)
        
        Returns:
            image: (batch, channels, H, W)
        """
        pass

    @abstractmethod
    def img_to_ts(self, img):
        """
        Convert image back to time series.
        
        Args:
            img: (batch, channels, H, W)
        
        Returns:
            signal: (batch, seq_len, channels)
        """
        pass


class DelayEmbedder(TsImgEmbedder):
    """
    Delay Embedding Transformation - matches ImagenTime exactly.
    
    Creates 2D images from 1D time series using overlapping delay windows.
    Each column contains a delay-embedded window (receptive field).
    """

    def __init__(self, device, seq_len, delay, embedding):
        """
        Args:
            device: "cpu" or "cuda"
            seq_len: Length of time series
            delay: Step size between windows
            embedding: Window size (height of image)
        """
        super().__init__(device, seq_len)
        self.delay = delay
        self.embedding = embedding
        self.img_shape = None

    def pad_to_square(self, x, mask=0):
        """
        Pads tensor to make it square along last two dimensions.
        
        Args:
            x: (batch, channels, height, width)
            mask: Padding value
        
        Returns:
            Padded square tensor
        """
        _, _, cols, rows = x.shape
        max_side = max(cols, rows)
        padding = (0, max_side - rows, 0, max_side - cols)
        x_padded = torch.nn.functional.pad(x, padding, mode='constant', value=mask)
        return x_padded

    def unpad(self, x, original_shape):
        """
        Remove padding to restore original shape.
        
        Args:
            x: Padded tensor
            original_shape: Original shape before padding
        
        Returns:
            Unpadded tensor
        """
        _, _, original_cols, original_rows = original_shape
        return x[:, :, :original_cols, :original_rows]

    def ts_to_img(self, signal, pad=True, mask=0):
        """
        Convert time series to image via delay embedding.
        
        Args:
            signal: (batch, seq_len, channels)
            pad: Whether to pad to square
            mask: Padding value
        
        Returns:
            image: (batch, channels, H, W) possibly padded to square
        """
        batch, length, features = signal.shape
        
        # Handle variable sequence lengths
        if self.seq_len != length:
            self.seq_len = length

        # Initialize image tensor
        x_image = torch.zeros((batch, features, self.embedding, self.embedding), device=self.device)
        
        # Fill columns with delay-embedded windows
        i = 0
        while (i * self.delay + self.embedding) <= self.seq_len:
            start = i * self.delay
            end = start + self.embedding
            x_image[:, :, :, i] = signal[:, start:end].permute(0, 2, 1)
            i += 1

        # Handle remaining incomplete window
        if i * self.delay != self.seq_len and i * self.delay + self.embedding > self.seq_len:
            start = i * self.delay
            end = signal[:, start:].permute(0, 2, 1).shape[-1]
            x_image[:, :, :end, i] = signal[:, start:].permute(0, 2, 1)
            i += 1

        # Cache shape before padding for reconstruction
        self.img_shape = (batch, features, self.embedding, i)
        x_image = x_image[:, :, :, :i]

        if pad:
            x_image = self.pad_to_square(x_image, mask)

        return x_image

    def img_to_ts(self, img):
        """
        Reconstruct time series from delay-embedded image.
        
        Args:
            img: (batch, channels, H, W) possibly padded
        
        Returns:
            signal: (batch, seq_len, channels)
        """
        # Remove padding
        img_non_square = self.unpad(img, self.img_shape)

        batch, channels, rows, cols = img_non_square.shape

        # Initialize reconstruction tensor
        reconstructed_ts = torch.zeros((batch, channels, self.seq_len), device=img.device)

        # Reconstruct from delay windows (overlap averaging for better reconstruction)
        for i in range(cols - 1):
            start = i * self.delay
            end = start + self.embedding
            reconstructed_ts[:, :, start:end] = img_non_square[:, :, :, i]

        # Handle last window (partial)
        start = (cols - 1) * self.delay
        end = reconstructed_ts[:, :, start:].shape[-1]
        reconstructed_ts[:, :, start:] = img_non_square[:, :, :end, cols - 1]

        # Permute back to (batch, seq_len, channels)
        return reconstructed_ts.permute(0, 2, 1)


class STFTEmbedder(TsImgEmbedder):
    """
    STFT Embedding Transformation - matches ImagenTime exactly.
    
    Uses Short-Time Fourier Transform to convert time series to frequency domain images.
    Real and imaginary parts are concatenated as separate channels.
    """

    def __init__(self, device, seq_len, n_fft=64, hop_length=16):
        """
        Args:
            device: "cpu" or "cuda"
            seq_len: Length of time series
            n_fft: FFT window size
            hop_length: Hop size for sliding window
        """
        super().__init__(device, seq_len)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.min_real = None
        self.max_real = None
        self.min_imag = None
        self.max_imag = None

    def cache_min_max_params(self, train_data):
        """
        Pre-compute normalization parameters from training data.
        MUST be called once before training starts.
        
        Args:
            train_data: Training time series (batch, seq_len, channels)
        """
        real, imag = self.stft_transform(train_data)
        
        # Compute min/max for real and imaginary parts
        self.min_real = real.min()
        self.max_real = real.max()
        self.min_imag = imag.min()
        self.max_imag = imag.max()

    def stft_transform(self, data):
        """
        Compute STFT real and imaginary parts.
        
        Args:
            data: (batch, seq_len, channels)
        
        Returns:
            (real, imag): Each is (batch, channels, freq_bins, time_steps)
        """
        # Permute to (batch, channels, seq_len) for torchaudio
        data_perm = torch.permute(data, (0, 2, 1))
        
        # Compute complex spectrogram
        spec_transform = T.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            center=True,
            power=None
        ).to(data.device)
        
        spec = spec_transform(data_perm)
        return spec.real, spec.imag

    def ts_to_img(self, signal):
        """
        Convert time series to STFT image.
        
        Args:
            signal: (batch, seq_len, channels)
        
        Returns:
            image: (batch, 2*channels, freq_bins, time_steps)
        """
        assert self.min_real is not None, "Must call cache_min_max_params() before ts_to_img()"
        
        # Compute STFT
        real, imag = self.stft_transform(signal)
        
        # MinMax normalize to [0, 1]
        real_norm = MinMaxArgs(real, self.min_real.to(self.device), self.max_real.to(self.device))
        imag_norm = MinMaxArgs(imag, self.min_imag.to(self.device), self.max_imag.to(self.device))
        
        # Scale to [-1, 1]
        real_scaled = (real_norm - 0.5) * 2
        imag_scaled = (imag_norm - 0.5) * 2
        
        # Concatenate as separate channels
        stft_img = torch.cat([real_scaled, imag_scaled], dim=1)
        return stft_img

    def img_to_ts(self, x_image):
        """
        Convert STFT image back to time series.
        
        Args:
            x_image: (batch, 2*channels, freq_bins, time_steps)
        
        Returns:
            signal: (batch, seq_len, channels)
        """
        # Split real and imaginary parts
        split = torch.split(x_image, x_image.shape[1] // 2, dim=1)
        real_scaled, imag_scaled = split[0], split[1]
        
        # Inverse scaling from [-1, 1] to [0, 1]
        real_norm = (real_scaled / 2) + 0.5
        imag_norm = (imag_scaled / 2) + 0.5
        
        # Denormalize
        min_real = self.min_real.to(self.device)
        max_real = self.max_real.to(self.device)
        min_imag = self.min_imag.to(self.device)
        max_imag = self.max_imag.to(self.device)
        
        real_denorm = real_norm * (max_real - min_real) + min_real
        imag_denorm = imag_norm * (max_imag - min_imag) + min_imag
        
        # Reconstruct complex spectrogram
        spec_complex = torch.complex(real_denorm, imag_denorm)
        
        # Inverse STFT
        ispec_transform = T.InverseSpectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            center=True
        ).to(self.device)
        
        ts_reconstructed = ispec_transform(spec_complex, self.seq_len)
        
        # Permute back to (batch, seq_len, channels)
        return torch.permute(ts_reconstructed, (0, 2, 1))
