"""
Image transformation utilities for time series (following ImagenTime approach).
Converts 1D time series to 2D images and back using delay embedding.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from scipy import signal as sp_signal
from scipy.fft import fft
# torchaudio imported lazily inside STFTEmbedder to avoid hard dependency

class TsImgEmbedder(nn.Module, ABC):
    """Abstract base class for time series to image transformations."""
    
    def __init__(self, device: str, seq_len: int):
        super().__init__()
        self.device = device
        self.seq_len = seq_len
    
    @abstractmethod
    def ts_to_img(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Convert time series to image.
        
        Args:
            signal: (batch, seq_len, channels)
        
        Returns:
            image: (batch, channels, height, width)
        """
        pass
    
    @abstractmethod
    def img_to_ts(self, img: torch.Tensor) -> torch.Tensor:
        """
        Convert image back to time series.
        
        Args:
            img: (batch, channels, height, width)
        
        Returns:
            signal: (batch, seq_len, channels)
        """
        pass


class DelayEmbedder(TsImgEmbedder):
    """
    Delay Embedding matching ImagenTime implementation exactly.
    Converts time series to 2D image using overlapping sliding windows.
    """
    
    def __init__(self, device: str, seq_len: int, delay: int, embedding: int):
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
    
    def ts_to_img(self, signal: torch.Tensor, pad: bool = True, mask_val: float = 0.0) -> torch.Tensor:
        """
        Convert time series to 2D image via delay embedding (ImagenTime style).
        
        Args:
            signal: (batch, seq_len, channels)
            pad: Whether to pad to square
            mask_val: Padding value
        
        Returns:
            img: (batch, channels, embedding, width) possibly padded to square
        """
        batch, length, channels = signal.shape
        
        # Update seq_len if needed
        if self.seq_len != length:
            self.seq_len = length
        
        # Initialize image
        x_image = torch.zeros(
            (batch, channels, self.embedding, self.embedding),
            dtype=signal.dtype,
            device=self.device
        )
        
        # Fill columns with delay windows
        i = 0
        while (i * self.delay + self.embedding) <= self.seq_len:
            start = i * self.delay
            end = start + self.embedding
            x_image[:, :, :, i] = signal[:, start:end].permute(0, 2, 1)
            i += 1
        
        # Handle remaining incomplete window
        if i * self.delay != self.seq_len and i * self.delay + self.embedding > self.seq_len:
            start = i * self.delay
            end_len = signal[:, start:].permute(0, 2, 1).shape[-1]
            x_image[:, :, :end_len, i] = signal[:, start:].permute(0, 2, 1)
            i += 1
        
        # Trim to actual used columns
        x_image = x_image[:, :, :, :i].to(self.device)
        
        # Cache shape before padding
        self.img_shape = (batch, channels, self.embedding, i)
        
        if pad:
            x_image = self._pad_to_square(x_image, mask_val)
        
        return x_image
    
    def img_to_ts(self, img: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct time series from delay image (ImagenTime style).
        
        Args:
            img: (batch, channels, height, width) possibly padded
        
        Returns:
            signal: (batch, seq_len, channels)
        """
        # Remove padding
        if self.img_shape is not None:
            img = self._unpad(img, self.img_shape)
        
        batch, channels, rows, cols = img.shape
        
        # Initialize reconstruction
        reconstructed = torch.zeros(
            (batch, channels, self.seq_len),
            dtype=img.dtype,
            device=img.device
        )
        
        # Reconstruct from delay windows
        for i in range(cols - 1):
            start = i * self.delay
            end = start + self.embedding
            reconstructed[:, :, start:end] = img[:, :, :, i]
        
        # Handle last window (partial)
        start = (cols - 1) * self.delay
        end = reconstructed[:, :, start:].shape[-1]
        reconstructed[:, :, start:start + end] = img[:, :, :end, cols - 1]
        
        # Permute to (batch, seq_len, channels)
        return reconstructed.permute(0, 2, 1)
    
    def _pad_to_square(self, x: torch.Tensor, mask_val: float = 0.0) -> torch.Tensor:
        """Pad to square."""
        _, _, rows, cols = x.shape
        max_side = max(rows, cols)
        padding = (0, max_side - cols, 0, max_side - rows)
        return torch.nn.functional.pad(x, padding, mode='constant', value=mask_val)
    
    def _unpad(self, x: torch.Tensor, img_shape: Tuple) -> torch.Tensor:
        """Remove padding."""
        _, _, orig_rows, orig_cols = img_shape
        return x[:, :, :orig_rows, :orig_cols]


class PatchEmbedder(TsImgEmbedder):
    """
    Alternative implementation: Converts time series to image using patch-based approach.
    Each patch is a fixed window of the time series.
    """
    
    def __init__(self, device: str, seq_len: int, patch_size: int, img_size: int):
        """
        Args:
            device: "cpu" or "cuda"
            seq_len: Length of time series
            patch_size: Size of each patch (window)
            img_size: Height/Width of output square image
        """
        super().__init__(device, seq_len)
        self.patch_size = patch_size
        self.img_size = img_size
    
    def ts_to_img(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Convert to image using non-overlapping patches.
        
        Args:
            signal: (batch, seq_len, channels)
        
        Returns:
            img: (batch, channels, img_size, img_size)
        """
        batch, seq_len, channels = signal.shape
        
        # Reshape into patches
        num_patches = seq_len // self.patch_size
        signal_patched = signal[:, :num_patches * self.patch_size]
        signal_patched = signal_patched.reshape(batch, num_patches, self.patch_size, channels)
        
        # Each patch becomes a "pixel" 
        # Average within each patch to get (batch, channels, num_patches)
        signal_averaged = signal_patched.mean(dim=2)  # (batch, num_patches, channels)
        
        # Reshape to image
        img_size = int(num_patches ** 0.5)
        signal_averaged = signal_averaged[:, :img_size**2]  # Trim to perfect square
        
        img = signal_averaged.reshape(batch, img_size, img_size, channels)
        img = img.permute(0, 3, 1, 2)  # (batch, channels, height, width)
        
        # Pad/resize to target image size
        if img.shape[-1] != self.img_size:
            img = torch.nn.functional.interpolate(
                img, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False
            )
        
        return img.to(self.device)
    
    def img_to_ts(self, img: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct time series from patch-based image.
        
        Args:
            img: (batch, channels, img_size, img_size)
        
        Returns:
            signal: (batch, seq_len, channels)
        """
        batch, channels, h, w = img.shape
        
        # Reshape back
        signal = img.permute(0, 2, 3, 1)  # (batch, h, w, channels)
        signal = signal.reshape(batch, h * w, channels)
        
        # Expand each "pixel" back to patch_size
        num_patches = h * w
        expanded = signal.unsqueeze(2).expand(-1, -1, self.patch_size, -1)  # (batch, patches, patch_size, channels)
        expanded = expanded.reshape(batch, num_patches * self.patch_size, channels)
        
        # Trim to original sequence length
        return expanded[:, :self.seq_len, :]


class STFTEmbedder(TsImgEmbedder):
    """
    STFT embedding matching ImagenTime implementation.
    Uses torchaudio Spectrogram with real and imaginary parts.
    """
    
    def __init__(self, device: str, seq_len: int, n_fft: int = 16, hop_length: int = 4):
        super().__init__(device, seq_len)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.min_real = None
        self.max_real = None
        self.min_imag = None
        self.max_imag = None
    
    def cache_min_max_params(self, data: torch.Tensor):
        """Cache normalization parameters from training data."""
        real, imag = self._stft_transform(data)
        self.min_real = real.min()
        self.max_real = real.max()
        self.min_imag = imag.min()
        self.max_imag = imag.max()
    
    def _stft_transform(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute STFT real and imaginary parts.

        Args:
            data: (batch, seq_len, channels)

        Returns:
            real, imag: (batch, channels, freq_bins, time_steps) each
        """
        try:
            import torchaudio.transforms as T
        except ImportError:
            raise ImportError("STFTEmbedder requires torchaudio: pip install torchaudio")

        # Permute to (batch, channels, seq_len) for torchaudio
        data = data.permute(0, 2, 1)

        # Compute complex spectrogram
        spec_transform = T.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            center=True,
            power=None
        ).to(data.device)

        spec = spec_transform(data)  # (batch, channels, freq_bins, time_steps)
        return spec.real, spec.imag
    
    def ts_to_img(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Convert time series to STFT image (real + imaginary concatenated).
        
        Args:
            signal: (batch, seq_len, channels)
        
        Returns:
            image: (batch, 2*channels, freq_bins, time_steps)
        """
        real, imag = self._stft_transform(signal)
        
        # Normalize if parameters are cached
        if self.min_real is not None:
            real = (real - self.min_real) / (self.max_real - self.min_real + 1e-8)
            real = (real - 0.5) * 2  # Map to [-1, 1]
            imag = (imag - self.min_imag) / (self.max_imag - self.min_imag + 1e-8)
            imag = (imag - 0.5) * 2  # Map to [-1, 1]
        
        # Concatenate real and imaginary parts
        return torch.cat([real, imag], dim=1)  # (batch, 2*channels, freq, time)
    
    def img_to_ts(self, img: torch.Tensor) -> torch.Tensor:
        """
        Convert STFT image back to time series.
        
        Args:
            img: (batch, 2*channels, freq_bins, time_steps)
        
        Returns:
            signal: (batch, seq_len, channels)
        """
        batch, _, freq_bins, time_steps = img.shape
        
        # Split real and imaginary parts
        half = img.shape[1] // 2
        real_norm = img[:, :half, :, :]
        imag_norm = img[:, half:, :, :]
        
        # Denormalize
        if self.min_real is not None:
            real = (real_norm / 2 + 0.5) * (self.max_real - self.min_real) + self.min_real
            imag = (imag_norm / 2 + 0.5) * (self.max_imag - self.min_imag) + self.min_imag
        else:
            real = real_norm
            imag = imag_norm
        
        # Create complex spectrogram
        spec = torch.complex(real, imag)
        
        # Inverse STFT
        try:
            import torchaudio.transforms as T
        except ImportError:
            raise ImportError("STFTEmbedder requires torchaudio: pip install torchaudio")
        ispec_transform = T.InverseSpectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            center=True
        ).to(img.device)
        
        # Reconstruct (batch, channels, seq_len)
        ts = ispec_transform(spec, self.seq_len)
        
        # Permute back to (batch, seq_len, channels)
        return ts.permute(0, 2, 1)


class MRTIEmbedder(TsImgEmbedder):
    """
    Multi-Resolution Time Imaging (MRTI) from TimeMixer++.
    Converts time series to multi-resolution 2D images using FFT-based period detection.
    """
    
    def __init__(self, device: str, seq_len: int, num_scales: int = 3, num_periods: int = 3):
        super().__init__(device, seq_len)
        self.num_scales = num_scales  # M in the paper
        self.num_periods = num_periods  # K in the paper
    
    def ts_to_img(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Convert time series to multi-resolution time images.
        
        Args:
            signal: (batch, seq_len, channels)
        
        Returns:
            image: (batch, channels * (M+1) * K, height, width)
        """
        batch, seq_len, channels = signal.shape
        all_images = []
        
        for c in range(channels):
            channel_data = signal[:, :, c]  # (batch, seq_len)
            
            # Get coarsest scale (m=num_scales) to find top-K periods via FFT
            x_coarse = channel_data[:, :].cpu().numpy()
            
            # Compute FFT on coarsest scale (use first sample for frequency detection)
            fft_vals = np.fft.fft(x_coarse[0])
            freqs = np.fft.fftfreq(seq_len)
            amplitudes = np.abs(fft_vals)
            
            # Find top-K frequencies
            top_k_indices = np.argsort(amplitudes)[-self.num_periods:][::-1]
            top_freqs = np.abs(freqs[top_k_indices])
            
            # Convert frequencies to periods
            periods = np.where(top_freqs > 0, seq_len / (top_freqs + 1e-8), seq_len).astype(int)
            periods = np.clip(periods, 1, seq_len)
            
            channel_images = []
            
            # For each scale m
            for m in range(self.num_scales + 1):
                scale_factor = 2 ** m
                # Downsample
                x_m = x_coarse[:, ::scale_factor if scale_factor <= seq_len else 1]
                x_m_len = x_m.shape[1]
                
                # For each period k
                for k, period in enumerate(periods):
                    # Pad time series
                    pad_len = period * int(np.ceil(x_m_len / period))
                    x_padded = np.pad(x_m, ((0, 0), (0, max(0, pad_len - x_m_len))), mode='constant')
                    
                    # Reshape to 2D (period x columns)
                    num_cols = x_padded.shape[1] // period
                    if num_cols > 0:
                        x_reshaped = x_padded[:, :period * num_cols].reshape(batch, period, num_cols)
                        # Convert to image, store as (batch, period, num_cols)
                        img = torch.from_numpy(x_reshaped).float().to(self.device)
                        channel_images.append(img)
            
            all_images.extend(channel_images)
        
        # Combine all images - stack them with padding to same size
        if len(all_images) > 0:
            # Find max dimensions
            max_h = max(img.shape[1] for img in all_images)
            max_w = max(img.shape[2] for img in all_images)
            
            padded_images = []
            for img in all_images:
                h, w = img.shape[1], img.shape[2]
                pad_img = torch.nn.functional.pad(img, (0, max_w - w, 0, max_h - h))
                padded_images.append(pad_img)
            
            # Stack: (batch, num_images, max_h, max_w)
            stacked = torch.stack(padded_images, dim=1).reshape(batch, -1, max_h, max_w)
            return stacked
        else:
            return torch.zeros(batch, 1, 1, 1, device=self.device)
    
    def img_to_ts(self, img: torch.Tensor) -> torch.Tensor:
        """
        Convert multi-resolution images back to time series (approximate).
        
        Args:
            img: (batch, num_images, height, width)
        
        Returns:
            signal: (batch, seq_len, channels)
        """
        batch = img.shape[0]
        
        # Infer number of channels from image count
        # Each channel has (num_scales + 1) * num_periods images
        total_images_per_channel = (self.num_scales + 1) * self.num_periods
        num_channels = img.shape[1] // total_images_per_channel
        
        reconstructed_channels = []
        
        for c in range(max(1, num_channels)):
            # Get images for this channel
            start_idx = c * total_images_per_channel
            end_idx = min((c + 1) * total_images_per_channel, img.shape[1])
            channel_imgs = img[:, start_idx:end_idx, :, :]  # (batch, images_per_channel, height, width)
            
            # Use first image as reconstruction (coarsest scale, first period)
            first_img = channel_imgs[:, 0, :, :]  # (batch, height, width)
            
            # Flatten back to 1D
            ts_reconstructed = first_img.reshape(batch, -1)  # (batch, flattened)
            
            # Trim or pad to original sequence length
            if ts_reconstructed.shape[1] >= self.seq_len:
                ts_reconstructed = ts_reconstructed[:, :self.seq_len]
            else:
                ts_reconstructed = torch.nn.functional.pad(
                    ts_reconstructed, 
                    (0, self.seq_len - ts_reconstructed.shape[1])
                )
            
            reconstructed_channels.append(ts_reconstructed)
        
        # Stack channels: (batch, seq_len, channels)
        if reconstructed_channels:
            result = torch.stack(reconstructed_channels, dim=2)  # (batch, seq_len, channels)
        else:
            result = torch.ones(batch, self.seq_len, 1, device=img.device)
        
        return result
