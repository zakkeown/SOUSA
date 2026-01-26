"""
Feature Extraction for SOUSA Dataset
=====================================

Comprehensive feature extraction utilities for drum audio analysis.

Features:
- Mel-spectrogram extraction with librosa/torchaudio
- Onset detection features
- RMS energy over time
- Pretrained Wav2Vec2 encoder features
- Feature caching for faster iteration

Usage:
    from examples.feature_extraction import FeatureExtractor

    extractor = FeatureExtractor(
        sample_rate=16000,
        feature_type="mel_spectrogram"
    )

    features = extractor.extract("audio.flac")
    # or
    features = extractor.extract_from_array(audio_array)
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Literal

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import torchaudio
    import torchaudio.transforms as T
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class MelSpectrogramExtractor:
    """
    Extract mel-spectrograms optimized for drum audio.

    Args:
        sample_rate: Audio sample rate
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length between frames
        f_min: Minimum frequency
        f_max: Maximum frequency
        power: Spectrogram power (1 for energy, 2 for power)
        normalized: Whether to normalize output
        use_torchaudio: Use torchaudio (True) or librosa (False)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 256,
        f_min: float = 20.0,
        f_max: float = 8000.0,
        power: float = 2.0,
        normalized: bool = True,
        use_torchaudio: bool = True,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.power = power
        self.normalized = normalized
        self.use_torchaudio = use_torchaudio and TORCHAUDIO_AVAILABLE

        if self.use_torchaudio:
            self.mel_transform = T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                f_min=f_min,
                f_max=f_max,
                power=power,
            )
            self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)

    def __call__(self, audio: np.ndarray | torch.Tensor) -> np.ndarray:
        """Extract mel-spectrogram from audio array."""
        if self.use_torchaudio:
            return self._extract_torchaudio(audio)
        elif LIBROSA_AVAILABLE:
            return self._extract_librosa(audio)
        else:
            raise ImportError("Neither torchaudio nor librosa available")

    def _extract_torchaudio(self, audio: np.ndarray | torch.Tensor) -> np.ndarray:
        """Extract using torchaudio."""
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        mel = self.mel_transform(audio)
        mel_db = self.amplitude_to_db(mel)

        # Remove batch/channel dimension
        mel_db = mel_db.squeeze()

        if self.normalized:
            mel_db = self._normalize(mel_db)

        return mel_db.numpy()

    def _extract_librosa(self, audio: np.ndarray) -> np.ndarray:
        """Extract using librosa."""
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
            power=self.power,
        )

        mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80)

        if self.normalized:
            mel_db = self._normalize_np(mel_db)

        return mel_db

    def _normalize(self, mel_db: torch.Tensor) -> torch.Tensor:
        """Per-channel normalization."""
        mean = mel_db.mean(dim=-1, keepdim=True)
        std = mel_db.std(dim=-1, keepdim=True) + 1e-6
        return (mel_db - mean) / std

    def _normalize_np(self, mel_db: np.ndarray) -> np.ndarray:
        """Per-channel normalization (numpy)."""
        mean = mel_db.mean(axis=-1, keepdims=True)
        std = mel_db.std(axis=-1, keepdims=True) + 1e-6
        return (mel_db - mean) / std


class OnsetDetector:
    """
    Extract onset-related features for drum analysis.

    Args:
        sample_rate: Audio sample rate
        hop_length: Hop length for frame-level features
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        hop_length: int = 128,
    ):
        self.sample_rate = sample_rate
        self.hop_length = hop_length

        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa required for onset detection")

    def detect_onsets(self, audio: np.ndarray) -> dict:
        """
        Detect onsets and compute onset features.

        Returns:
            Dictionary with:
            - onset_times: Array of onset times in seconds
            - onset_frames: Array of onset frame indices
            - onset_strength: Onset strength envelope
            - num_onsets: Total number of detected onsets
        """
        # Onset strength envelope
        onset_env = librosa.onset.onset_strength(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
        )

        # Detect onsets
        onset_frames = librosa.onset.onset_detect(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            onset_envelope=onset_env,
        )

        onset_times = librosa.frames_to_time(
            onset_frames,
            sr=self.sample_rate,
            hop_length=self.hop_length,
        )

        return {
            "onset_times": onset_times,
            "onset_frames": onset_frames,
            "onset_strength": onset_env,
            "num_onsets": len(onset_frames),
        }

    def compute_ioi_features(self, onset_times: np.ndarray) -> dict:
        """
        Compute inter-onset interval (IOI) features.

        Args:
            onset_times: Array of onset times in seconds

        Returns:
            Dictionary with IOI statistics
        """
        if len(onset_times) < 2:
            return {
                "ioi_mean": 0.0,
                "ioi_std": 0.0,
                "ioi_cv": 0.0,  # Coefficient of variation
                "tempo_estimate": 0.0,
            }

        iois = np.diff(onset_times)

        ioi_mean = np.mean(iois)
        ioi_std = np.std(iois)
        ioi_cv = ioi_std / ioi_mean if ioi_mean > 0 else 0.0

        # Estimate tempo from IOI
        tempo_estimate = 60.0 / ioi_mean if ioi_mean > 0 else 0.0

        return {
            "ioi_mean": ioi_mean,
            "ioi_std": ioi_std,
            "ioi_cv": ioi_cv,
            "tempo_estimate": tempo_estimate,
        }


class RMSExtractor:
    """
    Extract RMS energy features.

    Args:
        frame_length: Frame length for RMS computation
        hop_length: Hop length between frames
    """

    def __init__(
        self,
        frame_length: int = 2048,
        hop_length: int = 512,
    ):
        self.frame_length = frame_length
        self.hop_length = hop_length

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """Extract RMS energy curve."""
        if LIBROSA_AVAILABLE:
            rms = librosa.feature.rms(
                y=audio,
                frame_length=self.frame_length,
                hop_length=self.hop_length,
            )[0]
        else:
            # Manual RMS computation
            rms = self._compute_rms(audio)

        return rms

    def _compute_rms(self, audio: np.ndarray) -> np.ndarray:
        """Compute RMS without librosa."""
        n_frames = 1 + (len(audio) - self.frame_length) // self.hop_length
        rms = np.zeros(n_frames)

        for i in range(n_frames):
            start = i * self.hop_length
            end = start + self.frame_length
            frame = audio[start:end]
            rms[i] = np.sqrt(np.mean(frame ** 2))

        return rms

    def compute_dynamics_features(self, rms: np.ndarray) -> dict:
        """
        Compute dynamics-related features from RMS curve.

        Returns:
            Dictionary with dynamics statistics
        """
        return {
            "rms_mean": np.mean(rms),
            "rms_std": np.std(rms),
            "rms_max": np.max(rms),
            "rms_min": np.min(rms),
            "dynamic_range": np.max(rms) - np.min(rms),
            "rms_cv": np.std(rms) / np.mean(rms) if np.mean(rms) > 0 else 0,
        }


class Wav2Vec2Extractor:
    """
    Extract features using pretrained Wav2Vec2 model.

    Args:
        model_name: HuggingFace model name
        layer: Which transformer layer to extract from (-1 = last)
        device: Device to run model on
        freeze: Whether to freeze model weights
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        layer: int = -1,
        device: str | None = None,
        freeze: bool = True,
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers required for Wav2Vec2. "
                "Install with: pip install transformers"
            )

        self.model_name = model_name
        self.layer = layer

        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Load model and feature extractor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.model.to(device)

        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    def __call__(
        self,
        audio: np.ndarray,
        return_all_layers: bool = False,
    ) -> np.ndarray | dict:
        """
        Extract Wav2Vec2 features.

        Args:
            audio: Audio array (should be 16kHz)
            return_all_layers: Return features from all transformer layers

        Returns:
            Feature array of shape (time_frames, hidden_dim)
            or dict of arrays if return_all_layers=True
        """
        # Prepare input
        inputs = self.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )

        input_values = inputs.input_values.to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                input_values,
                output_hidden_states=return_all_layers,
            )

        if return_all_layers:
            # Return all hidden states
            hidden_states = outputs.hidden_states
            return {
                f"layer_{i}": h.squeeze(0).cpu().numpy()
                for i, h in enumerate(hidden_states)
            }
        else:
            # Return specified layer
            if self.layer == -1:
                features = outputs.last_hidden_state
            else:
                features = outputs.hidden_states[self.layer]

            return features.squeeze(0).cpu().numpy()

    def extract_pooled(
        self,
        audio: np.ndarray,
        pooling: Literal["mean", "max", "first", "last"] = "mean",
    ) -> np.ndarray:
        """
        Extract pooled features (single vector per sample).

        Args:
            audio: Audio array
            pooling: Pooling strategy

        Returns:
            Feature vector of shape (hidden_dim,)
        """
        features = self(audio)  # (time, hidden_dim)

        if pooling == "mean":
            return features.mean(axis=0)
        elif pooling == "max":
            return features.max(axis=0)
        elif pooling == "first":
            return features[0]
        elif pooling == "last":
            return features[-1]
        else:
            raise ValueError(f"Unknown pooling: {pooling}")


class FeatureExtractor:
    """
    Unified feature extractor combining multiple feature types.

    Args:
        sample_rate: Target sample rate
        feature_type: Type of features to extract
        cache_dir: Directory for caching features (None to disable)
        **kwargs: Additional arguments passed to specific extractors
    """

    FEATURE_TYPES = [
        "mel_spectrogram",
        "onset",
        "rms",
        "wav2vec2",
        "all",
    ]

    def __init__(
        self,
        sample_rate: int = 16000,
        feature_type: str = "mel_spectrogram",
        cache_dir: str | Path | None = None,
        **kwargs,
    ):
        self.sample_rate = sample_rate
        self.feature_type = feature_type
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.kwargs = kwargs

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize extractors based on type
        self.extractors = {}

        if feature_type in ["mel_spectrogram", "all"]:
            self.extractors["mel"] = MelSpectrogramExtractor(
                sample_rate=sample_rate,
                **{k: v for k, v in kwargs.items() if k in [
                    "n_mels", "n_fft", "hop_length", "f_min", "f_max"
                ]}
            )

        if feature_type in ["onset", "all"] and LIBROSA_AVAILABLE:
            self.extractors["onset"] = OnsetDetector(sample_rate=sample_rate)

        if feature_type in ["rms", "all"]:
            self.extractors["rms"] = RMSExtractor()

        if feature_type in ["wav2vec2", "all"] and TRANSFORMERS_AVAILABLE:
            self.extractors["wav2vec2"] = Wav2Vec2Extractor(
                **{k: v for k, v in kwargs.items() if k in [
                    "model_name", "layer", "device", "freeze"
                ]}
            )

    def extract(self, audio_path: str | Path) -> dict:
        """
        Extract features from audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary of extracted features
        """
        audio_path = Path(audio_path)

        # Check cache
        if self.cache_dir:
            cache_key = self._get_cache_key(audio_path)
            cache_path = self.cache_dir / f"{cache_key}.pkl"
            if cache_path.exists():
                with open(cache_path, "rb") as f:
                    return pickle.load(f)

        # Load audio
        if TORCHAUDIO_AVAILABLE:
            waveform, sr = torchaudio.load(audio_path)
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            audio = waveform.squeeze().numpy()
        elif LIBROSA_AVAILABLE:
            audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        else:
            raise ImportError("Either torchaudio or librosa required")

        # Extract features
        features = self.extract_from_array(audio)

        # Cache results
        if self.cache_dir:
            with open(cache_path, "wb") as f:
                pickle.dump(features, f)

        return features

    def extract_from_array(self, audio: np.ndarray) -> dict:
        """
        Extract features from audio array.

        Args:
            audio: Audio array at target sample rate

        Returns:
            Dictionary of extracted features
        """
        features = {}

        if "mel" in self.extractors:
            features["mel_spectrogram"] = self.extractors["mel"](audio)

        if "onset" in self.extractors:
            onset_result = self.extractors["onset"].detect_onsets(audio)
            features["onset_times"] = onset_result["onset_times"]
            features["onset_strength"] = onset_result["onset_strength"]
            features["num_onsets"] = onset_result["num_onsets"]

            ioi_features = self.extractors["onset"].compute_ioi_features(
                onset_result["onset_times"]
            )
            features.update(ioi_features)

        if "rms" in self.extractors:
            rms = self.extractors["rms"](audio)
            features["rms"] = rms

            dynamics = self.extractors["rms"].compute_dynamics_features(rms)
            features.update(dynamics)

        if "wav2vec2" in self.extractors:
            features["wav2vec2"] = self.extractors["wav2vec2"](audio)
            features["wav2vec2_pooled"] = self.extractors["wav2vec2"].extract_pooled(
                audio, pooling="mean"
            )

        return features

    def _get_cache_key(self, audio_path: Path) -> str:
        """Generate cache key from audio path and extractor config."""
        config_str = f"{audio_path.name}_{self.feature_type}_{self.sample_rate}"
        return hashlib.md5(config_str.encode()).hexdigest()


def extract_batch_features(
    audio_paths: list[str | Path],
    feature_type: str = "mel_spectrogram",
    sample_rate: int = 16000,
    num_workers: int = 4,
    cache_dir: str | Path | None = None,
    **kwargs,
) -> list[dict]:
    """
    Extract features from multiple audio files in parallel.

    Args:
        audio_paths: List of audio file paths
        feature_type: Type of features to extract
        sample_rate: Target sample rate
        num_workers: Number of parallel workers
        cache_dir: Cache directory
        **kwargs: Additional extractor arguments

    Returns:
        List of feature dictionaries
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from functools import partial

    extractor = FeatureExtractor(
        sample_rate=sample_rate,
        feature_type=feature_type,
        cache_dir=cache_dir,
        **kwargs,
    )

    results = [None] * len(audio_paths)

    if num_workers == 1:
        # Sequential processing
        for i, path in enumerate(audio_paths):
            results[i] = extractor.extract(path)
    else:
        # Note: Wav2Vec2 may not work well with multiprocessing
        # due to CUDA/model loading issues
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(extractor.extract, path): i
                for i, path in enumerate(audio_paths)
            }

            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

    return results


# Example usage
if __name__ == "__main__":
    import sys

    print("Feature Extraction Examples")
    print("=" * 50)

    # Create dummy audio for testing
    sample_rate = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Simulate drum hits with decaying noise
    audio = np.zeros_like(t)
    hit_times = [0.5, 1.0, 1.5, 2.0, 2.5]
    for hit_time in hit_times:
        hit_idx = int(hit_time * sample_rate)
        decay = np.exp(-np.arange(4000) / 800)
        noise = np.random.randn(4000) * 0.5
        hit_samples = decay * noise
        if hit_idx + len(hit_samples) < len(audio):
            audio[hit_idx:hit_idx + len(hit_samples)] += hit_samples

    audio = audio / np.abs(audio).max()

    print(f"Created test audio: {len(audio)} samples at {sample_rate}Hz")

    # Test mel spectrogram
    print("\n1. Mel Spectrogram Extraction")
    if TORCHAUDIO_AVAILABLE or LIBROSA_AVAILABLE:
        mel_extractor = MelSpectrogramExtractor(sample_rate=sample_rate)
        mel = mel_extractor(audio)
        print(f"   Shape: {mel.shape} (n_mels x time_frames)")
    else:
        print("   Skipped: torchaudio or librosa required")

    # Test onset detection
    print("\n2. Onset Detection")
    if LIBROSA_AVAILABLE:
        onset_detector = OnsetDetector(sample_rate=sample_rate)
        onsets = onset_detector.detect_onsets(audio)
        print(f"   Detected {onsets['num_onsets']} onsets")
        print(f"   Onset times: {onsets['onset_times'][:5]}...")

        ioi = onset_detector.compute_ioi_features(onsets["onset_times"])
        print(f"   Estimated tempo: {ioi['tempo_estimate']:.1f} BPM")
    else:
        print("   Skipped: librosa required")

    # Test RMS
    print("\n3. RMS Energy Extraction")
    rms_extractor = RMSExtractor()
    rms = rms_extractor(audio)
    dynamics = rms_extractor.compute_dynamics_features(rms)
    print(f"   RMS shape: {rms.shape}")
    print(f"   Dynamic range: {dynamics['dynamic_range']:.4f}")

    # Test Wav2Vec2
    print("\n4. Wav2Vec2 Features")
    if TRANSFORMERS_AVAILABLE and TORCH_AVAILABLE:
        try:
            w2v_extractor = Wav2Vec2Extractor()
            features = w2v_extractor(audio)
            print(f"   Shape: {features.shape} (time_frames x hidden_dim)")

            pooled = w2v_extractor.extract_pooled(audio)
            print(f"   Pooled shape: {pooled.shape}")
        except Exception as e:
            print(f"   Error: {e}")
    else:
        print("   Skipped: transformers required")

    # Test unified extractor
    print("\n5. Unified Feature Extraction")
    try:
        extractor = FeatureExtractor(
            sample_rate=sample_rate,
            feature_type="mel_spectrogram",
        )
        features = extractor.extract_from_array(audio)
        print(f"   Features: {list(features.keys())}")
    except Exception as e:
        print(f"   Error: {e}")
