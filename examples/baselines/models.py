"""
Neural Network Models for SOUSA Baselines
==========================================

Production-ready model architectures for audio classification and regression
on the SOUSA drum rudiment dataset.

Models:
- Wav2Vec2Classifier: HuggingFace Wav2Vec2 encoder + classification head
- Wav2Vec2Regressor: HuggingFace Wav2Vec2 encoder + regression head
- AudioSpectrogramTransformer: AST-style model for mel-spectrograms
- MultiTaskModel: Shared encoder with task-specific heads

Features:
- Configurable layer freezing
- Proper weight initialization
- Dropout regularization
- Support for attention masks (variable-length audio)

Usage:
    from examples.baselines.models import Wav2Vec2Classifier
    from examples.baselines.config import ModelConfig

    config = ModelConfig(
        model_name="facebook/wav2vec2-base",
        freeze_encoder_layers=6,
        hidden_dim=256,
        dropout=0.1,
    )
    model = Wav2Vec2Classifier(config, num_classes=4)
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import Wav2Vec2Model, Wav2Vec2Config
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import torchaudio.transforms as T
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False

from .config import ModelConfig


def _init_weights(module: nn.Module):
    """Initialize weights following best practices for transformers."""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class ClassificationHead(nn.Module):
    """
    MLP classification head with configurable hidden layers.

    Args:
        input_dim: Input feature dimension
        num_classes: Number of output classes
        hidden_dim: Hidden layer dimension
        num_layers: Number of hidden layers (0 = linear projection)
        dropout: Dropout probability
        activation: Activation function ("gelu", "relu", "silu")
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        # Get activation function
        activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
        }
        act_fn = activations.get(activation, nn.GELU())

        # Build layers
        layers = []
        in_dim = input_dim

        for i in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                act_fn,
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        # Final projection
        layers.append(nn.Linear(in_dim, num_classes))

        self.classifier = nn.Sequential(*layers)
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class RegressionHead(nn.Module):
    """
    MLP regression head for score prediction.

    Args:
        input_dim: Input feature dimension
        num_outputs: Number of regression targets (default 1)
        hidden_dim: Hidden layer dimension
        num_layers: Number of hidden layers
        dropout: Dropout probability
        activation: Activation function
        output_activation: Final activation ("sigmoid", "none")
    """

    def __init__(
        self,
        input_dim: int,
        num_outputs: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
        activation: str = "gelu",
        output_activation: str = "sigmoid",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_outputs = num_outputs

        activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
        }
        act_fn = activations.get(activation, nn.GELU())

        layers = []
        in_dim = input_dim

        for i in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                act_fn,
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, num_outputs))

        self.regressor = nn.Sequential(*layers)

        # Output activation
        if output_activation == "sigmoid":
            self.output_activation = nn.Sigmoid()
        elif output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = nn.Identity()

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.regressor(x)
        return self.output_activation(x)


class Wav2Vec2Encoder(nn.Module):
    """
    Wav2Vec2 encoder wrapper with freezing support.

    Extracts features from raw audio waveforms.

    Args:
        config: ModelConfig with model settings
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library required. Install with: pip install transformers"
            )

        self.config = config

        # Load pretrained model
        if config.pretrained:
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(config.model_name)
        else:
            wav2vec_config = Wav2Vec2Config.from_pretrained(config.model_name)
            self.wav2vec2 = Wav2Vec2Model(wav2vec_config)

        self.hidden_size = self.wav2vec2.config.hidden_size

        # Apply freezing strategy
        self._freeze_layers(config)

    def _freeze_layers(self, config: ModelConfig):
        """Freeze feature extractor and specified encoder layers."""
        # Freeze feature extractor (CNN layers)
        if config.freeze_feature_extractor:
            for param in self.wav2vec2.feature_extractor.parameters():
                param.requires_grad = False
            for param in self.wav2vec2.feature_projection.parameters():
                param.requires_grad = False

        # Freeze first N encoder layers
        if config.freeze_encoder_layers > 0:
            for i, layer in enumerate(self.wav2vec2.encoder.layers):
                if i < config.freeze_encoder_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Extract features from audio waveform.

        Args:
            input_values: Audio waveform (B, T) at 16kHz
            attention_mask: Optional attention mask (B, T)

        Returns:
            Hidden states (B, T', hidden_size) where T' is the compressed length
        """
        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return outputs.last_hidden_state

    def get_output_length(self, input_length: int) -> int:
        """Calculate output sequence length given input length."""
        # Wav2Vec2 CNN downsamples by factor of ~320
        return self.wav2vec2._get_feat_extract_output_lengths(input_length)


class Wav2Vec2Classifier(nn.Module):
    """
    Wav2Vec2-based audio classifier.

    Uses HuggingFace Wav2Vec2 as encoder with a classification head.

    Args:
        config: ModelConfig with architecture settings
        num_classes: Number of output classes
    """

    def __init__(self, config: ModelConfig, num_classes: int):
        super().__init__()

        self.config = config
        self.num_classes = num_classes

        # Encoder
        self.encoder = Wav2Vec2Encoder(config)

        # Pooling strategy
        self.pooling = "mean"  # mean, first, or attention

        # Classification head
        self.classifier = ClassificationHead(
            input_dim=self.encoder.hidden_size,
            num_classes=num_classes,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_hidden_layers,
            dropout=config.dropout,
            activation=config.activation,
        )

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_values: Audio waveform (B, T)
            attention_mask: Attention mask (B, T) - 1 for valid, 0 for padding
            labels: Optional labels for computing loss

        Returns:
            Dictionary with logits, loss (if labels provided), and hidden states
        """
        # Extract features
        hidden_states = self.encoder(input_values, attention_mask)

        # Pool to single vector
        if attention_mask is not None:
            # Compute output mask (account for CNN downsampling)
            output_lengths = self.encoder.get_output_length(
                attention_mask.sum(dim=1).cpu()
            )
            output_mask = torch.zeros(
                hidden_states.shape[:2],
                dtype=torch.bool,
                device=hidden_states.device,
            )
            for i, length in enumerate(output_lengths):
                output_mask[i, :length] = True

            # Masked mean pooling
            hidden_states = hidden_states.masked_fill(~output_mask.unsqueeze(-1), 0)
            pooled = hidden_states.sum(dim=1) / output_mask.sum(dim=1, keepdim=True)
        else:
            # Simple mean pooling
            pooled = hidden_states.mean(dim=1)

        # Classify
        logits = self.classifier(pooled)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": pooled,
        }

    @torch.no_grad()
    def predict(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get class predictions."""
        outputs = self.forward(input_values, attention_mask)
        return outputs["logits"].argmax(dim=-1)

    @torch.no_grad()
    def predict_proba(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get class probabilities."""
        outputs = self.forward(input_values, attention_mask)
        return F.softmax(outputs["logits"], dim=-1)


class Wav2Vec2Regressor(nn.Module):
    """
    Wav2Vec2-based audio regressor.

    Uses HuggingFace Wav2Vec2 as encoder with a regression head.
    Outputs are in [0, 1] range (scores are normalized).

    Args:
        config: ModelConfig with architecture settings
        num_outputs: Number of regression targets (default 1)
    """

    def __init__(self, config: ModelConfig, num_outputs: int = 1):
        super().__init__()

        self.config = config
        self.num_outputs = num_outputs

        # Encoder
        self.encoder = Wav2Vec2Encoder(config)

        # Regression head
        self.regressor = RegressionHead(
            input_dim=self.encoder.hidden_size,
            num_outputs=num_outputs,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_hidden_layers,
            dropout=config.dropout,
            activation=config.activation,
            output_activation="sigmoid",
        )

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_values: Audio waveform (B, T)
            attention_mask: Attention mask (B, T)
            labels: Optional labels (B,) or (B, num_outputs) for loss computation

        Returns:
            Dictionary with predictions, loss (if labels), and hidden states
        """
        # Extract features
        hidden_states = self.encoder(input_values, attention_mask)

        # Pool to single vector
        if attention_mask is not None:
            output_lengths = self.encoder.get_output_length(
                attention_mask.sum(dim=1).cpu()
            )
            output_mask = torch.zeros(
                hidden_states.shape[:2],
                dtype=torch.bool,
                device=hidden_states.device,
            )
            for i, length in enumerate(output_lengths):
                output_mask[i, :length] = True

            hidden_states = hidden_states.masked_fill(~output_mask.unsqueeze(-1), 0)
            pooled = hidden_states.sum(dim=1) / output_mask.sum(dim=1, keepdim=True)
        else:
            pooled = hidden_states.mean(dim=1)

        # Regress
        predictions = self.regressor(pooled)

        # Squeeze if single output
        if self.num_outputs == 1:
            predictions = predictions.squeeze(-1)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = F.mse_loss(predictions, labels)

        return {
            "predictions": predictions,
            "loss": loss,
            "hidden_states": pooled,
        }

    @torch.no_grad()
    def predict(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get predictions."""
        outputs = self.forward(input_values, attention_mask)
        return outputs["predictions"]


class MultiTaskModel(nn.Module):
    """
    Multi-task model with shared Wav2Vec2 encoder.

    Supports simultaneous skill tier classification, rudiment classification,
    and score regression with task-specific heads.

    Args:
        config: ModelConfig with architecture settings
        num_skill_classes: Number of skill tier classes (default 4)
        num_rudiment_classes: Number of rudiment classes (default 40)
    """

    def __init__(
        self,
        config: ModelConfig,
        num_skill_classes: int = 4,
        num_rudiment_classes: int = 40,
    ):
        super().__init__()

        self.config = config

        # Shared encoder
        self.encoder = Wav2Vec2Encoder(config)

        # Task-specific heads
        hidden_size = self.encoder.hidden_size
        task_dims = config.task_hidden_dims

        self.skill_head = ClassificationHead(
            input_dim=hidden_size,
            num_classes=num_skill_classes,
            hidden_dim=task_dims.get("skill_tier", 256),
            num_layers=config.num_hidden_layers,
            dropout=config.dropout,
        )

        self.rudiment_head = ClassificationHead(
            input_dim=hidden_size,
            num_classes=num_rudiment_classes,
            hidden_dim=task_dims.get("rudiment", 512),
            num_layers=config.num_hidden_layers,
            dropout=config.dropout,
        )

        self.score_head = RegressionHead(
            input_dim=hidden_size,
            num_outputs=1,
            hidden_dim=task_dims.get("overall_score", 128),
            num_layers=config.num_hidden_layers,
            dropout=config.dropout,
            output_activation="sigmoid",
        )

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        skill_labels: torch.Tensor | None = None,
        rudiment_labels: torch.Tensor | None = None,
        score_labels: torch.Tensor | None = None,
        task_weights: dict[str, float] | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for all tasks.

        Args:
            input_values: Audio waveform (B, T)
            attention_mask: Attention mask (B, T)
            skill_labels: Skill tier labels (B,)
            rudiment_labels: Rudiment labels (B,)
            score_labels: Score labels (B,)
            task_weights: Optional weights for loss combination

        Returns:
            Dictionary with all outputs and combined loss
        """
        # Extract features
        hidden_states = self.encoder(input_values, attention_mask)

        # Pool
        if attention_mask is not None:
            output_lengths = self.encoder.get_output_length(
                attention_mask.sum(dim=1).cpu()
            )
            output_mask = torch.zeros(
                hidden_states.shape[:2],
                dtype=torch.bool,
                device=hidden_states.device,
            )
            for i, length in enumerate(output_lengths):
                output_mask[i, :length] = True

            hidden_states = hidden_states.masked_fill(~output_mask.unsqueeze(-1), 0)
            pooled = hidden_states.sum(dim=1) / output_mask.sum(dim=1, keepdim=True)
        else:
            pooled = hidden_states.mean(dim=1)

        # Task predictions
        skill_logits = self.skill_head(pooled)
        rudiment_logits = self.rudiment_head(pooled)
        score_pred = self.score_head(pooled).squeeze(-1)

        # Compute losses
        losses = {}
        if skill_labels is not None:
            losses["skill_loss"] = F.cross_entropy(skill_logits, skill_labels)
        if rudiment_labels is not None:
            losses["rudiment_loss"] = F.cross_entropy(rudiment_logits, rudiment_labels)
        if score_labels is not None:
            losses["score_loss"] = F.mse_loss(score_pred, score_labels)

        # Combine losses
        total_loss = None
        if losses:
            weights = task_weights or {"skill_tier": 1.0, "rudiment": 1.0, "overall_score": 1.0}
            total_loss = torch.tensor(0.0, device=pooled.device)
            if "skill_loss" in losses:
                total_loss = total_loss + weights.get("skill_tier", 1.0) * losses["skill_loss"]
            if "rudiment_loss" in losses:
                total_loss = total_loss + weights.get("rudiment", 1.0) * losses["rudiment_loss"]
            if "score_loss" in losses:
                total_loss = total_loss + weights.get("overall_score", 1.0) * losses["score_loss"]

        return {
            "skill_logits": skill_logits,
            "rudiment_logits": rudiment_logits,
            "score_prediction": score_pred,
            "loss": total_loss,
            "losses": losses,
            "hidden_states": pooled,
        }


class AudioSpectrogramTransformer(nn.Module):
    """
    Audio Spectrogram Transformer (AST) style model.

    Processes mel-spectrograms using Vision Transformer architecture.
    Based on the AST paper: https://arxiv.org/abs/2104.01778

    Args:
        config: ModelConfig with architecture settings
        num_classes: Number of output classes (0 for regression)
        sample_rate: Audio sample rate
        n_mels: Number of mel bands
        n_fft: FFT size
        hop_length: Hop length for STFT
        max_length_sec: Maximum audio length in seconds
    """

    def __init__(
        self,
        config: ModelConfig,
        num_classes: int = 4,
        sample_rate: int = 16000,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 160,
        max_length_sec: float = 5.0,
    ):
        super().__init__()

        if not TORCHAUDIO_AVAILABLE:
            raise ImportError("torchaudio required for AudioSpectrogramTransformer")

        self.config = config
        self.num_classes = num_classes
        self.sample_rate = sample_rate

        # Spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )

        # Calculate expected spectrogram dimensions
        max_samples = int(max_length_sec * sample_rate)
        max_time_frames = max_samples // hop_length + 1

        # Patch embedding (treat spectrogram as image)
        self.patch_size = (16, 16)
        patch_dim = n_mels // self.patch_size[0]
        time_patches = max_time_frames // self.patch_size[1]

        self.num_patches = patch_dim * time_patches
        patch_embed_dim = self.patch_size[0] * self.patch_size[1]

        # Embedding dimension
        self.embed_dim = config.hidden_dim * 2

        # Patch embedding projection
        self.patch_embed = nn.Linear(patch_embed_dim, self.embed_dim)

        # CLS token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            dim_feedforward=self.embed_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # Classification/regression head
        self.head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, num_classes if num_classes > 0 else 1),
        )

        self.is_regression = num_classes == 0

        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_weights)

    def _patchify(self, spec: torch.Tensor) -> torch.Tensor:
        """Convert spectrogram to patch embeddings."""
        B, C, F, T = spec.shape  # (batch, 1, n_mels, time)

        # Ensure dimensions are divisible by patch size
        F_pad = (self.patch_size[0] - F % self.patch_size[0]) % self.patch_size[0]
        T_pad = (self.patch_size[1] - T % self.patch_size[1]) % self.patch_size[1]
        if F_pad > 0 or T_pad > 0:
            spec = F.pad(spec, (0, T_pad, 0, F_pad))

        B, C, F, T = spec.shape

        # Extract patches
        patches = spec.unfold(2, self.patch_size[0], self.patch_size[0])
        patches = patches.unfold(3, self.patch_size[1], self.patch_size[1])
        # Shape: (B, C, num_f_patches, num_t_patches, patch_h, patch_w)

        num_f_patches = F // self.patch_size[0]
        num_t_patches = T // self.patch_size[1]

        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(B, num_f_patches * num_t_patches, -1)

        return patches

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_values: Audio waveform (B, T)
            attention_mask: Attention mask (B, T) - not used for spectrogram
            labels: Optional labels for loss computation

        Returns:
            Dictionary with logits/predictions, loss, and hidden states
        """
        # Compute mel spectrogram
        spec = self.mel_transform(input_values)  # (B, n_mels, time)
        spec = torch.log(spec + 1e-8)  # Log scale
        spec = (spec - spec.mean()) / (spec.std() + 1e-8)  # Normalize
        spec = spec.unsqueeze(1)  # (B, 1, n_mels, time)

        # Patchify
        patches = self._patchify(spec)  # (B, num_patches, patch_dim)

        # Embed patches
        x = self.patch_embed(patches)  # (B, num_patches, embed_dim)

        # Add CLS token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embeddings (truncate if needed)
        if x.shape[1] <= self.pos_embed.shape[1]:
            x = x + self.pos_embed[:, :x.shape[1]]
        else:
            # Interpolate position embeddings for longer sequences
            pos = F.interpolate(
                self.pos_embed.permute(0, 2, 1),
                size=x.shape[1],
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1)
            x = x + pos

        # Transformer
        x = self.transformer(x)

        # Use CLS token for classification
        cls_output = x[:, 0]

        # Head
        output = self.head(cls_output)

        if self.is_regression:
            output = torch.sigmoid(output.squeeze(-1))

        # Compute loss
        loss = None
        if labels is not None:
            if self.is_regression:
                loss = F.mse_loss(output, labels)
            else:
                loss = F.cross_entropy(output, labels)

        return {
            "logits" if not self.is_regression else "predictions": output,
            "loss": loss,
            "hidden_states": cls_output,
        }


def create_model(
    config: ModelConfig,
    task: Literal["skill_classification", "score_regression", "rudiment_classification", "multitask"],
    num_rudiment_classes: int = 40,
) -> nn.Module:
    """
    Factory function to create appropriate model for task.

    Args:
        config: ModelConfig with architecture settings
        task: Task type
        num_rudiment_classes: Number of rudiment classes (for rudiment task)

    Returns:
        Configured model instance
    """
    if task == "skill_classification":
        return Wav2Vec2Classifier(config, num_classes=4)
    elif task == "rudiment_classification":
        return Wav2Vec2Classifier(config, num_classes=num_rudiment_classes)
    elif task == "score_regression":
        return Wav2Vec2Regressor(config, num_outputs=1)
    elif task == "multitask":
        return MultiTaskModel(
            config,
            num_skill_classes=4,
            num_rudiment_classes=num_rudiment_classes,
        )
    else:
        raise ValueError(f"Unknown task: {task}")


# Example usage
if __name__ == "__main__":
    print("Testing SOUSA baseline models...")

    config = ModelConfig(
        model_name="facebook/wav2vec2-base",
        freeze_encoder_layers=6,
        hidden_dim=256,
        dropout=0.1,
    )

    # Test classifier
    print("\n1. Testing Wav2Vec2Classifier...")
    classifier = Wav2Vec2Classifier(config, num_classes=4)
    print(f"   Encoder hidden size: {classifier.encoder.hidden_size}")

    # Dummy input (batch=2, 5 seconds at 16kHz)
    x = torch.randn(2, 80000)
    mask = torch.ones(2, 80000)

    outputs = classifier(x, mask)
    print(f"   Logits shape: {outputs['logits'].shape}")

    # Test regressor
    print("\n2. Testing Wav2Vec2Regressor...")
    regressor = Wav2Vec2Regressor(config, num_outputs=1)

    outputs = regressor(x, mask)
    print(f"   Predictions shape: {outputs['predictions'].shape}")

    # Test multi-task
    print("\n3. Testing MultiTaskModel...")
    multitask = MultiTaskModel(config, num_skill_classes=4, num_rudiment_classes=40)

    outputs = multitask(x, mask)
    print(f"   Skill logits shape: {outputs['skill_logits'].shape}")
    print(f"   Rudiment logits shape: {outputs['rudiment_logits'].shape}")
    print(f"   Score prediction shape: {outputs['score_prediction'].shape}")

    # Test AST (if torchaudio available)
    if TORCHAUDIO_AVAILABLE:
        print("\n4. Testing AudioSpectrogramTransformer...")
        ast_model = AudioSpectrogramTransformer(config, num_classes=4)

        outputs = ast_model(x)
        print(f"   Logits shape: {outputs['logits'].shape}")

    print("\nAll models tested successfully!")
