import torch
import torch.nn as nn
import numpy as np
import time
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class PolarQuantConfig:
    head_dim: int = 128
    num_sign_bits: int = 0  # auto-calculated if 0
    num_correction_bits: int = 2
    num_corrections: int = 0  # auto-calculated if 0
    target_bits_per_element: float = 3.0
    radius_bits: int = 8
    jl_seed: int = 42

    def __post_init__(self):
        if self.num_sign_bits == 0 or self.num_corrections == 0:
            total_budget = self.target_bits_per_element * self.head_dim
            remaining = total_budget - self.radius_bits
            # 80/20 split between sign bits and corrections
            self.num_sign_bits = int(remaining * 0.80)
            correction_budget = remaining - self.num_sign_bits
            self.num_corrections = int(correction_budget / self.num_correction_bits)

    @property
    def actual_bits_per_element(self) -> float:
        total_bits = (
            self.radius_bits
            + self.num_sign_bits * 1
            + self.num_corrections * self.num_correction_bits
        )
        return total_bits / self.head_dim


class JLProjector:
    """JL random projection for angular preservation via sign bits."""

    def __init__(self, input_dim: int, output_dim: int, seed: int = 42):
        self.input_dim = input_dim
        self.output_dim = output_dim

        rng = torch.Generator()
        rng.manual_seed(seed)
        # scaled by 1/sqrt(m) for norm preservation per JL lemma
        self.projection_matrix = torch.randn(
            output_dim, input_dim, generator=rng
        ) / np.sqrt(output_dim)

    def project(self, vectors: torch.Tensor) -> torch.Tensor:
        device = vectors.device
        proj = self.projection_matrix.to(device)
        return torch.matmul(vectors, proj.T)

    def get_signs(self, vectors: torch.Tensor) -> torch.Tensor:
        projected = self.project(vectors)
        return (projected > 0).to(torch.int8)

    def to(self, device):
        self.projection_matrix = self.projection_matrix.to(device)
        return self


class PolarQuantizer:
    """
    Quantizes KV cache via polar decomposition + JL projection.
    v -> (radius, signs, corrections) at ~3 bits/element.
    """

    def __init__(self, config: PolarQuantConfig):
        self.config = config
        self.jl = JLProjector(
            input_dim=config.head_dim,
            output_dim=config.num_sign_bits,
            seed=config.jl_seed,
        )
        self._reconstruction_matrix = None

    def _get_reconstruction_matrix(self, device: torch.device) -> torch.Tensor:
        if self._reconstruction_matrix is None or self._reconstruction_matrix.device != device:
            proj = self.jl.projection_matrix.to(device)
            self._reconstruction_matrix = torch.linalg.pinv(proj)
        return self._reconstruction_matrix

    def encode(self, vectors: torch.Tensor) -> dict:
        # polar decomposition: magnitude + direction
        radius = torch.norm(vectors, dim=-1, keepdim=True)
        direction = vectors / (radius + 1e-8)

        # quantize radius to 8 bits
        radius_squeezed = radius.squeeze(-1)
        r_min = radius_squeezed.min()
        r_max = radius_squeezed.max()
        r_scale = (r_max - r_min) / 255.0
        radius_quantized = ((radius_squeezed - r_min) / (r_scale + 1e-8)).clamp(0, 255).to(torch.uint8)

        # JL projection -> sign bits
        signs = self.jl.get_signs(direction)

        # reconstruct from signs to compute residual
        recon_matrix = self._get_reconstruction_matrix(vectors.device)
        signs_float = (2.0 * signs.float() - 1.0)
        direction_approx = torch.matmul(signs_float, recon_matrix.T)
        direction_approx = direction_approx / (
            torch.norm(direction_approx, dim=-1, keepdim=True) + 1e-8
        )

        # top-k residual corrections
        residual = direction - direction_approx
        k = self.config.num_corrections
        _, top_indices = torch.topk(residual.abs(), k=k, dim=-1)

        corrections_raw = torch.gather(residual, -1, top_indices)
        c_max = corrections_raw.abs().max() + 1e-8
        corrections_quantized = ((corrections_raw / c_max + 1) * 1.5).clamp(0, 3).to(torch.uint8)

        return {
            "radius_quantized": radius_quantized,
            "radius_min": r_min,
            "radius_scale": r_scale,
            "signs": signs,
            "corrections": corrections_quantized,
            "correction_indices": top_indices,
            "correction_scale": c_max,
        }

    def decode(self, encoded: dict) -> torch.Tensor:
        device = encoded["signs"].device

        radius = (
            encoded["radius_quantized"].float() * encoded["radius_scale"]
            + encoded["radius_min"]
        ).unsqueeze(-1)

        recon_matrix = self._get_reconstruction_matrix(device)
        signs_float = (2.0 * encoded["signs"].float() - 1.0)
        direction = torch.matmul(signs_float, recon_matrix.T)
        direction = direction / (torch.norm(direction, dim=-1, keepdim=True) + 1e-8)

        corrections_float = (
            encoded["corrections"].float() / 1.5 - 1.0
        ) * encoded["correction_scale"]

        correction_full = torch.zeros_like(direction)
        correction_full.scatter_(-1, encoded["correction_indices"], corrections_float)
        direction = direction + correction_full
        direction = direction / (torch.norm(direction, dim=-1, keepdim=True) + 1e-8)

        return radius * direction

    def compute_memory_bytes(self, batch_size: int, seq_len: int) -> dict:
        cfg = self.config

        radius_bytes = batch_size * seq_len * 1 + 8
        sign_bytes = batch_size * seq_len * cfg.num_sign_bits // 8
        correction_bytes = (
            batch_size * seq_len * cfg.num_corrections * cfg.num_correction_bits // 8
        )
        index_bytes = batch_size * seq_len * cfg.num_corrections * 2

        total = radius_bytes + sign_bytes + correction_bytes + index_bytes
        baseline = batch_size * seq_len * cfg.head_dim * 2

        return {
            "compressed_bytes": total,
            "baseline_fp16_bytes": baseline,
            "compression_ratio": baseline / total,
            "bits_per_element": (total * 8) / (batch_size * seq_len * cfg.head_dim),
            "breakdown": {
                "radius": radius_bytes,
                "signs": sign_bytes,
                "corrections": correction_bytes,
                "indices": index_bytes,
            },
        }


class QuantizedKVCache:
    """Drop-in KV cache that stores keys/values in compressed polar form."""

    def __init__(self, config: PolarQuantConfig, num_layers: int = 32):
        self.config = config
        self.quantizer = PolarQuantizer(config)
        self.num_layers = num_layers
        self._cache = {}

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ):
        B, H, S, D = key_states.shape

        keys_flat = key_states.reshape(B * H, S, D)
        values_flat = value_states.reshape(B * H, S, D)

        encoded_keys = self.quantizer.encode(keys_flat)
        encoded_values = self.quantizer.encode(values_flat)

        self._cache[layer_idx] = {
            "keys": encoded_keys,
            "values": encoded_values,
            "shape": (B, H, S, D),
        }

    def get(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        entry = self._cache[layer_idx]
        B, H, S, D = entry["shape"]

        keys_flat = self.quantizer.decode(entry["keys"])
        values_flat = self.quantizer.decode(entry["values"])

        keys = keys_flat.reshape(B, H, S, D)
        values = values_flat.reshape(B, H, S, D)

        return keys, values

    def memory_usage(self) -> dict:
        total_compressed = 0
        total_baseline = 0

        for layer_idx, entry in self._cache.items():
            B, H, S, D = entry["shape"]
            mem = self.quantizer.compute_memory_bytes(B * H, S)
            total_compressed += mem["compressed_bytes"] * 2
            total_baseline += mem["baseline_fp16_bytes"] * 2

        return {
            "total_compressed_mb": total_compressed / (1024 * 1024),
            "total_baseline_fp16_mb": total_baseline / (1024 * 1024),
            "compression_ratio": total_baseline / (total_compressed + 1e-8),
            "num_layers_cached": len(self._cache),
        }


def evaluate_reconstruction_quality(
    config: PolarQuantConfig,
    num_vectors: int = 1000,
    seq_len: int = 512,
    device: str = "cpu",
) -> dict:
    torch.manual_seed(config.jl_seed)

    quantizer = PolarQuantizer(config)
    quantizer.jl.to(device)

    # realistic KV vectors: gaussian with varying magnitudes
    vectors = torch.randn(num_vectors, seq_len, config.head_dim, device=device)
    magnitudes = torch.exp(torch.randn(num_vectors, seq_len, 1, device=device) * 0.5)
    vectors = vectors * magnitudes

    t_start = time.perf_counter()
    encoded = quantizer.encode(vectors)
    encode_time = time.perf_counter() - t_start

    t_start = time.perf_counter()
    reconstructed = quantizer.decode(encoded)
    decode_time = time.perf_counter() - t_start

    cos_sim = torch.nn.functional.cosine_similarity(
        vectors.reshape(-1, config.head_dim),
        reconstructed.reshape(-1, config.head_dim),
        dim=-1,
    )

    l2_error = torch.norm(vectors - reconstructed, dim=-1)
    l2_norm = torch.norm(vectors, dim=-1)
    relative_error = (l2_error / (l2_norm + 1e-8)).mean()

    mse = torch.mean((vectors - reconstructed) ** 2)
    mem = quantizer.compute_memory_bytes(num_vectors, seq_len)

    return {
        "cosine_similarity_mean": cos_sim.mean().item(),
        "cosine_similarity_std": cos_sim.std().item(),
        "cosine_similarity_min": cos_sim.min().item(),
        "relative_l2_error": relative_error.item(),
        "mse": mse.item(),
        "encode_time_ms": encode_time * 1000,
        "decode_time_ms": decode_time * 1000,
        "bits_per_element": mem["bits_per_element"],
        "compression_ratio": mem["compression_ratio"],
        "memory": mem,
    }


def run_bit_sweep(device: str = "cpu"):
    print(f"{'Bits/Elem':>10} {'Cosine Sim':>12} {'Rel L2 Err':>12} {'Compression':>12}")
    print("-" * 50)

    results = []
    for target_bits in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]:
        config = PolarQuantConfig(head_dim=128, target_bits_per_element=target_bits)
        metrics = evaluate_reconstruction_quality(config, num_vectors=100, seq_len=256, device=device)
        results.append({"bits": target_bits, **metrics})
        print(
            f"{target_bits:>10.1f} "
            f"{metrics['cosine_similarity_mean']:>12.4f} "
            f"{metrics['relative_l2_error']:>12.4f} "
            f"{metrics['compression_ratio']:>12.2f}x"
        )

    return results


def demo_kv_cache(device: str = "cpu"):
    config = PolarQuantConfig(head_dim=128, target_bits_per_element=3.0)
    cache = QuantizedKVCache(config, num_layers=32)

    batch_size = 1
    num_heads = 32
    seq_len = 4096
    head_dim = 128

    print(f"\n  Config: batch={batch_size}, heads={num_heads}, "
          f"seq={seq_len}, dim={head_dim}")
    print(f"  Target: {config.actual_bits_per_element:.1f} bits/element")
    print(f"  JL dims: {config.num_sign_bits}, corrections: {config.num_corrections}")

    print("\n  Filling cache...")
    total_encode_time = 0

    for layer in range(32):
        keys = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        values = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

        t0 = time.perf_counter()
        cache.update(keys, values, layer)
        total_encode_time += time.perf_counter() - t0

    mem = cache.memory_usage()
    print(f"\n    Baseline (FP16): {mem['total_baseline_fp16_mb']:.1f} MB")
    print(f"    Compressed:      {mem['total_compressed_mb']:.1f} MB")
    print(f"    Ratio:           {mem['compression_ratio']:.2f}x")
    print(f"    Encode time:     {total_encode_time*1000:.1f} ms (all 32 layers)")

    # reconstruction quality check
    keys_orig = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    values_orig = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    cache.update(keys_orig, values_orig, 0)

    keys_recon, values_recon = cache.get(0)

    cos_sim_k = torch.nn.functional.cosine_similarity(
        keys_orig.reshape(-1, head_dim), keys_recon.reshape(-1, head_dim), dim=-1
    ).mean()
    cos_sim_v = torch.nn.functional.cosine_similarity(
        values_orig.reshape(-1, head_dim), values_recon.reshape(-1, head_dim), dim=-1
    ).mean()

    print(f"    Key cosine sim:   {cos_sim_k:.4f}")
    print(f"    Value cosine sim: {cos_sim_v:.4f}")

    return mem


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    config = PolarQuantConfig(head_dim=128, target_bits_per_element=3.0)
    print(f"PolarQuant evaluation (3 bits/element)")
    print(f"  Head dim: {config.head_dim}")
    print(f"  Sign bits: {config.num_sign_bits}, corrections: {config.num_corrections} x {config.num_correction_bits}b")
    print(f"  Actual bits/element: {config.actual_bits_per_element:.2f}")

    metrics = evaluate_reconstruction_quality(config, num_vectors=200, seq_len=512, device=device)
    print(f"\n  Cosine similarity: {metrics['cosine_similarity_mean']:.4f} "
          f"(+/-{metrics['cosine_similarity_std']:.4f})")
    print(f"  Relative L2 error: {metrics['relative_l2_error']:.4f}")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  Compression ratio: {metrics['compression_ratio']:.2f}x")
    print(f"  Encode: {metrics['encode_time_ms']:.1f} ms, Decode: {metrics['decode_time_ms']:.1f} ms")

    print("\nBit-rate sweep:")
    sweep_results = run_bit_sweep(device)

    print("\nKV cache demo (32-layer transformer):")
    demo_kv_cache(device)

    print("\nDone.")


if __name__ == "__main__":
    main()
