import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pathlib import Path
from polar_quant import PolarQuantConfig, PolarQuantizer, QuantizedKVCache

RESULTS_DIR = Path("results")

MODEL_CONFIGS = {
    "LLaMA-7B": {"num_layers": 32, "num_heads": 32, "head_dim": 128},
    "LLaMA-13B": {"num_layers": 40, "num_heads": 40, "head_dim": 128},
    "LLaMA-70B": {"num_layers": 80, "num_heads": 64, "head_dim": 128},
    "Mistral-7B": {"num_layers": 32, "num_heads": 32, "head_dim": 128},
}

CONTEXT_LENGTHS = [2048, 4096, 8192, 16384, 32768]


def compute_kv_cache_memory_fp16(num_layers, num_heads, head_dim, seq_len, batch_size=1):
    # 2 (K+V) x layers x batch x heads x seq x dim x 2 bytes
    return 2 * num_layers * batch_size * num_heads * seq_len * head_dim * 2


def compute_kv_cache_memory_int8(num_layers, num_heads, head_dim, seq_len, batch_size=1):
    return 2 * num_layers * batch_size * num_heads * seq_len * head_dim * 1


def compute_kv_cache_memory_int4(num_layers, num_heads, head_dim, seq_len, batch_size=1):
    return 2 * num_layers * batch_size * num_heads * seq_len * head_dim // 2


def benchmark_memory_scaling():
    results = []

    for model_name, cfg in MODEL_CONFIGS.items():
        print(f"\n  {model_name}: layers={cfg['num_layers']}, heads={cfg['num_heads']}, dim={cfg['head_dim']}")

        for seq_len in CONTEXT_LENGTHS:
            fp16_bytes = compute_kv_cache_memory_fp16(
                cfg["num_layers"], cfg["num_heads"], cfg["head_dim"], seq_len
            )
            int8_bytes = compute_kv_cache_memory_int8(
                cfg["num_layers"], cfg["num_heads"], cfg["head_dim"], seq_len
            )
            int4_bytes = compute_kv_cache_memory_int4(
                cfg["num_layers"], cfg["num_heads"], cfg["head_dim"], seq_len
            )

            pq_config = PolarQuantConfig(
                head_dim=cfg["head_dim"], target_bits_per_element=3.0
            )
            quantizer = PolarQuantizer(pq_config)
            pq_mem = quantizer.compute_memory_bytes(cfg["num_heads"], seq_len)
            pq_bytes = pq_mem["compressed_bytes"] * cfg["num_layers"] * 2

            results.append({
                "model": model_name,
                "context_length": seq_len,
                "fp16_gb": fp16_bytes / (1024**3),
                "int8_gb": int8_bytes / (1024**3),
                "int4_gb": int4_bytes / (1024**3),
                "polarquant_gb": pq_bytes / (1024**3),
                "pq_vs_fp16": fp16_bytes / pq_bytes,
            })

    df = pd.DataFrame(results)

    print("\nMemory usage (GB), batch=1:\n")

    for model_name in MODEL_CONFIGS:
        model_df = df[df["model"] == model_name]
        print(f"  {model_name}:")
        print(f"  {'Context':>8} {'FP16':>8} {'INT8':>8} {'INT4':>8} "
              f"{'Polar(3b)':>10} {'Savings':>8}")
        print(f"  {'-'*56}")

        for _, row in model_df.iterrows():
            print(
                f"  {row['context_length']:>8,} "
                f"{row['fp16_gb']:>8.2f} "
                f"{row['int8_gb']:>8.2f} "
                f"{row['int4_gb']:>8.2f} "
                f"{row['polarquant_gb']:>10.2f} "
                f"{row['pq_vs_fp16']:>7.1f}x"
            )

    return df


def benchmark_reconstruction_vs_context(device: str = "cpu"):
    config = PolarQuantConfig(head_dim=128, target_bits_per_element=3.0)
    quantizer = PolarQuantizer(config)
    quantizer.jl.to(device)

    results = []
    print(f"\n  {'Seq Len':>8} {'Cos Sim':>10} {'Rel L2':>10} {'Enc (ms)':>10} {'Dec (ms)':>10}")
    print(f"  {'-'*52}")

    for seq_len in [128, 512, 1024, 2048, 4096, 8192]:
        vectors = torch.randn(4, seq_len, 128, device=device)
        magnitudes = torch.exp(torch.randn(4, seq_len, 1, device=device) * 0.5)
        vectors = vectors * magnitudes

        t0 = time.perf_counter()
        encoded = quantizer.encode(vectors)
        enc_time = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        reconstructed = quantizer.decode(encoded)
        dec_time = (time.perf_counter() - t0) * 1000

        cos_sim = torch.nn.functional.cosine_similarity(
            vectors.reshape(-1, 128), reconstructed.reshape(-1, 128), dim=-1
        ).mean().item()

        rel_err = (
            torch.norm(vectors - reconstructed, dim=-1)
            / (torch.norm(vectors, dim=-1) + 1e-8)
        ).mean().item()

        results.append({
            "seq_len": seq_len,
            "cosine_sim": cos_sim,
            "rel_l2_error": rel_err,
            "encode_ms": enc_time,
            "decode_ms": dec_time,
        })

        print(f"  {seq_len:>8,} {cos_sim:>10.4f} {rel_err:>10.4f} "
              f"{enc_time:>10.1f} {dec_time:>10.1f}")

    return results


def generate_plots(memory_df, quality_results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    llama70b = memory_df[memory_df["model"] == "LLaMA-70B"]
    axes[0].plot(llama70b["context_length"], llama70b["fp16_gb"], "o-", label="FP16", linewidth=2)
    axes[0].plot(llama70b["context_length"], llama70b["int8_gb"], "s-", label="INT8", linewidth=2)
    axes[0].plot(llama70b["context_length"], llama70b["int4_gb"], "^-", label="INT4", linewidth=2)
    axes[0].plot(llama70b["context_length"], llama70b["polarquant_gb"], "D-",
                 label="PolarQuant (3-bit)", linewidth=2, color="red")
    axes[0].set_xlabel("Context Length")
    axes[0].set_ylabel("KV Cache Memory (GB)")
    axes[0].set_title("LLaMA-70B: KV Cache Memory vs Context")
    axes[0].legend()
    axes[0].set_xscale("log", base=2)
    axes[0].grid(True, alpha=0.3)

    ctx_32k = memory_df[memory_df["context_length"] == 32768]
    models = ctx_32k["model"].tolist()
    x = np.arange(len(models))
    width = 0.2

    axes[1].bar(x - 1.5*width, ctx_32k["fp16_gb"], width, label="FP16", alpha=0.8)
    axes[1].bar(x - 0.5*width, ctx_32k["int8_gb"], width, label="INT8", alpha=0.8)
    axes[1].bar(x + 0.5*width, ctx_32k["int4_gb"], width, label="INT4", alpha=0.8)
    axes[1].bar(x + 1.5*width, ctx_32k["polarquant_gb"], width,
                label="PolarQuant", alpha=0.8, color="red")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m.replace("LLaMA-", "L") for m in models], rotation=15)
    axes[1].set_ylabel("Memory (GB)")
    axes[1].set_title("KV Cache at 32K Context")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3, axis="y")

    if quality_results:
        seq_lens = [r["seq_len"] for r in quality_results]
        cos_sims = [r["cosine_sim"] for r in quality_results]
        axes[2].plot(seq_lens, cos_sims, "o-", color="green", linewidth=2)
        axes[2].axhline(y=0.99, color="gray", linestyle="--", alpha=0.5, label="0.99 threshold")
        axes[2].set_xlabel("Sequence Length")
        axes[2].set_ylabel("Cosine Similarity")
        axes[2].set_title("Reconstruction Quality vs Context Length")
        axes[2].set_ylim(0.95, 1.0)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "kv_cache_benchmark.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved: {RESULTS_DIR / 'kv_cache_benchmark.png'}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    print("KV cache memory comparison:")
    memory_df = benchmark_memory_scaling()

    print("\nReconstruction quality vs context length:")
    quality_results = benchmark_reconstruction_vs_context(device)

    print("\n  Generating plots...")
    generate_plots(memory_df, quality_results)

    memory_df.to_csv(RESULTS_DIR / "memory_comparison.csv", index=False)
    print(f"  CSV saved: {RESULTS_DIR / 'memory_comparison.csv'}")

    llama70b_32k = memory_df[
        (memory_df["model"] == "LLaMA-70B") & (memory_df["context_length"] == 32768)
    ].iloc[0]
    print(f"\n  LLaMA-70B @ 32K:")
    print(f"    FP16:       {llama70b_32k['fp16_gb']:.1f} GB")
    print(f"    PolarQuant: {llama70b_32k['polarquant_gb']:.1f} GB")
    print(f"    Savings:    {llama70b_32k['pq_vs_fp16']:.1f}x")


if __name__ == "__main__":
    main()
