"""Benchmark CPU latency and memory footprint for ONNX and INT8 quantized models."""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def get_process_memory_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def benchmark_model(
    model_path: str,
    task: str = "gec",
    seq_lengths: list[int] = [8, 16, 32],
    warmup: int = 5,
    runs: int = 50,
) -> dict:
    file_size_mb = Path(model_path).stat().st_size / (1024 * 1024)
    gc.collect()
    mem_before = get_process_memory_mb()

    t_load_0 = time.perf_counter()
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = min(4, os.cpu_count() or 4)
    sess = ort.InferenceSession(model_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])
    load_time_ms = (time.perf_counter() - t_load_0) * 1000

    mem_after = get_process_memory_mb()
    mem_delta_mb = max(0.0, mem_after - mem_before)

    results_by_seq = {}
    for seq_len in seq_lengths:
        token_ids = np.ones((1, seq_len), dtype=np.int64)
        char_ids = np.zeros((1, seq_len, 16), dtype=np.int64)
        inputs = {"token_ids": token_ids, "char_ids": char_ids}

        for _ in range(warmup):
            sess.run(None, inputs)

        latencies = []
        for _ in range(runs):
            t0 = time.perf_counter()
            sess.run(None, inputs)
            latencies.append((time.perf_counter() - t0) * 1000)

        results_by_seq[seq_len] = {
            "mean_ms": float(np.mean(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
        }

    return {
        "model": Path(model_path).name,
        "task": task,
        "file_size_mb": file_size_mb,
        "load_time_ms": load_time_ms,
        "mem_delta_mb": mem_delta_mb,
        "latencies": results_by_seq,
    }


def main() -> None:
    models = [
        ("export/lm_stage1.onnx", "lm"),
        ("export/lm_stage1_int8.onnx", "lm"),
        ("export/gec_stage1.onnx", "gec"),
        ("export/gec_stage1_int8.onnx", "gec"),
    ]

    print("\n======================= ONNX CPU BENCHMARK =======================")
    print("Testing realistic keystroke latencies (batch=1) on CPU Execution Provider\n")

    for path, task in models:
        if not Path(path).exists():
            continue
        res = benchmark_model(path, task)
        print(f"--- {res['model']} ({res['task'].upper()}) ---")
        print(f"  File Size:      {res['file_size_mb']:.1f} MB")
        print(f"  Load Time:      {res['load_time_ms']:.1f} ms")
        if res["mem_delta_mb"] > 0:
            print(f"  RAM Delta:      {res['mem_delta_mb']:.1f} MB")
        for seq_len, lat in res["latencies"].items():
            print(
                f"  SeqLen {seq_len:2d}:  mean={lat['mean_ms']:.2f} ms | "
                f"p50={lat['p50_ms']:.2f} ms | p95={lat['p95_ms']:.2f} ms"
            )
        print()


if __name__ == "__main__":
    main()
