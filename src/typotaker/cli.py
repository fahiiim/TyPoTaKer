from __future__ import annotations

import argparse
from pathlib import Path

from typotaker.config import ModelConfig, TrainConfig
from typotaker.data.prepare import build_gec_from_clean, train_tokenizer_from_jsonl, write_smoke_corpus
from typotaker.models.growth import grow_model
from typotaker.models.typoformer import TyPoFormer, count_parameters
from typotaker.train.checkpoint import find_latest, load_checkpoint, run_dir, save_checkpoint
from typotaker.train.trainer import train


def _repo_root() -> Path:
    return Path.cwd()


def cmd_prepare_smoke(_args: argparse.Namespace) -> None:
    write_smoke_corpus(_repo_root())


def cmd_build_corpus(args: argparse.Namespace) -> None:
    from typotaker.data.download import build_pretrain_corpus

    build_pretrain_corpus(
        args.out,
        tinystories=args.tinystories,
        fineweb_chunks=args.fineweb_chunks,
    )


def cmd_train_tokenizer(args: argparse.Namespace) -> None:
    train_tokenizer_from_jsonl(args.inputs, args.out, args.vocab_size)
    print(f"tokenizer -> {args.out}")


def cmd_inject_gec(args: argparse.Namespace) -> None:
    build_gec_from_clean(
        args.clean,
        args.tokenizer,
        args.out,
        k_replace=args.k_replace,
        k_append=args.k_append,
        seed=args.seed,
        corruption_prob=args.corruption_prob,
        min_ops=args.min_ops,
        max_ops=args.max_ops,
        max_unit_chars=args.max_unit_chars,
    )


def cmd_train(args: argparse.Namespace) -> None:
    cfg = TrainConfig.from_yaml(args.config)
    if args.run_name:
        cfg.run_name = args.run_name
    if args.max_hours is not None:
        cfg.train.max_hours = args.max_hours
    if args.no_compile:
        cfg.train.compile = False
    train(cfg)


def cmd_grow(args: argparse.Namespace) -> None:
    src_root = Path(args.src)
    latest = find_latest(src_root) if src_root.is_dir() else src_root
    if latest is None:
        raise SystemExit(f"no checkpoint at {args.src}")
    blob = load_checkpoint(latest)
    src_cfg = ModelConfig(**blob["meta"]["model"])
    src = TyPoFormer(src_cfg)
    src.load_state_dict(blob["model"])
    dst_cfg = ModelConfig.from_yaml(args.config)
    dst = grow_model(src, dst_cfg)
    print(f"grew {count_parameters(src):,} -> {count_parameters(dst):,} params")
    dest = run_dir(args.checkpoint_dir, args.run_name)
    save_checkpoint(
        root=dest,
        step=0,
        samples_seen=0,
        model=dst,
        optimizer=__import__("torch").optim.AdamW(dst.parameters(), lr=1e-4),
        scheduler=None,
        scaler=None,
        model_cfg=dst_cfg,
        extra={"grown_from": str(latest), "note": "optimizer reset after growth; resume this run"},
        keep_last=2,
    )
    print(f"stage checkpoint -> {dest / 'latest.json'}")
    print("Next: point train YAML init_from/resume at this directory and train for up to 6h.")


def cmd_export(args: argparse.Namespace) -> None:
    from typotaker.export import export_onnx, quantize_onnx

    exported = export_onnx(args.run, args.out, task=args.task, seq_len=args.seq_len)
    if getattr(args, "quantize", False):
        quantize_onnx(exported, getattr(args, "quantize_out", None))



def cmd_suggest(args: argparse.Namespace) -> None:
    from typotaker.infer import next_words, load_run

    model, tok = load_run(args.run, args.tokenizer)
    for piece, p in next_words(model, tok, args.text, k=args.k):
        print(f"{p:.3f}\t{piece!r}")


def cmd_correct(args: argparse.Namespace) -> None:
    from typotaker.infer import correct_sentence, load_run

    model, tok = load_run(args.run, args.tokenizer)
    print(
        correct_sentence(
            model,
            tok,
            args.text,
            args.maps,
            min_edit_prob=args.min_edit_prob,
            edit_margin=args.edit_margin,
        )
    )


def cmd_count(args: argparse.Namespace) -> None:
    cfg = ModelConfig.from_yaml(args.config)
    model = TyPoFormer(cfg)
    print(f"{cfg.name}: {count_parameters(model):,} parameters")


def main() -> None:
    p = argparse.ArgumentParser(prog="typotaker", description="TyPoTaKer training and inference CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("prepare-smoke", help="Write a tiny built-in LM corpus for dry runs")
    s.set_defaults(func=cmd_prepare_smoke)

    s = sub.add_parser("build-corpus", help="Download TinyStories + FineWeb-Edu and mix a train JSONL")
    s.add_argument("--out", default="data/processed/lm/train.jsonl")
    s.add_argument("--tinystories", type=int, default=80_000)
    s.add_argument("--fineweb-chunks", type=int, default=60_000)
    s.set_defaults(func=cmd_build_corpus)

    s = sub.add_parser("train-tokenizer")
    s.add_argument("--inputs", nargs="+", required=True)
    s.add_argument("--out", default="data/tokenizer/tokenizer.json")
    s.add_argument("--vocab-size", type=int, default=16000)
    s.set_defaults(func=cmd_train_tokenizer)

    s = sub.add_parser("inject-gec")
    s.add_argument("--clean", required=True)
    s.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    s.add_argument("--out", default="data/processed/gec/train.jsonl")
    s.add_argument("--k-replace", type=int, default=2048)
    s.add_argument("--k-append", type=int, default=2048)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--corruption-prob", type=float, default=0.85)
    s.add_argument("--min-ops", type=int, default=1)
    s.add_argument("--max-ops", type=int, default=2)
    s.add_argument("--max-unit-chars", type=int, default=384)
    s.set_defaults(func=cmd_inject_gec)

    s = sub.add_parser("train")
    s.add_argument("--config", required=True)
    s.add_argument("--run-name")
    s.add_argument("--max-hours", type=float)
    s.add_argument("--no-compile", action="store_true")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("grow", help="Expand a saved stage into a larger model and store step_00000000")
    s.add_argument("--src", required=True, help="checkpoint run directory")
    s.add_argument("--config", default="configs/model/stage1_40m.yaml")
    s.add_argument("--run-name", default="lm_stage1")
    s.add_argument("--checkpoint-dir", default="checkpoints")
    s.set_defaults(func=cmd_grow)

    s = sub.add_parser("export")
    s.add_argument("--run", required=True)
    s.add_argument("--out", default="export/model.onnx")
    s.add_argument("--task", choices=["lm", "gec"], default="lm")
    s.add_argument("--seq-len", type=int, default=128)
    s.add_argument("--quantize", action="store_true", help="Quantize exported ONNX model to INT8")
    s.add_argument("--quantize-out", help="Optional output path for INT8 quantized model")
    s.set_defaults(func=cmd_export)


    s = sub.add_parser("suggest")
    s.add_argument("--run", required=True)
    s.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    s.add_argument("--text", required=True)
    s.add_argument("-k", type=int, default=5)
    s.set_defaults(func=cmd_suggest)

    s = sub.add_parser("correct")
    s.add_argument("--run", required=True)
    s.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    s.add_argument("--maps", default="data/processed/gec/edit_maps.json")
    s.add_argument("--min-edit-prob", type=float, default=0.0)
    s.add_argument("--edit-margin", type=float, default=0.0)
    s.add_argument("--text", required=True)
    s.set_defaults(func=cmd_correct)

    s = sub.add_parser("count-params")
    s.add_argument("--config", default="configs/model/stage0_25m.yaml")
    s.set_defaults(func=cmd_count)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
