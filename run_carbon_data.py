from __future__ import annotations

import itertools
import pathlib
import shlex
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class RunSpec:
    seq_len: int
    label_len: int
    pred_len: int


@dataclass(frozen=True)
class SiteConfig:
    name: str           # Human/readable and used as model_id/data
    data_path: str      # CSV filename under root_path
    enc_in: int
    dec_in: int
    c_out: int


# Shared grid
RUN_SPECS = (
    RunSpec(seq_len=96, label_len=24, pred_len=96),
    # RunSpec(seq_len=96, label_len=24, pred_len=192),
    # RunSpec(seq_len=96, label_len=24, pred_len=336),
)

# Multi-GPU configuration for all runs
USE_MULTI_GPU = False
DEVICES = "0,1,2"
PATCH_LENS = (16,)

# Models to evaluate (folder per model for logs)
MODELS = (
    'PatchMLP',
    'TimeBridge',
    'DLinear',
    'Informer',
    'iTransformer',
    'TimeMixer',
    'TSMixer',
    'TimeXer',
    'CarbonMix',
)


# Sites with their specific channel sizes
SITES = (
    # SiteConfig(name="BaoTianMan",     data_path="BaoTianMan.csv",     enc_in=10, dec_in=10, c_out=10),
    # SiteConfig(name="DangXiong",         data_path="DangXiong.csv",         enc_in=10, dec_in=10, c_out=10),
    SiteConfig(name="DuoLun",        data_path="DuoLun.csv",        enc_in=11, dec_in=11, c_out=11),
    # SiteConfig(name="HaiBei",         data_path="HaiBei.csv",         enc_in=10, dec_in=10, c_out=10),
    # SiteConfig(name="MaoWuSu",     data_path="MaoWuSu.csv",     enc_in=10, dec_in=10, c_out=10),
    # SiteConfig(name="QianYanZhou",   data_path="QianYanZhou.csv",   enc_in=9,  dec_in=9,  c_out=9),
    # SiteConfig(name="XiaoLangDi",         data_path="XiaoLangDi.csv",         enc_in=11, dec_in=11, c_out=11),
    # SiteConfig(name="XiLin",         data_path="XiLin.csv",         enc_in=10, dec_in=10, c_out=10),
    # SiteConfig(name="XiShuangBanNa", data_path="XiShuangBanNa.csv", enc_in=9,  dec_in=9,  c_out=9),
    # SiteConfig(name="YuCheng",         data_path="YuCheng.csv",         enc_in=10, dec_in=10, c_out=10),
)


def base_cmd_for(site: SiteConfig, model: str) -> list[str]:
    return [
        sys.executable, "-u", "run.py",
        "--task_name", "long_term_forecast",
        "--is_training", "1",
        "--root_path", "./dataset/carbon/",
        "--data_path", site.data_path,
        "--target", "NEE",
        "--train_epochs", "1",
        "--model_id", site.name,
        "--model", model,
        "--data", site.name,
        "--features", "M",
        "--e_layers", "2",
        "--factor", "3",
        "--enc_in", str(site.enc_in),
        "--dec_in", str(site.dec_in),
        "--c_out", str(site.c_out),
        "--d_model", "512",
        "--d_ff", "512",
        "--des", "Carbon",
        "--itr", "1",
        "--batch_size","64",
        "--var_rank", "4",
        "--ctx_len", "8",
        "--gpu","3"
    ]


def build_command(site: SiteConfig, spec: RunSpec, patch_len: int, model: str) -> list[str]:
    cmd = list(base_cmd_for(site, model))
    cmd.extend([
        "--seq_len", str(spec.seq_len),
        "--label_len", str(spec.label_len),
        "--pred_len", str(spec.pred_len),
        "--patch_len", str(patch_len),
    ])
        # Append multi-GPU flags properly (store_true flag without value)
    if USE_MULTI_GPU:
        cmd.append("--use_multi_gpu")
        cmd.extend(["--devices", DEVICES])
    return cmd


def run_and_log(cmd: list[str], log_path: pathlib.Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure all args are strings before quoting/executing
    cmd = [str(t) for t in cmd]
    printable_cmd = " ".join(shlex.quote(t) for t in cmd)
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# Command\n{printable_cmd}\n\n")
        fh.flush()
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True)
        fh.write(f"\n# Exit code: {proc.returncode}\n")
        return proc.returncode

 
def run_site(site: SiteConfig) -> int:
    exit_code = 0
    dataset_root = pathlib.Path("./logs")
    logs_root = dataset_root / "carbon"
    for model in MODELS:
        model_log_dir = logs_root / model
        for spec, patch_len in itertools.product(RUN_SPECS, PATCH_LENS):
            log_name = f"{site.name}_{spec.seq_len}_{spec.pred_len}_patch{patch_len}.log"
            log_path = model_log_dir / log_name
            print(f"[run_all_search] {site.name}/{model}: Running {log_name}...")
            code = run_and_log(build_command(site, spec, patch_len, model), log_path)
            if code != 0:
                print(f"[run_all_search] {site.name}/{model}: Failed with code {code}, see {log_path}")
                exit_code = code
    return exit_code


def main() -> int:
    overall = 0
    for site in SITES:
        code = run_site(site)
        if code != 0:
            overall = code
    return overall


if __name__ == "__main__":
    raise SystemExit(main())
