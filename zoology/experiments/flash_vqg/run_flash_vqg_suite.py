import argparse
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
GENERATED_DIR = Path(__file__).resolve().parent / "generated"


def _parse_csv_ints(raw: str) -> list[int]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("dmodels 不能为空.")
    return [int(v) for v in values]


def _parse_csv_floats(raw: str) -> list[float]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("learning rates 不能为空.")
    return [float(v) for v in values]


def _slugify_dmodels(dmodels: list[int]) -> str:
    return "d" + "-".join(str(v) for v in dmodels)


def _build_filename(*, backend: str, include_gdn: bool, block_len: int, dmodels: list[int]) -> str:
    scope = "with-gdn" if include_gdn else "flash-only"
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return f"flash_vqg__{backend}__{scope}__block{block_len}__{_slugify_dmodels(dmodels)}__{timestamp}.py"


def _render_generated_config(
    *,
    backend: str,
    include_gdn: bool,
    block_len: int,
    dmodels: list[int],
    learning_rates: list[float],
    wandb_project: str,
    wandb_entity: str,
    max_epochs: int,
) -> str:
    lines = [
        "# -*- coding: utf-8 -*-",
        "# 此文件由 run_flash_vqg_suite.py 自动生成.",
        "# 如需调整实验参数, 请修改 wrapper 入参后重新生成.",
        "",
        "from zoology.experiments.flash_vqg.flash_vqg_suite import build_configs",
        "",
        "configs = build_configs(",
        f"    flash_backend={backend!r},",
        f"    include_gdn={include_gdn!r},",
        f"    block_len={block_len!r},",
        f"    dmodels={dmodels!r},",
        f"    learning_rates={learning_rates!r},",
        f"    wandb_project={wandb_project!r},",
        f"    wandb_entity={wandb_entity!r},",
        f"    max_epochs={max_epochs!r},",
        ")",
        "",
    ]
    return "\n".join(lines)


def _build_launch_command(
    *,
    generated_config_path: Path,
    name: str,
    outdir: str | None,
    parallelize: bool,
    gpus: str | None,
) -> list[str]:
    cmd = [sys.executable, "-m", "zoology.launch", str(generated_config_path), "--name", name]
    if outdir is not None:
        cmd.extend(["--outdir", outdir])
    if parallelize:
        cmd.append("-p")
    if gpus is not None:
        cmd.extend(["--gpus", gpus])
    return cmd


def main():
    parser = argparse.ArgumentParser(description="生成并启动 Flash-VQG MQAR 实验配置.")
    parser.add_argument("--backend", choices=["accel", "torch"], default="accel")
    parser.add_argument("--include-gdn", dest="include_gdn", action="store_true", default=True)
    parser.add_argument("--flash-only", dest="include_gdn", action="store_false")
    parser.add_argument("--block-len", type=int, default=8)
    parser.add_argument("--dmodels", type=str, default="128")
    parser.add_argument("--learning-rates", type=str, default="1e-4,3e-4,1e-3,3e-3")
    parser.add_argument("--project", type=str, default="flash_vqg_vs_gdn")
    parser.add_argument("--entity", type=str, default="scu-mclab")
    parser.add_argument("--max-epochs", type=int, default=32)
    parser.add_argument("--name", type=str, default="flash-vqg-suite")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("-p", "--parallelize", action="store_true")
    args = parser.parse_args()

    dmodels = _parse_csv_ints(args.dmodels)
    learning_rates = _parse_csv_floats(args.learning_rates)

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    generated_filename = _build_filename(
        backend=args.backend,
        include_gdn=args.include_gdn,
        block_len=args.block_len,
        dmodels=dmodels,
    )
    generated_path = GENERATED_DIR / generated_filename
    generated_path.write_text(
        _render_generated_config(
            backend=args.backend,
            include_gdn=args.include_gdn,
            block_len=args.block_len,
            dmodels=dmodels,
            learning_rates=learning_rates,
            wandb_project=args.project,
            wandb_entity=args.entity,
            max_epochs=args.max_epochs,
        ),
        encoding="utf-8",
    )

    launch_cmd = _build_launch_command(
        generated_config_path=generated_path,
        name=args.name,
        outdir=args.outdir,
        parallelize=args.parallelize,
        gpus=args.gpus,
    )
    print(f"已生成配置脚本: {generated_path}")
    print("即将执行命令:")
    print(" ".join(shlex.quote(part) for part in launch_cmd))
    subprocess.run(launch_cmd, check=True, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
