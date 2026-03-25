import argparse

from zoology.analysis.flash_vqg.flash_vqg_analysis_suite import (
    DEFAULT_ENTITY,
    DEFAULT_PROJECT,
    run_d128_analysis,
    run_dmodel_analysis,
)


def parse_args():
    parser = argparse.ArgumentParser(description="统一的 Flash-VQG 分析入口.")
    parser.add_argument("--mode", choices=["d128", "dmodel"], required=True, help="分析模式.")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="WandB project 名.")
    parser.add_argument("--entity", default=DEFAULT_ENTITY, help="WandB entity 名.")
    parser.add_argument("--launch-id", nargs="+", default=None, help="一个或多个 launch_id.")
    parser.add_argument("--sweep-id", nargs="+", default=None, help="一个或多个 sweep_id.")
    parser.add_argument("--output-dir", default=None, help="输出目录. 默认写到 flash_vqg/results 下的时间戳目录.")
    parser.add_argument("--selection-metric", default="valid/accuracy", help="选择最佳 lr 时使用的指标.")
    parser.add_argument("--metric", default=None, help="d128 模式的兼容别名. 若传入, 会覆盖 selection-metric.")
    parser.add_argument("--target-case", default="512x64", help="dmodel 模式重点画图的测试组合.")
    parser.add_argument("--expected-runs", type=int, default=None, help="期望 run 数. 只做提示, 不强制报错.")
    args = parser.parse_args()
    if not args.launch_id and not args.sweep_id:
        parser.error("至少传一个 --launch-id 或 --sweep-id.")
    return args


def main():
    args = parse_args()
    metric = args.metric or args.selection_metric
    if args.mode == "d128":
        run_d128_analysis(
            project=args.project,
            entity=args.entity,
            launch_ids=args.launch_id,
            sweep_ids=args.sweep_id,
            metric=metric,
            output_dir=args.output_dir,
            expected_runs=args.expected_runs if args.expected_runs is not None else 8,
        )
        return

    run_dmodel_analysis(
        project=args.project,
        entity=args.entity,
        launch_ids=args.launch_id,
        sweep_ids=args.sweep_id,
        selection_metric=metric,
        target_case=args.target_case,
        output_dir=args.output_dir,
        expected_runs=args.expected_runs if args.expected_runs is not None else 24,
    )


if __name__ == "__main__":
    main()
