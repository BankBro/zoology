import argparse

from zoology.analysis.flash_vqg.flash_vqg_analysis_suite import DEFAULT_SOURCE, run_launch_analysis


def parse_args():
    parser = argparse.ArgumentParser(description="Flash-VQG SwanLab analysis 入口.")
    parser.add_argument("--launch-id", required=True, help="目标 launch_id.")
    parser.add_argument("--source", choices=["remote", "local"], default=DEFAULT_SOURCE, help="数据源.")
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_launch_analysis(launch_id=args.launch_id, source=args.source)
    print(f"analysis 输出目录: {result['output_dir']}")


if __name__ == "__main__":
    main()
