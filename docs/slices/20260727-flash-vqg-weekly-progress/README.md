# Flash-VQG 周进展汇报稿

## 权威文件

- 内容规划: `outline.md`.
- 完整主源码: `beamer/weekly-progress.tex`.
- 公共导言: `beamer/shared-preamble.tex`.
- 最终产物: `final/weekly-progress.pdf`.

当前 P1–P19 的全部 frame 均集中在 `beamer/weekly-progress.tex`. 最终 PDF 共 19 页, 由原开题答辩 P1–P15 和周进展 P16–P19 组成. `outline.md` 中的 P20 自然语言下游评估页仍处于规划状态, 尚未写入主源码, 因此不在当前 PDF 中.

## 目录结构

```text
.
├── README.md
├── outline.md
├── beamer/
│   ├── weekly-progress.tex
│   ├── shared-preamble.tex
│   ├── build-final.sh
│   └── figures/
│       ├── p04/ ... p14/
│       ├── p18/
│       ├── p19/
│       ├── scu-brand/
│       └── ...
└── final/
    └── weekly-progress.pdf
```

`beamer/figures/p18/` 和 `beamer/figures/p19/` 分别保存对应页面使用的正式实验图. 图片已从 zoology artifact 原样复制并校验 SHA-256, 当前目录不再依赖外部图片路径.

## 完整编译

```bash
cd /home/lyj/mnt/project/zoology/docs/slices/20260727-flash-vqg-weekly-progress/beamer
./build-final.sh
```

脚本使用临时目录运行两遍 XeLaTeX, 只把最终 PDF 写入 `final/`, 不保留 LaTeX 辅助文件.

P16–P19 不再保留重复的独立单页源码. 后续修改直接在 `weekly-progress.tex` 中定位对应 frame, 再运行 `build-final.sh` 验证完整稿.
