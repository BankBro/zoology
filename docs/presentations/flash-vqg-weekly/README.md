# Flash-VQG 滚动周报

这里保存 Flash-VQG 系列展示稿的唯一可维护源码. `docs/slices/` 只保存已经发布的只读快照, 不再承担源码工作区职责.

## 当前谱系

| 内容 | 源码 | 全稿页码 |
|---|---|---:|
| 2026.05.23 开题答辩基线 | `sections/20260523-proposal-baseline.tex` | P1--P15 |
| 2026.07.20--07.26 周进展 | `weeks/20260720-0726/*.tex` | P16--P19 |

`docs/slices/20260420-081319/` 是更早的独立 HTML 基线稿, 不属于这条 Beamer 滚动谱系.

## 目录约定

```text
flash-vqg-weekly/
├── main.tex
├── shared-preamble.tex
├── page-preview.tex
├── build-page.sh
├── build-release.sh
├── sections/
│   └── 20260523-proposal-baseline.tex
├── weeks/
│   └── 20260720-0726/
│       ├── p16-title.tex
│       ├── p17-gd-residual-efficiency.tex
│       ├── p18-cross-gpu-longer-mqar.tex
│       └── p19-low-precision.tex
├── plans/
│   └── 20260720-0726.md
├── previews/                         # 本地生成, 不提交
│   └── 20260720-0726/
│       └── pNN-topic.pdf
└── figures/
    ├── common/
    ├── 20260523/
    └── 20260720-0726/
```

- `main.tex` 只管理各段的装配顺序.
- `sections/` 保存长期基线或其他稳定章节.
- `weeks/` 每周新增一个目录, 每张幻灯片使用一份独立 tex 源码. 已汇报的页面原则上不回写.
- `figures/` 按来源周次分目录, 跨周复用的素材放 `common/`.
- `plans/` 保存每周内容规划, 不进入编译.

## 每周工作流

1. 新建 `weeks/YYYYMMDD-MMDD/` 和对应的 `figures/YYYYMMDD-MMDD/`.
2. 每页新建一个 `pNN-topic.tex`, 文件内只放一个 frame.
3. 在 `main.tex` 末尾按展示顺序逐页追加 `\input{weeks/.../pNN-topic.tex}`.
4. 单页设计阶段通过 `build-page.sh` 只编译当前页面, 不需要完整渲染全稿.
5. 定稿时从 `main.tex` 完整编译两遍, 发布到新的 `docs/slices/<date-topic>/final/`.
6. 在该发布目录写入 `release.md`, 记录父版本, 新增页, 页数, 源码位置, Git commit 和 SHA-256.
7. 发布后的 `final/` 与 `release.md` 视为不可变快照. 后续修订应创建新的 release, 不覆盖旧稿.

## 单页编译

例如只渲染 P18:

```bash
cd /home/lyj/mnt/project/zoology/docs/presentations/flash-vqg-weekly
./build-page.sh weeks/20260720-0726/p18-cross-gpu-longer-mqar.tex
```

默认输出为 `previews/20260720-0726/p18-cross-gpu-longer-mqar.pdf`. `previews/` 按周次组织, 是本地预览目录, 不提交 Git.

## 完整编译和发布

从仓库根目录运行:

```bash
docs/presentations/flash-vqg-weekly/build-release.sh \
  docs/slices/20260727-flash-vqg-weekly-progress \
  weekly-progress.pdf
```

脚本在临时目录中运行两遍 XeLaTeX, 检查错误及版面溢出, 最终只把 PDF 写入目标 release 的 `final/`.
