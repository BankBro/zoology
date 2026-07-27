# Flash-VQG 展示稿协作规范

- `main.tex` 只负责装配; `weeks/<date-range>/pNN-topic.tex` 每个文件只包含一个 frame.
- 周次图片放在 `figures/<date-range>/`; 跨周素材放在 `figures/common/`.
- 修改页面后先运行 `build-page.sh`; 调整页数或顺序后运行 `build-release.sh` 完整编译两遍.
- 编译必须检查 `LaTeX Error`, `Fatal error`, `Overfull` 和 `Underfull`, 并目视检查改动页.
- `previews/` 是本地生成物, 不提交; `docs/slices/` 中的已发布快照不覆盖, 修订时创建新 release.
- 操作命令和目录说明以本目录 `README.md` 为准.
