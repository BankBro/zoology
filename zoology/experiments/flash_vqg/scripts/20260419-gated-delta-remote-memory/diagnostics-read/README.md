# Read Diagnostics

这里只有在 `gated-delta` 写入本身不给正信号时才启用.

当前只保留两档旧 reader 放松配置:

- `score_only + shared_local_den + top2`
- `score_only + residual_add + top2`

目的不是正式晋升新 reader, 而是判断 `B4 gated-delta` 是否主要卡在 write-read mismatch.
