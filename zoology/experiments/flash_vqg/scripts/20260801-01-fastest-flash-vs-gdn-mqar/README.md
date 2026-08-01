# 当前最快 Flash 与 GDN 的 MQAR 正式对照

本目录实现`20260801-01-fastest-flash-vs-gdn-mqar`的三组BF16正式实验. 详细协议见[Plan](../../../../../docs/plans/20260801-01-fastest-flash-vs-gdn-mqar-plan.md).

正式入口:

```bash
export MQAR_FASTEST_GDN_RUN_TAG=20260801-fastest-gdn-mqar-01
bash zoology/experiments/flash_vqg/scripts/20260801-01-fastest-flash-vs-gdn-mqar/start_queue.sh
```

队列依次执行`preflight -> smoke/resume -> Q0 -> batch profile -> 9-run formal -> 13-case eval -> repro -> analysis`.
