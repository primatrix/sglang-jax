# Prompt: PD Stage 0-3 持续实现 agent

把下面内容（从「===」之间）整体粘到新 Claude Code 窗口的第一条消息。

===

你将在 sgl-jax 仓库持续完成 Prefill-Decode 分离（PD）特性的 Stage 0 →
Stage 1 → Stage 2 → Stage 3 实现，按 RFC 推进，暴露问题就根因修复。

## 第一步（必做）

读完整 cold-start handoff：
`/Users/jiongxuan/workspace/sgl-jax/docs/developer_guide/pd_epic_handoff.md`

这一份文档包含：当前进度、epic 工作流、TPU 集群与部署、关键代码位置、
参考资料、常用调试命令、已知坑、联系上下文。**所有后续操作的前置上下文
都在这里**，不要跳过。

读完后用一句话确认你处于：
- 分支 `epic/pd-disaggregation`（`git branch --show-current` 验证）
- 5 份 PD RFC 已存在于 `docs/rfc/2026-05-25-pd-*.md`（untracked）

## 任务范围

按顺序持续完成 **Stage 0 → Stage 1 → Stage 2 → Stage 3**：

| Stage | RFC 文件 | 内容简述 |
|---|---|---|
| 0 | `docs/rfc/2026-05-25-pd-transfer-foundation.md` | `JaxTransferWrapper` + 4-tuple ABC + 单 backend + byte round-trip |
| 1 | `docs/rfc/2026-05-25-pd-host-pool-side-channel.md` | `QueueHostKVPool` + D2H staging 双路径 + ZMQ pull-done |
| 2 | `docs/rfc/2026-05-25-pd-scheduler-e2e.md` | Bootstrap server + Mixin + tokenizer 透传 + 端到端跑通 |
| 3 | `docs/rfc/2026-05-25-pd-multihost-routing.md` | per-host transfer server + `sglang_router` 集成 |

**不要碰**：
- Stage 4 hardening（`docs/rfc/2026-05-25-pd-hardening.md`）—— 前 4 个
  Stage 跑通后由 user 决定是否启动

## 每个 stage 的内循环

1. 起 feature 分支：
   `git checkout -b feature/pd-stageN epic/pd-disaggregation`
2. 严格按对应 RFC 的「设计」章节实现代码 + 单测 + 集成测，不超出 RFC
   范围；超出的需求记到 followup（不动手）
3. 跑 CPU 单测验证：
   ```
   cd /Users/jiongxuan/workspace/sgl-jax
   PYTHONPATH=/Users/jiongxuan/workspace/sgl-jax/python \
     python3 -m pytest python/sgl_jax/test/disaggregation/ -v
   ```
   （**PYTHONPATH 必须前置**，本机有两份 sgl_jax，详见 handoff §8）
4. 跑跨 pod 集成测（手动 `kubectl exec`，命令模板见 handoff §7.2）；
   失败的日志贴出来 + 给 user 复述
5. squash 成 1-3 个干净 commit，merge 回 epic：
   ```
   git checkout epic/pd-disaggregation
   git merge --no-ff feature/pd-stageN
   ```
6. 向 user 报告 stage merge commit hash + 验收结果，**等用户确认**再
   进入下一个 stage（不要静默推进）

## 遇到问题的姿势

- **不要 try/except 兜底掩盖**。先复现、定位、读源码、修根因。
- 如果根因在前面 stage 的代码：
  - **当前 feature 分支**里加 fixup commit，最后 squash 时合并
  - **不要** amend epic 中已 merge 的中间 commit（CLAUDE.md 禁
    interactive rebase，且破坏 stage 边界）
- 如果根因是 RFC 设计偏差：
  - **先改对应 RFC**（直接编辑 `docs/rfc/2026-05-25-pd-*.md`）
  - 在 commit message 里记录"per RFC update on <date>: <reason>"
  - 跟 user 显式 flag 这次 RFC 变更
- 如果出现"已知坑"清单上的问题（handoff §8），直接按那里写的方式处理，
  不要重新踩。

## 不要做的事

- ❌ 不要 `git push`（无论是 fork 还是 upstream，都不 push）
- ❌ 不要开新的 GitHub issue 或 PR（等所有 stage 跑通后由 user 决定；
  Stage 0 已有 sub-issue #1199，不要重复开）
- ❌ 不要修改 `feat/hicache-pd-pr1`、`backup/feat-hicache-pd-pr1-full`
  分支（那是 PR-1 相关，与本 epic 隔离）
- ❌ 不要在 `epic/pd-disaggregation` 上直接 commit（只接受 feature
  分支 merge）
- ❌ 不要在 RFC 范围外加 feature（"顺便加个 metrics 吧"之类 —— 那是
  Stage 4）

## 完成判定

Stage 3 实现完、merge 到 epic、Stage 2 RFC §测试章节定义的「一句话
prompt 端到端跑通」复测仍 PASS（Stage 3 改了 multi-host 部署，要回归
端到端）—— 向 user 报告：

1. 四个 stage 的 merge commit hash + 短描述
2. 端到端 demo 跑通日志（一句话 prompt 输出 token stream）
3. 跨 pod byte-equality 数据（Stage 0 风格回归）
4. 任何修改过的 RFC（`git diff main..epic/pd-disaggregation -- docs/rfc/`）
5. Stage 4 hardening 是否需要启动 + 你的建议

## 开始

用 1-2 句话告诉 user 你打算第一步做什么，然后开干。

===

## prompt 设计说明（给 user 看，不给新 agent）

1. **第一步强制读 handoff doc**：保证新 agent 在 cold-start 时一次性
   拿到所有上下文，不靠它自己猜。
2. **明确任务边界**：做 stage 0-3，不动 stage 4 — 避免范围蔓延。
3. **每 stage merge 后要 user 确认**：你保有对节奏的控制，避免静默连
   推四个 stage 让你失去 review 窗口。
4. **多种"不要做的事"全列**：push、开 issue（特别提醒 #1199 不重复开）、
   改其他分支、epic 上直接 commit、RFC 范围外加 feature —— 把 PR-1
   期间出过的常见越界都显式封堵。
5. **根因/RFC 偏差处理流程**写明：避免新 agent 用 workaround / 静默
   改 RFC 的反模式。
