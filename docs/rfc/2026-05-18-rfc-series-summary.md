# sgl-jax HiCache + PD 分离 RFC 系列：交付总结

**日期**: 2026-05-18  
**Author**: john  
**Status**: 全部 5 份文档产出 + 通过 spec review

---

## 一、交付清单

### 调研文档（2 份）

| 文件 | 行数 | 内容 |
|---|---|---|
| `docs/research/2026-05-18-sglang-cache-pd-organization.md` | 721 | sglang 组织结构调研（HiCache + PD + UnifiedRadixCache 视角），含 sgl-jax 对照位置表 |
| `docs/research/2026-05-18-tpu-inference-jax-api-survey.md` | 634 | tpu-inference JAX/Pallas API 调研（按场景分类 + 与 SGLang CUDA API 对照表） |

### RFC 文档（3 份）

| 文件 | 行数 | 范围 |
|---|---|---|
| `docs/rfc/2026-05-18-rfc-0-unified-cache-and-kv-infra.md` | 1389 | **共同基础**：UnifiedRadixCache port + kv_cache_builder + 5 个 ABC 接口 |
| `docs/rfc/2026-05-18-rfc-1-hicache.md` | 1569 | **HiCache**：LRUHostKVPool + TPUKVCacheController + LocalFileStorage 实现 |
| `docs/rfc/2026-05-18-rfc-2-pd-disaggregation.md` | 1849 | **PD 分离**：JaxTransferKVEngine + BootstrapServer + Scheduler mixin |

**总计**: 5 份文档，6162 行。

---

## 二、关键决策汇总

### 战略层

| 决策 | 出处 | 备注 |
|---|---|---|
| **UnifiedRadixCache 默认且唯一** | RFC-0 ADR-1 | 删除 RadixCache / SWARadixCache；ChunkCache 保留作并列选项（非 fallback） |
| **TPU only，不为 GPU 留抽象** | RFC-0 ADR-2 | 接口直接用 JAX 原语（jax.device_put / pinned_host / jax.experimental.transfer） |
| **双 ABC 独立**（HiCacheStorage + KVTransferEngine） | RFC-0 ADR-3 | 协议层抽象本质不同：L3 内容寻址 vs PD P2P-routed |
| **kv_cache_builder 替换 scheduler if-elif** | RFC-0 ADR-4 | 与 sglang origin/main 一致 |
| **ChunkCache 一等公民 + D 节点第一版仅 ChunkCache** | RFC-0 ADR-5 + RFC-2 ADR-9 | 与 sglang 默认行为对齐：sglang 默认 D 走 ChunkCache 且有 opt-in flag 切 RadixCache；**sgl-jax 第一版不引入 opt-in flag**（不实现 D + RadixCache 路径），未来如有需求按 sglang 模式补加。PD 与 cache 类型在接口层正交，在部署层 D 节点 cache 选择被第一版限制。 |
| **兼容性兜底** | RFC-0 ADR-6 | chunked prefill / overlap / DP / mixed chunk / retract / partial rollout / SWA / Mamba / MLA 全保留 |

### HiCache 实现层

| 决策 | 出处 | 备注 |
|---|---|---|
| **jax.device_put 而非 Pallas** | RFC-1 ADR-1 | HiCache 不在延迟关键路径；Pallas 留作 profile 后优化点 |
| **ThreadPoolExecutor 异步** | RFC-1 ADR-2 | JAX C++ 调用释放 GIL，主线程 forward + 后台 D2H 真正并行 |
| **只做 write_through + write_back**（不做 selective） | RFC-1 ADR-3 | YAGNI |
| **不做 layer-wise overlap** | RFC-1 ADR-4 | XLA 静态编译不支持；改做 step-level overlap |
| **只做 file backend** | RFC-1 ADR-5 | 验证 ABC 设计；GPU 库（mooncake/nixl）不纳入 |
| **每进程独立 HiCache，不跨进程协调** | RFC-1 ADR-6 | sgl-jax 单 scheduler + thread-based worker；与 sglang 多 rank 进程模型不同 |
| **复用 KVCache 现有 get_cpu_copy / load_cpu_copy** | RFC-1 §6.3.1 | 不引入新 D2H/H2D 原语 |

### PD 实现层

| 决策 | 出处 | 备注 |
|---|---|---|
| **基于 jax.experimental.transfer** | RFC-2 ADR-1 | 唯一 backend；不引入 mooncake/NIXL/MoRI |
| **transfer_server 进程内单例** | RFC-2 ADR-2 | 与 tpu-inference 实践一致 |
| **进程级 role**（disaggregation_mode） | RFC-2 ADR-3 | 与 sglang 一致 |
| **HTTP-based Bootstrap Server** | RFC-2 ADR-4 | 调试方便；与 sglang 一致 |
| **D2H staging 单机默认启用** | RFC-2 ADR-5 | 与 tpu-inference 实践一致 |
| **Mixin 模式分隔事件循环** | RFC-2 ADR-6 | 不污染 Scheduler 主类 |
| **PD 主路径绕过 tree_cache** | RFC-2 ADR-7 | 与 sglang §3.6.2 一致；保证 ChunkCache + PD 兼容 |
| **不支持 partial KV pulling** | RFC-2 ADR-8 | jax.experimental.transfer 限制（已知） |

---

## 三、调研中的关键事实修正

### sglang 主分支事实（基于 origin/main HEAD `f04c52253`）

| 之前误判 | 实际事实 |
|---|---|
| 「UnifiedRadixCache + HiCache 集成被 revert」 | **未被 revert**。`bb306bc62` 只 revert 了 PR #24346（CI 测试）。一长串 PR (#23316/#23391/#24585/#24691/#24972/#25088/#25277/#25348) 全部 Merged |
| 「unified_radix_cache.py 1024 行」 | 实际 **1960 行**（origin/main） |
| 「TreeComponent 3 个 HiCache hook 是空 ABC」 | 三个 component（full/swa/mamba）**全部已实装** 3 个 hook |
| 「scheduler.py 大段 if-elif」 | 已重构为 **builder 模式**（`kv_cache_builder.build_kv_cache`） |
| 「HiCache 与 UnifiedRadixCache 互斥」 | **不互斥**——UnifiedRadixCache 内部已实装 HiCache（`init_hicache()`） |
| 「sgl-jax 可能成为先行者」 | **应作为跟随者/port 者**；L3/PD/Spec 兼容才是 upstream 真正窗口 |

### 关键 insight 修正

**用户提出的"P→D 是 L3 操作"猜想**：经代码核实只成立 30%。
- 共享（30%）：底层传输引擎（如 mooncake `TransferEngine`）可同进程共享
- 独立（70%）：UnifiedRadixCache **0 处**提及 PD/disagg/kv_send；PD 完全绕过 tree_cache 走 `KVSender.send`
- 唯一现存融合：`DecodeKVCacheOffloadManager`（D 把自己生成的 KV offload 到 L3，**不是** P 写 L3 / D 读 L3）

详见 `docs/research/2026-05-18-sglang-cache-pd-organization.md` §3.6。

---

## 四、Spec Review 过程

| RFC | 一轮 review | 应用修订 | 二轮 review | 最终状态 |
|---|---|---|---|---|
| RFC-0 | Approved + 5 advisory | 4/5 应用 | — | **Approved** |
| RFC-1 | Issues Found (7 critical) | 全部应用 | 6 resolved + 1 partial + 2 new contradictions | **Approved (after 2nd round fixes)** |
| RFC-2 | Issues Found (9 critical) | 全部应用 | — | **Approved (in current state)** |

### RFC-1 二轮修订要点
- §6 cache_controller 改用 `get_cpu_copy/load_cpu_copy` 替代不存在的 `gather_kv/scatter_kv`
- `device_pool.sharding` → `device_pool.kv_sharding`（sgl-jax 真实属性名）
- 加 `device_allocator` 独立依赖（KVCache 没有 allocator）
- 加 `tree_cache_evict_callback`（eviction 是 tree_cache 职责）
- §12 SPMD 章节重写（去掉错误的 leader+broadcast，改为每进程独立 HiCache）
- §7.2 file backend 从 per-token 改为 page-granular
- RFC-0 `HostBufferHandle.buffer` 改为 `Optional[jax.Array]` + 加 `read_indices/write_indices` 到 ABC（消除 §6.3.1 越界改 ABC 的争议）
- RFC-0 `MemoryPools` 加 `token_allocator` 字段（让 RFC-1 cache_controller 拿到 allocator）

### RFC-2 修订要点
- §4.1 `register_runner` 改为显式 scheduler 注入（KVCache 不持 runner 反向引用）
- §4.1 `await_pull` docstring 与实现对齐（明确阻塞语义）
- §4.2 ZmqPullNotifier 加 lock 保护 `pending_callbacks`
- §5.2 `copy_from_device` 改为 in-place update（不替换 buffer，明确 stable 语义）
- §6.2 BootstrapServer 加 health/heartbeat/unregister + TTL 淘汰
- §7.2 `JaxTransferKVManager.__init__` 补 `_send_executor/_recv_executor` 初始化
- §9.3 scheduler wiring 补 `_create_kv_manager` + 显式 register_runner
- §16.2 `generate_bootstrap_room` 明确 64-bit；说明 bootstrap_room 在 P/D/proxy 间流通路径
- §17 配置项补充

---

## 五、tree_cache × PD × HiCache 正交矩阵（贯穿三份 RFC）

| tree_cache | + PD | + HiCache | RFC-0 ChunkCache 处理 | RFC-1 实施 | RFC-2 实施 |
|---|---|---|---|---|---|
| **ChunkCache** | ✓ | ✗ | 保留（无 prefix 复用） | ChunkCache 不接入 HiCache | ChunkCache + PD 由 §12 验证 |
| **UnifiedRadixCache** | ✓ | ✓ | 默认 | 完整 HiCache hooks 集成 | UnifiedRadixCache + PD 由 §13 验证 |
| ~~RadixCache~~ | — | — | **删除**（被 UnifiedRadixCache + FullComponent 取代） | — | — |
| ~~SWARadixCache~~ | — | — | **删除**（被 UnifiedRadixCache + SWAComponent 取代） | — | — |

---

## 六、实施路线整合

### 阶段总览

```
RFC-0 (M0 → M1 → M2)  ──┬→  RFC-1 (H0 → H1 → H2 → H3)
                        │
                        └→  RFC-2 (D0 → D1 → D2 → D3)

RFC-1 和 RFC-2 在 RFC-0 M0 完成后可并行起步
（H0/D0 不依赖 UnifiedRadixCache port）
```

### RFC-0 阶段
- **M0** 基础设施 + ABC 落地（仅重构 scheduler 入口，无功能变化）
- **M1** UnifiedRadixCache + FullComponent（替代 RadixCache，Llama/Qwen KL 等价）
- **M2** SWAComponent + RecurrentStateComponent（与 SWARadixCache 并存，MiMo-V2 / KDA KL 等价；M3 阶段独立清理 legacy）

### RFC-1 阶段
- **H0** HostKVPool + Allocator（单元测试）
- **H1** TPUKVCacheController + D2H/H2D 端到端
- **H2** LocalFileStorage（L3 toy backend）
- **H3** SWA/Mamba HiCache 覆盖 + 性能验证

### RFC-2 阶段
- **D0** JaxTransferKVEngine + BootstrapServer（单元测试）
- **D1** 4-tuple ABC + Scheduler Mixin（ChunkCache + PD KL 等价）
- **D2** 部署 + Proxy + UnifiedRadixCache + PD 端到端
- **D3** PD + HiCache 三者兼容性 + 多主机文档

---

## 七、明确划清的范围边界

### 不在三份 RFC 范围内的工作（明确为「未来工作」）

| 项目 | 原因 |
|---|---|
| mooncake / NIXL / aibrix / hf3fs / eic / simm 等 storage backend | GPU 专用，TPU 不能直接用（你的明确指导） |
| Layer-wise H2D overlap | XLA 静态编译不支持 |
| LMCache 集成 | 与 HiCache 互斥的替代方案，不重复 |
| `write_through_selective` 策略 | YAGNI |
| Partial KV pulling（PD） | `jax.experimental.transfer` 不支持 |
| Pipeline Parallel + PD | sgl-jax 暂未支持 PP |
| Speculative decoding + PD | sgl-jax 暂未支持 spec dec |
| SWA / Mamba 模型 + PD | RFC-2 仅支持 standard MHA + MLA（state_type 字段已预留） |
| 跨进程共享 host pool | 复杂度高，sglang 也没做 |
| Ray 集群启动逻辑 | RFC-2 仅预留 backend 接口，部署留文档 |
| HA bootstrap server | 短期接受单点 |

### 这些工作如何接入

每个未来工作都可以**不破坏现有 ABC** 的情况下加入：
- 新 storage backend → 实现 `HiCacheStorage` ABC，注册到 builder
- Layer-wise overlap → 在 `TPUKVCacheController` 内部改异步策略，不动 ABC
- 新 PD backend → 实现 `KVTransferEngine` ABC，加 enum 分支
- PD + spec dec → Mixin 模式天然兼容，加新 event loop

---

## 八、风险与未决问题汇总

### 跨 RFC 共同风险

| Risk | 影响 | 缓解 |
|---|---|---|
| **`jax.experimental.transfer` API 变更** | PD backend 失效 | ABC 隔离 + 锁 JAX 版本 + fallback 计划（ZMQ + 显式 device_get/put） |
| **sglang upstream 持续 rebase** | port 工作量持续 | 锁定 sglang commit hash 做 port，落地后再 rebase |
| **测试覆盖 SWA/Mamba 端到端** | 不易复现 | 在 v6e-4 pod 跑实际模型推理（参考 [[v6e_pod_paths]]） |

### 未决问题（不阻塞 RFC 完成，留实施时决策）

| 问题 | 在哪里决策 |
|---|---|
| host pool 总大小 ratio vs absolute size 优先级 | RFC-1 实施 |
| PD bootstrap server 是否合并到 proxy | RFC-2 实施 |
| 多 host PD 部署用 Ray 还是 K8s | RFC-2 实施 |
| L3 file backend SHA256 key scheme 跨 backend 兼容性 | RFC-1 实施时验证 |
| PD + HiCache 同启时的资源分配（HBM、host memory） | RFC-1/RFC-2 联合实施时 benchmark |

---

## 九、需要审阅的关键点

请审阅时重点关注：

### RFC-0
1. ADR-1 (UnifiedRadixCache 默认且唯一) + ADR-5 (ChunkCache 一等公民) 是否准确反映你的意图
2. §1.3 tree_cache × PD × HiCache 正交矩阵
3. §3.2 文件路径变更清单（新增 / 修改 / 删除）
4. §16 实施路线（M0→M1→M2）

### RFC-1
1. ADR-1 (jax.device_put 而非 Pallas)、ADR-4 (无 layer-wise overlap) 的取舍
2. §6 TPUKVCacheController 与 sgl-jax 现有 KVCache 的对接（复用 get_cpu_copy / load_cpu_copy）
3. §7 LocalFileStorage 设计（page-granular + atomic write）
4. §12 SPMD 章节（每进程独立 HiCache 模型）
5. RFC-0 `MemoryPools` 加 `token_allocator` 字段是否合理

### RFC-2
1. ADR-1 (jax.experimental.transfer 唯一 backend) + ADR-8 (无 partial pull) 的限制接受度
2. §4 JaxTransferKVEngine 实现（ZMQ 侧通道 + 双路径 D2H staging）
3. §6 BootstrapServer 设计（HTTP + TTL + heartbeat）
4. §9 Mixin 模式（Prefill / Decode 各自独立 event loop）
5. §12/§13 ChunkCache + PD 和 UnifiedRadixCache + PD 的兼容性验证
6. §16 单机部署示例（4-process 模型：bootstrap + P + D + proxy）

---

## 十、Memory 已存的关键设计原则

为防止后续 conversation 中遗漏，已写入 memory：

| Memory 文件 | 内容 |
|---|---|
| `memory/hicache_pd_rfc_design_principles.md` | 抽象优先、简单实现、参考 tpu-inference 接口、不重投 GPU 库 |
| `memory/research_doc_no_version_history.md` | 调研/RFC 文档偏好：直接写最终事实，不留 v1/v2 修正历史 |
| `memory/cache_pd_orthogonal_matrix.md` | tree_cache × PD × HiCache 正交组合矩阵（ChunkCache+PD ✓，HiCache 必须基于 prefix-tree） |

---

## 十一、下一步建议

1. **你 review 三份 RFC + 反馈**（特别是 §九 列出的关键点）
2. RFC 落地后：
   - 创建 GitHub project board 跟踪 M0/H0/D0 → 完整阶段
   - 建立 owner 分配（你 / 团队）
   - 锁定 sglang origin/main 一个 commit hash 作为 port 基准（建议 `f04c52253`，2026-05-18）
3. 如果接受 RFC，建议 commit 三份 RFC + 两份调研到 main 分支，建立 RFC 评审记录

---

**End of Summary**
