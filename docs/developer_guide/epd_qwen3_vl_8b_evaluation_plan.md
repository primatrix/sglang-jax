# Qwen3-VL-8B-Instruct EPD 评测计划

状态：已提交非 EPD 摸底实验，当前处理服务启动问题；性能数值与 SLO 待定。

## 目标与顺序

当前任务是先采集非 EPD baseline 数据，了解 Qwen3-VL-8B-Instruct 在现有 TPU 资源上的正确性、稳定性、吞吐和延迟分布，判断模型及请求配置是否适合本次实验。现在不设 SLO 数值、分位数选取规则、阈值倍数或校准负载公式。看完实测数据后，再决定三档 SLO，最后用确定的标准比较 EPD。

原讨论中的 `strict、strict、relaxed` 暂解释为 `strict、moderate、relaxed`。档位名称暂作占位，具体指标、阈值与达标率要求均待 baseline 测量后决定；届时定义的是实验比较标准，不是生产业务承诺。性能实验不能单独证明模型适合某种业务任务。

## 硬件与部署

- 硬件：用户提供的 v7x slice，拓扑 `2×2×1`，4 chips，每 chip 2 devices，共 8 devices。
- 模型：`Qwen/Qwen3-VL-8B-Instruct`；保存确切模型与 tokenizer/processor revision。
- 精度：首轮计划 BF16；启动后核实实际 dtype，各组一致。
- 保存代码 commit、JAX/jaxlib/libtpu 版本、启动命令、设备映射、并行配置、上下文长度、KV cache 配置、processor 像素限制、CPU 与传输配置。
- 资源优先按完整 chip 分配；下表是资源预算，不等同于必须采用相同数值的 TP。

当前固定一个非 EPD baseline，加两种 E/PD 资源比例；不再扫描 baseline 的 DP/TP 配置。三组均使用全部 8 devices。

| ID | 部署 | 资源分配 | 语言模型并行 | 顺序 |
|---|---|---|---|---|
| N1 | 非 EPD，合并 E/P/D | 4 chips / 8 devices | DP4 × 有效 TP2 | 首先测数据，再定 SLO |
| E1 | E 与合并 PD 分离 | E 2 chips / 4 devices；PD 2 chips / 4 devices | PD：DP2 × 有效 TP2 | SLO 确定后比较 |
| E2 | E 与合并 PD 分离 | E 1 chip / 2 devices；PD 3 chips / 6 devices | PD：DP3 × 有效 TP2 | 与 E1 使用相同请求比较 |

E1、E2 都是 E 与 PD 分离，不表示 P、D 也独立部署。E2 检查较少 encoder 资源是否已经足够，以及把余下资源用于 PD 是否改善整体表现。不能预先假定 encoder 一定不构成瓶颈；多图和高分辨率请求可能改变最优比例。

### 固定的 DP / TP 配置

非 EPD N1：

```bash
--tp-size 8 --dp-size 4 --vision-encoder-parallel dp
```

| 配置 / 进程 | CLI tp-size | CLI dp-size | 说明 |
|---|---:|---:|---|
| N1 合并服务 | 8 | 4 | 8 devices，DP4 × TP2，视觉 DP |
| E1 encoder | 4 | 4 | 4 devices，视觉 DP |
| E1 PD | 4 | 2 | 4 devices，DP2 × TP2 |
| E2 encoder | 2 | 2 | 2 devices，视觉 DP |
| E2 PD | 6 | 3 | 6 devices，DP3 × TP2 |

Encoder 使用 `--encoder-only --vision-encoder-parallel dp`；启动时还需配置 encoder 与 PD 的连接及传输参数。上表是并行参数，不是完整启动命令。

当前 model runner 的 `attention_tp_size = tp_size // dp_size`，因此 E2 的 `--tp-size 6 --dp-size 3` 表示有效 TP2，不是有效 TP6。三组保持相同总资源及有效 TP2，DP 数随 E/PD 分配变化。内部 DP 不等同于独立服务副本。

代码中的设备子集 mesh 路径支持 reshape 为 `(3, 2)`，scheduler 也按 dp_size 分配请求；这说明 E2 可作为待验证配置，不代表完整模型、编译与传输路径已在 TPU 上验证。部署时通过 `--device-indexes` 明确选择互不重叠的 E/PD 设备子集，核实物理 chip 与 device 的映射，尽量让每个 TP2 组位于同一 chip。不能只设置 tp-size 而不限制进程使用的设备。

N1 的起始配置参考仓库同拓扑 Qwen3-VL-32B cookbook；8B 和两种 EPD 布局都需实际验证。视觉 `dp` 在所选 mesh devices 上分配图片，与语言模型 dp-size 是不同控制项。

各组保持模型精度、请求数据、总并发、缓存策略和可比的全局调度预算一致。检查 DP=3 时批次/预编译 buckets、token padding 和请求预算的整除约束；scheduler 会将 max-running-requests 向下调整为 dp_size 的倍数，因此应选各组都可整除的全局预算并记录实际值。先测 N1 的 A、C 请求曲线，再决定 SLO；不预设阈值。

## 阶段 0：判断模型是否可用于实验

1. 在 N1 上加载模型，核实实际设备使用、精度和显存占用。
2. 用 10–20 个不同的真实图文请求检查单图、多图和流式输出，允许正常 EOS。检查描述是否与图片相符，并与可信的合并实现或模型参考输出对照；允许浮点差异，不要求生成文本逐字相同。
3. 逐一验证下表 A–D 的形状和输出长度，检查图片顺序、视觉 token 数及上下文长度是否正确，无截断、OOM、超时或异常结果。
4. 对每种形状及待测并发预跑，确认正式计时阶段没有 JAX 编译；正式性能运行关闭 profiler。
5. 后续启动 E1/E2 前，分别验证设备隔离、PD mesh、编译和 encoder 传输；尤其检查 E2 的 DP3 配置。若失败，记录具体原因，不能将未运行的配置作为实测结果。

输出判断分为：模型功能是否可用、部署是否稳定、负载是否能区分 EPD 的价值。如果模型可运行但负载始终被 decode 或客户端限制，不能据此断言 EPD 无收益，应先定位瓶颈。

## 请求配置

第一轮使用现有 `sgl_jax.bench_serving --dataset-name image` 生成可控图文请求。

| 组别 | 每请求图片 | 文本输入目标 tokens | 输出 tokens | 目的 |
|---|---|---:|---:|---|
| A | 1 张 512×512 | 256 | 128 | 轻视觉负载与分离开销 |
| B | 1 张 1024×1024 | 256 | 128 | 高分辨率负载 |
| C | 4 张 512×512 | 256 | 128 | 多图编码、batch 与传输 |
| D | 4 张 512×512 | 256 | 512 | 编码与长生成重叠 |
| balance | 1 张 512×512 | 2048 | 1024 | 用户追加的较长输入/输出非 EPD 实验 |

- 优先完成 A、C，再扩展 B、D。
- 固定图片数量，不启用 `--random-image-count`。
- 使用 JPEG、`--image-content random` 和固定 seed；首轮 seed=42，重复实验可用 42/43/44，但所有部署使用配对 seed 和相同请求数。
- 设置 `--random-range-ratio 1.0`，固定采样长度。当前实现的默认值 0.0 会在 1 到目标长度之间采样。
- 性能测试保持默认 ignore EOS，核实实际生成长度；正确性测试允许正常 EOS。
- 文本目标长度不等于包含 chat template 和视觉输入后的总长度。保存实际文本、视觉与输出 token 数。
- B 与 C 总输入像素数相同，但不假定预处理后 token 数或计算量完全相同。
- 增加视觉 tokens 同时增加语言模型 prefill 工作，不能将全部变化归因于 encoder。
- 缓存策略在各部署间保持一致，记录命中率；独立图片避免重复视觉缓存，文本前缀缓存仍需检查。

第二轮补充几百张不同的真实照片、文档或图表，提前准备并保存同一份请求列表。统一问题可用“请描述图片中的主要内容、可见文字，以及各对象之间的关系。”避免在线图片下载混入实验。随机噪声 JPEG 的压缩率和预处理成本未必代表真实图片，因此分别报告合成与真实负载结果。

## 阶段 1：非 EPD 并发扫描

- 本阶段只运行固定的 N1（8 devices，DP4 × TP2），不扫描其他 DP/TP 配置。
- 总并发扫描 `1、2、4、8、16、32`；吞吐仍显著增长则继续 64。
- 使用 `--request-rate inf --max-concurrency C`，这是固定并发实验。
- 每点先用 100 个请求摸底。正式运行增加请求数，使每点持续至少约 1–3 分钟，每点重复三次；100 个请求不用于可靠估计 p99。
- 保存失败、超时、完成吞吐、输出 tokens/s、TTFT、TPOT、ITL、端到端延迟、缓存命中率与客户端资源情况。
- 达到吞吐平台、排队/延迟急剧增长或资源限制时停止加压并记录原因。若未观察到平台，只报告已测范围，不宣称得到最大容量。

示例：N1 已在 `127.0.0.1:30000` 启动，A 组，并发 8 的摸底运行：

```bash
python -m sgl_jax.bench_serving \
  --backend sglang-oai-chat \
  --host 127.0.0.1 \
  --port 30000 \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --tokenizer Qwen/Qwen3-VL-8B-Instruct \
  --dataset-name image \
  --image-count 1 \
  --image-resolution 512x512 \
  --image-format jpeg \
  --image-content random \
  --random-input-len 256 \
  --random-output-len 128 \
  --random-range-ratio 1.0 \
  --request-rate inf \
  --max-concurrency 8 \
  --num-prompts 100 \
  --warmup-requests 16 \
  --seed 42 \
  --output-details \
  --output-file qwen3vl8b_N1_A_c8_seed42.jsonl
```

按请求组修改图片数量、尺寸和输出长度。输出文件名必须区分部署、请求组、并发/到达率、seed 和重复次数。模型为本地路径或自定义 served name 时同步调整参数。16 次 warmup 不保证覆盖全部编译形状，仍需检查服务日志。

## 阶段 2：先看数据，再决定 SLO（待 baseline 完成）

本阶段目前只列出需要查看的数据，不预设如何从数据计算 SLO。

1. 整理各 workload 的 N1 在不同并发下的完成吞吐、TTFT/TPOT 分布、ITL、失败率及资源占用，并保留重复运行的波动。
2. 观察低负载延迟、吞吐增长区间、吞吐平台及延迟开始明显恶化的位置。需要时根据曲线选有限请求到达率补测，确认排队和稳定性；具体到达率看数据后选。
3. 判断 Qwen3-VL-8B 与当前请求规模是否适合评测：是否稳定可运行，是否能覆盖轻载到饱和，瓶颈在视觉编码、语言模型还是客户端。必要时先调整请求范围并重测。
4. 展示 baseline 表格和曲线后，再与用户确定三档 SLO 的指标组合、绝对阈值、分位数依据和是否要求某个达标比例。不自动套用 p95 或固定倍数，不把测得的延迟直接视为业务可接受标准。
5. 将选定标准及其数据依据写入本文，在正式比较 EPD 前固定；相同 workload 下，各部署使用相同标准。是否不同 workload 使用不同阈值，也在查看数据后决定。

| 等级（暂定名称） | 指标组合 | 延迟阈值 | 达标率要求 | 选择依据 |
|---|---|---|---|---|
| strict | 待 baseline 后决定 | 待定 | 待定 | 待实测数据 |
| moderate | 待 baseline 后决定 | 待定 | 待定 | 待实测数据 |
| relaxed | 待 baseline 后决定 | 待定 | 待定 | 待实测数据 |

### 原始数据保存要求

TPOT 使用每请求首 token 之后的平均输出 token 时间，按当前 benchmark 的 `(请求完成延迟 − TTFT) / (输出 tokens − 1)` 口径计算。它不能代替 ITL 尾延迟；后者单独报告。

本次已给 `--output-details` 补充逐请求 `latencies`、`tpots`、`successes`，与已有 `ttfts`、`itls`、`output_lens`、`errors` 一起保存。时间字段单位为秒；失败或输出不足两个 tokens 的请求 TPOT 为 null。这些字段用于之后按实测数据选择指标和阈值，本次不设 SLO。

摸底 runner 为 `scripts/disaggregation/bench_non_epd.py`，默认固定 N1，运行 A/C 和并发 1/2/4/8/16/32；每点独立预热并保存原始 JSONL、日志和毫秒单位的分位数汇总。默认每点 100 个请求、一次运行，只用于探索，不作为可靠 p99 或最终稳定性结论。正式重复测量根据摸底结果扩大请求数。

在已准备好的 TPU 环境中运行：

```bash
python scripts/disaggregation/bench_non_epd.py \
  --launch-server \
  --output-dir /tmp/qwen3vl-n1-baseline \
  --num-prompts 100
```

输出目录必须尚不存在。全局最大运行请求数为 96，上下文长度为 16384，BF16 模型/KV，chunked prefill 为 4096，关闭 radix cache。所有命令和服务信息保存在输出目录；发现失败或输出长度偏离指定预算（默认 128，balance 为 1024）时停止并保留日志。图文任务正确性还需单独检查真实图片输出，合成负载不能证明任务质量。

## 阶段 3：使用冻结 SLO 比较 EPD

- E1、E2 分别完成正确性和预热验证，再复用与 N1 相同的请求配置及配对 seed。三组总资源均为 8 devices。
- 保留固定并发吞吐—延迟曲线；有限到达率的扫描范围待 baseline 数据分析后确定，在转折处补点。各部署使用相同到达率，不按各自容量单独缩放。
- 有限到达率实验不加会限制实际发出的并发上限；如必须设置保护上限，记录客户端排队和实际发送率，不能将其当作未受限到达率实验。
- SLO 确定后，对三档分别报告 goodput（满足选定全部条件的成功请求数 / 测量时间）及达标率（达标数 / 总提交数）。失败、超时均不达标，不能从分母剔除。
- 固定请求数实验的测量时间从首个正式请求发出，到最后一个请求完成或超时，包含 drain；保持客户端超时设置一致。若额外报告稳态窗口，明确窗口与请求归属，不与全程结果混用。
- 报告每档已测到的最高 goodput、对应到达率和达标率；若届时选择了达标率要求，再报告满足该要求的吞吐范围。
- 同时保留原始吞吐、延迟分位数、失败率和资源占用，便于读者使用其他延迟要求解读结果。

## 性能原因分析与交付物

选择一个 EPD 收益点和一个退化点，用 `scripts/disaggregation/profile_epd.py` 单独采集 trace，分析 encoder 排队、设备计算、embedding 传输、语言模型排队、prefill 和 decode 重叠。JAX host dispatch 时间不等于设备计算时间；跨进程/主机阶段计时需要核实时间基准。

最终保存：

1. 模型功能与稳定性检查结果，以及 N1/E1/E2 的可行性。
2. 所有启动命令、环境信息、请求配置/真实数据清单、原始 benchmark JSONL 和请求明细。
3. baseline 实测数据与分析，以及之后根据数据确定的 SLO 表、选择依据和离线统计方法。
4. 吞吐—TTFT、吞吐—TPOT 曲线及三档 goodput—到达率曲线。
5. 收益/退化的 trace 解释与限制；不将单个 slice 的结果直接推广为跨主机扩展能力。

TPU 实验的最新状态见文末执行记录；模型可用性需要实际启动及请求结果支持，当前未生成任何实测 SLO。

## 参考 vLLM EPD 文章后的评测补充

参考：[Encoder Disaggregation for Scalable Multimodal Model Serving](https://vllm.ai/blog/2025-12-15-vllm-epd)，2025-12-15。以下文章设置只作方法参考，性能数字和阈值不迁移到本 TPU 实验。

### 文章的实验设置

文章在 4×A100 80GB 上测 Qwen3-VL-4B-Instruct，对比 1E+3PD 与四路合并 DP；文本输入约 400/2000 tokens，输出 150 tokens，每请求 1–4 张 640×640 图片，扫描请求到达率。它按 P99 TTFT、P99 TPOT 同时满足阈值时的最大可持续请求率定义 goodput，采用 TTFT 20000 ms、TPOT 100 ms 的实验阈值。来源见上述文章的 Performance Results。

### 本实验采纳的方法与需要验证的假设

以下是针对当前仓库和 TPU 的实验设计，不是文章报告的 TPU 结论：

- 保留已确定的三组部署。E2 的 1 chip E + 3 chips PD 在资源比例上对应文章的 1:3 思路；我们的每 chip 有两个 devices、语言模型有效 TP2，因此不能将其视为相同硬件或实例实现。
- 第一轮 A/C 与 balance 保持已提交参数，不追溯修改请求。A/C 探索视觉负载差异；balance 的 2048/1024 是用户指定的长输入、长输出配置，不能称为复现文章的 2000/150，也不因名字 balance 就断言三阶段计算均衡。
- baseline 摸底后，优先补“短/长文本 × 图片数 1/2/3/4”的受控矩阵，输出长度和图片分辨率在比较文本长度时固定。若需要贴近文章，再单列 400/2000 文本 tokens、150 输出 tokens、640×640 图片的参考组；这不是当前已提交实验。
- 用有限请求到达率扫描检查吞吐平台、排队和 P99 曲线，范围根据本机实测容量选取，不复制文章的 QPS。固定并发扫描仍用于第一轮找负载范围。
- 在 E1/E2 对比中记录 encoder 队列、编码批量、传输耗时、PD 队列和 ITL 尾延迟，解释 E 资源减少后是否成为瓶颈。语言输入变长会增加 prefill 工作，输出变长会增加 decode 工作；实际主导阶段由 trace 判定。
- 若后续要验证请求间干扰，单独增加文本请求与图文请求混合流量，并分别统计两类请求延迟。若实现支持 encoder 输出缓存，再单列缓存实验，记录命中率；不与无复用的主实验合并。
- 保留合并部署的视觉 DP，使 baseline 本身具备合理的视觉并行。EPD 收益应从设备计算、排队和传输证据中解释，不预先假定合并部署完全没有任何重叠，或 EPD 消除了所有排队。

### goodput 的两种口径分别命名

1. **逐请求达标吞吐（req/s）**：成功且同时满足选定请求级条件的请求数 / 全程测量时间；同时报告达标率，失败与超时仍计入提交总数。
2. **P99 约束容量（req/s）**：在正式选定 P99 TTFT/TPOT 阈值后，报告两个分位数均达标且运行稳定的已测最高到达率，同时报告实际完成吞吐、失败率和队列趋势。这是文章采用的 goodput 思路；有限测试只能给出已测容量，不保证数学意义上的最大值。

这两者不能互相替代：成功请求的 P99 达标也不等于所有提交请求的联合达标率达到 99%。P99 约束容量还需预先记录失败容忍度和稳定性判定方法，不能通过排除超时/失败制造较好的尾延迟。本轮每点 100 请求仅用于探索，不能据此宣布满足可靠的 P99 容量标准。

**SLO 仍然等 baseline 数据出来后再定**，包括是否采用 P99 约束容量、三档阈值及稳定性要求。文章的 20000/100 ms 仅作为来源记录，不作为本次默认值。

## 2026-09-08 执行记录

使用 Falcon 集群 `gke-tpu-train-us-central1-1-prod`，每个实验独占一个 4-chip / 8-device slice。

| 实验 | Falcon ID | 请求配置 | 提交状态 |
|---|---|---|---|
| N1 A/C 摸底 | `exp-jnnjyqru5b` | ISL文本256 / OSL128，1图和4图 | TPU 8 devices 检查通过；服务启动失败，无性能数据 |
| N1 A/C 诊断重试 | `exp-gkuhhpmjso` | 相同负载，补全启动日志 | 定位为旧参数 `--mm-io-worker-num`，已移除 |
| N1 balance 摸底 | `exp-u3nis3hkoz` | ISL文本2048 / OSL1024，1图 | TPU 8 devices 检查通过；服务启动失败，无性能数据 |

两组均扫并发 1/2/4/8/16/32，每点 100 个正式请求并单独预热。balance 的 ISL2048 不含视觉 tokens 与模板开销，实际输入量另存。当前提交不代表模型启动成功或性能验证完成，SLO 仍待定。

### 使用用户指定镜像重新运行

参考 `exp-oejdl0md4d`，镜像固定为 `ghcr.io/yanko-7/sglang-jax:py312-048b037-r1`。保留镜像的 JAX/PyTorch 依赖，安装本仓库使用 `pip install --no-deps -e ./python`；显式检查 `torchcodec.decoders.decode_image`，缺失时安装 CPU `torchcodec>=0.16.0`，必要时补 FFmpeg，并执行实际 JPEG 解码预检。

参考实验已有 benchmark 输出，最终失败于归档目录不存在；本次在退出复制前创建 artifact 目录。基准结果写入本地 scratch，结束后统一归档，避免测量期间频繁写挂载目录。代码基线为 `2b1c144b588166e80d783f392258e5382d937a07`，额外 runner 和客户端修改的哈希记录在实验 config。

| 实验 | Falcon ID | 状态 |
|---|---|---|
| N1 A/C，指定镜像 | `exp-aa007xfs7h` | 启动 payload 过长，未执行 Python |
| N1 balance，指定镜像 | `exp-mie7i7s4rm` | 启动 payload 过长，未执行 Python |
| N1 A/C，压缩 payload | `exp-hp1afp46t0` | 已提交；通过图片解码与 8-device 检查，等待测量 |
| N1 balance，压缩 payload | `exp-h4y47w3um2` | 已提交，性能数据待收集 |

两组都要求流式响应返回服务端 usage，并保存逐请求 usage；实际 completion tokens 必须等于指定 OSL。ISL 参数仍表示文本目标长度，同时报告服务端总 prompt tokens，避免将客户端估算当作精确服务端计数。

### 后续统一用 Git commit 部署

按用户要求，将 runner 和客户端变更推送到独立分支，后续 Falcon `sources` 直接拉取固定 commit，不再内嵌源码或传输代码包。环境入口为 `scripts/disaggregation/run_non_epd.sh`，在指定镜像中完成 TorchCodec 检查、代码安装和结果归档。

```bash
# A/C 摸底
bash scripts/disaggregation/run_non_epd.sh --groups A C --num-prompts 100
# balance 摸底（单图，文本 ISL2048 / OSL1024）
bash scripts/disaggregation/run_non_epd.sh --groups A --input-len 2048 --output-len 1024 --num-prompts 100
```

每个命令在独立 Falcon 实验中执行，沿用同一镜像、相同模型和 DP4×TP2；`ARTIFACT_LOCAL_DIR` 由 Falcon 提供。
