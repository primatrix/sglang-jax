# VLM 分享配图

原有九张图对应 Outline 文档的第 1、2.1、2.2、3.1、4、5、7 节；后续新增配图仅保存在本地仓库：
https://outline.infiscale-tech.com/doc/vlm-eosNGzaACl

- `04-vlm-component-comparison`：原有 LLM 组件、原组件扩展与 VLM 新增模块。
- `00-vlm-patches-placeholders`：图片 patchify、占位符展开与视觉特征对齐。
- `01-vlm-request-flow`：真实照片与提问示例、请求链路与视觉 embedding 合并。
- `02-vlm-lane-packing`：动态 patch 数、lane 分配与 bucket 容量。
- `03-vlm-chunked-prefill`：跨 chunk 索引映射和完整 item 复用。

- `05-vlm-overlap`：VLM overlap 时间线，包含 TokenizerManager / Scheduler 进程、调度 / forward 线程与数据上设备位置。

- `06-epd-why`：E / P / D 的资源差异、分离动机与传输边界。
- `07-epd-implementation`：Dispatcher、EncoderRuntime、Raiden 接收池和 PD 衔接。
- `08-epd-overlap`：线程泳道与预处理 / encode / transfer / completed 队列，以及 Encode 与 Raiden 传输并行。

- `09-encoder-dp-tp`：完整 item 的 lane 分配、DP 权重复制 / TP 权重分片、层内通信与输出汇合；[三句取舍说明](../encoder-dp-tp.md)。

- `10-encoder-input-sharding`：输入数组第 0 维的切分对比：DP 为 DP × TP 个整宽切片，TP 为 DP 个整宽切片，每条 lane 均展示有效 patches 与 bucket padding；[说明](../encoder-input-sharding.md)。

- `11-vlm-chunk-embedding-integration`：ScheduleBatch 的任务筛选、完整 item 与 merge mapping 载荷、Runner 的 embedding overlay 与语言模型接入；[说明](../chunk-embedding-integration.md)。

- `12-vlm-core-principle`：用于文档开头的小横图，展示视觉特征接入统一 embedding 序列及 LLM 推理框架复用；[说明](../vlm-core-principle.md)。

- `13-encoder-replicated-output`：DP 编码后的容量不变量、原序整 item 分 lane 的溢出反例，以及全复制输出 / pool 的编译与存储取舍；[说明](../encoder-replicated-output.md)。

每张图提供 `.excalidraw`（可编辑源文件）、`.svg` 与 `.png`。
将 `.excalidraw` 文件拖入 https://excalidraw.com 即可修改。图形、文字和箭头均为独立元素；节点文字与框体已分组。

前六张图依据分享文档与代码提交 `32684367`；EPD 三张图依据本地代码 `2b1c144b588166e80d783f392258e5382d937a07`。图中时间和数字为教学示意，不代表性能实测结果。

全部文本元素的 Excalidraw 字体设为 Comic Shanns（fontFamily=8）。该字体不含中文字形，中文由渲染环境回退显示。

请求链路图中的照片取自 [Qwen2.5-VL 官方示例](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)，[原图](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg)。照片已嵌入 `.excalidraw` 文件，无需额外加载远程图片。提问为教学示例，未展示模型实测回答。

DP / TP 对比图依据同一提交中的 Qwen3-VL 实现，未上传至 Outline。
