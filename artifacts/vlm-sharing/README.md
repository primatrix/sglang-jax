# VLM 分享文档与配图

[阅读本地文档](document.md) · [在线文档](https://outline.infiscale-tech.com/doc/vlm-eosNGzaACl)

`document.md` 使用本地图片，可以直接在仓库中阅读。`links.json` 保存在线文档、参考文档、benchmark PR、图片来源，以及 Outline 附件网址与本地文件的对应关系。

## 可编辑图源

将 `.excalidraw` 文件拖入 [Excalidraw](https://excalidraw.com) 即可继续编辑。图源内嵌所需照片，文字字体均为 Comic Shanns。

| 图 | Excalidraw | 预览 | 矢量图 |
|---|---|---|---|
| 00-vlm-patches-placeholders | [图源](excalidraw/00-vlm-patches-placeholders.excalidraw) | [PNG](excalidraw/00-vlm-patches-placeholders.png) | [SVG](excalidraw/00-vlm-patches-placeholders.svg) |
| 01-vlm-request-flow | [图源](excalidraw/01-vlm-request-flow.excalidraw) | [PNG](excalidraw/01-vlm-request-flow.png) | [SVG](excalidraw/01-vlm-request-flow.svg) |
| 02-vlm-lane-packing | [图源](excalidraw/02-vlm-lane-packing.excalidraw) | [PNG](excalidraw/02-vlm-lane-packing.png) | [SVG](excalidraw/02-vlm-lane-packing.svg) |
| 03-vlm-chunked-prefill | [图源](excalidraw/03-vlm-chunked-prefill.excalidraw) | [PNG](excalidraw/03-vlm-chunked-prefill.png) | [SVG](excalidraw/03-vlm-chunked-prefill.svg) |
| 04-vlm-component-comparison | [图源](excalidraw/04-vlm-component-comparison.excalidraw) | [PNG](excalidraw/04-vlm-component-comparison.png) | [SVG](excalidraw/04-vlm-component-comparison.svg) |
| 05-vlm-overlap | [图源](excalidraw/05-vlm-overlap.excalidraw) | [PNG](excalidraw/05-vlm-overlap.png) | [SVG](excalidraw/05-vlm-overlap.svg) |
| 06-epd-why | [图源](excalidraw/06-epd-why.excalidraw) | [PNG](excalidraw/06-epd-why.png) | [SVG](excalidraw/06-epd-why.svg) |
| 07-epd-implementation | [图源](excalidraw/07-epd-implementation.excalidraw) | [PNG](excalidraw/07-epd-implementation.png) | [SVG](excalidraw/07-epd-implementation.svg) |
| 08-epd-overlap | [图源](excalidraw/08-epd-overlap.excalidraw) | [PNG](excalidraw/08-epd-overlap.png) | [SVG](excalidraw/08-epd-overlap.svg) |
| 09-encoder-dp-tp | [图源](excalidraw/09-encoder-dp-tp.excalidraw) | [PNG](excalidraw/09-encoder-dp-tp.png) | [SVG](excalidraw/09-encoder-dp-tp.svg) |
| 10-encoder-input-sharding | [图源](excalidraw/10-encoder-input-sharding.excalidraw) | [PNG](excalidraw/10-encoder-input-sharding.png) | [SVG](excalidraw/10-encoder-input-sharding.svg) |
| 11-vlm-chunk-embedding-integration | [图源](excalidraw/11-vlm-chunk-embedding-integration.excalidraw) | [PNG](excalidraw/11-vlm-chunk-embedding-integration.png) | [SVG](excalidraw/11-vlm-chunk-embedding-integration.svg) |
| 12-vlm-core-principle | [图源](excalidraw/12-vlm-core-principle.excalidraw) | [PNG](excalidraw/12-vlm-core-principle.png) | [SVG](excalidraw/12-vlm-core-principle.svg) |
| 13-encoder-replicated-output | [图源](excalidraw/13-encoder-replicated-output.excalidraw) | [PNG](excalidraw/13-encoder-replicated-output.png) | [SVG](excalidraw/13-encoder-replicated-output.svg) |

[下载全部配图](vlm-excalidraw-diagrams.zip)。`attachments/` 保存在线文档中另外上传的图片；完整的网址与文件校验值见 `links.json`。

第 7.3 节的图源含线程泳道、队列任务卡片与连续 Encode / 传输时间线。已从在线文档移除的 EPD 动机图也保留在本地素材中。

新增本地材料：[DP-Encoder / TP-Encoder 对比与取舍](encoder-dp-tp.md)，联系 lane 分配与 bucket 编译复用，未上传至 Outline。

[输入数组切分对比](encoder-input-sharding.md)：用数组块与分界线展示 DP / TP 对第 0 维的不同切分，并标出每条 lane 内补齐到 bucket 的 padding，仅保存在本地仓库。

[Chunk 任务与 embedding 的架构接入](chunk-embedding-integration.md)：任务筛选、完整 item 载荷、按映射 overlay，以及语言模型 forward 前的接入点。

[文档开头的小横图](vlm-core-principle.md)：图片转视觉 embedding、按占位位置与文本合并、复用 LLM 推理框架。

[全 Mesh 复制输出的取舍](encoder-replicated-output.md)：原序分组的单 lane 容量问题、固定总容量紧密重排与 pool 写入。
