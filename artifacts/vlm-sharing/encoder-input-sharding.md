# Encoder 输入分片与 lane

![逻辑输入第 0 维到设备 mesh 的映射](excalidraw/10-encoder-input-sharding.png)

[可编辑 Excalidraw](excalidraw/10-encoder-input-sharding.excalidraw) · [SVG](excalidraw/10-encoder-input-sharding.svg)

DP-Encoder 用 `data` 和 `tensor` 两个 mesh 轴共同切分逻辑输入第 0 维，形成 `DP × TP` 条 lane，每个设备独立编码一条 lane。TP-Encoder 只沿 `data` 切分输入，形成 `DP` 条 lane；每条 lane 对应固定一个 `data` 坐标下的 TP 设备组，组内设备持有相同的 lane 输入，并在编码层内沿 `tensor` 切分主要线性权重与 attention heads。

```python
from jax.sharding import NamedSharding, PartitionSpec as P

# DP-Encoder：两个 mesh 轴对应同一个数组维度。
input_sharding = NamedSharding(mesh, P(("data", "tensor")))

# TP-Encoder：输入仅沿 data 分片，在 tensor 轴上复制。
input_sharding = NamedSharding(mesh, P("data"))
```

这里 `DP`、`TP` 分别表示 mesh 的 `data`、`tensor` 轴大小；“每条 lane 对应一个 TP 设备组”比“每条 lane 是一个 DP 组”更准确。图展示 packing 后的逻辑形状 `[lanes, bucket, patch_dim]`；代码在 `device_put` 前将数组展平，仍按完整 lane 的连续切片分配给设备。

可放在文档介绍 lane 数与 `input_sharding` 的位置，与[权重、通信及输出汇合对比图](encoder-dp-tp.md)互补；实现依据同页所列的 `VisionShardSpecs` 与 `pack_lanes`。本页及配图仅保存在仓库，未上传至 Outline。
