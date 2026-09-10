# Encoder 输入数组的切分区别

![DP 与 TP 对输入数组第 0 维的切分](excalidraw/10-encoder-input-sharding.png)

[可编辑 Excalidraw](excalidraw/10-encoder-input-sharding.excalidraw) · [SVG](excalidraw/10-encoder-input-sharding.svg)

以同一个二维数组 `X[N, F]` 对照分片规则：DP-Encoder 的 `P(("data", "tensor"))` 将两个 mesh 轴放在同一个位置，共同切分第 0 维，得到 `DP × TP` 个整宽切片。TP-Encoder 的输入使用 `P("data")`，仅将第 0 维切成 `DP` 份，`tensor` 轴不进一步切分输入；两种模式下输入的 `F` 维都完整保留。

```python
from jax.sharding import NamedSharding, PartitionSpec as P

# 两个 mesh 轴共同切分数组第 0 维。
input_sharding = NamedSharding(mesh, P(("data", "tensor")))

# 只有 data 切分数组第 0 维。
input_sharding = NamedSharding(mesh, P("data"))
```

图中蓝色实线表示 data 分界，紫色虚线表示 tensor 在 data 分块内沿第 0 维进一步划分；这是分片关系的示意，不表示先后执行两次数据搬运。每个切片对应一条 lane，实际容量由 bucket 决定；图中保持相同 `N`、`F` 仅为对照切法，不表示两种模式实际 packing 后的填充总量相同，代码在 `device_put` 前还会将输入展平。

本图聚焦输入数组。TP 的层内特征、权重分片及通信另见[DP / TP 完整对比](encoder-dp-tp.md)；实现依据同页所列的 `VisionShardSpecs` 与 `pack_lanes`。本页及配图仅保存在仓库，未上传至 Outline。
