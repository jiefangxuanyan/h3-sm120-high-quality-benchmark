# Nsight Systems 去噪阶段统计

`integration/summarize_nsys.py` 离线分析导出的 SQLite，不启动 GPU 工作，仅依赖 Python 标准库。导出时，SQLite 文件名应与对应配置目录名一致：

```bash
mkdir -p /path/to/analysis/sqlite
pixi run --frozen --no-install nsys export --type sqlite \
  --output /path/to/analysis/sqlite/4-nccl-0123.sqlite \
  /path/to/results/profiling/4-nccl-0123/4-nccl-0123.nsys-rep
python3 integration/summarize_nsys.py /path/to/analysis/sqlite/*.sqlite \
  --run-root /path/to/results/profiling --output /path/to/analysis
```

提供 `--run-root` 时，从各配置模型日志读取 `skip_first_steps`、`skip_first_layers=0` 和正式请求的 Cache-DiT 记录，启用阶段分析；省略时仍输出整体分解。默认矩阵和八端点矩阵采用同一统计口径。

## 稠密、稀疏和缓存命中

本配方的 dense 为 step 0–9，sparse 为 step 10–48。这是去噪步骤阶段，不是模型中两组固定层。Sparse 指受路由控制的视频 attention；音频 attention、projection/MLP GEMM 仍可为 dense。

Sparse 包含缓存命中，所以 dense+sparse 覆盖完整去噪。`sparse_compute` 和 `cache_hits` 进一步拆分 sparse。报告配方实际命中 12 步、未命中 27 步；缓存短步仍有部分 block 工作，不能视为零计算。实际命中索引读取自日志，不根据耗时阈值推断。

## 分母和时间单位

整体分母为各 GPU worker 去噪循环墙钟时间之和。绝对时间取单次采集中各 GPU 的平均值，不是多轮计时均值。各阶段的分类占比以该阶段的总 GPU 时间为分母；`denoise_fraction_percent` 则表示该阶段占完整去噪的比例。

这些区间排除加载、预热、输入编码、解码和输出保存，不能与引擎内部请求总耗时直接混用。

## 互斥分区

- **Attention**：CAKE/FlashAttention、稀疏路由及 CAKE 布局操作。
- **GEMM**：矩阵乘，包括 projection 和 MLP。
- **Communication**：NCCL GPU kernel 区间，加上成对 opening/closing `UlyssesPcieBarrier` 的 GPU 包络。后者包括主机推进和等待；不同后端边界不同，不是严格等价的纯传输时间。
- **Overlap**：通信与 attention/GEMM 同时执行的交集，只计一次。
- **Other**：其余已记录 GPU kernel、copy、memset，包括量化、打包、归一化和逐元素操作。
- **Gap**：没有记录到 GPU 活动、也不在已定义通信区间内的时间，保持未归因。

JSON 另保留 attention/GEMM 交集及各类 inclusive 区间并集。Inclusive 项之间不能直接相加。同类区间逐 GPU 合并后再求交和分区，不累加可能重叠的 kernel 耗时。执行时长也不是 SM 利用率或 FLOP 利用率。

阶段边界采用各 worker 的 CPU step-NVTX 起点切分 GPU 时间线。步骤间隙归前一步，循环头尾归首末步；跨边界的异步 GPU 区间按时间截断。这不是按 kernel 启动关联精确归属步骤。Dense+sparse 必须精确重建整体时长和每个分类。

## 校验与输出

默认校验每 worker 49 步、worker/GPU 映射、完整采集范围、偶数个 barrier 及各卡一致的交换次数。更换 kernel 实现后，需要重新核对名称分类和串行 opening/closing barrier 配对假设。未知 kernel 归入 Other，保留完整名称供检查。

输出包括逐配置 JSON、`summary.json`、`summary.md`，提供 `--run-root` 时另有 `phases.md`。Markdown 分类单元格同时给出秒数和百分比，与报告一致。逐配置 JSON 包含逐卡区间、方法及脚本哈希、代表 GPU 的完整 kernel 名称。

性能比较以不带 profiling 的重复计时为准；一次 capture 只是单次观察。
