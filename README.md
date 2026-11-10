# MiniMax-H3 高质量方案复现

本仓库用于在 Linux x86-64、SM120 GPU 上复现约 5 秒的 FL2VA 高质量视频与音频生成，支持 NCCL、FlashInfer RDMA、PCIe P2P，以及单卡本地 attention。GPU 数量、放置方式、网卡端点数量和共享方式分别记录，不把它们混为一个变量。

## 环境与模型

两个源码子模块指向 GitHub 私有仓库 [sglang-h3-sm120](https://github.com/jiefangxuanyan/sglang-h3-sm120) 和 [flashinfer-h3-sm120](https://github.com/jiefangxuanyan/flashinfer-h3-sm120)，固定已验证的提交。克隆前需要两个仓库的访问权限及 GitHub SSH 认证。初始化子模块及其依赖，安装锁定的 Pixi 环境，再下载模型：

```bash
git submodule update --init --recursive
pixi install --locked
pixi run --as-is download-model
```

环境固定 Python 3.12、CUDA 13.0 / nvcc 13.0.88、PyTorch 2.13.0+cu130、Nsight Systems CLI 2026.3.2.476，最低要求 glibc 2.35，可用于 Ubuntu 22.04 及更新系统；锁定依赖沿用 RTX PRO 5000 实验环境。模型为 `MiniMaxAI/MiniMax-H3`，revision 为 `42ed227ee7df40d41602854ae760620d6eb651fe`。下载任务只获取 FL2VA 文件和配置 JSON；模型权重不随仓库分发。

为模型、Python 环境及编译缓存预留足够空间。可在下载前设置 `HF_HUB_CACHE`；`FLASHINFER_WORKSPACE_BASE` 下保存 `.cache/flashinfer`，`SGLANG_CACHE_DIR` 和 `TRITON_CACHE_DIR` 分别指定其它编译缓存位置。这些是存储位置，不是性能调参。运行入口使用当前检出的两个子仓库源码。

源码构建使用默认 PEP 517 隔离环境，先安装声明的构建依赖，避免在全新环境尚无 setuptools 时直接构建。运行时软件版本仍由 `pixi.lock` 固定。目标系统或 GPU 显存不同，可能需要单独适配环境或驻留策略，不能把端点模板视为整机配置相同的证明。

## 工作负载与计时

输入为 `integration/examples/t2va.json`：1344×768、约 5 秒视频及音频、batch=1、seed=1101。配置 50 步，实际执行 49 次去噪。DiT 使用普通 FP8 权重，VAE 为 BF16；TP=1、Ring=1，Ulysses degree 等于 GPU 数。TP=1 指 DiT；`encoder_parallel=auto` 仍可能在多卡组中分片文本编码器。DiT 不做 CPU offload，文本编码器启用 CPU offload，video VAE 常驻；5000 单卡配置进一步将文本编码器设为逐层 offload。

Sage 前十步为 dense，后续进入 sparse；原生 Cache-DiT 配方为 `(4, 0.04, 1)`，AdaLN 主机缓存预算为 2 GiB。每个样本新建进程，在同一组 worker 上先完整预热一次，再测第二次请求。完成计时和指标导出后显式关闭生成器。

主指标 `total_request_s` 是引擎内部生成墙钟时间，包含输入编码、去噪、解码及内部通信和同步，排除进程启动、模型加载、预热、输出文件编码保存和请求/响应传输。性能比较采用不带 profiling 的重复计时；profiling 单独采集和统计。

## 默认矩阵：四活动端点平台

`integration/examples/benchmark-cases.json` 包含 5500 高质量基线的 12 个配置。八卡三配置各测八次，其余九配置各测四次，共 **60 次正式请求**，每次都有独立完整预热。每对轮次交替正序、倒序，平衡组内位置；不宣称完整拉丁方均衡。

```bash
# 只查看计划，不访问 GPU。
pixi run --frozen --no-install python -m integration.benchmark_high

# 全部执行；NIC0..NIC3 替换为实际 RDMA 设备名。
pixi run --frozen --no-install python -m integration.benchmark_high \
  --nics NIC0,NIC1,NIC2,NIC3 --execute --output /path/to/results/timing

# 本机小规模验证：一次完整预热加一次正式请求。
pixi run --frozen --no-install python -m integration.benchmark_high \
  --case 4-nccl-0123 --repeats 1 --execute --output /path/to/results/check
```

`--nics` 填写不同的 RDMA 设备名，例如 `mlx5_0`，不是以太网接口名。默认四端点模板中，槽位 0、1、2、3 分别对应 GPU 对 01、23、45、67 就近使用的活动端点。

GPU 编号是物理设备编号。`CUDA_VISIBLE_DEVICES` 可不设置，也可为包含所有被测 GPU 的顺序前缀，例如 `0,1,2,3`；会重排或隐藏被测物理编号的 mask 会被拒绝。

## 八端点机器及少端点对照

**保留八端点独享和在八端点机器上只允许部分端点的能力。** 可选模板 `integration/examples/benchmark-cases-8-endpoints.json` 包含 5000 平台的 15 个多卡配置及 1 个单卡配置：

| 卡数 | 配置 |
| --- | --- |
| 8 卡 | NCCL 允许八端点、NCCL 允许四端点、RDMA 八端点独享、RDMA 四端点共享、PCIe |
| 4 卡 | NCCL、RDMA 共享、RDMA 相邻端点独享、RDMA 分散网卡独享、PCIe |
| 2 卡 | NCCL、RDMA 共享、RDMA 相邻端点独享、RDMA 分散网卡独享、PCIe |
| 1 卡 | 本地 attention、文本编码器逐层 offload |

默认共 **84 次正式请求**；profiling 为 **16 次独立采集**。单卡配置为 `1-local-layerwise-offload`，四轮分别使用 GPU 0、1、2、3，profiling 仅使用 GPU 0。5500 默认矩阵的单卡仍保留原驻留方案；5000 单卡应选择本模板的逐层卸载配置。

模板假设端点槽位 0–7 分别就近对应 GPU 0–7，且端点对 01、23、45、67 各属于同一张物理适配器。请按实际拓扑调整 `--nics` 顺序或复制修改 JSON，不要根据 `mlx5_N` 编号猜测归属。例如，报告中的一种八端点排列是 `mlx5_2,mlx5_3,mlx5_0,mlx5_1,mlx5_6,mlx5_7,mlx5_4,mlx5_5`；它不是通用默认值。

```bash
# 八端点全套计划，不需要机器具备八张卡或八个活动端点。
pixi run --frozen --no-install python -m integration.benchmark_high \
  --cases integration/examples/benchmark-cases-8-endpoints.json

# 同一台八端点机器上，对照八端点与四端点 NCCL。
# NIC0..NIC7 按上文的 GPU 就近关系替换为实际设备名。
pixi run --frozen --no-install python -m integration.benchmark_high \
  --cases integration/examples/benchmark-cases-8-endpoints.json \
  --case 8-nccl-8endpoints --case 8-nccl-4endpoints \
  --nics NIC0,NIC1,NIC2,NIC3,NIC4,NIC5,NIC6,NIC7 \
  --execute --output /path/to/results/nccl-endpoints
```

对照 RDMA 时选 `8-rdma-8endpoints` 和 `8-rdma-4endpoints`。省略 `--case` 执行该文件全部配置；添加 `--profile` 则每配置只采集一次。整个过程只设置本次进程的选路和驻留参数，不关闭宿主机接口，也不修改其它任务。

5000 单卡计时无需提供 NIC：

```bash
pixi run --frozen --no-install python -m integration.benchmark_high \
  --cases integration/examples/benchmark-cases-8-endpoints.json \
  --case 1-local-layerwise-offload --execute --output /path/to/results/single
# 在同一命令中增加 --profile，并使用新的输出目录，采集一次单卡报告。
```

### 配置字段与选路口径

- `gpus`：按 rank 顺序列出物理 GPU 编号。
- `nic_slots`：仅用于主 RDMA 传输，每个 rank 一个槽位。重复槽位表示共享端点。
- `nccl_nic_slots`：可选的 NCCL 允许端点集合，必须互不重复。用于 NCCL 主传输及 RDMA 场景的辅助 NCCL；通过 `NCCL_IB_HCA` 的精确名称列表生效。
- `component_residency`：可选的组件驻留策略，例如 `{"video_vae":"resident","text_encoder":"layerwise-offload"}`；执行后核对引擎实际配置。
- `gpu_selections`：可选的单卡采样列表，计时逐轮循环选择，profiling 使用第一项。

八端点模板中，八卡 RDMA 独享使用槽位 `[0,1,2,3,4,5,6,7]`，四端点共享使用 `[0,0,2,2,4,4,6,6]`；它们的辅助 NCCL 允许集合严格等于主 RDMA 实际使用集合。四端点 NCCL 只允许 `[0,2,4,6]`。四卡 NCCL 只允许 `[0,2]`，双卡 NCCL 只允许 `[0]`，与对应共享 RDMA 组的端点集合一致。

“独享端点”不等于独享物理网卡。例如双卡 01 分别使用端点 0、1，仍共用一张物理适配器；双卡 02 使用端点 0、2，才是分散到不同物理适配器。模板同时保留这两种比较。

默认 5500 矩阵未填写 `nccl_nic_slots`，保持原口径：主 RDMA 显式指定端点，辅助 NCCL 自动选路；手动设置的 `NCCL_IB_HCA` 会被继承并记录。填写该字段时，以配置为准，覆盖继承值，支持在同一批次逐配置切换允许集合。

限制端点不等于强制 NCCL 全部走 RDMA，也不保证用满允许集合中的每个端点；P2P/SHM 仍由 NCCL 选择。`NCCL_IB_HCA` 限制 IB/RoCE 数据传输的设备，不限制 bootstrap TCP。实际选路需查看连接日志。PCIe 配置固定主传输为 P2P，辅助 NCCL 禁用 NET/RDMA、仍可用 P2P/SHM，因此不接受额外的 NCCL 端点集合。

## 执行检查与产物

可重复 `--case` 选择多个配置，用 `--cases FILE` 指定自定义矩阵。硬件、GPU 放置或驻留策略改变后的数据应单独标明，不能当作原报告平台的重复样本。

驱动只检查被测 GPU 是否空闲，不终止已有进程；仅检查实际会使用的 RDMA 端点是否 ACTIVE，未选中的 DOWN 端点不阻碍少端点实验。输出目录已存在时拒绝覆盖；每完成一个样本就保存进度，失败即停止，不自动重试。它校验实际 GPU/NIC/peer 路径、缓存命中一致性，并在内核提供计数器时记录全机 pin 增长。该计数包含其它用户，不用它判定本实验失败；主机在运行中重启仍会中止。不要求调度器或固定主机名。

产物包含冻结的请求、源码哈希、可用时的 Git revision 与已跟踪改动标记、实际配置、逐次性能/日志/视频、`trials.json` 和 `summary.json`。汇总提供均值、样本标准差和 CV；单次样本的标准差和 CV 为 null。`launch.json` 记录实际启动命令及相关端点限制。原始运行产物描述其生成机器，外发前应单独检视。

## Profiling 与分析

```bash
# 只采集第二次请求；第一次仍完整预热。
pixi run --frozen --no-install python -m integration.benchmark_high \
  --case 4-nccl-0123 --profile --execute --output /path/to/results/profiling
```

默认矩阵去掉 `--case`、提供对应 `--nics` 即采集全部 12 份；选择八端点模板则为 16 份。采集项为 CUDA、NVTX、OS runtime、NCCL 和 NIC metrics，并保留逐层标记。范围由 `H3_MEASURED_REQUEST` 控制，模型加载和预热不在其中。报告先在本地临时目录生成，成功转换后回收到输出目录；只清理临时文件。最终报告为 `<case>/<case>.nsys-rep`。

整体、dense、sparse 分解方法和命令见 [NSYS_ANALYSIS.md](integration/NSYS_ANALYSIS.md)。

## 检查命令

```bash
pixi run --frozen --no-install python -m pytest -q \
  integration/test_benchmark_high.py integration/test_warmup_generate.py \
  integration/test_summarize_nsys.py
pixi run --frozen --no-install python integration/check_adaln_cache_dispatch.py
pixi run --frozen --no-install python integration/check_adaln_contract.py
```

CPU 检查不代表 GPU 正确性或性能已经实测。两个源码树应保留原有许可证；性能结论应注明实际硬件、拓扑和选路配置。
