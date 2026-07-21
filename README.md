# comfyui-dit-watermark

面向 ComfyUI 图像 DiT 工作流的 GROW 渐进式频域水印节点，支持无 diffusion inversion 检测、弹性 Reed–Solomon 帧、几何 robust search 和 RGB PSNR 测量。

算法基于 [luopengchen/GROW](https://github.com/luopengchen/GROW)，并针对 ComfyUI 通用 `SAMPLER` 接口、Flux2 Klein 4B Distilled 四维 latent、Qwen Image Edit 单帧五维 latent 和 inference mode 做了适配。

## 三图效果验证

测试使用当前唯一 UI 工作流 `workflows/flux2_klein_image_edit_grow.json` 的模型、seed、采样器、频带和提示词：

```text
Keep the input image exactly unchanged. Preserve every pixel, color, texture, composition, and detail.
```

公开输入字段名是 `watermark`，本次三张图实际写入的内容统一为 `zhangp36512345`，`secret_key=watermark`。

| 图片 | Clean → marked PSNR | ECC 检测 |
|---|---:|---|
| Dog | **41.212684 dB** | 成功，0 symbol 修正 |
| Claw | **40.970031 dB** | 成功，0 symbol 修正 |
| Girl | **38.193383 dB** | 成功，0 symbol 修正 |

三张图均完整恢复 `zhangp36512345`，且水印增量 PSNR 均高于新的 35 dB 门槛。当前唯一工作流统一使用：

| `strength` | `guidance_scale` | `start_ratio` | 引导步数 |
|---:|---:|---:|---:|
| 1.20 | 4000 | 0.00 | 4/4 |

PSNR 以相同输入、prompt、seed、模型和采样参数的 clean 输出为 reference，只衡量 GROW 引入的变化。源图片还会经过 Flux2/VAE 重建，源图到输出的变化不能算作水印失真。

## 节点

### GROW DiT Sampler

```text
KSamplerSelect → GROW DiT Sampler → SamplerCustomAdvanced
```

节点在选定的去噪步骤拦截 DiT 预测的干净 latent `x0`，对 secret-key 控制的中频坐标计算 fp32 FFT/DCT-proxy sign-margin loss，再将引导后的 `x0` 交还原 sampler。默认 `start_ratio=0`，Flux2 四步采样从第一步开始、共引导 4/4 steps。模型、conditioning、scheduler、noise 和 VAE 权重均不修改。

| 输入 | 作用 | Flux2 默认值 |
|---|---|---:|
| `watermark` | UTF-8 水印内容，最多 250 bytes | `zhangp36512345` |
| `secret_key` | 决定频率坐标顺序 | `watermark` |
| `strength` | 最小有符号频率间隔 | `1.20` |
| `guidance_scale` | 水印梯度步长 | `4000` |
| `start_ratio` | 开始引导的采样比例 | `0.00` |
| `dct_min`, `dct_max` | 归一化中频范围 | `0.15`, `0.45` |
| `max_channels` | 使用的 latent 通道数 | `8` |
| `channel_start` | 连续通道 profile 的起始 latent channel | `0` |
| `center_ratio` | 使用的中心 latent 比例 | `1.0` |

`strength` 在 UI 中按 0.01 步进显示，`guidance_scale` 按整数步进显示。超过 32 UTF-8 bytes 会发出长水印警告：帧越长，每个 bit 的频率重复次数越少，攻击鲁棒性越低，盲检候选长度也越多。

### GROW Watermark Detect

输入生成结果 `IMAGE` 和同一个 `VAE`。检测节点无需预先提供水印内容；弹性帧包含长度和 Reed–Solomon 校验信息，但必须使用相同 secret key、频带、起始通道、通道数和中心比例。`max_watermark_bytes` 控制盲检长度上限，`robust_mode` 可选择 `none`、`rotation`、`crop_scale` 或组合搜索。

#### 嵌入与检测参数必须一致

`secret_key`、`dct_min`、`dct_max`、`max_channels`、`channel_start`、`center_ratio` 共同定义 bit 到 latent 频率坐标的布局。检测端任一项不一致，就会读取不同系数或用不同方式拆分 bit，通常无法通过 RS 校验。水印内容无需传给 detector，但布局参数不是可从当前帧中任意盲推的。

当前工作流应把 sampler 与 detector 的这些值保持一致。更稳妥的后续节点设计是增加一个 `GROW Watermark Config` 节点，输出单个 `GROW_CONFIG` 对象，同时连接 sampler 和 detector，消除两份 UI 参数漂移。检测外部攻击图片时，可对少量预定义 profile（例如 0–4）做有界搜索并以 RS 有效性和 vote margin 选优；不建议对连续的 `dct_min/dct_max` 任意暴力搜索，因为组合数、VAE 次数和误判面都会快速增加。若要让图片完全自描述，需要在固定、不随 profile 改变的 pilot 频带写入布局 ID/header，再据此读取主 payload。

Claw/profile 1 的单参数错配实测也验证了这一点：完全匹配时原始码字零错误；分别只把 `channel_start` 改为 0、`max_channels` 改为 4、`dct_min` 改为 0.10 或 `dct_max` 改为 0.50，四种情况均 `ecc_valid=False`。注意错配 `max_channels` 时 vote margin 仍可达到 1.0，因此 margin 高不代表布局正确，必须以 RS/帧校验为准。

输出包括 `decoded_message`、`ecc_valid`、`corrected_symbols`、`min_vote_margin` 和 `raw_codeword_hex`。检测流程为：

```text
IMAGE → VAE encode → keyed latent frequency signs → majority vote → RS decode
```

它不加载额外扩散模型，也不执行 diffusion inversion。

### GROW Image PSNR

输入相同尺寸的 reference 和 watermarked `IMAGE`，输出 `[0,1]` RGB PSNR。

## Reed–Solomon 纠错

每个水印编码成最短的自描述弹性帧：

```text
1-byte UTF-8 length + N-byte payload + 4 RS parity bytes
```

总长度为 `N+5` bytes；本次 14-byte 水印从旧协议的 32 bytes 缩短到 19 bytes，因而每 bit 获得更多频率重复。四个 parity symbols 可纠正任意两个损坏的 byte symbols。超过纠错能力或帧结构异常时返回 `ecc_valid=False`。ASCII 的两个字符对应两个 symbols；中文字符是多字节 UTF-8，因此能力按损坏 bytes 计算。默认检测上限 64 bytes，并兼容旧 32-byte 零填充帧。

## 防攻击性能参考

### Qwen Image Edit 2511 + Claw 初步实测

Qwen 工作流 `workflows/image_qwen_image_edit_2511.json` 使用 `qwen_image_edit_2511_fp8mixed.safetensors`，不加载 LoRA；原标准 `KSampler` 已等价展开为：

```text
RandomNoise + CFGGuider + BasicScheduler + KSamplerSelect
                                      ↓
                            GROW DiT Sampler
                                      ↓
                         SamplerCustomAdvanced
```

工作流保留 seed `1030828666996942`、20 steps、CFG 1、Euler、simple scheduler、denoise 1，并将 2048×2048 Claw 通过 `ImageScaleToTotalPixels` 缩放到约 0.6 MP，避免超大输入显存峰值。Qwen VAE/denoiser 的 `[B,C,1,H,W]` 单帧五维 latent 会在 GROW 内部压缩时间维执行原四维频域算法，再恢复原形状；多帧 latent 会明确拒绝，避免把视频维误当空间维。

本轮写入 `zhangp36512345`，`secret_key=watermark`，profile 1（channels 4–11），`strength=1.2`，20/20 步从第一步引导。考虑 20-step 累计引导，Qwen 的 `guidance_scale` 使用 **500**，而不是 Flux 四步工作流的 4000。最终 clean → marked PSNR 为 **36.7807 dB**；无攻击检测精确恢复，ECC 有效，0 symbol 修正。

这是仅使用一张 Claw 的初步攻击测试，不能与下面 Flux 三图 aggregate 混合。16 个样本中完整恢复 **12/16（75.00%）**，平均 Bit Accuracy **98.73%**，最低 **93.42%**。

| 攻击组 | 完整恢复 | 平均 Bit Accuracy | 最低 | 结论 |
|---|---:|---:|---:|---|
| AVIF → JPG | 1/2 | 98.68% | 97.37% | q95 成功，q80 失败 |
| HEIF → JPG | **2/2** | **100.00%** | **100.00%** | q80、q95 原始码字零错误 |
| JXL → JPG | 1/2 | 98.68% | 98.03% | q95 经 1 symbol 修正成功，q80 失败 |
| JPEG quality | 4/6 | 97.70% | 93.42% | q95/q90/q80/q70 成功；q50、q30 失败 |
| Rotate 10°/30° | **2/2** | **100.00%** | **100.00%** | 反向角搜索均恢复且原始码字零错误 |
| Crop 90%/75% | **2/2** | 99.34% | 98.68% | crop-scale 搜索均恢复 |

JPEG q70 及以上全部成功；q50、q30 分别只有 94.08%、93.42% bit accuracy，损坏超过 4-parity RS 的两-symbol纠错能力，均 `ecc_valid=False`。现代压缩中 AVIF q80 与 JXL q80 也失败，而对应 q95 成功；这反映在 36.78 dB 质量门槛下，压缩鲁棒余量低于先前更强但不满足 PSNR 的参数。测试脚本、API prompt、生成图、攻击图和逐样本原始结果位于忽略的 `validation/qwen_claw/`，不纳入版本控制。

### 最新 Flux2 + ECC 实测

基准图是 Dog、Claw、Girl 使用 **profile 1（channels 4–11）**生成的 watermarked PNG，clean → marked PSNR 分别为 39.58、39.74、36.78 dB，均满足 35 dB。共生成并检测 48 个攻击样本：AVIF/HEIF/JXL q80、q95 回转 JPG，JPEG quality 95/90/80/70/50/30，旋转 10°/30°，中心裁剪 90%/75% 后放大。所有样本都由服务器上最新部署的 `GROWWatermarkDetect` 使用相同 `channel_start=4`、`max_channels=8` 检测。

判定标准：`ecc_valid=True` 且解码内容等于 `zhangp36512345` 才算完整成功；Bit Accuracy 比较完整 19-byte / 152-bit 弹性 RS frame。旋转样本使用 `rotation` 搜索，裁剪样本使用 `crop_scale` 搜索。

| 攻击组 | 完整恢复 | 平均 Bit Accuracy | 最低 | 结论 |
|---|---:|---:|---:|---|
| AVIF → JPG | **6/6** | **100.00%** | **100.00%** | q80、q95 全部原始码字零错误 |
| HEIF → JPG | **6/6** | **100.00%** | **100.00%** | q80、q95 全部成功 |
| JXL → JPG | **6/6** | **100.00%** | **100.00%** | q80、q95 全部原始码字零错误 |
| JPEG quality | **17/18** | 99.82% | 97.37% | 仅 Claw q30 失败；q50 及以上全成功 |
| Rotate 10°/30° | 2/6 | 67.87% | 51.32% | Claw 10°、30°均由反向角搜索恢复 |
| Crop 90%/75% | **6/6** | **100.00%** | **100.00%** | 对应 0.90/0.75 scale 搜索全部原始码字零错误 |
| **合计** | **43/48** | **95.92%** | **51.32%** | 完整恢复率 **89.58%** |

### Resize 与 crop+upscale 对照（Flux2 profile 1）

为区分“缩放插值损失”和“检测尺寸变化”，对 Dog（752×1360）、Claw（1024×1024）、Girl（768×1344）三张已验证 profile 1 水印图新增了无损 PNG 攻击。攻击和恢复统一使用 Pillow `LANCZOS`，并保持 `channel_start=4`、`max_channels=8` 及其他 detector 布局参数不变。

| 攻击 | 样本数 | 完整恢复 | 结论 |
|---|---:|---:|---|
| 中心 crop 90%/75% 后放大回原尺寸 | 6 | **6/6** | `crop_scale` 搜索均恢复 |
| 纯 resize 至 99%/90%/75%，以缩小后的尺寸检测 | 9 | **0/9** | 三档、三图均 `ecc_valid=False` |
| 纯 resize 至 99%/90%/75%，再精确恢复原宽高后检测 | 9 | **9/9** | 不需要 robust search 即可恢复 |

这表明本轮 profile 1 + LANCZOS 条件下，纯缩放失败的主因是检测时尺寸改变：VAE latent 的空间网格随之改变，而 keyed frequency layout 由该网格重建。它**不**说明下采样再上采样本身已经擦除了水印。实际产品应把嵌入时图像宽高作为检测配置的一部分；对未知来源图片则应搜索有限的规范尺寸候选。缩放软件、插值核、长边对齐与取整规则都可能改变结果，因此不能把本表外推为所有 resize 实现都能恢复。逐项尺寸、检测输出及 margin 见 `validation/resize_attack_2026-07-21/report.md`。

同一矩阵随后以 Pillow `BILINEAR`、`BICUBIC` 和 `NEAREST` 复测；四种核（含 LANCZOS）均得到“缩小尺寸检测 **0/9**、缩小再精确恢复原宽高 **9/9**”。恢复后最低 vote margin 分别为 0.684211、0.789474、0.684211、0.894737；核会影响余量，但未改变本轮的 ECC 成败边界。

JPEG “percent/quality”系列的失败边界：

| JPEG quality | 完整恢复 | 平均 Bit Accuracy | 失败图片 |
|---:|---:|---:|---|
| 95% | **3/3** | **100.00%** | 无 |
| 90% | **3/3** | **100.00%** | 无 |
| 80% | **3/3** | **100.00%** | 无 |
| 70% | **3/3** | **100.00%** | 无 |
| 50% | **3/3** | **100.00%** | 无 |
| 30% | 2/3 | 98.90% | Claw |

按原图汇总：Dog 14/16、Girl 14/16、Claw **15/16**。profile 1 将 Claw 在旧 profile 0 下的 AVIF q80/q95、JXL q80、JPEG q80/q70/q50 失败全部消除；完整矩阵中只剩 JPEG q30 失败。

旋转和裁剪会改变固定频率坐标。robust search 已把中心裁剪恢复到 6/6；Claw 的 10°、30°旋转也能精确恢复，但 Dog/Girl 因插值和边角信息丢失仍失败，后续需要更强同步模板或软判决。本轮未把旧 SD2.1 数据集训练出的 SyncSeal recovery 结果冒充为新 Flux2 结果；MLIC++ 兼容运行环境/权重也不在当前工作区，所以这两项明确记为未复测。本地逐样本 benchmark、攻击图片和执行报告位于 `validation/`，作为测试过程产物不纳入版本控制。

### 历史 SD2.1 参考

旧 SD2.1、120-bit、无 RS 版本曾得到：AVIF 4/4、HEIF 4/4、JXL 1/4、MLIC++ 1/4、percent 系列 0/12；历史 SyncSeal 将旋转平均 Bit Accuracy 从 49.17% 恢复到 98.33%，裁剪从 50.63% 恢复到 65.42%。协议、模型和图片均不同，只作为历史对照，不能与上表直接合并。

## 安装

将仓库放入：

```text
ComfyUI/custom_nodes/comfyui-dit-watermark
```

然后重启 ComfyUI。运行时只依赖 ComfyUI 自带的 PyTorch，不需要额外 ECC 包。本次远程部署路径为 `/app/custom_nodes/comfyui-dit-watermark`。

## 工作流

仓库保留两份面向不同模型的 UI 工作流，不保存重复的 API 工作流 JSON：

```text
workflows/flux2_klein_image_edit_grow.json
workflows/image_qwen_image_edit_2511.json
```

两者都包含 identity prompt、`GROW DiT Sampler`、无需预知水印内容的检测节点，以及 `watermark=zhangp36512345`。Qwen 版本使用 fp8 模型且不加载 LoRA，并通过高级采样组件为原标准 KSampler 增加可接入的 `SAMPLER` 输入。

工作流注入：

```powershell
python scripts/inject_workflow.py input.json output.json `
  --watermark zhangp36512345 `
  --secret-key watermark `
  --strength 1.2 `
  --guidance-scale 4000
```

## 兼容性与限制

- 已在 ComfyUI 0.26.0、Flux2 Klein 4B Distilled、Euler 四步采样上验证。
- 返回四维 `[B,C,H,W]` 或单帧五维 `[B,C,1,H,W]` latent 的图像 DiT 可复用 sampler wrapper；多帧视频 latent 当前不支持。
- 检测节点已支持旋转与裁剪/尺度 robust search；裁剪实测稳定，旋转仍有明显残余错误。
- 抗压缩能力与图像内容相关，应为实际图片保留纠错余量。
- secret key 会以明文保存在工作流 JSON 中，不应把生产密钥放入公开工作流。

## 测试

```powershell
python -m unittest discover -s tests -v
```

测试覆盖 GROW guidance、inference mode、弹性帧、旧帧兼容、双 symbol 纠错、三 symbol 拒绝、盲长度检测、几何候选、公开 `watermark` 接口、工作流注入和攻击汇总。
