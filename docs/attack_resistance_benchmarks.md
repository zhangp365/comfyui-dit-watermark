# 防攻击性能测试

本文集中记录 GROW 水印在 Flux2 Klein、Qwen Image Edit 和历史 SD2.1 测试中的防攻击表现。README 只保留面向用户的结果摘要；这里保留测试条件、完整矩阵、失败边界和适用限制。

## Flux2 Klein + ECC 三图实测

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

### Resize 与 crop+upscale 对照

为区分“缩放插值损失”和“检测尺寸变化”，对 Dog（752×1360）、Claw（1024×1024）、Girl（768×1344）三张已验证 profile 1 水印图新增了无损 PNG 攻击。攻击和恢复统一使用 Pillow `LANCZOS`，并保持 `channel_start=4`、`max_channels=8` 及其他 detector 布局参数不变。

| 攻击 | 样本数 | 完整恢复 | 结论 |
|---|---:|---:|---|
| 中心 crop 90%/75% 后放大回原尺寸 | 6 | **6/6** | `crop_scale` 搜索均恢复 |
| 纯 resize 至 99%/90%/75%，以缩小后的尺寸检测 | 9 | **0/9** | 三档、三图均 `ecc_valid=False` |
| 纯 resize 至 99%/90%/75%，再精确恢复原宽高后检测 | 9 | **9/9** | 不需要 robust search 即可恢复 |

这表明本轮 profile 1 + LANCZOS 条件下，纯缩放失败的主因是检测时尺寸改变：VAE latent 的空间网格随之改变，而 keyed frequency layout 由该网格重建。它**不**说明下采样再上采样本身已经擦除了水印。实际产品应把嵌入时图像宽高作为检测配置的一部分；对未知来源图片则应搜索有限的规范尺寸候选。缩放软件、插值核、长边对齐与取整规则都可能改变结果，因此不能把本表外推为所有 resize 实现都能恢复。逐项尺寸、检测输出及 margin 见本地测试产物 `validation/resize_attack_2026-07-21/report.md`。

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

旋转和裁剪会改变固定频率坐标。robust search 已把中心裁剪恢复到 6/6；Claw 的 10°、30°旋转也能精确恢复，但 Dog/Girl 因插值和边角信息丢失仍失败，后续需要更强的几何定位或软判决。MLIC++ 兼容运行环境/权重不在当前工作区，所以明确记为未复测。本地逐样本 benchmark、攻击图片和执行报告位于 `validation/`，作为测试过程产物不纳入版本控制。

## Qwen Image Edit 2511 + Claw 初步实测

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

这是仅使用一张 Claw 的初步攻击测试，不能与 Flux 三图 aggregate 混合。16 个样本中完整恢复 **12/16（75.00%）**，平均 Bit Accuracy **98.73%**，最低 **93.42%**。

| 攻击组 | 完整恢复 | 平均 Bit Accuracy | 最低 | 结论 |
|---|---:|---:|---:|---|
| AVIF → JPG | 1/2 | 98.68% | 97.37% | q95 成功，q80 失败 |
| HEIF → JPG | **2/2** | **100.00%** | **100.00%** | q80、q95 原始码字零错误 |
| JXL → JPG | 1/2 | 98.68% | 98.03% | q95 经 1 symbol 修正成功，q80 失败 |
| JPEG quality | 4/6 | 97.70% | 93.42% | q95/q90/q80/q70 成功；q50、q30 失败 |
| Rotate 10°/30° | **2/2** | **100.00%** | **100.00%** | 反向角搜索均恢复且原始码字零错误 |
| Crop 90%/75% | **2/2** | 99.34% | 98.68% | crop-scale 搜索均恢复 |

JPEG q70 及以上全部成功；q50、q30 分别只有 94.08%、93.42% bit accuracy，损坏超过 4-parity RS 的两-symbol纠错能力，均 `ecc_valid=False`。现代压缩中 AVIF q80 与 JXL q80 也失败，而对应 q95 成功；这反映在 36.78 dB 质量门槛下，压缩鲁棒余量低于先前更强但不满足 PSNR 的参数。测试脚本、API prompt、生成图、攻击图和逐样本原始结果位于忽略的 `validation/qwen_claw/`，不纳入版本控制。

