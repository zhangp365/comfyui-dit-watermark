# comfyui-dit-watermark

面向 ComfyUI 图像 DiT 工作流的 GROW 渐进式频域水印节点，支持无 diffusion inversion 检测、固定帧 Reed–Solomon 纠错和 RGB PSNR 测量。

算法基于 [luopengchen/GROW](https://github.com/luopengchen/GROW)，并针对 ComfyUI 通用 `SAMPLER` 接口、Flux2 Klein 4B Distilled 的四步采样以及 inference mode 做了适配。

## 效果

测试提示词：

```text
Keep the input image exactly unchanged. Preserve every pixel, color, texture, composition, and detail. Add only the invisible watermark.
```

相同输入、prompt、seed、模型和采样参数下，只有右图经过 `GROW DiT Sampler`：

| 无水印 identity 基准图 | GROW + ECC 水印结果 |
|---|---|
| ![Identity clean reference](images/identity_clean.png) | ![Identity watermarked](images/identity_watermarked.png) |

| 指标 | 结果 |
|---|---:|
| Clean → watermarked RGB PSNR | **42.748101 dB** |
| ECC 解码 | `watermark` |
| ECC 状态 | 有效 |
| 自动纠正 | **1 个错误字节符号** |
| 最小投票间隔 | 0.090909 |

`GROWImagePSNR` 节点和独立 NumPy 计算均得到 `42.748101 dB`。该指标使用同一次确定性 DiT 生成的无水印结果作为 reference，能够单独衡量水印引入的失真，并满足 PSNR > 40 dB。

实际上传给工作流的源图片如下：

![Workflow source input](images/source_input.png)

源文件为 736×1312，Flux2 工作流会将其调整为 752×1360 并重新生成。即使使用 identity prompt，无水印输出相对缩放后源图也只有 23.60 dB；源图到最终水印结果为 23.49 dB。这部分主要是 Flux2/VAE 重建误差，不能作为水印本身的失真。README 同时给出两种口径，避免把模型重绘误差计入水印 PSNR。

## 节点

### GROW DiT Sampler

连接方式：

```text
KSamplerSelect → GROW DiT Sampler → SamplerCustomAdvanced
```

该节点在去噪后半程拦截 DiT 预测的干净 latent `x0`，对 secret-key 控制的中频坐标计算 fp32 FFT/DCT-proxy sign-margin loss，再将引导后的 `x0` 交还原 sampler。DiT、conditioning、scheduler、noise 和 VAE 权重均不修改。

| 输入 | 作用 | Flux2 默认值 |
|---|---|---:|
| `message` | UTF-8 水印，最多 27 bytes | `watermark` |
| `secret_key` | 决定频率坐标顺序 | `watermark` |
| `strength` | 最小有符号频率间隔 | `1.20` |
| `guidance_scale` | 水印梯度步长 | `4000` |
| `start_ratio` | 开始引导的采样比例 | `0.50` |
| `dct_min`, `dct_max` | 归一化中频范围 | `0.15`, `0.45` |
| `max_channels` | 使用的 latent 通道数 | `8` |
| `center_ratio` | 使用的中心 latent 比例 | `1.0` |

`strength` 在 UI 中保留两位小数，`guidance_scale` 使用整数步长，避免显示无意义的小数位。

### GROW Watermark Detect

输入生成结果 `IMAGE` 和同一个 `VAE`。检测节点不再要求 message，因为帧中包含长度头和 Reed–Solomon 校验信息；它只需要与嵌入端一致的 secret key、频带、通道数和中心比例。

输出：

- `decoded_message`
- `ecc_valid`
- `corrected_symbols`
- `min_vote_margin`
- `raw_codeword_hex`

检测过程为 `IMAGE → VAE encode → keyed latent frequency signs → majority vote → RS decode`，不加载额外扩散模型，也不执行 diffusion inversion。

### GROW Image PSNR

输入相同尺寸的 reference 和 watermarked `IMAGE`，输出 `[0,1]` RGB PSNR。该节点是 output node，可直接在 ComfyUI/API 执行结果中查看数值。

## Reed–Solomon 纠错

每条 message 编码成固定 32-byte / 256-bit 帧：

```text
1-byte UTF-8 length + 27-byte payload/padding + 4 RS parity bytes
```

四个 RS 校验符号保证：

- 任意 1 个错误字节可纠正；
- 任意 2 个错误字节可纠正；
- 超过纠错能力或帧结构异常时返回 `ecc_valid=False`，不会假装成功。

自动测试覆盖消息区、长度字节、padding 区和 parity 区的双符号错误组合。本次 Flux2 实测原始提取存在 1 个错误字节，检测节点已自动纠正并恢复 `watermark`。

这里的“两个字符”按 Reed–Solomon byte symbol 定义：ASCII 字符对应一个 symbol；UTF-8 中文字符由多个 bytes 组成，因此纠错能力按损坏 bytes 计算。

## 防攻击性能参考

此前对原始 SD2.1 GROW 版本进行过 28 张 JPG 攻击测试。该测试使用旧的 120-bit payload 和多数投票，不包含本仓库新增的 RS 纠错：

| 攻击 | Exact Match | 平均 Bit Accuracy | 结论 |
|---|---:|---:|---|
| AVIF → JPG | 4/4 | 100% | 全部成功 |
| HEIF → JPG | 4/4 | 100% | 全部成功 |
| JXL → JPG | 1/4 | 99.38% | 3 张只错 1 bit |
| MLIC++ | 1/4 | 98.96% | 失败样本只错 1–2 bit |
| Percent 系列 | 0/12 | 66.39% | 全部失败，部分接近随机 |

JXL/MLIC++ 的近成功样本错误集中在极少数字节内，新 RS 帧能够纠正不超过两个错误 byte symbols 的情况；但 `percent 0.95–0.98` 这类大规模频域破坏不能靠四个 parity bytes 修复。

另一次 20 张几何攻击/SyncSeal 恢复测试结果：

| 分组 | Attacked 平均 Bit Accuracy | Recovered 平均 Bit Accuracy |
|---|---:|---:|
| 旋转 | 49.17% | **98.33%** |
| 裁剪/放大 | 50.63% | **65.42%** |
| 全部 recovered | — | 80.83% |

旋转恢复非常有效，裁剪恢复仍不稳定。上述数字是历史 GROW baseline，用于说明已观察到的攻击特征，并不是新 Flux2 + RS 节点的直接攻击基准；对新协议做严格比较时应重新生成同一批攻击样本。

详细历史报告见：

- `../GROW/attack_results_jpg_only/watermark_extraction_report.md`
- `../wmar/out/syncseal_recovery/processed/grow/grow_detection_report.md`

## 安装

将仓库放入：

```text
ComfyUI/custom_nodes/comfyui-dit-watermark
```

然后重启 ComfyUI。运行时只依赖 ComfyUI 自带的 PyTorch，不需要安装额外 ECC 包。

本次远程部署位置：

```text
/app/custom_nodes/comfyui-dit-watermark
```

## 工作流

仓库只保留一份修改后的 UI 工作流：

```text
workflows/flux2_klein_image_edit_grow.json
```

该工作流已经包含：

- identity prompt；
- `GROW DiT Sampler`；
- 无 message 输入的 `GROW Watermark Detect`；
- message `watermark`；
- `strength=1.20`、`guidance_scale=4000`。

工作流注入工具：

```powershell
python scripts/inject_workflow.py input.json output.json `
  --message watermark `
  --secret-key watermark `
  --strength 1.2 `
  --guidance-scale 4000
```

## 兼容性与限制

- 已在 ComfyUI 0.26.0、Flux2 Klein 4B Distilled、Euler 四步采样上验证。
- 其他返回四维 `[B,C,H,W]` latent 的图像 DiT 可复用同一 sampler wrapper。
- video/nested latent 和非四维预测会被明确拒绝。
- 旋转和裁剪会改变固定频率坐标，应在检测前增加对齐/恢复步骤。
- secret key 会以明文保存在工作流 JSON 中，不应把生产密钥放进公开工作流。

## 测试

```powershell
python -m unittest discover -s tests -v
```

测试覆盖 GROW guidance、inference mode、固定帧、双 symbol 纠错、三 symbol 拒绝、节点 API prompt 和混合格式 ComfyUI 子图注入。
