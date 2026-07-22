# Flux2 Klein 4B/9B 原图水印对比测试报告

## 1. 测试目标

使用 `images/test` 中的 10 张图片，对比 Flux2 Klein 4B 和 9B 在原图编辑场景中的：

1. 原图到 clean 输出的 PSNR，用于衡量模型自身的图像还原能力；
2. clean 输出到 watermarked 输出的 PSNR，用于衡量 GROW 水印单独引入的失真；
3. 水印 `zhangp365123456` 是否能被完整恢复；
4. 每张图片的 `min_vote_margin`；
5. 将约 1 MP 输入缩放工作流与保持原图尺寸工作流进行对比。

本报告仅使用当前工作流对应的 profile 1，即 `channel_start=4`。早期误用 `channel_start=0` 的结果不纳入报告。

## 2. 测试环境与统一参数

| 项目 | Flux2 Klein 4B | Flux2 Klein 9B |
|---|---|---|
| Diffusion model | `flux-2-klein-4b-fp8.safetensors` | `flux-2-klein-9b-fp8.safetensors` |
| Text encoder / CLIP | `qwen_3_4b.safetensors` | `qwen_3_8b_fp8mixed.safetensors` |
| VAE | `flux2-vae.safetensors` | `flux2-vae.safetensors` |
| ComfyUI | 0.26.0 | 0.26.0 |
| GPU | NVIDIA GeForce RTX 4090 24 GB | NVIDIA GeForce RTX 4090 24 GB |

两模型除 diffusion model 和 CLIP 外使用相同配置：

| 参数 | 数值 |
|---|---:|
| 水印 | `zhangp365123456` |
| `secret_key` | `watermark` |
| `strength` | 1.2 |
| `guidance_scale` | 4000 |
| `start_ratio` | 0.0 |
| 引导步数 | 4/4 |
| Sampler | Euler |
| `dct_min` / `dct_max` | 0.15 / 0.45 |
| `channel_start` | 4 |
| `max_channels` | 8 |
| 使用通道 | 4–11 |
| `center_ratio` | 1.0 |
| Seed | 167626463082108 |

Identity prompt：

```text
Keep the input image exactly unchanged. Preserve every pixel, color, texture, composition, and detail.
```

## 3. 指标与判定口径

### 3.1 原图到 Clean PSNR

该指标衡量模型执行 identity image edit 时对输入图片的还原能力。

- 缩放测试：使用 `ImageScaleToTotalPixels` 实际输出的约 1 MP 图片作为 reference；
- 原尺寸测试：跳过 `ImageScaleToTotalPixels`，使用原图作为 reference；
- 如果宽高不是 Flux/VAE 支持的倍数，则只做必要的中心裁剪，使 reference 与模型输出对齐；
- 不将不同分辨率的图片直接计算 PSNR。

### 3.2 Clean 到 Watermarked PSNR

Clean 和 watermarked 使用完全相同的模型、CLIP、VAE、输入、prompt、seed 和采样参数，唯一差异是是否启用 GROW sampler。因此该 PSNR 只衡量水印引入的变化。

### 3.3 水印成功判定

只有同时满足以下条件才视为成功：

```text
ecc_valid = True
decoded_message = zhangp365123456
```

`min_vote_margin` 不能单独作为成功依据。ECC 无效时，即使 margin 较高，也不能认为水印存在。

## 4. 约 1 MP 缩放测试

### 4.1 测试方式

输入先经过：

```text
ImageScaleToTotalPixels(megapixels=1.0, upscale_method=nearest-exact)
```

然后分别运行 4B/9B clean 和 watermarked 工作流。原图到 clean PSNR 中的“原图”指工作流缩放后的实际输入，而不是磁盘上的不同尺寸原文件。

### 4.2 逐图结果

| 图片 | 4B 原图→Clean PSNR | 4B Clean→水印 PSNR | 4B 检测 | 4B margin | 9B 原图→Clean PSNR | 9B Clean→水印 PSNR | 9B 检测 | 9B margin |
|---|---:|---:|---|---:|---:|---:|---|---:|
| blonde-woman-portrait.png | 32.839 | 45.370 | 成功 | 1.000000 | 31.456 | 45.968 | 成功 | 1.000000 |
| claw.jpg | 30.545 | 39.286 | 成功 | 0.882353 | 33.806 | 40.776 | 成功 | 0.882353 |
| genshin-tartaglia-character-poster.png | 30.893 | 33.235 | 成功 | 1.000000 | 21.361 | 32.924 | 成功 | 1.000000 |
| girl.png | 32.614 | 36.814 | 成功 | 1.000000 | 30.439 | 37.853 | 成功 | 1.000000 |
| sars-cov-2-origin-infographic.png | 26.855 | 26.538 | 成功 | 0.529412 | 17.013 | 26.299 | 成功 | 0.411765 |
| snowy-woman-with-horse.jpg | 32.543 | 34.668 | 成功 | 1.000000 | 25.196 | 34.704 | 成功 | 1.000000 |
| source_input.png | 34.441 | 39.305 | 成功 | 1.000000 | 32.414 | 39.351 | 成功 | 1.000000 |
| sun-protection-hood-product-ad.png | 33.418 | 37.119 | 成功 | 1.000000 | 23.163 | 36.250 | 成功 | 1.000000 |
| white-handbag-product-shot.png | 36.054 | 44.931 | 成功 | 1.000000 | 31.225 | 45.315 | 成功 | 1.000000 |
| yellow-blue-abstract-logo.png | 24.948 | 46.313 | 成功 | 0.411765 | 31.260 | 47.190 | **失败** | 0.200000 |

### 4.3 汇总

| 模型 | 原图→Clean 平均 PSNR | 原图→Clean 范围 | Clean→水印平均 PSNR | Clean→水印范围 | 水印成功率 | margin 范围 |
|---|---:|---:|---:|---:|---:|---:|
| Flux2 Klein 4B | **31.5149 dB** | 24.9478–36.0545 | 38.3579 dB | 26.5378–46.3135 | **10/10** | 0.411765–1.000000 |
| Flux2 Klein 9B | 27.7331 dB | 17.0125–33.8063 | **38.6629 dB** | 26.2992–47.1897 | **9/10** | 0.200000–1.000000 |

缩放测试中，4B 的平均原图还原 PSNR 比 9B 高约 3.78 dB。两者水印失真接近，9B 的 Clean→水印平均 PSNR 高约 0.31 dB，但 9B 未能从黄蓝 Logo 中恢复有效水印。该失败项的 `margin=0.2` 是无效候选的 margin，不是有效水印置信度。

完整原始记录：`validation/flux2_4b_9b_ten_images_channel4/results.json`。

## 5. 保持原图尺寸测试

### 5.1 测试方式

该测试完全绕过 `ImageScaleToTotalPixels`：

```text
LoadImage → GetImageSize / VAEEncode
```

不执行插值缩放。Flux/VAE 对尺寸有对齐要求，因此对不能整除的宽高进行最小中心裁剪。最大图片 `claw.jpg` 保持 2048×2048 原始尺寸运行。

### 5.2 实际尺寸

| 图片 | 原图尺寸 | 模型输出尺寸 |
|---|---:|---:|
| blonde-woman-portrait.png | 729×1227 | 720×1216 |
| claw.jpg | 2048×2048 | 2048×2048 |
| genshin-tartaglia-character-poster.png | 968×1624 | 960×1616 |
| girl.png | 768×1344 | 768×1344 |
| sars-cov-2-origin-infographic.png | 1152×1366 | 1152×1360 |
| snowy-woman-with-horse.jpg | 880×1168 | 880×1168 |
| source_input.png | 736×1312 | 736×1312 |
| sun-protection-hood-product-ad.png | 1122×1402 | 1120×1392 |
| white-handbag-product-shot.png | 675×1209 | 672×1200 |
| yellow-blue-abstract-logo.png | 747×744 | 736×736 |

### 5.3 逐图结果

| 图片 | 4B 原图→Clean PSNR | 4B Clean→水印 PSNR | 4B 检测 | 4B margin | 9B 原图→Clean PSNR | 9B Clean→水印 PSNR | 9B 检测 | 9B margin |
|---|---:|---:|---|---:|---:|---:|---|---:|
| blonde-woman-portrait.png | 29.759 | 44.009 | 成功 | 1.000000 | 36.318 | 44.540 | 成功 | 1.000000 |
| claw.jpg | 25.599 | 50.102 | 成功 | 0.126761 | 29.341 | 52.598 | 成功 | 0.267606 |
| genshin-tartaglia-character-poster.png | 32.129 | 38.320 | 成功 | 1.000000 | 33.847 | 39.067 | 成功 | 1.000000 |
| girl.png | 32.630 | 37.011 | 成功 | 1.000000 | 26.022 | 37.419 | 成功 | 1.000000 |
| sars-cov-2-origin-infographic.png | 26.567 | 30.059 | 成功 | 0.111111 | 18.190 | 30.188 | 成功 | 0.185185 |
| snowy-woman-with-horse.jpg | 32.241 | 34.483 | 成功 | 1.000000 | 27.911 | 34.696 | 成功 | 1.000000 |
| source_input.png | 25.490 | 37.082 | 成功 | 1.000000 | 32.914 | 37.625 | 成功 | 1.000000 |
| sun-protection-hood-product-ad.png | 31.304 | 41.583 | 成功 | 1.000000 | 33.650 | 42.335 | 成功 | 1.000000 |
| white-handbag-product-shot.png | 32.279 | 42.925 | 成功 | 1.000000 | 37.312 | 43.378 | 成功 | 1.000000 |
| yellow-blue-abstract-logo.png | 28.157 | 40.895 | 成功 | 0.111111 | 29.708 | 41.137 | 成功 | 0.111111 |

所有成功项均为 `ecc_valid=True`、完整恢复 `zhangp365123456`，且 `corrected_symbols=0`。

### 5.4 汇总

| 模型 | 原图→Clean 平均 PSNR | 原图→Clean 范围 | Clean→水印平均 PSNR | Clean→水印范围 | 水印成功率 | margin 范围 |
|---|---:|---:|---:|---:|---:|---:|
| Flux2 Klein 4B | 29.6155 dB | 25.4897–32.6300 | 39.6469 dB | 30.0589–50.1023 | **10/10** | 0.111111–1.000000 |
| Flux2 Klein 9B | **30.5213 dB** | 18.1903–37.3124 | **40.2981 dB** | 30.1875–52.5979 | **10/10** | 0.111111–1.000000 |

完整原始记录：`validation/flux2_4b_9b_ten_images_channel4_original_size/results.json`。

## 6. 缩放与原尺寸对比

| 模型 | 输入方式 | 原图→Clean 平均 PSNR | Clean→水印平均 PSNR | 水印成功率 |
|---|---|---:|---:|---:|
| 4B | 约 1 MP 缩放 | **31.5149 dB** | 38.3579 dB | 10/10 |
| 4B | 保持原尺寸 | 29.6155 dB | **39.6469 dB** | 10/10 |
| 9B | 约 1 MP 缩放 | 27.7331 dB | 38.6629 dB | 9/10 |
| 9B | 保持原尺寸 | **30.5213 dB** | **40.2981 dB** | **10/10** |

主要观察：

1. 保持原尺寸后，4B 和 9B 的 Clean→水印平均 PSNR 都有所提高，分别提高约 1.29 dB 和 1.64 dB；
2. 原尺寸提供了更多频率载波，9B 黄蓝 Logo 从缩放测试中的 ECC 失败变为完整恢复；
3. 原尺寸测试中两模型均达到 10/10，且全部无需 RS symbol 修正；
4. 原图还原能力高度依赖画面内容。9B 原尺寸平均值略高于 4B，但在 SARS 信息图上只有 18.19 dB，在人像、商品图和部分海报上则明显优于 4B；
5. `min_vote_margin` 会受到分辨率和每 bit 频率重复次数影响，不应跨尺寸孤立比较；最终成功标准必须是 ECC 有效且解码内容完全一致；
6. 固定 `guidance_scale=4000`、`strength=1.2` 时，信息图在两种尺寸策略中均是水印 PSNR 最低的样本，说明水印视觉失真具有明显的内容依赖性。

## 7. 复现资料

批量测试脚本：

```text
scripts/benchmark_flux2_models.py
```

约 1 MP 缩放测试：

```powershell
python scripts/benchmark_flux2_models.py `
  --output-dir validation/flux2_4b_9b_ten_images_channel4
```

保持原尺寸测试：

```powershell
python scripts/benchmark_flux2_models.py `
  --preserve-input-size `
  --output-dir validation/flux2_4b_9b_ten_images_channel4_original_size
```

测试产物目录包含每个模型、每张图片的 reference、clean 和 watermarked PNG，以及 ComfyUI prompt ID、PSNR、ECC、原始码字和 margin 结果。
