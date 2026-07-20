# comfyui-dit-watermark：Flux2 Klein 验收报告

## 1. 验收结论

自定义节点已部署到 `10.168.1.168:8189` 的 ComfyUI 容器，并完成 Flux2 Klein 4B Distilled 端到端验证。

| 验收项 | 要求 | 实测 | 结论 |
|---|---:|---:|---|
| 水印内容 | `watermark` | 无先验 message 解码为 `watermark` | **通过** |
| ECC | 可纠正两个错误符号 | 单元测试覆盖任意 1 个及代表性 2 个 byte symbol；实图修正 1 个 | **通过** |
| 水印增量 PSNR | > 40 dB | **42.748101 dB** | **通过** |
| 工作流 | 仅保留一个 UI JSON | 1 个 | **通过** |
| 节点加载 | 3 个节点可发现 | 3/3 | **通过** |

这里的 42.748101 dB 比较同一模型、输入、prompt、seed 和采样参数下的无水印输出与水印输出，只度量水印带来的变化。

## 2. 远程环境

| 项目 | 值 |
|---|---|
| Host / HTTP | `10.168.1.168:8189` |
| ComfyUI path | `/app` |
| Custom node path | `/app/custom_nodes/comfyui-dit-watermark` |
| ComfyUI | 0.26.0 |
| Python | 3.11.11 |
| PyTorch | 2.6.0+cu124 |
| GPU | NVIDIA GeForce RTX 4090 24 GB |

注册节点：`GROWDiTSampler`、`GROWWatermarkDetect`、`GROWImagePSNR`。

## 3. 最终工作流参数

| 参数 | 值 |
|---|---|
| Diffusion model | `flux-2-klein-4b-fp8.safetensors` |
| Text encoder | `qwen_3_4b.safetensors` |
| VAE | `flux2-vae.safetensors` |
| Prompt | `Keep the input image exactly unchanged. Preserve every pixel, color, texture, composition, and detail. Add only the invisible watermark.` |
| Seed | `167626463082108` |
| Sampler / steps | Euler / 4 |
| Message / secret key | `watermark` / `watermark` |
| Strength | `1.20` |
| Guidance scale | `4000` |
| Start ratio | `0.50` |
| Frequency band | `[0.15, 0.45)` |
| Max channels / center ratio | `8` / `1.0` |

UI 中 `strength` 按 0.01 步进和两位小数显示，`guidance_scale` 按整数步进显示。检测节点没有 message 输入，只用 secret key 和布局参数恢复帧。

## 4. 图像与 PSNR

| Identity clean reference | GROW + ECC watermarked |
|---|---|
| ![identity clean](outputs/identity_clean.png) | ![identity watermarked](outputs/identity_watermarked.png) |

检测结果：

```text
decoded_message='watermark'
ecc_valid=True
corrected_symbols=1
min_vote_margin=0.090909
```

独立计算和 `GROWImagePSNR` 节点结果一致：

```text
MSE  = 0.00005311166263032
PSNR = 42.748101030269 dB
```

原始上传图如下：

![source input](outputs/source_input.png)

源图为 736×1312，工作流输出为 752×1360。Flux2/VAE 即使在 identity prompt 下也会产生重建变化：缩放后的源图到 clean 为 23.60 dB，到 watermarked 为 23.49 dB。因此验收门槛采用 clean → watermarked 的 42.748101 dB，以排除模型重建误差。

## 5. 纠错实现

message 使用固定 32-byte Reed–Solomon 帧：

```text
1 byte UTF-8 length + 27 bytes payload/padding + 4 bytes parity
```

四个 parity symbols 可纠正任意两个损坏的 byte symbols；超过能力或帧长度/UTF-8 校验失败时返回 `ecc_valid=False`。ASCII 的两个字符对应两个 symbols；中文字符是多字节 UTF-8，纠错能力按 byte symbol 计算。

## 6. 历史防攻击测试

旧 SD2.1 GROW、120-bit payload 的 28 张压缩/percent 攻击结果：

| 攻击 | Exact Match | 平均 Bit Accuracy |
|---|---:|---:|
| AVIF → JPG | 4/4 | 100% |
| HEIF → JPG | 4/4 | 100% |
| JXL → JPG | 1/4 | 99.38% |
| MLIC++ | 1/4 | 98.96% |
| Percent 系列 | **0/12** | 66.39% |

Percent 系列全部失败；它们造成的大规模频域破坏超出两个 RS symbols 的纠错范围。JXL/MLIC++ 的部分失败样本只错少量 bit，新 ECC 有机会修复落在不超过两个 byte symbols 内的错误。

SyncSeal 几何恢复历史结果：

| 分组 | Attacked 平均 Bit Accuracy | Recovered 平均 Bit Accuracy |
|---|---:|---:|
| 旋转 | 49.17% | **98.33%** |
| 裁剪/放大 | 50.63% | **65.42%** |
| 全部 recovered | — | 80.83% |

这些数据是历史 GROW baseline，不是 Flux2 + RS 新协议的直接攻击基准。旋转恢复表现较好，裁剪恢复仍不稳定。

## 7. 产物

- 唯一仓库工作流：`workflows/flux2_klein_image_edit_grow.json`
- Downloads 工作流：`C:\Users\zhangsongbo\Downloads\image_flux2_klein_image_edit_4b_distilled (2).json`
- 原始工作流备份：`C:\Users\zhangsongbo\Downloads\image_flux2_klein_image_edit_4b_distilled (2).json.bak`
- 最终图片：`validation/outputs/identity_clean.png`、`identity_watermarked.png`

UI sampler 链路：

```text
KSamplerSelect (61) → GROWDiTSampler (124) → SamplerCustomAdvanced (64)
```

`GROWWatermarkDetect (125)` 连接 `VAEDecode` 的 IMAGE 和 `VAELoader` 的 VAE，无 message 输入。

## 8. 自动化验证

```powershell
python -m unittest discover -s tests -v
```

测试覆盖 guidance、inference mode、固定帧编解码、1/2-symbol 纠错、3-symbol 拒绝、节点 API 拓扑、img2img API 拓扑及 UI 子图注入。

最终结果：本机系统 Python 与远程 ComfyUI 容器均为 **20/20 通过**。
