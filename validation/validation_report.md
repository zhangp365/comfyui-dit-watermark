# ComfyUI-GROW-DiT Flux2 Klein 验收报告

## 1. 验收结论

自定义节点已在 `10.168.1.168:8189` 的 ComfyUI 容器中部署并通过端到端测试。

| 验收项 | 要求 | 实测 | 结论 |
|---|---:|---:|---|
| GROW message Exact Match | 完整恢复 | `zhangp365123456`，120/120 bit | **通过** |
| Bit Accuracy | 100% | 100% | **通过** |
| Watermarked vs clean PSNR | > 40 dB | **46.115162 dB** | **通过** |
| 工作流集成 | 可运行 | sampler 与 detector 已加入 Flux2 子图 | **通过** |
| ComfyUI 节点加载 | 3 个节点可发现 | 3/3 | **通过** |

最终 PSNR 比最低要求高 `6.115162 dB`。Clean 工作流重复运行得到逐像素完全相同的 PNG（MSE=0，PSNR=∞），因此水印对比不受随机漂移影响。

## 2. 远程运行环境

| 项目 | 值 |
|---|---|
| Host | `10.168.1.168` |
| ComfyUI HTTP | `http://10.168.1.168:8189` |
| Container | `a993c199f853` (`gallant_wilson`) |
| ComfyUI path | `/app` |
| Custom node path | `/app/custom_nodes/ComfyUI-GROW-DiT` |
| ComfyUI version | `0.26.0` |
| ComfyUI commit | `5db3311946d5109434b9c9623d37c05195612416` |
| Python | 3.11.11 |
| PyTorch | `2.6.0+cu124` |
| GPU | NVIDIA GeForce RTX 4090, 24 GB |

启动日志确认 `/app/custom_nodes/ComfyUI-GROW-DiT` 导入耗时 0.0 秒且无 traceback。`/object_info` 中已注册：

- `GROWDiTSampler`
- `GROWWatermarkDetect`
- `GROWImagePSNR`

## 3. Flux2 工作流参数

Clean 与 watermarked 两次运行只有 sampler 路径是否经过 `GROWDiTSampler` 这一项不同，其余输入完全一致。

| 参数 | 值 |
|---|---|
| Diffusion model | `flux-2-klein-4b-fp8.safetensors` |
| Text encoder | `qwen_3_4b.safetensors` |
| VAE | `flux2-vae.safetensors` |
| Prompt | `Change the dog color to blue.` |
| Input | `generation-b1e59042-91a9-4338-8308-5acb024f7c5a.png` |
| Seed | `167626463082108` |
| Sampler | Euler |
| Flux2 scheduler steps | 4 |
| CFG | 1.0 |
| Output size | 752×1360 RGB |

最终 GROW 参数：

| 参数 | 值 |
|---|---:|
| Message | `zhangp365123456` |
| Secret key | `watermark` |
| Strength / sign margin | `1.0` |
| Guidance scale | `2000.0` |
| Start ratio | `0.5` |
| DCT/FFT band | `[0.15, 0.45)` |
| Max latent channels | `8` |
| Center ratio | `1.0` |
| Repetitions | 每个 bit 使用奇数次重复，避免多数投票平局 |

## 4. 技术实现

节点沿用 GROW 的核心流程：在去噪后半程拦截 DiT 预测的干净 latent `x0`，对 keyed 中频区域执行正交归一化二维 FFT 实部运算，通过频域损失求取梯度，再把更新后的 `x0` 返回给原 ComfyUI sampler。

针对 Flux2 Klein 的 4 步蒸馏工作流做了两项必要适配：

1. ComfyUI sampler 运行在 inference mode 中，节点只在 4D latent 的频域 loss/gradient 小区域临时关闭 inference mode并启用 fp32 autograd，不改变 DiT 推理精度。
2. 原 GROW 绝对 MSE 会把已经符号正确的大系数也拉向固定幅度。这里改为单边 sign-margin loss，只更新符号错误或间隔不足的系数，减少不可见水印造成的图像变化。

检测节点使用同一个 Flux2 VAE 对输出图片编码，不需要 Stable Diffusion pipeline，也不进行 diffusion inversion。它使用相同 key 重建坐标，对重复系数做多数投票，并同时报告完整字符串与 bit 级结果。

## 5. 最终结果

ComfyUI detector 输出：

```text
decoded='zhangp365123456'
exact=True
bits=120/120
accuracy=1.000000
min_vote_margin=0.043478
```

PSNR 使用 `[0,1]` RGB 张量计算：

```text
MSE  = 0.00002446153997604016
PSNR = 10 * log10(1 / MSE)
     = 46.115162054336736 dB
```

ComfyUI 内部 `GROWImagePSNR` 节点独立输出：

```text
PSNR=46.115162 dB
```

两种计算结果在 0.000001 dB 精度内一致。

## 6. 产物

| 文件 | SHA-256 |
|---|---|
| `validation/outputs/clean_final_00001_.png` | `E8C1A90659E7B3DA86D20FF218E68E5569EF5DAA074CF027B45B56C79E65220E` |
| `validation/outputs/watermarked_final_00001_.png` | `36470DE6A8841A001A66241011EE90672F8DD677318F1914863BD4962FC2D36A` |
| `workflows/flux2_klein_image_edit_grow.json` | `C8D356F6A10ECEFEBA2360FE62F893F215A9A4C2ADB2B1574C35F61B67EF453F` |

工作流位置：

- 用户 Downloads 工作流：`C:\Users\zhangsongbo\Downloads\image_flux2_klein_image_edit_4b_distilled (2).json`
- 原始工作流备份：`C:\Users\zhangsongbo\Downloads\image_flux2_klein_image_edit_4b_distilled (2).json.bak`
- 仓库 UI 工作流：`workflows/flux2_klein_image_edit_grow.json`
- 仓库 API 工作流：`workflows/flux2_klein_image_edit_grow_api.json`

UI 工作流中的 sampler 链路已经从：

```text
KSamplerSelect (61) → SamplerCustomAdvanced (64)
```

变为：

```text
KSamplerSelect (61) → GROWDiTSampler (124) → SamplerCustomAdvanced (64)
```

`GROWWatermarkDetect (125)` 同时连接到 `VAEDecode` 的 IMAGE 输出和 `VAELoader` 的 VAE 输出。

## 7. 自动化测试

本地系统 Python 3.10 和远程容器 Python 3.11 均运行同一套 unittest。最终测试覆盖：

- UTF-8 payload 编解码；
- secret key 布局确定性；
- payload 容量校验；
- 奇数重复与无平局多数投票；
- fp32 频域 guidance 损失下降；
- ComfyUI inference mode 回归；
- sampler 后半程启用逻辑；
- clean/watermarked API prompt 拓扑；
- 旧式根图 link 与新式子图 link 的混合工作流注入。

最终本地测试结果：14 项全部通过。最终完整仓库远程部署测试结果：14 项全部通过。
