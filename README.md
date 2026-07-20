# ComfyUI-GROW-DiT

GROW progressive latent-frequency watermarking for ComfyUI diffusion transformers (DiT), with inversion-free VAE detection and an RGB PSNR metric.

The algorithm is based on [luopengchen/GROW](https://github.com/luopengchen/GROW), adapted from Stable Diffusion/DDIM to ComfyUI's model-agnostic sampler interface and short distilled DiT schedules.

The sampler node wraps ComfyUI's selected sampler. During the configured final denoising steps it intercepts each predicted clean latent (`x0`), builds a secret-keyed repeated payload in a mid-frequency FFT/DCT-proxy band, computes an fp32 frequency loss, and returns one guided `x0` step. The underlying DiT, conditioning, scheduler, noise, and VAE are unchanged.

## Nodes

### GROW DiT Sampler

Connect it between `KSamplerSelect` and `SamplerCustomAdvanced`:

```text
KSamplerSelect → GROW DiT Sampler → SamplerCustomAdvanced
```

Inputs:

| Input | Meaning | Default |
|---|---|---:|
| `message` | UTF-8 payload | `zhangp365123456` |
| `secret_key` | Deterministically shuffles frequency coordinates | `watermark` |
| `strength` | Minimum signed frequency margin | `1.0` |
| `guidance_scale` | Gradient step multiplier | `2000` |
| `start_ratio` | Fraction of sampling completed before guidance starts | `0.5` |
| `dct_min`, `dct_max` | Normalized 2D frequency band | `0.15`, `0.45` |
| `max_channels` | Leading latent channels used | `8` |
| `center_ratio` | Central latent area used | `1.0` |

FFT, loss, and gradient calculations always use fp32; the guided latent is converted back to the DiT's original dtype. The loss is a one-sided sign-margin form of GROW's keyed frequency objective: coefficients that already have the correct sign and enough margin are not pulled toward a fixed amplitude. This is important for high-PSNR distilled DiT workflows with only a few denoising steps.

### GROW Watermark Detect

Connect the generated `IMAGE` and the same `VAE`. Use exactly the same message, secret key, frequency band, channel count, and center ratio as the sampler. The node VAE-encodes the image and returns:

- decoded message;
- exact-match boolean;
- bit accuracy and correct/total bits;
- minimum majority-vote margin.

Detection does not load a Stable Diffusion pipeline and does not perform diffusion inversion.

### GROW Image PSNR

Connect deterministic clean and watermarked images of identical shape. The node reports RGB PSNR in dB for tensors in `[0,1]`.

## Install

Copy or clone this repository into `ComfyUI/custom_nodes/ComfyUI-GROW-DiT`, then restart ComfyUI. PyTorch supplied by ComfyUI is the only runtime dependency.

## Flux2 Klein workflow

The included injector understands current ComfyUI object-link workflow JSON, including nested subgraphs:

```powershell
python scripts/inject_workflow.py input.json output.json \
  --message zhangp365123456 \
  --secret-key watermark \
  --strength 1.0 \
  --guidance-scale 2000
```

It inserts the GROW sampler into the sampler path and adds a detector connected to `VAEDecode` and `VAELoader`. When input and output paths are identical, it creates `input.json.bak` once.

Ready-to-use files:

- `workflows/flux2_klein_image_edit_grow.json`: UI workflow with GROW sampler and detector.
- `workflows/flux2_klein_image_edit_grow_api.json`: deterministic API workflow used for acceptance.
- `validation/validation_report.md`: remote environment, parameters, exact extraction result, PSNR and artifact hashes.

## Reproducible quality validation

Generate clean and watermarked images with the same model, input image, prompt, dimensions, scheduler, sampler, seed and precision. The only graph difference should be the GROW sampler wrapper. Save losslessly as PNG, detect the payload through the workflow VAE, and calculate PSNR against the clean output.

The defaults are the validated Flux2 Klein 4B Distilled four-step preset. Small strength and guidance values preserve quality but reduce robustness. Tune them together for other models and require both exact payload recovery and the desired PSNR; a high bit accuracy alone is not an exact message recovery.

## Compatibility and limitations

- Designed around ComfyUI's `SAMPLER` interface and tested with Flux2 Klein 4B Distilled using Euler sampling.
- Other image DiTs with four-dimensional latent tensors should work without model-specific code.
- Video/nested latents and samplers returning non-4D predictions are rejected explicitly.
- Geometric transforms change frequency coordinates; use an alignment/recovery stage before detection.
- A workflow JSON contains the secret key in plain text. Do not distribute a production secret inside a public workflow.

## Tests

```powershell
python -m unittest discover -s tests -v
```
