# Handoff — llm-training
**Date:** 2026-04-22  
**Session ended:** ~12:05 MST

---

## Where I left off

Trying to start training for `game-dev` (Qwen3.6-27B). Got blocked on ROCm not seeing the GPU. Spent the session diagnosing and fixing the environment. **The machine has NOT been rebooted yet** — training hasn't started.

---

## Next physical action

**Reboot the machine.** The grub menu will show for 3 seconds — let it boot into `6.17.0-22-generic` (top of list, already set as default).

After boot:
```bash
export PATH=/opt/rocm/bin:$PATH
rocminfo   # should show Strix Halo iGPU + RX 7900 XTX as GPU agents

# If rocminfo works, start training:
cd ~/git/llm-training
uv run llm-train --base qwen3.6-27b --specialization game-dev
```

If `rocminfo` still fails with "Unable to open /dev/kfd read-write: Invalid argument", select `6.8.0-110-generic` from the grub menu instead (it's in the list, the menu stays for 3 seconds).

---

## What was done this session

**ROCm upgraded: 6.4 → 7.2.2**
- Root cause of GPU issue: kernel 6.17.0-20 + ROCm 6.4 KFD interface mismatch
- Installed `amdgpu-install_7.2.2.70202-1_all.deb`, updated apt sources to 7.2.2
- `hsa-rocr` upgraded 1.15 → 1.18, `rocm-core` 6.4 → 7.2.2
- KFD topology IS fully initialized (sysfs has 4 nodes: CPU + Strix Halo iGPU + 3× RX 7900 XTX) — just the open() call fails on the current kernel

**Grub configured:**
- Default: `6.17.0-22-generic` (already installed, likely has the KFD fix)
- Fallback: `6.8.0-110-generic` (stable Ubuntu HWE, also installed)
- Timeout: 3 seconds (was hidden/instant before)
- `/etc/default/grub`: `GRUB_DEFAULT=saved`

**train.py fixed for ROCm:**
- Auto-detects CUDA vs ROCm vs CPU
- ROCm path: runs bf16 LoRA (skips BNB 4-bit — BNB doesn't support ROCm reliably)
- Optimizer fallback: `adamw_torch_fused` when `paged_adamw_8bit` can't be used (BNB not active)
- No unsloth — standard peft/transformers pipeline (unsloth is CUDA-only)

**game-dev.yaml updated with correct LoRA target modules for Qwen3.6-27B:**
- The model is a hybrid architecture: GatedDeltaNet (3 of every 4 layers) + full attention (every 4th)
- GatedDeltaNet uses: `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj`
- Full attention uses: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- MLP (all layers): `gate_proj`, `up_proj`, `down_proj`
- Without this fix, only the 25% full-attention layers would get LoRA coverage

---

## Hardware confirmed

- **GPU[0] Strix Halo iGPU** (gfx1151, 0x1586): 80 SIMDs, ~124GB unified RAM as local GPU memory
- **GPU[1] RX 7900 XTX** (gfx1100, 0x744c): 192 SIMDs × 3 sub-devices, 24GB GDDR6

For 27B model training on ROCm:
- Strix Halo path: bf16 LoRA (~54GB model + overhead, fits in 124GB) ← **this is the plan**
- RX 7900 XTX path: would need QLoRA (BNB on ROCm is messy), not the path we're taking

---

## Open questions

- Does 6.17.0-22 actually fix the KFD EINVAL? (unknown, have to reboot and test)
- If neither 6.17.22 nor 6.8.0-110 fixes KFD: report bug to AMD, may need to wait for ROCm 7.3
- Training speed on Strix Halo iGPU: unknown, will learn from first run
- Shader (Devstral-24B) and FIM (Qwen2.5-Coder-1.5B) training: ready to start after game-dev confirms

---

## What I almost forgot

- The datasets are fully curated and ready: `datasets/processed/game-dev/train.jsonl` has 12,234 examples
- All base models already downloaded: `models/bases/qwen3.6-27b/`, `devstral-small-2-24b/`, `qwen2.5-coder-1.5b/`
- `curate.py` and `download.py` were already run in a previous session — don't re-run them

---

## Don't forget about (cross-project)

- `normandy-sr2/configs/bridges/local.yaml` was open in the IDE at session end — may have been something you were about to work on
- Store-marketing dataset generation was running in the background (rate-limited, ~670/1,666 at last check) — probably finished or stalled by now
- game-dev-github extraction: only 1,256/15,000 records — still needs to run more
