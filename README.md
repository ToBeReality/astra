# 🌌 Astra

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)]()

> **Introduction** Efficient video understanding remains a major challenge for Video Language Models (VLMs), as processing massive visual tokens scales quadratically during the prefill stage. Recent token pruning methods alleviate this burden by compressing spatiotemporal redundancy across frames, while which often rely on a uniform budget and overlook diverse temporal dynamics and semantic relevance. In this work, we propose modeling video token pruning as an adaptive, dynamics-aware process. We introduce three core mechanisms: (i) STV-Guided Dynamic Budget Allocation, which assigns frame-level quotas based on inter-frame variations and query relevance; (ii) Backward Adaptive Temporal Merging, which eliminates cross-frame redundancy by absorbing static backgrounds into earlier anchors to preserve motion continuity; and (iii) Dual-Perspective Token Selection, which preserves visual topology while retrieving text-guided fine-grained semantics. Together, these mechanisms preserve chronological causality and query-specific details while aggressively reducing computation. Evaluated on VideoMME, LongVideoBench, MVBench, and MLVU with diverse VLMs, our method consistently outperforms prior baselines under extreme budgets. Notably, it achieves an exceptional 99.7\% relative accuracy on LLaVA-OneVision at a 20\% token retention ratio, demonstrating superior efficiency and performance.

---

## 💎 Architectural Nomenclature

The framework implements several novel strategies to optimize the trade-off between computational efficiency and model performance:

* **STV-Guided Dynamic Budget Allocation**
    * Dynamic resource management guided by Spatio-Temporal Variance.
* **Adaptive Backward Temporal Merging**
    * Context-aware merging of temporal tokens to minimize redundancy.
* **Dual-Perspective Token Selection**
    * A robust mechanism for selecting critical tokens from multiple analytical perspectives.

## ✂️ Pruning Strategies

Advanced pruning mechanisms designed for Large Multi-modal Models:

* **Visual-Guided Pruning**
    * Saliency-based visual token recycling.
* **Semantic Recycle Pruning**
    * High-level semantic importance-driven pruning and feature recovery.

---

## 📂 Directory Structure

| Path | Description |
| :--- | :--- |
| `astra_core/` | Implementation of core methodologies and algorithms. |
| `llava/` | Local instance of the LLaVA codebase. |
| `lmms-eval/` | Framework for comprehensive LMM evaluation. |
| `run_llava_onevision.sh` | Minimal evaluation script with fixed parameters. |
| `setup_astra.sh` | One-click environment initialization script. |
| `requirements_Astra.txt` | List of mandatory Python dependencies. |

---

## ⚙️ Environment Variables

Configure the following variables to ensure optimal execution:

```bash
# Multi-GPU Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Video Decoding Acceleration
export DECORD_NUM_THREADS=8

# Python Path Mapping
export PYTHONPATH=<astra_root>:<astra_root>/lmms-eval
