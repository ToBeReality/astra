# 🌌 Astra

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)]()

> **Astra** is a high-performance codebase independently refactored from `dynamicvid`. It provides a streamlined, modular architecture for research in dynamic video processing, comprising three maintainable pillars: `astra_core`, `llava`, and `lmms-eval`.

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
