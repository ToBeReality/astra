# Astra

Astra 是从 `dynamicvid` 独立重构出的代码库版本，当前包含可独立维护的 `astra_core`、`llava`、`lmms-eval` 三部分。

## 架构命名

- STV-Guided Dynamic Budget Allocation
- Adaptive Backward Temporal Merging
- Dual-Perspective Token Selection

## 剪枝命名

- Visual-Guided Pruning：视觉回收
- Semantic Recycle Pruning：语义回收剪枝

## 目录

- `astra_core/`：核心方法实现
- `llava/`：LLaVA 代码副本
- `lmms-eval/`：评测框架代码副本
- `run_llava_onevision.sh`：最小评测脚本（固定参数）
- `setup_astra.sh`：一键环境配置
- `requirements_Astra.txt`：依赖清单

## 环境变量

- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
- `DECORD_NUM_THREADS=8`
- `PYTHONPATH=<astra_root>:<astra_root>/lmms-eval`

## 一键配置与运行

```bash
bash setup_astra.sh
source env_astra.sh
bash run_llava_onevision.sh
```
