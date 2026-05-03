Astra
Astra is an independently refactored codebase derived from dynamicvid. It currently consists of three independently maintainable components: astra_core, llava, and lmms-eval.

Architectural Nomenclature
STV-Guided Dynamic Budget Allocation

Adaptive Backward Temporal Merging

Dual-Perspective Token Selection

Pruning Nomenclature
Visual-Guided Pruning: Visual Recycling

Semantic Recycle Pruning: Semantic Recycling/Pruning

Directory Structure
astra_core/: Implementation of core methodologies.

llava/: A local copy of the LLaVA codebase.

lmms-eval/: A local copy of the evaluation framework.

run_llava_onevision.sh: Minimal evaluation script (with fixed parameters).

setup_astra.sh: One-click environment configuration script.

requirements_Astra.txt: Dependency manifest.

Environment Variables
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

DECORD_NUM_THREADS=8

PYTHONPATH=<astra_root>:<astra_root>/lmms-eval
