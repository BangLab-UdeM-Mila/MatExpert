# MatExpert: Decomposing Materials Discovery By Mimicking Human Experts

Material discovery is a critical research area with profound implications for various industries. In this work, we introduce MatExpert, a novel framework that leverages Large Language Models (LLMs) and contrastive learning to accelerate the discovery and design of new solid-state materials. 

Inspired by the workflow of human materials design experts, our approach integrates three key stages:

1. **Retrieval**: MatExpert identifies an existing material that closely matches the desired criteria.
2. **Transition**: MatExpert outlines the necessary modifications to transform this material formulation to meet specific requirements outlined by the initial user query.
3. **Generation**: MatExpert performs detailed computations and structural generation to create a new material based on the provided information.

Our experimental results demonstrate that MatExpert outperforms state-of-the-art methods in material generation tasks, achieving superior performance across various metrics including validity, distribution, and stability. As such, MatExpert represents a meaningful advancement in computational material discovery using language-based generative models.

## Prerequisites

- Python 3.11 (Note: Python 3.12 is not supported)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd MatExpert
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

### Basic Training

1. For the initial training:
   ```bash
   llamafactory-cli train ~/intel/crystal-llm-retrieval/llama_stage/mp_train.yaml
   ```

2. For training with Llama2:
   ```bash
   llamafactory-cli train ~/research/crystal-llm-retrieval/llama_stage/mp_train_llama2.yaml
   ```

### AutoDL Training

For training in an AutoDL environment:
```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir /root/autodl-tmp/Llama-2-7b-chat-hf --token <your-hf-token>
llamafactory-cli train ~/research/crystal-llm-retrieval/llama_stage/mp_train_llama2_autodl.yaml
```

### Large Model Training

For training the 70B model:
```bash
llamafactory-cli train ~/intel/crystal-llm-retrieval/llama_stage/mp_train_70B.yaml
```

## Evaluation

To evaluate the trained models, follow these steps:

1. Set environment variables (if needed):
   ```bash
   export NCCL_P2P_DISABLE="1"
   export NCCL_IB_DISABLE="1"
   ```

2. Run the evaluation pipeline:
   ```bash
   llamafactory-cli train ~/research/crystal-llm-retrieval/llama_stage/autodl/mp_prediction_test_llama2.yaml
   python generate_test_data_stage_2.py
   llamafactory-cli train ~/research/crystal-llm-retrieval/llama_stage/autodl/mp_prediction_test_stage_2_llama2.yaml
   python sample_mp_llama2.py
   ```

3. Clean up and run basic evaluation:
   ```bash
   rm -rf data/basic/*.pkl
   python basic_eval.py --model_name mp_llama2_1 --samples_path /u/dingqian/intel/crystal-llm-retrieval/llama_stage/mp_llama2_1_samples.csv
   ```

## Directory Structure

- `data_second/`: Scripts and notebooks for data generation
- `llama_stage/`: Configuration files and scripts for model training and evaluation
  - `autodl/`: AutoDL-specific training and evaluation scripts
  - `mp_2/`: Alternative model configurations
- `retrieval/`: Scripts for data retrieval and processing
- `basic_eval.py`: Basic evaluation script
- `eval_util.py`: Evaluation utilities

## Citation

If you use this work, please cite it as follows:

```bibtex
@inproceedings{ICLR2025_7d6850f4,
 author = {Ding, Qianggang and Miret, Santiago and Liu, Bang},
 booktitle = {International Conference on Representation Learning},
 editor = {Y. Yue and A. Garg and N. Peng and F. Sha and R. Yu},
 pages = {50113--50132},
 title = {MatExpert: Decomposing Materials Discovery By Mimicking Human Experts},
 url = {https://proceedings.iclr.cc/paper_files/paper/2025/file/7d6850f4c82520793f738d98a72aab9d-Paper-Conference.pdf},
 volume = {2025},
 year = {2025}
}
```