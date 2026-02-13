# vLLM notes 

This benchmark is based on prompting models to look at some text and produce either "safe" or "unsafe" keywords to classify the given input (optionally with generating the list of violated policies).
We would like to benchmark models with the vLLM inference engine instead of the default transformers backend.
Since the benchmark consists of extracting logits for safe and unsafe keywords from the first generated token by an LLM, this requires transferring logits for the full vocab size over HTTP from vllm server to our client.
This is prohibitively expensive (vLLM becomes 6x slower than Transformers), therefore we only transfer Top_K logits (e.g. 10). Under the assumption that model will always assign large logits to safe/unsafe keywords (it has been trained to do so), and under the GuardBench evaluation pipeline where only the relative ratio of safe/unsafe logits is used to compute F1 and Recall metrics, we are guaranteed to have exactly the same score with Top_K logits as we would have with full dictionary logits. This has been verified and battle tested across all 40 benchmarks in the GuardBench repository.


## Example commands for meta-llama/Llama-Guard-4-12B

0. setup environment: `uv pip install -r llama4guard_vllm_requirements.txt`
1. serve the model with: `vllm serve meta-llama/Llama-Guard-4-12B -tp 1 --api-key EMPTY --logprobs-mode processed_logits --max-logprobs 10 --max-model-len 131072`
2. evaluate the model with: `python vllm_server_mm_eval.py --model meta-llama/Llama-Guard-4-12B --datasets all --output_dir output_dir/Llama-Guard-4-12B --top_logprobs 10`


<div align="center">
  <img src="https://repository-images.githubusercontent.com/837144095/8190ad0e-e9ff-4dda-9116-644d62d6b886">
</div>

<p align="center">
  <!-- Python -->
  <a href="https://www.python.org" alt="Python"><img src="https://badges.aleen42.com/src/python.svg"></a>
  <!-- Version -->
  <a href="https://pypi.org/project/guardbench/"><img src="https://img.shields.io/pypi/v/guardbench?color=light-green" alt="PyPI version"></a>
  <!-- Docs -->
  <a href="https://github.com/AmenRa/guardbench/tree/main/docs"><img src="https://img.shields.io/badge/docs-passing-<COLOR>.svg" alt="Documentation Status"></a>
  <!-- Black -->
  <a href="https://github.com/psf/black" alt="Code style: black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
  <!-- License -->
  <a href="https://interoperable-europe.ec.europa.eu/sites/default/files/custom-page/attachment/2020-03/EUPL-1.2%20EN.txt"><img src="https://img.shields.io/badge/license-EUPL-blue.svg" alt="License: EUPL-1.2"></a>
</p>

# GuardBench

## 🔥 News

- [October 9, 2025] GuardBench now supports four additional datasets: [JBB Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors), [NicheHazardQA](https://huggingface.co/datasets/SoftMINER-Group/NicheHazardQA), [HarmEval](https://huggingface.co/datasets/SoftMINER-Group/HarmEval), and [TechHazardQA](https://huggingface.co/datasets/SoftMINER-Group/TechHazardQA). Also, it now allows for choosing the metrics to show at the end of the evaluation. Supported metrics are: `precision` (Precision), `recall` (Recall), `f1` (F1), `mcc` (Matthews Correlation Coefficient), `auprc` (AUPRC), `sensitivity` (Sensitivity), `specificity` (Specificity), `g_mean` (G-Mean), `fpr` (False Positive Rate), `fnr` (False Negative Rate).

## ⚡️ Introduction
[`GuardBench`](https://github.com/AmenRa/guardbench) is a Python library for the evaluation of guardrail models, i.e., LLMs fine-tuned to detect unsafe content in human-AI interactions.
[`GuardBench`](https://github.com/AmenRa/guardbench) provides a common interface to 40 evaluation datasets, which are downloaded and converted into a [standardized format](docs/data_format.md) for improved usability.
It also allows to quickly [compare results and export](docs/report.md) `LaTeX` tables for scientific publications.
[`GuardBench`](https://github.com/AmenRa/guardbench)'s benchmarking pipeline can also be leveraged on [custom datasets](docs/custom_dataset.md).

[`GuardBench`](https://github.com/AmenRa/guardbench) was featured in [EMNLP 2024](https://2024.emnlp.org).
The related paper is available [here](https://aclanthology.org/2024.emnlp-main.1022.pdf).

[`GuardBench`](https://github.com/AmenRa/guardbench) has a public [leaderboard](https://huggingface.co/spaces/AmenRa/guardbench-leaderboard) available on HuggingFace.

You can find the list of supported datasets [here](docs/datasets.md).
A few of them requires authorization. Please, read [this](docs/get_datasets.md).

If you use [`GuardBench`](https://github.com/AmenRa/guardbench) to evaluate guardrail models for your scientific publications, please consider [citing our work](#-citation).

## ✨ Features
- [40 datasets](docs/datasets.md) for guardrail models evaluation.
- Automated evaluation pipeline.
- User-friendly.
- [Extendable](docs/custom_dataset.md).
- Reproducible and sharable evaluation.
- Exportable [evaluation reports](docs/report.md).

## 🔌 Requirements
```bash
python>=3.10
```

## 💾 Installation 
```bash
pip install guardbench
```

## 💡 Usage
```python
from guardbench import benchmark

def moderate(
    conversations: list[list[dict[str, str]]],  # MANDATORY!
    # additional `kwargs` as needed
) -> list[float]:
    # do moderation
    # return list of floats (unsafe probabilities)

benchmark(
    moderate=moderate,  # User-defined moderation function
    model_name="My Guardrail Model",
    batch_size=1,              # Default value
    datasets="all",            # Default value
    metrics=["f1", "recall"],  # Default value
    # Note: you can pass additional `kwargs` for `moderate`
)
```

### 📖 Examples
- Follow our [tutorial](docs/llama_guard.md) on benchmarking [`Llama Guard`](https://arxiv.org/pdf/2312.06674) with [`GuardBench`](https://github.com/AmenRa/guardbench).
- More examples are available in the [`scripts`](scripts/effectiveness) folder.

## 📚 Documentation
Browse the documentation for more details about:
- The [datasets](docs/datasets.md) and how to [obtain them](docs/get_datasets.md).
- The [data format](data_format.md) used by [`GuardBench`](https://github.com/AmenRa/guardbench).
- How to use the [`Report`](docs/report.md) class to compare models and export results as `LaTeX` tables.
- How to leverage [`GuardBench`](https://github.com/AmenRa/guardbench)'s benchmarking pipeline on [custom datasets](docs/custom_dataset.md).

## 🏆 Leaderboard
You can find [`GuardBench`](https://github.com/AmenRa/guardbench)'s leaderboard [here](https://huggingface.co/spaces/AmenRa/guardbench-leaderboard). If you want to submit your results, please contact us.
<!-- All results can be reproduced using the provided [`scripts`](scripts/effectiveness).   -->

## 👨‍💻 Authors
- Elias Bassani (European Commission - Joint Research Centre)

## 🎓 Citation
```bibtex
@inproceedings{guardbench,
    title = "{G}uard{B}ench: A Large-Scale Benchmark for Guardrail Models",
    author = "Bassani, Elias  and
      Sanchez, Ignacio",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.1022",
    doi = "10.18653/v1/2024.emnlp-main.1022",
    pages = "18393--18409",
}
```

## 🎁 Feature Requests
Would you like to see other features implemented? Please, open a [feature request](https://github.com/AmenRa/guardbench/issues/new?assignees=&labels=enhancement&template=feature_request.md&title=%5BFeature+Request%5D+title).

## 📄 License
[GuardBench](https://github.com/AmenRa/guardbench) is provided as open-source software licensed under [EUPL v1.2](https://github.com/AmenRa/guardbench/blob/master/LICENSE).
