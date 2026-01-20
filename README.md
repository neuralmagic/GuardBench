# Eldar README

This benchmark is based on prompting models to look at some text and produce either "safe" or "unsafe" keywords to classify the given input (optionally with generating the list of violated policies).
We would like to benchmark models with the vLLM inference engine instead of the default transformers backend.
We have two options to use vLLM: (1) via vllm server, or (2) via python interface (LLM class).

## (1) vLLM server
Since the benchmark consists of extracting logits for safe and unsafe keywords from the first generated token by an LLM, this requires transferring logits for the full vocab size over HTTP from vllm server to our client. After running this version with the full vocabulary size (no concurrent requests, just batch-size=1) we notice that the full 40 datasets eval takes 12 hours compared to 2 hours via transformers backend. The slowness comes from transferring all vocab size logits over HTTP. We validate this via benchmarking. Serve the model with `vllm serve meta-llama/LlamaGuard-7b -tp 1 --api-key EMPTY --logprobs-mode raw_logits --max-logprobs 32000` and measure time to transfer full vocab `python vllm_benchmarking_toplogprobs.py --microbench --model meta-llama/LlamaGuard-7b --datasets all --batch_size 1 --output_dir test` versus only top 2 logits (for safe and unsafe keywords) `python vllm_eval.py --microbench --model meta-llama/LlamaGuard-7b --datasets all --batch_size 1 --output_dir test --top_logprobs 2 --logit_bias_strength 100` (we add logit bias to safe and unsafe keywords and request only top 2 logits; here we assume that our bias is large enough so that safe/unsafe will always be in the top 2; this doesn't change their relative probabilities because we add the same bias to both).
The results are as follows:
```bash
microbench_moderate results
  iters=20 warmup=5
  top_logprobs=32000 logit_bias_strength=0.0
  total_s: mean=0.4481 p50=0.4449 p90=0.4647
  rpc_s:   mean=0.4414 p50=0.4380 p90=0.4580
  parse_s: mean=0.0000 p50=0.0000 p90=0.0000

microbench_moderate results
  iters=20 warmup=5
  top_logprobs=2 logit_bias_strength=100.0
  total_s: mean=0.0183 p50=0.0184 p90=0.0186
  rpc_s:   mean=0.0180 p50=0.0181 p90=0.0183
  parse_s: mean=0.0000 p50=0.0000 p90=0.0000
```
So, `p50(top_logprobs=32000) / p50(top_logprobs=2) = 25` times slower!

TLDR: With vLLM server approach, we should run evaluations by adding logit-bias to safe/unsafe keywords and requesting only top 2 logits over HTTP. The most important note here is that the model should be served with `vllm serve meta-llama/LlamaGuard-7b -tp 1 --api-key EMPTY --logprobs-mode processed_logits --max-logprobs 32000` (`processed_logits` returns logits after the bias is applied). And we run the evaluation script as: `python vllm_server_eval.py --model meta-llama/LlamaGuard-7b --datasets all --output_dir top2_logs --top_logprobs 2 --logit_bias_strength 100`.

The results `(F1, Recall)` are fully matching results with transformers backend (baseline):
| Dataset                   | **Transformers** | **vLLM (Full Vocab)** | **vLLM (Top 2 with Bias)** |
| ------------------------- | ----------------------------- | ---------------------------------- | --------------------------------------- |
| AART                      | (0.904, 0.824)                | (0.903, 0.824)                     | (0.903, 0.824)                          |
| AdvBench Behaviors        | (0.911, 0.837)                | (0.911, 0.837)                     | (0.911, 0.837)                          |
| AdvBench Strings          | (0.894, 0.808)                | (0.892, 0.805)                     | (0.892, 0.805)                          |
| BeaverTails 330k          | (0.685, 0.545)                | (0.687, 0.546)                     | (0.687, 0.546)                          |
| Bot-Adversarial Dialogue  | (0.634, 0.729)                | (0.634, 0.730)                     | (0.634, 0.730)                          |
| CatQA                     | (0.890, 0.802)                | (0.888, 0.798)                     | (0.888, 0.798)                          |
| ConvAbuse                 | (0.437, 0.312)                | (0.437, 0.312)                     | (0.437, 0.312)                          |
| DecodingTrust Stereotypes | (0.932, 0.873)                | (0.934, 0.877)                     | (0.934, 0.877)                          |
| DICES 350                 | (0.272, 0.215)                | (0.270, 0.215)                     | (0.270, 0.215)                          |
| DICES 990                 | (0.411, 0.348)                | (0.415, 0.355)                     | (0.415, 0.355)                          |
| Do Anything Now Questions | (0.660, 0.492)                | (0.662, 0.495)                     | (0.662, 0.495)                          |
| DoNotAnswer               | (0.487, 0.322)                | (0.485, 0.321)                     | (0.485, 0.321)                          |
| DynaHate                  | (0.804, 0.842)                | (0.804, 0.842)                     | (0.804, 0.842)                          |
| HarmEval                  | (0.516, 0.347)                | (0.520, 0.351)                     | (0.520, 0.351)                          |
| HarmBench Behaviors       | (0.650, 0.481)                | (0.650, 0.481)                     | (0.650, 0.481)                          |
| HarmfulQ                  | (0.947, 0.900)                | (0.945, 0.895)                     | (0.945, 0.895)                          |
| HarmfulQA Questions       | (0.579, 0.408)                | (0.577, 0.405)                     | (0.577, 0.405)                          |
| HarmfulQA                 | (0.171, 0.094)                | (0.174, 0.095)                     | (0.174, 0.095)                          |
| HateCheck                 | (0.943, 0.979)                | (0.943, 0.979)                     | (0.943, 0.979)                          |
| Hatemoji Check            | (0.862, 0.808)                | (0.863, 0.810)                     | (0.863, 0.810)                          |
| HEx-PHI                   | (0.830, 0.710)                | (0.830, 0.710)                     | (0.830, 0.710)                          |
| I-CoNa                    | (0.956, 0.916)                | (0.956, 0.916)                     | (0.956, 0.916)                          |
| I-Controversial           | (0.947, 0.900)                | (0.947, 0.900)                     | (0.947, 0.900)                          |
| I-MaliciousInstructions   | (0.883, 0.790)                | (0.883, 0.790)                     | (0.883, 0.790)                          |
| I-Physical-Safety         | (0.147, 0.080)                | (0.147, 0.080)                     | (0.147, 0.080)                          |
| JBB Behaviors             | (0.789, 0.730)                | (0.789, 0.730)                     | (0.789, 0.730)                          |
| MaliciousInstruct         | (0.901, 0.820)                | (0.901, 0.820)                     | (0.901, 0.820)                          |
| MITRE                     | (0.296, 0.174)                | (0.296, 0.174)                     | (0.296, 0.174)                          |
| NicheHazardQA             | (0.511, 0.343)                | (0.508, 0.340)                     | (0.508, 0.340)                          |
| OpenAI Moderation Dataset | (0.745, 0.686)                | (0.747, 0.690)                     | (0.747, 0.690)                          |
| ProsocialDialog           | (0.518, 0.360)                | (0.519, 0.360)                     | (0.519, 0.360)                          |
| SafeText                  | (0.134, 0.073)                | (0.139, 0.076)                     | (0.139, 0.076)                          |
| SimpleSafetyTests         | (0.925, 0.860)                | (0.925, 0.860)                     | (0.925, 0.860)                          |
| StrongREJECT Instructions | (0.908, 0.831)                | (0.908, 0.831)                     | (0.908, 0.831)                          |
| TDCRedTeaming             | (0.889, 0.800)                | (0.889, 0.800)                     | (0.889, 0.800)                          |
| TechHazardQA              | (0.777, 0.635)                | (0.777, 0.636)                     | (0.777, 0.636)                          |
| Toxic Chat                | (0.558, 0.434)                | (0.565, 0.439)                     | (0.565, 0.439)                          |
| ToxiGen                   | (0.783, 0.761)                | (0.782, 0.764)                     | (0.782, 0.764)                          |
| XSTest                    | (0.817, 0.825)                | (0.814, 0.820)                     | (0.814, 0.820)                          |

The expected runtime on a single H100 GPU is: Transformers backend (2h), vLLM server w/ full vocab (12h), vLLM server with top 2 logits and bias (30mins).

## (2) Python interface (LLM class)
We also implement a version with vLLM's Python interface via LLM class. This one has a very similar structure as the original transformers-based implementation. Run with: `python vllm_pyclass_eval.py --model meta-llama/LlamaGuard-7b --datasets all --output_dir my_logs --batch_size -1`. To run all evals, it takes 4h on a single H100 GPU.

| Dataset                   | **Transformers** | **vLLM server (Full Vocab)** | **vLLM (LLM class)** |
| ------------------------- | ----------------------------- | ---------------------------------- | --------------------------------- |
| AART                      | (0.904, 0.824)                | (0.903, 0.824)                     | (0.903, 0.824)                    |
| AdvBench Behaviors        | (0.911, 0.837)                | (0.911, 0.837)                     | (0.912, 0.838)                    |
| AdvBench Strings          | (0.894, 0.808)                | (0.892, 0.805)                     | (0.893, 0.807)                    |
| BeaverTails 330k          | (0.685, 0.545)                | (0.687, 0.546)                     | (0.687, 0.546)                    |
| Bot-Adversarial Dialogue  | (0.634, 0.729)                | (0.634, 0.730)                     | (0.636, 0.731)                    |
| CatQA                     | (0.890, 0.802)                | (0.888, 0.798)                     | (0.886, 0.795)                    |
| ConvAbuse                 | (0.437, 0.312)                | (0.437, 0.312)                     | (0.446, 0.320)                    |
| DecodingTrust Stereotypes | (0.932, 0.873)                | (0.934, 0.877)                     | (0.934, 0.876)                    |
| DICES 350                 | (0.272, 0.215)                | (0.270, 0.215)                     | (0.283, 0.228)                    |
| DICES 990                 | (0.411, 0.348)                | (0.415, 0.355)                     | (0.411, 0.348)                    |
| Do Anything Now Questions | (0.660, 0.492)                | (0.662, 0.495)                     | (0.662, 0.495)                    |
| DoNotAnswer               | (0.487, 0.322)                | (0.485, 0.321)                     | (0.487, 0.322)                    |
| DynaHate                  | (0.804, 0.842)                | (0.804, 0.842)                     | (0.804, 0.843)                    |
| HarmEval                  | (0.516, 0.347)                | (0.520, 0.351)                     | (0.518, 0.349)                    |
| HarmBench Behaviors       | (0.650, 0.481)                | (0.650, 0.481)                     | (0.650, 0.481)                    |
| HarmfulQ                  | (0.947, 0.900)                | (0.945, 0.895)                     | (0.942, 0.890)                    |
| HarmfulQA Questions       | (0.579, 0.408)                | (0.577, 0.405)                     | (0.580, 0.408)                    |
| HarmfulQA                 | (0.171, 0.094)                | (0.174, 0.095)                     | (0.175, 0.096)                    |
| HateCheck                 | (0.943, 0.979)                | (0.943, 0.979)                     | (0.943, 0.979)                    |
| Hatemoji Check            | (0.862, 0.808)                | (0.863, 0.810)                     | (0.863, 0.810)                    |
| HEx-PHI                   | (0.830, 0.710)                | (0.830, 0.710)                     | (0.830, 0.710)                    |
| I-CoNa                    | (0.956, 0.916)                | (0.956, 0.916)                     | (0.956, 0.916)                    |
| I-Controversial           | (0.947, 0.900)                | (0.947, 0.900)                     | (0.947, 0.900)                    |
| I-MaliciousInstructions   | (0.883, 0.790)                | (0.883, 0.790)                     | (0.883, 0.790)                    |
| I-Physical-Safety         | (0.147, 0.080)                | (0.147, 0.080)                     | (0.147, 0.080)                    |
| JBB Behaviors             | (0.789, 0.730)                | (0.789, 0.730)                     | (0.785, 0.730)                    |
| MaliciousInstruct         | (0.901, 0.820)                | (0.901, 0.820)                     | (0.901, 0.820)                    |
| MITRE                     | (0.296, 0.174)                | (0.296, 0.174)                     | (0.296, 0.174)                    |
| NicheHazardQA             | (0.511, 0.343)                | (0.508, 0.340)                     | (0.511, 0.343)                    |
| OpenAI Moderation Dataset | (0.745, 0.686)                | (0.747, 0.690)                     | (0.748, 0.690)                    |
| ProsocialDialog           | (0.518, 0.360)                | (0.519, 0.360)                     | (0.519, 0.361)                    |
| SafeText                  | (0.134, 0.073)                | (0.139, 0.076)                     | (0.134, 0.073)                    |
| SimpleSafetyTests         | (0.925, 0.860)                | (0.925, 0.860)                     | (0.925, 0.860)                    |
| StrongREJECT Instructions | (0.908, 0.831)                | (0.908, 0.831)                     | (0.908, 0.831)                    |
| TDCRedTeaming             | (0.889, 0.800)                | (0.889, 0.800)                     | (0.889, 0.800)                    |
| TechHazardQA              | (0.777, 0.635)                | (0.777, 0.636)                     | (0.778, 0.636)                    |
| Toxic Chat                | (0.558, 0.434)                | (0.565, 0.439)                     | (0.566, 0.439)                    |
| ToxiGen                   | (0.783, 0.761)                | (0.782, 0.764)                     | (0.783, 0.764)                    |
| XSTest                    | (0.817, 0.825)                | (0.814, 0.820)                     | (0.814, 0.820)                    |


## Llama-Guard-4-12B

Now we report `(F1, Recall)` scores for the new model `meta-llama/Llama-Guard-4-12B` with both, transformers and vllm backend. Transformers results are obtained by running `python transformers_mm_eval.py` and vllm results by serving the model with `vllm serve meta-llama/Llama-Guard-4-12B -tp 1 --api-key EMPTY --logprobs-mode processed_logits --max-logprobs 10 --max-model-len 131072` and `python vllm_server_mm_eval.py --model meta-llama/Llama-Guard-4-12B --datasets all --output_dir output_dir/Llama-Guard-4-12B --top_logprobs 10 --logit_bias_strength 0.0`. In the `vllm_server_mm_eval.py` we can't use the logit-bias trick as we did in the section above because `Llama-Guard-4-12B` tends not to produce `safe/unsafe` as the very first token but rather some formatting tokens like `\n\n`. Because of this we generate at most 5 tokens and iterate through them until we find the first instance of `safe/unsafe`, which we then use to calculate probs for F1/Recall.

| Dataset                   | Transformers | vLLM server |
|---------------------------|--------------|-----------------|
| AART                      | (0.874, 0.776) | (0.874, 0.776) |
| AdvBench Behaviors        | (0.964, 0.931) | (0.964, 0.931) |
| AdvBench Strings          | (0.830, 0.709) | (0.830, 0.709) |
| BeaverTails 330k          | (0.732, 0.591) | (0.732, 0.591) |
| Bot-Adversarial Dialogue  | (0.513, 0.377) | (0.513, 0.376) |
| CatQA                     | (0.932, 0.873) | (0.932, 0.873) |
| ConvAbuse                 | (0.237, 0.148) | (0.241, 0.148) |
| DecodingTrust Stereotypes | (0.589, 0.418) | (0.591, 0.419) |
| DICES 350                 | (0.118, 0.063) | (0.118, 0.063) |
| DICES 990                 | (0.219, 0.135) | (0.219, 0.135) |
| Do Anything Now Questions | (0.746, 0.595) | (0.746, 0.595) |
| DoNotAnswer               | (0.549, 0.378) | (0.546, 0.376) |
| DynaHate                  | (0.604, 0.481) | (0.603, 0.481) |
| HarmEval                  | (0.560, 0.389) | (0.560, 0.389) |
| HarmBench Behaviors       | (0.961, 0.925) | (0.959, 0.922) |
| HarmfulQ                  | (0.860, 0.755) | (0.860, 0.755) |
| HarmfulQA Questions       | (0.588, 0.416) | (0.588, 0.416) |
| HarmfulQA                 | (0.375, 0.231) | (0.374, 0.231) |
| HateCheck                 | (0.782, 0.667) | (0.782, 0.667) |
| Hatemoji Check            | (0.626, 0.475) | (0.625, 0.474) |
| HEx-PHI                   | (0.964, 0.930) | (0.966, 0.933) |
| I-CoNa                    | (0.833, 0.713) | (0.837, 0.719) |
| I-Controversial           | (0.596, 0.425) | (0.596, 0.425) |
| I-MaliciousInstructions   | (0.824, 0.700) | (0.824, 0.700) |
| I-Physical-Safety         | (0.493, 0.340) | (0.493, 0.340) |
| JBB Behaviors             | (0.866, 0.870) | (0.860, 0.860) |
| MaliciousInstruct         | (0.953, 0.910) | (0.953, 0.910) |
| MITRE                     | (0.664, 0.497) | (0.663, 0.495) |
| NicheHazardQA             | (0.460, 0.299) | (0.460, 0.299) |
| OpenAI Moderation Dataset | (0.740, 0.789) | (0.739, 0.787) |
| ProsocialDialog           | (0.426, 0.275) | (0.427, 0.276) |
| SafeText                  | (0.375, 0.257) | (0.372, 0.254) |
| SimpleSafetyTests         | (0.985, 0.970) | (0.985, 0.970) |
| StrongREJECT Instructions | (0.910, 0.836) | (0.910, 0.836) |
| TDCRedTeaming             | (0.947, 0.900) | (0.947, 0.900) |
| TechHazardQA              | (0.759, 0.611) | (0.758, 0.610) |
| Toxic Chat                | (0.433, 0.519) | (0.433, 0.519) |
| ToxiGen                   | (0.459, 0.315) | (0.460, 0.315) |
| XSTest                    | (0.834, 0.780) | (0.834, 0.780) |


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
