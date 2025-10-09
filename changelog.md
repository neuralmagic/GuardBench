# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Fixed issue when reading CSV files in `hex_phi.py`.

### Added
- Added support for [JBB Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors).
- Added support for [NicheHazardQA](https://huggingface.co/datasets/SoftMINER-Group/NicheHazardQA).
- Added support for [HarmEval](https://huggingface.co/datasets/SoftMINER-Group/HarmEval).
- Added support for [TechHazardQA](https://huggingface.co/datasets/SoftMINER-Group/TechHazardQA).

### Changed
- `evaluate` now also compute Matthews Correlation Coefficient (MCC) score.
- `benchmark` now allows for choosing the metrics to show at the end of the evaluation. Supported metrics are: `precision` (Precision), `recall` (Recall), `f1` (F1), `mcc` (Matthews Correlation Coefficient), `auprc` (AUPRC), `sensitivity` (Sensitivity), `specificity` (Specificity), `g_mean` (G-Mean), `fpr` (False Positive Rate), `fnr` (False Negative Rate).
- `benchmark`'s `batch_size` parameter now defaults to 1.