# Diffusion Data Augmentation: Prompt-based Image Generation for Imbalanced Learning

## Motivation and Problem Statement
Supervised visual recognition models are notoriously vulnerable to severe class imbalance, where underrepresented minority categories suffer from biased decision boundaries and degraded recall. Standard geometric and photometric data augmentation techniques apply rigid pixel-level perturbations that fail to inject novel semantic contexts or diverse intra-class variance.

This project investigates an end-to-end multi-stage Prompt-based Generative Data Augmentation framework designed to mitigate severe class imbalance in image classification. Rather than manually writing prompts or copying existing pixels, this pipeline leverages generative AI to synthesize entirely new training samples that are realistic, highly diverse, and semantically consistent with the minority class.

## Architecture and Methodology
The automated framework consists of three main stages to augment the dataset:

1. **Image Captioning:** Utilizes the Vision-Language Model **BLIP** to extract descriptive textual captions from the original minority class images.
2. **Textual Augmentation:** Employs the Large Language Model **Qwen-2.5-7B-Instruct** to systematically mutate peripheral attributes (e.g., background, lighting) of the captions while strictly preserving the core semantic subject.
3. **Image Synthesis & Editing:** Uses advanced conditional diffusion models to generate new samples. This project evaluates and compares two state-of-the-art architectures: **FlowAlign** and **FLUX.2-klein-4B**.

The methodology is evaluated on a binary classification task derived from the Oxford-IIIT Pet benchmark under a controlled 10:1 imbalance ratio. Downstream generalization is quantified by training ResNet-18 classifiers from scratch under a 5-Fold Stratified Cross-Validation protocol to prevent out-of-fold data leakage.

## Repository Structure
The repository is organized into the following main files and notebooks, designed to isolate different stages of the pipeline:

* `dataset_splitter.ipynb`: Notebook dedicated to data preparation, creating the artificial imbalance split, and configuring the cross-validation folds.
* `pipeline_flowalign.ipynb`: Implementation of the generative augmentation pipeline based on the FlowAlign architecture.
* `pipeline_flux.ipynb`: Implementation of the generative augmentation pipeline based on the FLUX.2 Klein architecture.
* `utils.ipynb`: Collection of shared support functions and utilities for training and evaluation.
* `2026_SEAI_project_Bacherotti_Ceccarelli_Meazzini.pdf`: Comprehensive academic documentation containing detailed analysis and theoretical formulations.

## Experimental Results
Empirical results demonstrate that generative augmentation effectively regularizes minority representation learning. Compared to a baseline trained exclusively with conventional data augmentation, our pipeline (especially when using FLUX) yields consistent improvements in:
* Balanced Accuracy and Macro F1-Score.
* Decision boundary stability (measured via ROC-AUC and Average Precision).
* Statistically significant increases in minority-class Recall (verified via exact McNemar's test and binomial tests).

## Authors
* Nicolò Bacherotti
* Leonardo Ceccarelli
* Michele Meazzini
