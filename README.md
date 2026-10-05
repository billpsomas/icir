<div align="center">
<h1>i-CIR: Instance-Level Composed Image Retrieval (NeurIPS 2025)</h1>

**Bill Psomas<sup>1</sup>†, George Retsinas<sup>2</sup>†, Nikos Efthymiadis<sup>1</sup>, Panagiotis Filntisis<sup>2,4</sup>**  
**Yannis Avrithis, Petros Maragos<sup>2,3,4</sup>, Ondrej Chum<sup>1</sup>, Giorgos Tolias<sup>1</sup>**

<sup>1</sup>Visual Recognition Group, FEE, Czech Technical University in Prague <sup>2</sup>Robotics Institute, Athena Research Center  
<sup>3</sup>National Technical University of Athens  <sup>4</sup>HERON - Hellenic Robotics Center of Excellence

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/billpsomas/icir)
[![Project Page](https://img.shields.io/badge/-Project_Page-green.svg?colorA=333&logo=html5)](https://vrg.fel.cvut.cz/icir/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-yellow)](https://huggingface.co/papers/2510.25387)
[![arXiv](https://img.shields.io/badge/arXiv-2510.25387-b31b1b.svg)](https://arxiv.org/abs/2510.25387)
[![OpenReview](https://img.shields.io/badge/OpenReview-Paper-yellow.svg)](https://openreview.net/forum?id=7NEP4jGKwA)

[![Dataset Version](https://img.shields.io/badge/Dataset-v1.0.0-blue.svg)](#)
[![Dataset License](https://img.shields.io/badge/Dataset%20License-CC%20BY--NC--SA%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Code License: MIT](https://img.shields.io/badge/Code%20License-MIT-lightgray.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-lightgray.svg)](#)

</div>

Official implementation of our **B**aseline **A**pproach for **S**urpr**I**singly strong **C**omposition (**BASIC**) and the **i**nstance-level **c**omposed **i**mage **r**etrieval (**i-CIR**) dataset.   

**TL;DR**: We introduce **BASIC**, a training-free VLM-based **method** that centers and projects image embeddings, and **i-CIR**, a well-curated, instance-level composed image retrieval **benchmark** with *rich hard negatives* that is *compact yet really hard*.

# Contents
1. [News](#news)
2. [Overview](#overview)
3. [Download the i-CIR dataset](#download-the-i-cir-dataset)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Methods](#methods)
7. [Key Parameters](#key-parameters)
8. [Corpus Files](#corpus-files)
9. [Output](#output)
10. [Results](#results)
11. [Project Structure](#project-structure)
12. [Citation](#citation)
13. [License](#license)
14. [Acknowledgments](#acknowledgments)
15. [Contact](#contact)

# News
- **20/12/2025**: 🤗 HuggingFace WebDataset is now supported. You can now find i-CIR [[`here`](https://huggingface.co/datasets/billpsomas/icir)].
- **5/12/2025**: i-CIR is presented at NeurIPS 2025! 🎉 Go now through [[`poster`](.github/icir_poster.png)].

# Overview

This repository contains a clean implementation for performing composed image retrieval (CIR) on **i-CIR** dataset using vision-language models (CLIP/SigLIP). 


## Method (BASIC)

Our BASIC method decomposes multimodal queries into object and style components through:

1. **Feature Standardization**: Centering features using LAION-1M statistics
2. **Contrastive PCA Projection**: Separating information using positive and negative text corpora
3. **Query Expansion**: Refining queries with top-k similar database images
4. **Harris Corner Fusion**: Combining image and text similarities with geometric weighting

<p align="center">
<img width="85%" alt="EP illustration" src=".github/method.png">
</p>

## Dataset

### Well-curated

i-CIR is an instance-level composed image retrieval benchmark where each *instance* is a specific, visually indistinguishable object (e.g., Temple of Poseidon). Each query composes an image of the instance with a text modification. For every instance we curate a shared database and define composed positives plus a rich set of **hard negatives**—**visual** (same/similar object, wrong text), **textual** (right text semantics, different instance—often same category), and **composed** (nearly matches both parts but fails one).

<p align="center">
<img width="75%" alt="EP illustration" src=".github/dataset.png">
</p>

### Compact but hard

<img src=".github/hard.png" align="right" width="40%">

Built by combining human curation with automated retrieval from LAION, followed by filtering (quality/duplicates/PII) and manual verification of positives and hard negatives, **i-CIR** is compact yet challenging: it rivals searching with **>40M distractor images** for simple baselines, while keeping per-query databases manageable. **Key stats:**
- **Instances:** 202  
- **Total images:** ~750K  
- **Composed queries:** 1,883  
- **Image queries / instance:** 1–46 
- **Text queries / instance:** 1–5
- **Positives / composed query:** 1–127
- **Hard negatives / instance:** 951–10,045
- **Avg database size / query:** ~3.7K images  

<br clear="right"/>

### Truly compositional

Performance peaks at interior text–image fusion weights ($\lambda$) and shows large **composition gains** over the best uni-modal baselines—evidence that both modalities *must* work together.

<p align="center">
<img width="80%" alt="EP illustration" src=".github/compositional.png">
</p>

# Download the i-CIR dataset

i-CIR is available in two equivalent formats:

## **Option A — Direct tarball (local folder layout)**
i-CIR is stored [here](https://vrg.fel.cvut.cz/icir/icir_v1.0.0.tar.gz).

```bash
# Download 
wget https://vrg.fel.cvut.cz/icir/icir_v1.0.0.tar.gz -O icir_v1.0.0.tar.gz
# Extract
tar -xzf icir_v1.0.0.tar.gz
# Verify
sha256sum -c icir_v1.0.0.sha256   # should print OK
```

**Reulting layout (folder-based):**
```
icir/
├── database/
├── query/
├── database_files.csv
├── query_files.csv
├── VERSION.txt
├── LICENSE
└── checksums.sha256
```

## **Option B — Hugging Face Hub (WebDataset shards)**

You can also download i-CIR directly from the Hugging Face Hub as WebDataset tar shards (recommended for more robust downloading).

**CLI:**
```bash
# Install HF tooling
pip install -U huggingface_hub

# (Optional) login if the repo is gated/private
huggingface-cli login

# Download the dataset snapshot locally
huggingface-cli download billpsomas/icir \
  --repo-type dataset \
  --local-dir ./data/icir \
  --revision main
```

**Python (equivalent):**
```python
from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="billpsomas/icir",
    repo_type="dataset",
    revision="main",
    local_dir="./data/icir",
)
print("Downloaded to:", local_dir)
```

**Resulting layout (WebDataset-based):**

```
icir/
├── webdataset/
│   ├── query/
│   │   ├── query-000000.tar
│   │   ├── query-000001.tar
│   │   └── ...
│   └── database/
│       ├── database-000000.tar
│       ├── database-000001.tar
│       └── ...
├── annotations/
│   ├── query_files.csv
│   ├── database_files.csv
├── VERSION.txt
└── LICENSE
```

You do not need to extract images to a database/ and query/ folder for this option; feature extraction reads directly from the WebDataset shards.

# Installation

## Requirements
- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- (Optional, for Hugging Face / WebDataset mode) `huggingface_hub` + `webdataset`

## Setup

```bash
# Clone the repository
git clone https://github.com/billpsomas/icir.git
cd icir

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

# Quick Start

## 1. Prepare Data

Ensure you have the following structure:

```
icir/
├── data/
│   ├── icir/                       # i-CIR dataset (local folder layout or WebDataset shards)
│   └── laion_mean/                 # Pre-computed LAION means
├── corpora/
│   ├── generic_subjects.csv        # Positive corpus (objects)
│   └── generic_styles.csv          # Negative corpus (styles)
└── synthetic_data/                 # Score normalization data
    ├── dataset_1_sd_clip.pkl.npy
    └── dataset_1_sd_siglip.pkl.npy
```

## 2. Extract Features

Extract features for the i-CIR dataset and text corpora:

```bash
# Extract i-CIR dataset features (local folder layout)
python3 create_features.py --dataset icir --icir_source folder --backbone clip --batch 512 --gpu 0

# Extract i-CIR dataset features (WebDataset shards)
python3 create_features.py --dataset icir --icir_source wds --backbone clip --batch 512 --gpu 0

# Extract corpus features
python3 create_features.py --dataset corpus --backbone clip --batch 512 --gpu 0
```

Features will be saved to `features/{backbone}_features/`.

> **Note on the CLIP backbone.** The paper uses OpenAI's CLIP ViT-L/14 with its original QuickGELU activation. Since
> open_clip 2.29, `open_clip.create_model_from_pretrained("ViT-L/14", "openai")` builds this model with standard GELU
> instead and only prints a warning. GELU features differ from the original ones (cosine about 0.95) and change the
> results. The code passes `force_quick_gelu=True`, which loads the original model on every open_clip version, and
> prints which activation is in use. If you extracted features with an earlier version of this repository and
> open_clip 2.29 or newer, please extract them again.

## 3. Run Retrieval

The easiest way is to use method presets with `--use_preset`:

```bash
# Full BASIC method (recommended)
python3 run_retrieval.py --method basic --use_preset

# Baseline methods
python3 run_retrieval.py --method sum --use_preset
python3 run_retrieval.py --method product --use_preset
python3 run_retrieval.py --method image --use_preset
python3 run_retrieval.py --method text --use_preset
```

For advanced usage with custom parameters:

```bash
python3 run_retrieval.py \
  --method basic \
  --backbone clip \
  --dataset icir \
  --results_dir results/ \
  --specified_corpus generic_subjects \
  --specified_ncorpus generic_styles \
  --num_principal_components_for_projection 250 \
  --aa 0.2 \
  --standardize_features \
  --use_laion_mean \
  --project_features \
  --do_query_expansion \
  --contextualize \
  --normalize_similarities \
  --path_to_synthetic_data ./synthetic_data \
  --harris_lambda 0.1
```

# Methods

The codebase implements several retrieval methods:

- **basic**: Full decomposition method with all components (PCA projection, query expansion, Harris fusion)
- **sum**: Simple sum of image and text similarities
- **product**: Simple product of image and text similarities  
- **image**: Image-only retrieval (ignores text)
- **text**: Text-only retrieval (ignores image)

# Key Parameters

- `--method`: Retrieval method (`basic`, `sum`, `product`, `image`, `text`)
- `--backbone`: Vision-language model (`clip` for ViT-L/14, `siglip` for ViT-L-16-SigLIP-256)
- `--use_preset`: Use predefined method configurations (recommended)
- `--specified_corpus`: Positive corpus for projection (default: `generic_subjects`)
- `--specified_ncorpus`: Negative corpus for projection (default: `generic_styles`)
- `--num_principal_components_for_projection`: PCA components, >1 for exact count or <1 for energy threshold (default: 250)
- `--aa`: Negative corpus weight in contrastive PCA (default: 0.2)
- `--harris_lambda`: Harris fusion parameter (default: 0.1)
- `--contextualize`: Add corpus objects to the text query to contextualize the query
- `--standardize_features`: Center features before projection
- `--use_laion_mean`: Use pre-computed LAION mean for centering
- `--project_features`: Apply PCA projection
- `--do_query_expansion`: Expand queries with retrieved images
- `--normalize_similarities`: Apply score normalization using synthetic data

# Corpus Files

Text corpora define semantic spaces for PCA projection:

- **generic_subjects.csv**: General object/subject descriptions (positive corpus)
- **generic_styles.csv**: General style/attribute descriptions (negative corpus)

Corpora are CSV files with a single column of text descriptions, loaded from the `corpora/` directory.

# Output

Results are saved to the specified results directory (default: `results/`):

```
results/
└── icir/
    └── {method_variant}/
        └── mAP_table.csv          # Mean Average Precision results
```

Each result file includes:
- mAP score for the retrieval method
- Configuration parameters used (for basic method only)
- Timestamp of the experiment

# Results (mAP \%)

| Method            | ImageNet-R |  NICO | Mini-DN |  LTLL |  i-CIR |
|:------------------|-----------:|------:|--------:|------:|------:|
| Text              |      0.74 |  1.09 |    0.57 |  5.72 |  3.01 |
| Image             |      3.84 |  6.32 |    6.66 | 16.49 |  3.04 |
| Text + Image      |      6.21 |  9.30 |    9.33 | 17.86 |  8.20 |
| Text × Image      |      7.83 |  9.79 |    9.86 | 23.16 | 17.48 |
| WeiCom            |     10.47 | 10.54 |    8.52 | 26.60 | 18.03 |
| PicWord           |      7.88 |  9.76 |   12.00 | 21.27 | 19.36 |
| CompoDiff         |     12.88 | 10.32 |   22.95 | 21.61 |  9.63 |
| CIReVL            |     18.11 | 17.80 |   26.20 | 32.60 | 18.66 |
| Searle            |     14.04 | 15.13 |   21.78 | 25.46 | 19.90 |
| MCL               |      8.13 | 19.09 |   18.41 | 16.67 | 19.89 |
| MagicLens         |      9.13 | 19.66 |   20.06 | 24.21 | 27.35 |
| CoVR              |     11.52 | 24.93 |   27.76 | 24.68 | 28.50 |
| FREEDOM           |     29.91 | 26.10 |   37.27 | 33.24 | 17.24 |
| FREEDOM†          |     25.81 | 23.24 |   32.14 | 30.82 | 15.76 |
| **BASIC**  | **32.13** | **31.65** | **39.58** | **41.38** | 31.64 |
| **BASIC†**         |     27.54 | 28.90 |   35.75 | 38.22 | **34.35** |

† Without query expansion.

# Project Structure

```
icir/
├── run_retrieval.py           # Main retrieval script
├── create_features.py         # Feature extraction script
├── utils.py                   # General utilities (device setup, text processing, evaluation)
├── utils_features.py          # Feature I/O and model loading
├── utils_retrieval.py         # Core retrieval algorithms
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── LICENSE                    # MIT License
├── data/                      # Dataset and normalization data
├── corpora/                   # Text corpus files
├── features/                  # Extracted features (generated)
└── results/                   # Retrieval results (generated)
```

# Citation

If you found BASIC and/or i-CIR useful in your research, please consider starring ⭐ us on GitHub and citing 📚 us in your research!

```bibtex
@inproceedings{
    psomas2025instancelevel,
    title={Instance-Level Composed Image Retrieval},
    author={Bill Psomas and George Retsinas and Nikos Efthymiadis and Panagiotis Filntisis and Yannis Avrithis and Petros Maragos and Ondrej Chum and Giorgos Tolias},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year={2025}
}
```

# License

- This code is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
- This dataset is licensed under the CC-BY-NC-SA License - see dataset's LICENSE file for details.

# Acknowledgments

## Code and models

- Vision-language models via [OpenCLIP](https://github.com/mlfoundations/open_clip)
- LAION-1M statistics for feature standardization

## Funding

This work was supported by:

- The [Junior Star grant](https://starfos.tacr.cz/en/vysledky-vyzkumu?query=skbyaadwnkha) GM 21-28830M of the [Czech Science Foundation (GAČR)](https://gacr.cz/en/)
- The Czech National Recovery Plan — [CEDMO 2.0 NPO](https://cedmohub.eu/) (MPO 60273/24/21300/21000), provided by the [Ministry of Industry and Trade](https://www.mpo.gov.cz/en/) of the Czech Republic
- The EU [Horizon Europe](https://research-and-innovation.ec.europa.eu/funding/funding-opportunities/funding-programmes-and-open-calls/horizon-europe_en) programme, [MSCA Postdoctoral Fellowship RAVIOLI](https://cordis.europa.eu/project/id/101205297) (No. 101205297) and [HERON – Hellenic Robotics Center of Excellence](https://cordis.europa.eu/project/id/101136568) (No. 101136568)
- The National Recovery and Resilience Plan [“Greece 2.0”](https://greece20.gov.gr/en/) / [NextGenerationEU](https://next-generation-eu.europa.eu/index_en) project “Applied Research for Autonomous Robotic Systems” (MIS 5200632)
- [Czech Technical University in Prague](https://www.cvut.cz/en) (SGS23/173/OHK3/3T/13 and institutional Future Fund)

## Computational resources

We acknowledge [VSB – Technical University of Ostrava](https://www.vsb.cz/en) and [IT4Innovations National Supercomputing Center](https://www.it4i.cz/en), Czech Republic, for awarding this project (OPEN-33-67) access to the [LUMI supercomputer](https://www.lumi-supercomputer.eu/), owned by the [EuroHPC Joint Undertaking](https://eurohpc-ju.europa.eu/), hosted by [CSC](https://csc.fi/en/) (Finland) and the LUMI consortium, through the [Ministry of Education, Youth and Sports of the Czech Republic](https://msmt.gov.cz/) via the [e-INFRA CZ](https://www.e-infra.cz/en) project (ID: 90254).

We also acknowledge the OP VVV project [“Research Center for Informatics”](https://rci.cvut.cz/) (CZ.02.1.01/0.0/0.0/16_019/0000765).

# Contact

For questions or issues, please open an issue on GitHub or contact Bill $\rightarrow$ vasileios.psomas@fel.cvut.cz.
