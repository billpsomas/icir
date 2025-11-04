# ICIR: Instance-Level Composed Image Retrieval

Official implementation of our composed image retrieval method using PCA-based feature decomposition for the ILCIR dataset.

## Overview

This repository contains a clean implementation for performing composed image retrieval (CIR) on the ILCIR dataset using vision-language models (CLIP/SigLIP). Our **basic** method decomposes multimodal queries into object and style components through:

1. **Feature Standardization**: Centering features using LAION-1M statistics
2. **Contrastive PCA Projection**: Separating information using positive and negative text corpora
3. **Query Expansion**: Refining queries with top-k similar database images
4. **Harris Corner Fusion**: Combining image and text similarities with geometric weighting

## Installation

### Requirements
- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

### Setup

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

## Quick Start

### 1. Prepare Data

Ensure you have the following structure:

```
icir/
├── data/
│   ├── ilcir202_censored/          # ILCIR dataset
│   └── laion_mean/                 # Pre-computed LAION means
├── corpora/
│   ├── generic_subjects.csv        # Positive corpus (objects)
│   └── generic_styles.csv          # Negative corpus (styles)
└── synthetic_data/                 # Min-max normalization data
    ├── dataset_1_sd_clip.pkl.npy
    └── dataset_1_sd_siglip.pkl.npy
```

### 2. Extract Features

Extract features for the ILCIR dataset and text corpora:

```bash
# Extract ILCIR dataset features
python3 create_features.py --dataset ilcir --backbone clip --batch 512 --gpu 0

# Extract corpus features
python3 create_features.py --dataset corpus --backbone clip --batch 512 --gpu 0
```

Features will be saved to `features/{backbone}_features/`.

### 3. Run Retrieval

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
  --dataset ilcir \
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

## Methods

The codebase implements several retrieval methods:

- **basic**: Full decomposition method with all components (PCA projection, query expansion, Harris fusion)
- **sum**: Simple sum of image and text similarities
- **product**: Simple product of image and text similarities  
- **image**: Image-only retrieval (ignores text)
- **text**: Text-only retrieval (ignores image)

## Key Parameters

- `--method`: Retrieval method (`basic`, `sum`, `product`, `image`, `text`)
- `--backbone`: Vision-language model (`clip` for ViT-L/14, `siglip` for ViT-L-16-SigLIP-256)
- `--use_preset`: Use predefined method configurations (recommended)
- `--specified_corpus`: Positive corpus for projection (default: `generic_subjects`)
- `--specified_ncorpus`: Negative corpus for projection (default: `generic_styles`)
- `--num_principal_components_for_projection`: PCA components, >1 for exact count or <1 for energy threshold (default: 250)
- `--aa`: Negative corpus weight in contrastive PCA (default: 0.2)
- `--harris_lambda`: Harris fusion parameter (default: 0.1)
- `--contextualize`: Add "a photo of a" prefix to text queries
- `--standardize_features`: Center features before projection
- `--use_laion_mean`: Use pre-computed LAION mean for centering
- `--project_features`: Apply PCA projection
- `--do_query_expansion`: Expand queries with retrieved images
- `--normalize_similarities`: Apply min-max normalization using synthetic data

## Corpus Files

Text corpora define semantic spaces for PCA projection:

- **generic_subjects.csv**: General object/subject descriptions (positive corpus)
- **generic_styles.csv**: General style/attribute descriptions (negative corpus)

Corpora are CSV files with a single column of text descriptions, loaded from the `corpora/` directory.

## Output

Results are saved to the specified results directory (default: `results/`):

```
results/
└── ilcir/
    └── {method_variant}/
        └── mAP_table.csv          # Mean Average Precision results
```

Each result file includes:
- mAP score for the retrieval method
- Configuration parameters used (for basic method only)
- Timestamp of the experiment

## Project Structure

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

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{
    psomas2025instancelevel,
    title={Instance-Level Composed Image Retrieval},
    author={Bill Psomas and George Retsinas and Nikos Efthymiadis and Panagiotis Filntisis and Yannis Avrithis and Petros Maragos and Ondrej Chum and Giorgos Tolias},
    booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Vision-language models via [OpenCLIP](https://github.com/mlfoundations/open_clip)
- LAION-1M statistics for feature standardization

## Contact

For questions or issues, please open an issue on GitHub.
