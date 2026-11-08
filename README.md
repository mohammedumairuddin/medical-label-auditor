# Medical Label Auditor

An agentic AI system for automated label auditing and mislabel detection in medical imaging datasets. Built as an MSc Artificial Intelligence dissertation project at the University of Surrey.

## Overview

Medical imaging datasets often contain incorrectly labelled images, which degrades the performance of clinical AI models. This project audits datasets for suspicious labels using frozen visual embeddings and statistical outlier detection, recommends how to handle flagged samples, and generates human-readable audit reports.

The system runs as a sequential pipeline of four agents:

1. **Data Inspector Agent**: image quality checks (sharpness, brightness, contrast)
2. **Label Auditor Agent**: extracts 512-dim embeddings from a frozen ResNet18 and scores each image with kNN label consistency, Mahalanobis distance (LedoitWolf covariance) and Isolation Forest, combined by rank fusion
3. **Dataset Label Optimizer Agent**: recommends removal, loss downweighting or keeping each image, and reports class balance impact
4. **LLM Label Explainer Agent**: uses GPT-4o-mini to write evidence-grounded explanations for the most suspicious images, compiled into an HTML report

## Datasets

Not included in this repository because of their size.

| Dataset | Domain | Subset used | Classes |
|---|---|---|---|
| [CheXpert (small)](https://www.kaggle.com/datasets/ashery/chexpert) | Chest X-ray | 8,776 images | 14 |
| [ISIC 2019](https://www.kaggle.com/datasets/nischaydnk/isic-2019-jpg-224x224-resized) | Dermoscopy | 9,158 images | 8 |

## Repository structure

```
notebooks/   Jupyter notebooks, run in numerical order
reports/     LLM explanations (CSV) and the HTML audit report
requirements.txt
```

## How to run

| Notebook | Purpose | Where |
|---|---|---|
| 00 | Environment check | Local |
| 01-02 | Dataset exploration | Local |
| 03-05 | Build manifests and extract the image subset | Local |
| 06 | Preprocessing and embedding extraction | Kaggle (GPU) |
| 07 | Label Auditor Agent | Local |
| 08 | Dataset Label Optimizer Agent | Local |
| 09 | LLM Label Explainer and HTML report | Local |
| 10A | Synthetic noise injection and detection evaluation | Local |
| 10B | Downstream classifier comparison (incl. simplified DivideMix) | Local |

Setup:

```bash
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```

Notebook 09 needs an OpenAI API key set as the environment variable `OPENAI_API_KEY`. Download the datasets, place them under `data/raw/`, and run notebook 06 on Kaggle to produce the embedding cache.

## Results

**Auditing (17,934 images):** 1,423 images (7.9%) recommended for removal, 4,522 (25.2%) for downweighting, 11,989 (66.9%) kept.

**Synthetic noise detection (20% corruption, 3,302 images):**

| Metric | Symmetric | Asymmetric |
|---|---|---|
| AUROC | 0.8559 | 0.8693 |
| Precision@50 | 0.96 | 0.62 |

**Downstream classification (macro F1, MLP on frozen embeddings):**

| Method | CheXpert | ISIC |
|---|---|---|
| Noisy baseline | 0.5244 | 0.6945 |
| Cleaned (hard removal) | 0.5244 | 0.6693 |
| Downweighted | 0.5515 | 0.6844 |
| DivideMix (simplified) | 0.4948 | 0.6732 |

Downweighting gave the best CheXpert result (+2.71 pp). ISIC's visually distinct classes were less sensitive to cleaning.

## Limitations

- No expert clinical validation of the flagged images
- Detection is evaluated on synthetic noise, which may not match real annotation errors
- A frozen ImageNet backbone lacks medical-domain features, which mainly limits CheXpert
- DivideMix is a simplified version without co-training or pseudo-labelling
- Synthetic noise detection uses classifier-confidence disagreement, while the deployed auditor uses embedding geometry

## Future work

Medical pretrained backbones (CheXNet, dermoscopy-pretrained EfficientNet), full DivideMix, per-label multilabel auditing for CheXpert, and expert validation of flagged images.

## Author

Mohammed Umairuddin, MSc Artificial Intelligence, University of Surrey. Supervisor: Gustavo Carneiro.
