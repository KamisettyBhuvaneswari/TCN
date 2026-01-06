TCN-BiLSTM Multimodal Storytelling (Notebook)
This project implements a multimodal sequence model for visual storytelling / scene reasoning using image sequences and text instructions/captions. The notebook (experiment.ipynb) builds a pipeline that combines a frozen ResNet50 visual encoder, a frozen BERT text encoder, a Temporal Convolutional Network (TCN) for temporal modeling, cross-modal multi-head attention to fuse text and visual signals, and a BiLSTM for sequence understanding, with decoders for frame reconstruction and caption generation.

Contents
Project Overview

Architecture

Dataset

Setup

How to Run

Outputs

Evaluation

Notes / Troubleshooting

Citation

Project Overview
The goal is to improve visual storytelling by learning representations from:

Frames (images from COCO)

Text (instructions/captions from LLaVA Instruct dataset)

Temporal context across a sequence of frames

The notebook is designed to run inside a GPU environment (Lightning AI / Colab-like) and includes:

Dependency installation

Configuration setup (YAML-style config dict saved to results folder)

Real-data loading with fallbacks (synthetic data if downloads fail)

Model component tests (shape checks) to verify correctness

Architecture
High-level components used in the notebook:

Visual Encoder: ResNet50 (pretrained, typically frozen) → visual feature vectors

Text Encoder: BERT base uncased (pretrained, typically frozen) → text feature vectors

Tag Embeddings: simple embedding lookup for tag tokens (if used)

Temporal Model: TCN blocks with causal padding + dilation (e.g., 1,2,4,8)

Fusion: Multi-head cross-modal attention (visual queries, text keys/values)

Sequence Model: BiLSTM over fused temporal features

Decoders:

Image decoder (upsampling / ConvTranspose2d) for predicting target frame

Text decoder (LSTM) for generating target caption tokens

Dataset
The notebook builds a dataset class that pairs:

LLaVA instruction JSON (captions/instructions)

COCO val2017 images (real images)

Downloaded resources (as used in the notebook)
LLaVA JSON: llava_instruct_80k.json (downloaded from Hugging Face in the notebook)

COCO val2017: val2017.zip (downloaded from the official COCO image server in the notebook)

Fallback behavior
If COCO download fails or image loading fails, the notebook falls back to:

a smaller COCO subset (test2015) OR

synthetic images and synthetic token sequences

Setup
Recommended environment
Python 3.9+

GPU runtime recommended (CUDA)

Install dependencies
The notebook installs (examples):

torch, torchvision, torchaudio

transformers, datasets

pytorch-msssim, lpips

nltk, rouge-score

numpy, pandas, matplotlib, seaborn, scikit-learn, tqdm, pillow

wandb (optional)

If running outside the notebook, create a virtual env and install similar packages.

How to Run
Open experiment.ipynb.

Run cells in order:

Setup + installs

Config creation

Dataset download + dataset build

Model component definitions and tests

Training / evaluation cells (if included in your later sections)

Typical paths (Lightning AI)
Your workspace paths may look like:

/teamspace/studios/this_studio/storytelling_project/

dataset folders created inside the project directory

If you renamed the folder, update the paths in the config and dataset cell.

Outputs
The notebook writes artifacts into configured folders (created automatically), e.g.:

checkpoints/ – saved model checkpoints (if training enabled)

results/ – config YAML, plots, metrics outputs

data/ – cached or prepared dataset files (optional)

Evaluation
The config indicates evaluation metrics such as:

SSIM (image similarity)

LPIPS (perceptual similarity)

BLEU (caption quality)

ROUGE (caption overlap)

Perplexity (language model fluency proxy)

Exact evaluation depends on which training/evaluation cells you run and whether metrics computation is enabled.

Notes / Troubleshooting
“remote origin already exists”: your Git remote is already added; use git remote -v then git push -u origin master (or main) depending on your branch.

COCO download is large: val2017.zip is big; if it times out, use the notebook’s smaller subset path or synthetic fallback.

Tokenizer: notebook uses a simple hashing tokenizer in places; results are mainly for pipeline validation unless you replace with a proper tokenizer/vocab.

Reproducibility: seeds are set (Python/NumPy/PyTorch), but GPU determinism can still vary.

Citation
If you use the dataset references in reports/presentations, cite:

COCO (Microsoft Common Objects in Context)

LLaVA Instruct dataset (as referenced by the notebook download)

Any course-provided dataset reference required in your assessment brief

