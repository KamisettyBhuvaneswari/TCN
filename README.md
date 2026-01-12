# TCN-BiLSTM Multimodal Storytelling

A deep learning pipeline for visual storytelling and scene reasoning that combines Temporal Convolutional Networks (TCN) and Bidirectional LSTM with cross-modal attention to generate coherent narratives from image sequences.

## Project Overview

This project implements a multimodal sequence model that learns rich representations from:
- **Visual frames**: Images from the COCO val2017 dataset
- **Text instructions/captions**: Annotations from the LLaVA Instruct 80K dataset
- **Temporal context**: Sequential dependencies across frames

The model predicts target frames and generates descriptive captions by fusing visual and textual information through temporal modeling and cross-modal attention mechanisms.

---

## Model Architecture

The pipeline consists of the following components:

### 1. **Frozen Encoders**
- **Visual Encoder**: Pretrained ResNet50 (frozen) → 2048-dim visual features
- **Text Encoder**: BERT base-uncased (frozen) → 768-dim text embeddings
- **Tag Embeddings**: Learnable embeddings for semantic tags (500 vocab, 128-dim)

### 2. **Temporal Modeling**
- **TCN (Temporal Convolutional Network)**:
  - 4 layers with channels: `[64, 64, 64, 64]`
  - Kernel size: 8
  - Dilation rates: `[1, 2, 4, 8]` (exponentially increasing receptive field)
  - Causal padding for sequential processing
  - Dropout: 0.2

### 3. **Sequence Understanding**
- **BiLSTM (Bidirectional LSTM)**:
  - Hidden size: 256
  - Num layers: 2
  - Dropout: 0.2
  - Captures bidirectional temporal dependencies

### 4. **Cross-Modal Fusion**
- **Multi-head Cross-Attention**:
  - Visual queries attend to text keys/values
  - Num heads: 8
  - Dropout: 0.1

### 5. **Decoders**
- **Image Decoder**: Transposed convolutions to reconstruct 224×224 RGB frames
- **Text Decoder**: LSTM-based caption generator (vocab: 5000, max length: 30)

**Total Parameters**: ~249M  
**Trainable Parameters**: ~116M

---

## Dataset

### Data Sources

1. **LLaVA Instruct 80K**
   - Contains instruction-following conversations with image context
   - Downloaded from: `https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K`
   - File: `llava_instruct_80k.json`
   
2. **COCO val2017**
   - High-quality images from Microsoft Common Objects in Context
   - Downloaded from: `http://images.cocodataset.org/zips/val2017.zip`
   - ~5000 validation images used for training

### Dataset Processing

- **Sequence Length**: 10 frames per sample
- **Image Size**: 224×224 pixels
- **Caption Tokenization**: Custom hash-based tokenizer (max length: 20 tokens)
- **Image Normalization**: ImageNet statistics (mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225])

**Dataset Split**:
- Train: 70% (350 samples)
- Validation: 15% (75 samples)
- Test: 15% (75 samples)

**Data Augmentation**: Minor perturbations (±0.01 noise) applied to create temporal frame sequences.

**Fallback Mechanism**: If downloads fail, synthetic images and token sequences are generated for pipeline validation.

---

## Environment Setup

### Requirements

**Python**: 3.9+  
**CUDA**: 12.4+ (for GPU acceleration)

### Dependencies

Core libraries (from `requirements.txt`):

```
# Deep Learning
torch==2.8.0+cu128
torchvision==0.23.0+cu128
torchaudio==2.8.0
transformers==4.57.3
datasets==4.4.2

# NLP & Evaluation
nltk==3.9.2
rouge-score==0.1.2

# Image Quality Metrics
pytorch-msssim==1.0.0
lpips==0.1.4

# Data Processing
numpy==1.26.4
pandas==2.1.4
pillow==12.0.0
PyYAML==6.0.3

# Visualization
matplotlib==3.8.2
seaborn==0.13.2
scikit-learn==1.3.2
tqdm==4.67.1

# Experiment Tracking (optional)
wandb==0.23.1
```

### Installation

1. **Clone the repository**:
   ```bash
   %cd TCN
   git clone https://github.com/KamisettyBhuvaneswari/TCN.git
   ```

2. **Create virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
   ```

4. **Download NLTK data**:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   ```

---

## Clone and Run in Google Colab

### Quick Start

1. **Open Google Colab**: [colab.research.google.com](https://colab.research.google.com)

2. **Enable GPU**:
   - Runtime → Change runtime type → Hardware accelerator: **GPU** (T4 recommended)

3. **Clone and install**:
   ```python
   # Clone repository
   !git clone https://github.com/KamisettyBhuvaneswari/TCN.git
   
   # Install dependencies
   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
   ```

4. **Run the notebook**:
   ```python
   # Open and run all cells in experiment-3.ipynb
   ```

### Expected Runtime

- **Dataset Download**: ~5-15 minutes (COCO val2017 is ~13GB)
- **Model Setup**: ~2 minutes
- **Training (5 epochs)**: ~15-20 minutes on T4 GPU
- **Evaluation**: ~2 minutes

### Colab-Specific Notes

- Ensure **GPU is enabled** (check with `torch.cuda.is_available()`)
- Session timeout: ~12 hours (may disconnect if idle)
- Storage: Colab provides ~100GB disk space
- If COCO download fails, the notebook will use a smaller subset or synthetic fallback data

---

## Training Configuration

Key hyperparameters (defined in `config` dict):

```yaml
training:
  max_epochs: 30
  learning_rate: 3e-4
  optimizer: adam
  weight_decay: 1e-5
  batch_size: 8
  gradient_clip_norm: 1.0
  
  lr_scheduler:
    type: plateau
    patience: 3
    factor: 0.5
    min_lr: 1e-6
  
  early_stopping_patience: 5
  
  loss_weights:
    image_loss: 0.5  # L1 loss for frame reconstruction
    text_loss: 0.5   # Cross-entropy for caption generation
```

### Loss Functions

1. **Image Loss**: L1 (Mean Absolute Error) between generated and target frames
2. **Text Loss**: Cross-entropy between predicted and target caption tokens
3. **Combined Loss**: Weighted sum of image and text losses

---

## Expected Results

### Performance Metrics

Based on 5-epoch training runs:

| Model Variant | SSIM (Image Quality) | Caption Accuracy |
|--------------|----------------------|------------------|
| **TCN-BiLSTM (Full)** | **0.248** | **0.183** |
| BiLSTM Only | 0.237 | 0.173 |
| TCN Only | 0.248 | 0.185 |
| LSTM Baseline | 0.000 | 0.000 |

**Evaluation Metrics**:
- **SSIM** (Structural Similarity Index): Measures perceptual image quality (0-1 scale, higher is better)
- **Caption Accuracy**: Token-level accuracy between predicted and ground-truth captions

### Training Curves

Typical convergence behavior:
- **Epoch 1**: Train Loss ~3.96, Val Loss ~3.33
- **Epoch 5**: Train Loss ~2.72, Val Loss ~2.92
- **Best Val Loss**: ~2.92 (achieved around epoch 5)

**Learning Rate Schedule**: Adaptive reduction when validation loss plateaus (factor: 0.5, patience: 3 epochs)

### Checkpoints

Best model saved to: `checkpoints/best_model.pth`  
Results saved to: `results/results_summary.json`

---

## Project Structure

```
tcn-bilstm-storytelling/
│
├── experiment-3.ipynb          # Main training notebook
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── checkpoints/                # Saved model weights
│   └── best_model.pth
│
├── results/                    # Training outputs
│   ├── config.yaml
│   ├── results_summary.json
│   ├── training_curves.png
│   └── model_comparison.png
│
└── data/                       # Dataset directory
    ├── llava_data/
    │   └── llava_instruct_80k.json
    └── coco_images/
        └── val2017/            # COCO images
```

---

## Key Features

### 1. **Real Data Integration**
- Seamlessly loads and pairs LLaVA captions with COCO images
- Robust fallback to synthetic data if downloads fail

### 2. **Modular Architecture**
- Each component (TCN, BiLSTM, attention) is independently testable
- Ablation studies demonstrate each module's contribution

### 3. **Multi-Task Learning**
- Jointly optimizes frame reconstruction (visual) and caption generation (textual)
- Cross-modal attention bridges visual-text modalities

### 4. **Temporal Modeling**
- TCN captures long-range dependencies with dilated convolutions
- BiLSTM refines temporal features bidirectionally

### 5. **Reproducibility**
- Fixed random seeds (Python, NumPy, PyTorch)
- Deterministic operations where possible
- Configuration saved with each experiment

---

## Future Work

### Planned Improvements

1. **Advanced Tokenization**
   - Replace custom hash tokenizer with pretrained tokenizers (e.g., GPT-2, T5)
   - Subword tokenization (BPE) for better vocabulary coverage

2. **Enhanced Evaluation Metrics**
   - **BLEU** (Bilingual Evaluation Understudy) for caption quality
   - **LPIPS** (Learned Perceptual Image Patch Similarity) for perceptual image quality
   - **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation) for text overlap

3. **Model Architecture**
   - Experiment with Transformer-based temporal modeling (e.g., Temporal Transformer Encoder)
   - Add residual connections between encoders and decoders
   - Explore cross-attention variants (scaled dot-product, linear attention)

4. **Dataset Expansion**
   - Integrate full VIST (Visual Storytelling) dataset
   - Use complete LLaVA Instruct 150K dataset
   - Augment with Flickr30k or Conceptual Captions

5. **Training Optimization**
   - Mixed precision training (FP16) for faster convergence
   - Distributed training across multiple GPUs
   - Curriculum learning: start with shorter sequences, gradually increase length

6. **Decoder Improvements**
   - Attention-based text decoder (instead of LSTM)
   - Progressive image generation (coarse-to-fine)
   - Multi-scale feature fusion in image decoder

7. **Interpretability**
   - Visualize attention weights to understand cross-modal interactions
   - Generate saliency maps for frame predictions
   - Ablate specific attention heads to study their roles

8. **Application Extensions**
   - Video captioning (extend to longer temporal sequences)
   - Story continuation (predict next N frames given first M frames)
   - Interactive storytelling (user-guided caption generation)

---

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` in config (e.g., from 8 to 4)
   - Enable gradient checkpointing
   - Clear GPU cache: `torch.cuda.empty_cache()`

2. **Dataset Download Failure**
   - Check internet connection
   - Manually download from URLs provided in notebook
   - Use fallback synthetic data for testing

3. **Slow Training**
   - Verify GPU is enabled: `torch.cuda.is_available()` should return `True`
   - Reduce `num_workers` in DataLoader (set to 0 for Colab)
   - Disable `wandb` logging if not needed

4. **Shape Mismatches**
   - Ensure captions are correctly reshaped: `(batch, seq_len, max_tokens)`
   - Use `.reshape()` instead of `.view()` for non-contiguous tensors

---

## Citation

If you use this code or the referenced datasets, please cite:

### COCO Dataset
```bibtex
@inproceedings{lin2014microsoft,
  title={Microsoft COCO: Common Objects in Context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and others},
  booktitle={ECCV},
  year={2014}
}
```

### LLaVA Dataset
```bibtex
@misc{liu2023llava,
  title={Visual Instruction Tuning},
  author={Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae},
  journal={arXiv preprint arXiv:2304.08485},
  year={2023}
}
```

---

## License

This project is released under the **MIT License**.

```
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Contact & Contributions

For questions, issues, or contributions, please open an issue or submit a pull request on the GitHub repository.

**Maintainer**: [Your Name]  
**Email**: [Your Email]  
**GitHub**: [Your GitHub Profile]

---

## Acknowledgments

- **LLaVA Team** for the instruction-following dataset
- **Microsoft COCO** for high-quality annotated images
- **Hugging Face Transformers** for pretrained BERT models
- **PyTorch Team** for the deep learning framework
- **Lightning AI / Google Colab** for GPU compute resources

---

**Happy Storytelling! 📖🎨**
