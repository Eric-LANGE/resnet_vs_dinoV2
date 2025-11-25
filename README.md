# Product image classification with DINOv2 and ResNet

A comparative study of Vision Transformer (DINOv2) and CNN (ResNet34) architectures for multi-class product image classification, with explainability analysis using Grad-CAM.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

## Project overview

This project classifies product images into 7 e-commerce categories using transfer learning. 
It compares two different deep learning approaches to evaluate the benefits of modern Vision Transformer architectures over traditional CNNs.

### Performance summary

| Model | Architecture | Accuracy | Misclassified Samples | Training Time (T4) |
|-------|-------------|----------|----------------------|--------------------|
| **DINOv2** | Vision Transformer (ViT-B/14 with registers) | **90%** | 20/210 (9.5%) | ~10 min |
| **ResNet34** | Convolutional Neural Network | 80% | 41/210 (19.5%) | ~15 min |


### Product categories

- Baby Care
- Beauty and Personal Care
- Computers
- Home Decor & Festive Needs
- Home Furnishing
- Kitchen & Dining
- Watches

---

## Dataset

- **Total images:** 1,050 (7 classes × 150 images each)
- **Train set:** 840 images (80%)
- **Validation set:** 210 images (20%)
- **Split strategy:** stratified by class (ensures balanced class distribution)
- **Image format:** JPG with variable resolution
- **Preprocessing:** resized to 224×224 pixels
- **Data balance:** perfectly balanced (30 samples per class in validation set)

---

## Architecture

### DINOv2 model

The project uses `dinov2_vitb14_reg` (ViT-Base with 14×14 patches and register tokens):

```
Input (224×224×3)
    ↓
DINOv2 Backbone (frozen)
    ↓
Patch Embedding (14×14 patch size)
    ↓
[CLS Token] + [4 Register Tokens] + [256 Patch Tokens]
    ↓
12 Transformer Blocks (768-dim embeddings)
    ↓
CLS Token Output (768-dim)
    ↓
Classification Head:
    - Dropout (0.5)
    - Linear (768 → 7)
```

**Key features:**
- **Pre-training:** self-supervised learning on 142M images (LVD-142M dataset)
- **Register tokens:** 4 learnable tokens that absorb attention artifacts, improving representation quality
- **Training strategy:** only the classification head is trained (backbone frozen)
- **Parameters:** ~86M (backbone) + ~5K (head) = ~86M total, but only ~5K trainable
- **Embedding dimension:** 768


### ResNet34 model

The ResNet34 model uses a **progressive unfreezing** strategy with two training phases:

```
Input (224×224×3)
    ↓
ResNet34 Backbone (ImageNet pre-trained)
    ↓
4 Residual Layer Groups:
    - layer1: 3 BasicBlocks (64 channels)
    - layer2: 4 BasicBlocks (128 channels)
    - layer3: 6 BasicBlocks (256 channels)
    - layer4: 3 BasicBlocks (512 channels)
    ↓
Global Average Pooling
    ↓
Classification Head:
    - Dropout (0.5)
    - Linear (512 → 7)
```

**Training strategy:**
- **Phase 1 (3 epochs):** trains classification head + BatchNorm layers (backbone frozen)
  - Learning rate: 1e-4
  - Trainable: FC head + all BatchNorm layers
  
- **Phase 2 (20 epochs):** unfreezes `layer3`, `layer4`, classification head, and all BatchNorm layers
  - Learning Rate: 1e-5 (10× reduction)
  - Trainable: layer3 + layer4 + FC head + all BatchNorm layers

**Total training:** 23 epochs (3 + 20)

**Why progressive unfreezing?**

ResNet's ImageNet pre-training is specialized for natural objects (1000 classes).

The progressive strategy:

1. first adapts the head to our 7 product classes
2. then fine-tunes deeper layers to learn product-specific features
3. keeps early layers frozen (they capture generic low-level features)

---

## Results

### DINOv2 performance (90% accuracy)

```
                            precision    recall  f1-score   support

                 Baby Care       0.96      0.73      0.83        30
  Beauty and Personal Care       0.96      0.77      0.85        30
                 Computers       0.94      0.97      0.95        30
Home Decor & Festive Needs       0.85      0.93      0.89        30
           Home Furnishing       0.78      0.97      0.87        30
          Kitchen & Dining       0.94      0.97      0.95        30
                   Watches       0.97      1.00      0.98        30

                  accuracy                           0.90       210
                 macro avg       0.91      0.90      0.90       210
              weighted avg       0.91      0.90      0.90       210
```

**Strengths:**
- **Watches:** near-perfect performance (100% recall, 97% precision)
- **Computers & Kitchen:** excellent performance (95%+ F1-score)
- **High Precision:** 96% for Baby Care and Beauty categories (low false positive rate)

**Challenges:**
- **Baby Care:** lower recall (73%) - some samples misclassified as Home Furnishing
- **Home Furnishing:** lower precision (78%) - receives false positives from other categories

### ResNet34 performance (80% accuracy)

```
                            precision    recall  f1-score   support

                 Baby Care       0.68      0.70      0.69        30
  Beauty and Personal Care       0.85      0.73      0.79        30
                 Computers       0.83      0.80      0.81        30
Home Decor & Festive Needs       0.75      0.80      0.77        30
           Home Furnishing       0.76      0.73      0.75        30
          Kitchen & Dining       0.82      0.90      0.86        30
                   Watches       0.97      0.97      0.97        30

                  accuracy                           0.80       210
                 macro avg       0.81      0.80      0.80       210
              weighted avg       0.81      0.80      0.80       210
```

**Strengths:**
- **Watches:** consistent performance (97% across the board)
- **Kitchen & Dining:** strong recall (90%)

**Challenges:**
- **Baby Care:** significant struggles (68% precision, 70% recall)
- **Home Furnishing:** confusion with Baby Care category

### Key observations

1. **DINOv2 outperforms ResNet34 by 10%** in overall accuracy
2. **Watches** is the easiest category for both models (97-100% recall)
3. **Baby care** shows the largest performance gap between models:
   - DINOv2: 96% precision (but 73% recall)
   - ResNet34: 68% precision, 70% recall
4. **Common confusion patterns** for both models:
   - Baby Care ↔ Home Furnishing (soft textures, pastel colors)
   - Beauty and Personal Care ↔ Computers (packaging similarity)
   - Home Decor & Festive Needs ↔ Kitchen & Dining (functional overlap)

---

## Error analysis

### DINOv2 - most confused classes

| True Class | Predicted As | Count | % of Class Errors | Visual Similarity |
|------------|--------------|-------|-------------------|-------------------|
| Baby Care | Home Furnishing | 5 | 30% | Soft textures, pastel colors, rounded shapes |
| Baby Care | Home Decor & Festive Needs | 2 | 10% | Similar aesthetic, decorative items |
| Beauty and Personal Care | Computers | 2 | 10% | Sleek packaging, compact form factor |
| Beauty and Personal Care | Home Furnishing | 2 | 10% | Product shape and material similarity |
| Beauty and Personal Care | Kitchen & Dining | 1 | 5% | Container shape confusion |
| Beauty and Personal Care | Watches | 1 | 5% | Luxury product aesthetic |
| Home Decor & Festive Needs | Home Furnishing | 1 | 5% | Decorative household items |
| Home Decor & Festive Needs | Kitchen & Dining | 2 | 10% | Functional household products |
| Kitchen & Dining | Home Decor & Festive Needs | 1 | 5% | Decorative tableware |
| Computers | Home Decor & Festive Needs | 1 | 5% | Tech accessories vs decor |

**Total errors:** 20/210 (9.5% error rate, 90.5% accuracy)

**Pattern analysis:**
- **Baby care confusion dominates:** 8/20 errors (40%) involve Baby Care
- **Most problematic:** Baby Care → Home Furnishing (6 errors)
- **Reason:** both categories feature soft materials, light colors, and rounded designs

### ResNet34 - most confused classes

| True Class | Predicted As | Count | % of Class Errors | Visual Similarity |
|------------|--------------|-------|-------------------|-------------------|
| Home Furnishing | Baby Care | 5 | 12% | Bidirectional texture/color confusion |
| Baby Care | Home Furnishing | 4 | 10% | Bidirectional texture/color confusion |
| Beauty and Personal Care | Computers | 3 | 7% | Packaging shape and finish |
| Home Decor & Festive Needs | Baby Care | 3 | 7% | Color palette overlap |
| Home Decor & Festive Needs | Kitchen & Dining | 2 | 5% | Functional household items |
| Baby Care | Kitchen & Dining | 2 | 5% | Product overlap (bottles, containers) |
| Baby Care | Home Decor & Festive Needs | 1 | 2% | Decorative items |
| Baby Care | Beauty and Personal Care | 1 | 2% | Personal care products |
| Beauty and Personal Care | Home Decor & Festive Needs | 1 | 2% | Aesthetic similarity |
| Beauty and Personal Care | Home Furnishing | 1 | 2% | Material/finish similarity |
| Beauty and Personal Care | Kitchen & Dining | 1 | 2% | Container shape |
| Computers | Beauty and Personal Care | 1 | 2% | Sleek design |
| Computers | Kitchen & Dining | 1 | 2% | Tech/appliance confusion |
| Computers | Home Decor & Festive Needs | 2 | 5% | Tech accessories |
| Computers | Watches | 1 | 2% | Tech wearables |
| Kitchen & Dining | Baby Care | 1 | 2% | Container shapes |
| Kitchen & Dining | Computers | 1 | 2% | Appliance confusion |
| Kitchen & Dining | Home Decor & Festive Needs | 1 | 2% | Decorative tableware |
| Watches | Home Furnishing | 1 | 2% | Luxury item aesthetic |
| Home Furnishing | Computers | 1 | 2% | Design similarity |
| Home Furnishing | Home Decor & Festive Needs | 1 | 2% | Household items |

**Total errors:** 41/210 (19.5% error rate, 80.5% accuracy)

**Pattern analysis:**
- **Baby care ↔ Home furnishing dominate:** 9 bidirectional errors (22% of all errors)
- **More scattered confusion:** ResNet shows errors across more category pairs
- **Texture dependence:** CNN appears to rely heavily on surface textures, causing confusion between categories with similar materials

### Comparative error analysis

| Confusion Pattern | DINOv2 Errors | ResNet34 Errors | Δ (Absolute) | Δ (Relative) |
|-------------------|---------------|-----------------|--------------|--------------|
| Baby Care → Home Furnishing | 6 | 4 | +2 | +50% |
| Home Furnishing → Baby Care | 1 | 5 | -5 | N/A |
| **Total BC ↔ HF** | **6** | **9** | **-3** | **-33%** |
| Beauty → Computers | 2 | 3 | -1 | -33% |
| Home Decor ↔ Kitchen | 2 | 2 | 0 | 0% |
| Home Decor → Baby Care | 2 | 3 | -1 | -33% |
| **Overall Error Rate** | **9.5%** | **19.5%** | **-10%** | **-51%** |

**Key insights:**

1. **DINOv2's semantic understanding:** the Transformer's self-attention mechanism captures global context and semantic relationships, this results in:

   - Fewer bidirectional confusions (6 vs 9 for BC ↔ HF)
   - More consistent predictions (errors are more unidirectional)
   - Better handling of categories with overlapping visual features

2. **ResNet's texture bias:** the CNN shows stronger reliance on low-level features (textures, colors), leading to:

   - More bidirectional confusion (both A→B and B→A errors)
   - Vulnerability to surface-level similarities
   - 9 total errors between Baby Care and Home Furnishing

3. **Self-Supervised pre-training advantage:** DINOv2's training on 142M diverse, unlabeled images (vs ResNet's 1M labeled ImageNet images) provides:
   - More robust visual representations
   - Better generalization to product images
   - 54% reduction in total errors (20 vs 41)

4. **Persistent challenges:**, both models struggle with:

   - Baby Care ↔ Home Furnishing (inherent visual similarity)
   - Beauty products (varied packaging styles)
   - Home categories (functional overlap)

These errors often reflect **legitimate ambiguity** in product categorization, where even human annotators might hesitate.

---

## Explainability with Grad-CAM

Both models include Grad-CAM (Gradient-weighted Class Activation Mapping) visualizations to understand model decisions and identify which image regions influence predictions.

### What is Grad-CAM?

Grad-CAM produces heatmaps that highlight the image regions most important for a model's classification decision. It works by:

1. Computing gradients of the target class with respect to feature maps
2. Weighting feature maps by these gradients
3. Creating a spatial heatmap showing "attention" regions
4. Overlaying the heatmap on the original image

### DINOv2 Grad-CAM setup

```
# Target the last transformer block's LayerNorm
target_layers = [model.backbone.blocks[-1].norm1]

# Custom reshape transform for ViT with register tokens
def create_dino_reshape_transform(model):
    """
    Adapts Grad-CAM for Vision Transformers with register tokens.
    
    DINOv2 sequence structure:
    [CLS (1)] + [Registers (4)] + [Patches (256)] = 261 tokens
    
    For Grad-CAM, we need to:
    1. Skip CLS token (index 0)
    2. Skip register tokens (indices 1-4)
    3. Reshape remaining 256 patches into 2D spatial grid
    """
    start_index = 1 + model.num_register_tokens  # = 1 + 4 = 5
    
    def reshape_transform(tensor):
        # tensor shape: (batch, 261, 768)
        result = tensor[:, start_index:, :]  # (batch, 256, 768)
        
        # Reshape patches into 2D grid
        # Calculation: sqrt(256 patches) = 16
        # Grid is 16x16 because: Image (224px) / Patch Size (14px) = 16
        height = width = int(result.shape[1]**0.5) 
        
        result = result.reshape(result.shape[0], height, width, result.shape[2])
        # Change to (batch, channels, height, width) for Grad-CAM
        return result.permute(0, 3, 1, 2)
    
    return reshape_transform

# Initialize Grad-CAM with custom transform
dynamic_reshape_transform = create_dino_reshape_transform(model)
cam = GradCAM(model=model, 
              target_layers=target_layers,
              reshape_transform=dynamic_reshape_transform)
```

### ResNet Grad-CAM setup

```python
# Target the last convolutional block
target_layers = [model.layer4[-1]]

# No reshape transform needed for CNNs
cam = GradCAM(model=model, target_layers=target_layers)
```

**Simpler setup:**
- ResNet naturally outputs 2D feature maps (7×7 spatial resolution)
- No token reshaping required
- Direct gradient computation from convolutional layers

### Grad-CAM enhancement: eigen-smooth

Both implementations use `eigen_smooth=True` for higher-quality heatmaps:

```python
grayscale_cam = cam(input_tensor=image, 
                    targets=targets, 
                    eigen_smooth=True)
```

**Benefits of eigen-smooth:**

- **Noise reduction:** applies PCA-based smoothing to feature maps
- **Preserves structure:** retains top eigenvectors that capture the most variance
- **Cleaner visualizations:** produces smoother, more interpretable heatmaps
- **Semantic preservation:** doesn't lose important spatial information

**Technical details:**
1. Computes principal components of the feature maps
2. Projects features onto top-k eigenvectors
3. Reconstructs feature maps with reduced noise
4. Results in cleaner activation maps

### Interpretation guidelines

**DINOv2 attention patterns:**
- Focuses on **object shape and global context**
- Captures **semantic relationships** (e.g., watch on wrist, product in setting)
- Highlights **entire object boundaries**
- Less distracted by background textures

**ResNet attention patterns:**
- Focuses on **local textures and patterns**
- Highlights **discriminative features** (e.g., metallic surfaces, logos)
- More sensitive to **color gradients**
- Can be distracted by background elements

---

### Performance benchmarks

| Hardware | DINOv2 Training | ResNet34 Training | Grad-CAM (per image) |
|----------|-----------------|-------------------|----------------------|
| Tesla T4 | ~10 min (25 epochs) | ~15 min (23 epochs) | ~0.5 sec |
| A100 (estimated) | ~4 min | ~6 min | ~0.2 sec |
| CPU-only (not recommended) | ~4 hours | ~2 hours | ~5 sec |


---

## Data augmentation

### DINOv2 training transforms

```python
A.Compose([
    # Resize with aspect ratio preservation
    A.SmallestMaxSize(max_size=256, interpolation=cv2.INTER_AREA),
    
    # Random crop to target size
    A.RandomCrop(height=224, width=224),
    
    # Horizontal flip for view invariance
    A.HorizontalFlip(p=0.5),
    
    # Color augmentation (critical for product images)
    A.ColorJitter(
        brightness=0.2,  # ±20% brightness variation
        contrast=0.2,    # ±20% contrast variation
        saturation=0.2,  # ±20% saturation variation
        hue=0.1,         # ±10% hue shift
        p=0.8            # Apply to 80% of images
    ),
    
    # Normalize to ImageNet statistics
    A.Normalize(mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)),
    
    # Convert to PyTorch tensor
    ToTensorV2()
])
```

**Rationale:**
- **Minimal augmentation:** DINOv2's self-supervised pre-training already learned invariance to transformations
- **Color jitter:** most important for product images with varied lighting conditions
- **No geometric distortions:** avoids disrupting product shapes

### ResNet34 training transforms

```python
A.Compose([
    # Resize with aspect ratio preservation
    A.SmallestMaxSize(max_size=256, interpolation=cv2.INTER_AREA),
    
    # Affine transformations (scale, rotate, translate)
    A.Affine(
        scale=(0.96, 1.04),      # ±4% zoom
        translate_percent=0.2,    # ±20% shift
        rotate=(-15, 15),         # ±15° rotation
        shear=0,                  # No shear (preserves product shape)
        border_mode=cv2.BORDER_REPLICATE,
        p=0.5
    ),
    
    # Random crop to target size
    A.RandomCrop(height=224, width=224),
    
    # Horizontal flip
    A.HorizontalFlip(),
    
    # Color augmentation (choice between two methods)
    A.OneOf([
        A.RandomBrightnessContrast(),
        A.ColorJitter(brightness=0.2, contrast=0.2, 
                      saturation=0.2, hue=0.2),
    ], p=1.0),
    
    # Noise augmentation (for robustness)
    A.OneOf([
        A.GaussNoise(p=0.3),
        A.GaussianBlur(p=0.3)
    ], p=0.3),
    
    # Normalize to ImageNet statistics
    A.Normalize(mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)),
    
    # Convert to PyTorch tensor
    ToTensorV2()
])
```

**Rationale:**
- **Aggressive augmentation:** ResNet's ImageNet pre-training is specialized for natural images
- **Geometric transforms:** compensate for domain shift (natural objects → products)
- **Noise/blur:** improve robustness to image quality variations

### Validation transforms (both models)

```python
A.Compose([
    # Resize with aspect ratio preservation
    A.SmallestMaxSize(max_size=224, interpolation=cv2.INTER_AREA),
    
    # Center crop (deterministic)
    A.CenterCrop(height=224, width=224),
    
    # Normalize to ImageNet statistics
    A.Normalize(mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225)),
    
    # Convert to PyTorch tensor
    ToTensorV2()
])
```

**Rationale:**
- **No randomness:** ensures reproducible validation metrics
- **Center crop:** focuses on main product (typically centered in e-commerce images)
- **Same normalization:** matches training statistics

---

## Training details

### DINOv2 training configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Optimizer** | AdamW | Decoupled weight decay for better regularization |
| **Learning Rate** | 1e-4 | Conservative LR for pre-trained features |
| **Weight Decay** | 1e-4 | L2 regularization to prevent overfitting |
| **Batch Size** | 32 | Balanced memory usage and gradient stability |
| **Scheduler** | ReduceLROnPlateau | Adaptive LR reduction on validation plateau |
| **Scheduler Factor** | 0.1 | 10× LR reduction when triggered |
| **Scheduler Patience** | 5 epochs | Wait 5 epochs before LR reduction |
| **Early Stopping** | patience=5, delta=1e-4 | Stop if no improvement for 5 epochs |
| **Training Strategy** | Head-only (backbone frozen) | Leverage pre-trained features |
| **Total Epochs** | 25
| **Training Time** | ~10 minutes (Tesla T4) | Fast due to frozen backbone |
| **Trainable Params** | ~5K (classification head only) | 0.006% of total parameters |

**Training curve:**
- Training loss: 2.06 → 0.45
- Validation loss: 1.74 → 0.48
- Validation accuracy: 37% → 90.5%
- Convergence: smooth and stable

### ResNet34 training configuration

#### Phase 1: head-only training (3 epochs)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Optimizer** | AdamW | Decoupled weight decay |
| **Learning Rate** | 1e-4 | Standard LR for transfer learning |
| **Weight Decay** | 1e-4 | L2 regularization |
| **Batch Size** | 32 | Stable gradients |
| **Trainable Layers** | FC head + all BatchNorm | Adapt statistics to product domain |
| **Frozen Layers** | layer1, layer2, layer3, layer4 | Preserve ImageNet features |
| **Scheduler** | ReduceLROnPlateau (factor=0.1, patience=5) | Adaptive LR |
| **Early Stopping** | patience=5, delta=1e-4 | Prevent overfitting |

**Phase 1 results:**
- Best validation loss: 1.75
- Best validation accuracy: 34%
- Training time: ~3 minutes

#### Phase 2: fine-tuning (20 epochs)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Optimizer** | AdamW | Same as Phase 1 |
| **Learning Rate** | 1e-5 (10× reduction) | Gentle fine-tuning of pre-trained layers |
| **Weight Decay** | 1e-4 | Consistent regularization |
| **Batch Size** | 32 | Unchanged |
| **Trainable Layers** | layer3, layer4, FC head, all BatchNorm | Adapt deeper features |
| **Frozen Layers** | layer1, layer2 | Preserve low-level features |
| **Scheduler** | ReduceLROnPlateau (factor=0.1, patience=5) | Adaptive LR |
| **Early Stopping** | patience=5, delta=1e-4 | Prevent overfitting |

**Phase 2 results:**
- Best validation loss: 0.62
- Best validation accuracy: 81%
- Training time: ~12 minutes
- Total training time: ~15 minutes (Phase 1 + Phase 2)

**Why 2 phases ?**
1. **Phase 1:** quickly adapts the classification head to 7 product classes
2. **Phase 2:** fine-tunes deeper layers to learn product-specific features
3. **Lower LR in phase 2:** prevents catastrophic forgetting of ImageNet features
4. **BatchNorm unfreezing:** adapts normalization statistics to product distribution

---


## License

This project uses pretrained models from [DINOv2](https://github.com/facebookresearch/dinov2) by Meta AI.

- **License:** Apache License 2.0
- **Obligations:** 
  - Include attribution to the original authors (Meta AI)
  - Provide a copy of the Apache 2.0 license text (see `DINOv2_LICENSE.txt`)
- **Implications for this project:**
  - Use in coursework and non-commercial contexts is fully permitted
  - Future commercial use is also permitted under Apache 2.0 terms, provided attribution and license inclusion are maintained

**No modifications** were made to the core DINOv2 code or model weights beyond integration into this classification project.

See [`LICENSE_USAGE.md`](LICENSE_USAGE.md) for detailed license compliance notes.

---

## Acknowledgments

- **[Meta AI - DINOv2](https://github.com/facebookresearch/dinov2)** for the pretrained Vision Transformer
  - Oquab, M., et al. (2023). "DINOv2: Learning Robust Visual Features without Supervision". arXiv:2304.07193
  - Darcet, T., et al. (2023). "Vision Transformers Need Registers". arXiv:2309.16588

- **[PyTorch Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam)** by Jacob Gildenblat for explainability tools
  - Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization". ICCV 2017

- **[Albumentations](https://albumentations.ai/)** for fast and flexible image augmentation
  - Buslaev, A., et al. (2020). "Albumentations: Fast and Flexible Image Augmentations". Information 11(2), 125

- **[PyTorch](https://pytorch.org/)** and **[torchvision](https://pytorch.org/vision/)** for deep learning framework
  - Paszke, A., et al. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library". NeurIPS 2019

- **[ResNet](https://arxiv.org/abs/1512.03385)** architecture
  - He, K., et al. (2016). "Deep Residual Learning for Image Recognition". CVPR 2016

---

## References

1. **Oquab, M., Darcet, T., Moutakanni, T., et al.** (2023)
   "DINOv2: Learning Robust Visual Features without Supervision"
   arXiv:2304.07193
   [Paper](https://arxiv.org/abs/2304.07193) | [Code](https://github.com/facebookresearch/dinov2)

2. **Darcet, T., Oquab, M., Mairal, J., & Bojanowski, P.** (2023)
   "Vision Transformers Need Registers"
   arXiv:2309.16588
   [Paper](https://arxiv.org/abs/2309.16588)

3. **Selvaraju, R. R., Cogswell, M., Das, A., et al.** (2017)
   "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
   ICCV 2017
   [Paper](https://arxiv.org/abs/1610.02391)



## Future Work

Potential improvements and extensions:

1. **Multi-modal fusion**

   - Incorporate product titles, descriptions, or specifications
   - Vision-Language models (e.g., CLIP) for text-image fusion


2. **Hierarchical classification**

   - Two-stage: coarse category → fine subcategory
   - Could improve accuracy on overlapping classes


