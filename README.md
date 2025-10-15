# PyTorch Training Framework

A comprehensive, modular neural network training framework built with PyTorch, featuring support for image classification and segmentation tasks.

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for datasets, models, losses, metrics, optimizers, and callbacks
- **Configuration Management**: YAML-based configuration with CLI overrides using OmegaConf
- **Multiple Task Support**: Image classification (single & multi-label) and segmentation
- **Model Zoo Integration**: Support for timm and torchvision backbones
- **Advanced Augmentations**: Albumentations-based augmentation pipelines with deterministic mode
- **Comprehensive Metrics**: torchmetrics integration with per-class, macro/micro reductions
- **Training Features**: AMP, gradient clipping, accumulation, early stopping, checkpointing
- **Visualization**: Automatic plotting of loss curves, confusion matrices, ROC/PR curves, sample grids
- **Experiment Tracking**: Structured logging with JSONL/CSV output and run organization
- **Inference Tools**: Batch inference with TTA support and model export (TorchScript/ONNX)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ML_Framework
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

## Quick Start

### 1. Prepare Your Data

For **classification**, organize your data as:
```
datasets/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── class2/
│       ├── image3.jpg
│       └── image4.jpg
└── val/
    ├── class1/
    └── class2/
```

For **segmentation**, organize your data as:
```
datasets/
├── images/
│   ├── image1.jpg
│   └── image2.jpg
└── masks/
    ├── image1.png
    └── image2.png
```

### 2. Train a Model

```bash
# Classification with ResNet-50
python scripts/train.py --config configs/classification_resnet50.yaml

# Segmentation with U-Net
python scripts/train.py --config configs/segmentation_unet.yaml

# With CLI overrides
python scripts/train.py --config configs/classification_resnet50.yaml model.num_classes=5 training.epochs=50
```

### 3. Evaluate a Model

```bash
# Evaluate on validation set
python scripts/eval.py --config configs/classification_resnet50.yaml --checkpoint runs/experiment_name/timestamp/checkpoints/best.pt

# Evaluate on test set
python scripts/eval.py --config configs/classification_resnet50.yaml --checkpoint runs/experiment_name/timestamp/checkpoints/best.pt --split test
```

### 4. Run Inference

```bash
# Inference on image folder
python scripts/infer.py --config configs/classification_resnet50.yaml --checkpoint runs/experiment_name/timestamp/checkpoints/best.pt --input ./test_images --output ./results

# Inference on CSV file
python scripts/infer.py --config configs/classification_resnet50.yaml --checkpoint runs/experiment_name/timestamp/checkpoints/best.pt --input ./test_data.csv --input-type csv --output ./results
```

## Configuration

The framework uses YAML configuration files with CLI override support. Here's a basic configuration structure:

```yaml
experiment:
  name: my_experiment
  seed: 42

data:
  dataset_type: classification
  data_dir: ./datasets/my_dataset
  num_classes: 10
  class_names: ['class1', 'class2', ...]
  train_split: 0.8
  val_split: 0.2

model:
  backbone: resnet50
  pretrained: true
  freeze_backbone: false
  dropout: 0.1

loss:
  name: cross_entropy
  label_smoothing: 0.1

optimizer:
  name: adamw
  lr: 1e-3
  weight_decay: 1e-4

scheduler:
  name: cosine_warmup
  warmup_epochs: 5
  total_epochs: 100

training:
  epochs: 100
  amp: true
  gradient_clip_norm: 1.0

callbacks:
  checkpoint:
    monitor: val_loss
    mode: min
    save_top_k: 3
  early_stopping:
    monitor: val_loss
    mode: min
    patience: 10
```

### CLI Overrides

You can override any configuration value using the `key.subkey=value` syntax:

```bash
python scripts/train.py --config configs/classification_resnet50.yaml \
  model.backbone=efficientnet_b0 \
  training.epochs=200 \
  optimizer.lr=0.001 \
  callbacks.early_stopping.patience=20
```

## Project Structure

```
ML_Framework/
├── configs/              # YAML configuration files
├── data/                 # Dataset adapters
├── datasets/             # Dataset definitions
├── transforms/           # Augmentation pipelines
├── models/               # Model architectures & registry
├── losses/               # Loss function registry
├── metrics/              # Metrics wrappers
├── optim/                # Optimizers & schedulers
├── callbacks/            # Training callbacks
├── engine/               # Trainer, Evaluator, Inferencer
├── utils/                # Utilities (seeding, logging, device, config)
├── scripts/              # CLI entry points
├── runs/                 # Experiment outputs (auto-created)
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## Available Models

### Classification Models
- **ResNet**: resnet18, resnet34, resnet50, resnet101, resnet152
- **EfficientNet**: efficientnet_b0 through efficientnet_b7, efficientnet_v2_s/m/l
- **ConvNeXt**: convnext_tiny, convnext_small, convnext_base, convnext_large
- **Vision Transformers**: vit_tiny, vit_small, vit_base, vit_large
- **Swin Transformers**: swin_tiny, swin_small, swin_base, swin_large
- **DenseNet**: densenet121, densenet161, densenet169, densenet201

### Segmentation Models
- **U-Net**: Custom encoder-decoder architecture
- **DeepLabV3**: Available through timm
- **FPN**: Feature Pyramid Network

## Available Loss Functions

- **Classification**: CrossEntropy, BCEWithLogits, Focal, Label Smoothing
- **Segmentation**: Dice, SoftDice, Lovász, Tversky, Combined Loss
- **Multi-label**: BCEWithLogits with class weights

## Available Metrics

- **Classification**: Accuracy, Precision, Recall, F1, AUROC, PR-AUC
- **Segmentation**: IoU, Dice, Pixel Accuracy, Mean IoU
- **Multi-label**: All classification metrics with multi-label support

## Available Optimizers

- SGD, Adam, AdamW, RMSprop, Adagrad, Adamax, Rprop, Adadelta

## Available Schedulers

- StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR
- CosineAnnealingWarmupLR, LinearWarmupLR, PolynomialWarmupLR
- OneCycleLR, ReduceLROnPlateau

## Callbacks

- **ModelCheckpoint**: Save best and last models
- **EarlyStopping**: Stop training when metric plateaus
- **LearningRateMonitor**: Track learning rate changes
- **SampleVisualizer**: Visualize predictions during training
- **ConfusionMatrixVisualizer**: Generate confusion matrices
- **MetricLogger**: Log metrics to JSONL/CSV

## Experiment Outputs

Each training run creates a structured output folder:

```
runs/experiment_name/timestamp/
├── config.yaml              # Resolved configuration
├── checkpoints/             # Model checkpoints
│   ├── best.pt
│   ├── last.pt
│   └── epoch_*.pt
├── plots/                   # Generated plots
│   ├── loss_curves.png
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   └── pr_curves.png
├── samples/                 # Sample visualizations
│   └── epoch_*/
├── logs.jsonl              # Structured logs
├── scalars.csv             # Metrics per epoch
└── training_history.json   # Complete training history
```

## Advanced Usage

### Custom Datasets

Create a custom dataset by inheriting from the base dataset classes:

```python
from datasets.classification import ImageClassificationDataset

class MyCustomDataset(ImageClassificationDataset):
    def __init__(self, data_dir, transform=None, **kwargs):
        super().__init__(data_dir, transform=transform, **kwargs)
        # Add custom logic here
```

### Custom Models

Register custom models in the model registry:

```python
from models.registry import model_registry

def my_custom_model(num_classes, pretrained=True, **kwargs):
    # Your model implementation
    return model

model_registry.register('my_custom_model', my_custom_model)
```

### Custom Loss Functions

Add custom loss functions to the loss registry:

```python
from losses.registry import loss_registry

class MyCustomLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # Your loss implementation
    
    def forward(self, inputs, targets):
        # Your forward pass
        return loss

loss_registry.register('my_custom_loss', MyCustomLoss)
```

### Custom Callbacks

Create custom callbacks by inheriting from the base callback class:

```python
from callbacks.base import Callback

class MyCustomCallback(Callback):
    def on_epoch_end(self, epoch, **kwargs):
        # Your custom logic
        pass
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use gradient accumulation
2. **Data loading errors**: Check data paths and file formats
3. **Model loading errors**: Verify checkpoint compatibility
4. **Configuration errors**: Check YAML syntax and required fields

### Debug Mode

Enable debug mode by setting `deterministic: true` in your configuration:

```yaml
experiment:
  deterministic: true
```

This will use deterministic algorithms for reproducibility.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [PyTorch](https://pytorch.org/) for the deep learning framework
- [timm](https://github.com/rwightman/pytorch-image-models) for model backbones
- [Albumentations](https://albumentations.ai/) for data augmentations
- [torchmetrics](https://torchmetrics.readthedocs.io/) for metrics computation
- [OmegaConf](https://omegaconf.readthedocs.io/) for configuration management
