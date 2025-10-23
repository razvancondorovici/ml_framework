## **Introduction Guide: PyTorch ML Framework**

This is a comprehensive learning path that will help newcommers understand and use the ML framework effectively:

### **🎯 Learning Path Overview**

The framework is a **production-ready, modular PyTorch training system** that supports both classification and segmentation tasks. It's designed with clean architecture principles and follows modern ML engineering practices.

---

## **Phase 1: Framework Overview & Setup**

### **1.1 What is this Framework?**

- **Purpose**: A complete ML training pipeline that handles everything from data loading to model deployment
- **Key Features**: 
  - Modular design (easy to extend)
  - YAML-based configuration (no code changes for experiments)
  - Support for both classification and segmentation
  - Built-in visualization and monitoring
  - Production-ready features (checkpointing, early stopping, etc.)

### **1.2 Project Structure Walkthrough**

```
ML_Framework/
├── configs/          # 🎛️ Experiment configurations
├── datasets/         # 📊 Data loading logic
├── models/           # 🧠 Model architectures
├── engine/           # 🚀 Training/inference engines
├── callbacks/        # 📈 Monitoring & visualization
├── scripts/          # 🖥️ Command-line tools
└── utils/            # 🔧 Helper functions
```

**Key Point**: Each module has a single responsibility - this makes the code easy to understand and modify.

### **1.3 Installation & Quick Test**
```powershell
# Create new evirtual nvironment
python -m venv venv

# Activate the environement
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Test installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

---

## **Phase 2: Configuration System**

### **2.1 YAML Configuration Deep Dive**
Examine `configs/classification_resnet50.yaml`:

```yaml
experiment:
  name: resnet50_classification
  seed: 42

data:
  dataset_type: classification
  data_dir: ./datasets/imagenet_subset
  num_classes: 10
  class_names: ['airplane', 'automobile', ...]

model:
  backbone: resnet50
  pretrained: true
  dropout: 0.1
```

**Key Concepts**:
- **No code changes needed** for different experiments
- **CLI overrides** allow quick parameter tuning
- **Hierarchical structure** keeps related settings together

### **2.2 CLI Override Examples**
```powershell
# Change model architecture
python scripts/train.py --config configs/classification_resnet50.yaml model.backbone=efficientnet_b0

# Adjust training parameters
python scripts/train.py --config configs/classification_resnet50.yaml training.epochs=200 optimizer.lr=0.001
```

---

## **Phase 3: Data Pipeline**

### **3.1 Dataset Classes**
Walk through `datasets/classification.py`:

```python
class ImageClassificationDataset(Dataset):
    def __init__(self, data_dir, annotations_file=None, class_names=None, transform=None):
        # Handles both folder-based and CSV-based data
        # Automatic class mapping
        # Transform pipeline integration
```

**Key Features**:
- **Flexible data sources** (folders or CSV files)
- **Automatic class mapping**
- **Transform integration**

### **3.2 Data Organization**
Expected data structure:

**Classification**:
```
datasets/
├── train/
│   ├── class1/
│   └── class2/
└── val/
    ├── class1/
    └── class2/
```

**Segmentation**:
```
datasets/
├── images/
│   ├── image1.jpg
│   └── image2.jpg
└── masks/
    ├── image1.png
    └── image2.png
```

### **3.3 Augmentation Pipeline**
Examine `transforms/augmentations.py` to see how Albumentations is integrated for robust data augmentation.

---

## **Phase 4: Model Architecture**

### **4.1 Model Registry System**
Check `models/registry.py`

```python
class ModelRegistry:
    def _register_timm_models(self):
        # Registers 50+ models from timm
        # ResNet, EfficientNet, Vision Transformers, etc.
```

**Key Benefits**:
- **50+ pre-trained models** available
- **Consistent interface** for all models
- **Easy to add custom models**

### **4.2 Model Building Process**
```python
# From scripts/train.py
model = build_classifier(
    backbone='resnet50',
    num_classes=10,
    pretrained=True,
    dropout=0.1
)
```

---

## **Phase 5: Training Engine**

### **5.1 Trainer Class Overview**
Examine `engine/trainer.py` - the heart of the framework:

```python
class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, config, device, callbacks):
        # Handles training loop, validation, metrics, etc.
    
    def fit(self, epochs, resume_from_checkpoint=None):
        # Main training loop with callbacks
```

**Key Features**:
- **Checkpointing**
- **Callback system**

### **5.2 Callback System**
Check `callbacks/base.py`:

```python
class Callback(ABC):
    def on_epoch_start(self, epoch, **kwargs): pass
    def on_epoch_end(self, epoch, **kwargs): pass
    def on_batch_start(self, batch_idx, **kwargs): pass
    # ... more hooks
```

**Available Callbacks**:
- `ModelCheckpoint` - Save best models
- `EarlyStopping` - Stop when no improvement
- `SampleVisualizer` - Visualize predictions
- `ConfusionMatrixVisualizer` - Generate confusion matrices
- `LearningRateVisualizer` - Track learning rate changes

---


---

## **Phase 6: Advanced Features**

### **6.1 Custom Components**
Extending the framework:

**Custom Dataset**:
```python
class MyCustomDataset(ImageClassificationDataset):
    def __init__(self, data_dir, transform=None, **kwargs):
        super().__init__(data_dir, transform=transform, **kwargs)
        # Add custom logic
```

**Custom Model**:
```python
from models.registry import model_registry

def my_custom_model(num_classes, pretrained=True, **kwargs):
    # Your model implementation
    return model

model_registry.register('my_custom_model', my_custom_model)
```

### **6.2 Inference Pipeline**
Check `scripts/infer.py` for model deployment:

```powershell
# Inference on image folder
python scripts/infer.py --config configs/classification_resnet50.yaml --checkpoint runs/.../best.pt --input ./test_images --output ./results
```

### **6.3 Experiment Tracking**
Check the `runs/` folder structure:
```
runs/experiment_name/timestamp/
├── config.yaml              # Complete configuration
├── checkpoints/             # Model checkpoints
├── plots/                   # Generated visualizations
├── samples/                 # Sample predictions
├── scalars.csv             # Metrics per epoch
└── training_history.json   # Complete training history
```

---

## **Phase 7: Best Practices & Tips**

### **7.1 Configuration Management**
- **Start with existing configs** and modify them
- **Use meaningful experiment names**
- **Version control your configs**
- **Use CLI overrides for quick experiments**

### **7.2 Debugging Tips**
- **Enable deterministic mode** for reproducibility
- **Use smaller datasets** for initial testing
- **Monitor GPU memory usage**
- **Check data loading with small batch sizes**

### **7.3 Performance Optimization**
- **Use appropriate batch sizes**
- **Enable mixed precision training**
- **Use multiple workers for data loading**
- **Monitor training curves for overfitting**

---

## **📚 Key Takeaways**

1. **Modular Design**: Each component has a clear responsibility
2. **Configuration-Driven**: Change experiments without code changes
3. **Production-Ready**: Built-in monitoring, checkpointing, and deployment tools
4. **Extensible**: Easy to add custom components
5. **Best Practices**: Follows modern ML engineering principles
