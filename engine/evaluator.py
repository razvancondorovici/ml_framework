"""Evaluation engine for PyTorch models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.preprocessing import label_binarize

from utils.device import get_device, move_to_device
from utils.logger import StructuredLogger
from utils.plotting import plot_confusion_matrix, plot_roc_curves, plot_pr_curves
from losses.registry import create_loss
from metrics.wrappers import create_metrics


class Evaluator:
    """Evaluation engine for PyTorch models."""
    
    def __init__(self,
                 model: nn.Module,
                 config: Optional[Dict[str, Any]] = None,
                 device: Optional[Union[str, torch.device]] = None):
        """Initialize evaluator.
        
        Args:
            model: PyTorch model to evaluate
            config: Evaluation configuration
            device: Device to evaluate on
        """
        self.model = model
        self.config = config or {}
        self.device = get_device(device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Set up evaluation components
        self._setup_evaluation_components()
        
        # Initialize logger
        self.logger = StructuredLogger(self.config.get('log_dir', 'logs'), 'evaluation')
    
    def _setup_evaluation_components(self):
        """Set up evaluation components from config."""
        # Loss function
        loss_config = self.config.get('loss', {'name': 'cross_entropy'})
        self.criterion = create_loss(loss_config)
        
        # Metrics
        metrics_config = self.config.get('metrics', {'task': 'multiclass', 'num_classes': 10})
        self.metrics = create_metrics(metrics_config)
        self.metrics.to(self.device)
        
        # Mixed precision
        self.use_amp = self.config.get('amp', True)
    
    def evaluate(self, 
                 dataloader: DataLoader,
                 class_names: Optional[List[str]] = None,
                 save_plots: bool = True,
                 save_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Evaluate model on given dataloader.
        
        Args:
            dataloader: Data loader for evaluation
            class_names: List of class names for visualization
            save_plots: Whether to save evaluation plots
            save_dir: Directory to save plots
            
        Returns:
            Evaluation results
        """
        self.model.eval()
        self.metrics.reset()
        
        # Initialize results
        all_predictions = []
        all_targets = []
        all_probabilities = []
        total_loss = 0.0
        num_batches = len(dataloader)
        
        self.logger.info(f"Starting evaluation on {num_batches} batches")
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader, desc="Evaluating")):
                # Move to device
                inputs = move_to_device(inputs, self.device)
                targets = move_to_device(targets, self.device)
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                # Update metrics
                self.metrics.update(outputs, targets)
                
                # Store predictions and targets
                if outputs.dim() == 4:  # Segmentation
                    predictions = torch.argmax(outputs, dim=1)
                    all_predictions.append(predictions.cpu())
                else:  # Classification
                    predictions = torch.argmax(outputs, dim=1)
                    probabilities = torch.softmax(outputs, dim=1)
                    
                    all_predictions.append(predictions.cpu())
                    all_probabilities.append(probabilities.cpu())
                
                all_targets.append(targets.cpu())
                
                # Update loss
                total_loss += loss.item()
        
        # Concatenate all predictions and targets
        all_predictions = torch.cat(all_predictions).numpy()
        all_targets = torch.cat(all_targets).numpy()
        
        if all_probabilities:
            all_probabilities = torch.cat(all_probabilities).numpy()
        
        # Compute metrics
        metrics = self.metrics.compute()
        metrics = {k: v.item() if hasattr(v, 'item') else v for k, v in metrics.items()}
        metrics['loss'] = total_loss / num_batches
        
        # Create evaluation results
        results = {
            'metrics': metrics,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities if all_probabilities else None
        }
        
        # Generate plots if requested
        if save_plots and save_dir:
            self._generate_plots(results, class_names, save_dir)
        
        # Log results
        self.logger.info("Evaluation completed", **metrics)
        
        return results
    
    def _generate_plots(self, 
                       results: Dict[str, Any], 
                       class_names: Optional[List[str]] = None,
                       save_dir: Union[str, Path]):
        """Generate evaluation plots.
        
        Args:
            results: Evaluation results
            class_names: List of class names
            save_dir: Directory to save plots
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        predictions = results['predictions']
        targets = results['targets']
        probabilities = results['probabilities']
        
        # Confusion matrix
        if len(np.unique(targets)) <= 20:  # Only for reasonable number of classes
            cm_path = save_dir / 'confusion_matrix.png'
            plot_confusion_matrix(
                y_true=targets,
                y_pred=predictions,
                class_names=class_names,
                save_path=cm_path,
                title="Confusion Matrix"
            )
        
        # ROC curves (for classification with probabilities)
        if probabilities is not None and len(np.unique(targets)) <= 10:
            roc_path = save_dir / 'roc_curves.png'
            plot_roc_curves(
                y_true=targets,
                y_scores=probabilities,
                class_names=class_names,
                save_path=roc_path,
                title="ROC Curves"
            )
            
            # PR curves
            pr_path = save_dir / 'pr_curves.png'
            plot_pr_curves(
                y_true=targets,
                y_scores=probabilities,
                class_names=class_names,
                save_path=pr_path,
                title="Precision-Recall Curves"
            )
    
    def evaluate_classification(self, 
                              dataloader: DataLoader,
                              class_names: Optional[List[str]] = None,
                              save_plots: bool = True,
                              save_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Evaluate classification model.
        
        Args:
            dataloader: Data loader for evaluation
            class_names: List of class names
            save_plots: Whether to save plots
            save_dir: Directory to save plots
            
        Returns:
            Classification evaluation results
        """
        results = self.evaluate(dataloader, class_names, save_plots, save_dir)
        
        # Add classification-specific metrics
        predictions = results['predictions']
        targets = results['targets']
        
        # Per-class accuracy
        num_classes = len(np.unique(targets))
        per_class_accuracy = []
        
        for class_id in range(num_classes):
            class_mask = targets == class_id
            if class_mask.sum() > 0:
                class_accuracy = (predictions[class_mask] == class_id).mean()
                per_class_accuracy.append(class_accuracy)
            else:
                per_class_accuracy.append(0.0)
        
        results['per_class_accuracy'] = per_class_accuracy
        
        # Top-k accuracy
        if results['probabilities'] is not None:
            probabilities = results['probabilities']
            top_k_accuracy = {}
            
            for k in [1, 3, 5]:
                if k <= probabilities.shape[1]:
                    top_k_preds = np.argsort(probabilities, axis=1)[:, -k:]
                    top_k_acc = np.mean([targets[i] in top_k_preds[i] for i in range(len(targets))])
                    top_k_accuracy[f'top_{k}_accuracy'] = top_k_acc
            
            results['top_k_accuracy'] = top_k_accuracy
        
        return results
    
    def evaluate_segmentation(self, 
                            dataloader: DataLoader,
                            class_names: Optional[List[str]] = None,
                            save_plots: bool = True,
                            save_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Evaluate segmentation model.
        
        Args:
            dataloader: Data loader for evaluation
            class_names: List of class names
            save_plots: Whether to save plots
            save_dir: Directory to save plots
            
        Returns:
            Segmentation evaluation results
        """
        results = self.evaluate(dataloader, class_names, save_plots, save_dir)
        
        # Add segmentation-specific metrics
        predictions = results['predictions']
        targets = results['targets']
        
        # Flatten for metric computation
        preds_flat = predictions.flatten()
        targets_flat = targets.flatten()
        
        # Per-class IoU
        num_classes = len(np.unique(targets))
        per_class_iou = []
        
        for class_id in range(num_classes):
            pred_mask = preds_flat == class_id
            target_mask = targets_flat == class_id
            
            intersection = (pred_mask & target_mask).sum()
            union = (pred_mask | target_mask).sum()
            
            if union > 0:
                iou = intersection / union
                per_class_iou.append(iou)
            else:
                per_class_iou.append(0.0)
        
        results['per_class_iou'] = per_class_iou
        results['mean_iou'] = np.mean(per_class_iou)
        
        # Per-class Dice coefficient
        per_class_dice = []
        
        for class_id in range(num_classes):
            pred_mask = preds_flat == class_id
            target_mask = targets_flat == class_id
            
            intersection = (pred_mask & target_mask).sum()
            union = pred_mask.sum() + target_mask.sum()
            
            if union > 0:
                dice = 2 * intersection / union
                per_class_dice.append(dice)
            else:
                per_class_dice.append(0.0)
        
        results['per_class_dice'] = per_class_dice
        results['mean_dice'] = np.mean(per_class_dice)
        
        return results
    
    def compare_models(self, 
                      models: List[nn.Module],
                      model_names: List[str],
                      dataloader: DataLoader,
                      class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compare multiple models on the same dataset.
        
        Args:
            models: List of models to compare
            model_names: List of model names
            dataloader: Data loader for evaluation
            class_names: List of class names
            
        Returns:
            Comparison results
        """
        results = {}
        
        for model, name in zip(models, model_names):
            self.logger.info(f"Evaluating model: {name}")
            
            # Temporarily replace model
            original_model = self.model
            self.model = model
            self.model.to(self.device)
            
            # Evaluate model
            model_results = self.evaluate(dataloader, class_names, save_plots=False)
            results[name] = model_results
            
            # Restore original model
            self.model = original_model
        
        # Create comparison summary
        comparison = {}
        for name, model_results in results.items():
            comparison[name] = model_results['metrics']
        
        results['comparison'] = comparison
        
        return results
