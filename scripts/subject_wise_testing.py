




import argparse
from datetime import datetime
import os
import sys
from pathlib import Path

import numpy as np
import torch
from typing import Dict, Any

from torch.utils.checkpoint import checkpoint

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.config import load_config, get_config_parser
from utils.device import get_device_info
from utils.logger import StructuredLogger
from models.registry import build_classifier, build_segmentation_model
from engine.inferencer import Inferencer
from metrics.wrappers import MetricsWrapper
from models.registry import ModelRegistry


def create_model(config: Dict[str, Any]) -> Any:
    """Create model from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Model instance
    """
    model_config = config['model']
    data_config = config['data']

    # Get model parameters
    backbone = model_config.get('backbone', 'resnet50')
    num_classes = data_config.get('num_classes', 10)

    model = build_classifier(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=True,
        freeze_backbone=True,
        dropout=config["model"]["dropout"],
        mode="test"
    )
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    return model

def main():
    """Main subject wise testing function."""
    parser = get_config_parser()
    args = parser.parse_args()

    datetime_now = datetime.now()
    new_test_dir = "_".join([str(datetime_now.month), str(datetime_now.day),
                             str(datetime_now.hour), str(datetime_now.minute),
                             str(datetime_now.second)])

    # Load configuration
    config = load_config(args.config)
    checkpoint_path = config.data.checkpoint
    # # Print device info
    device_info = get_device_info()
    print(f"Device info: {device_info}")

    config['output'] = os.path.join(os.getcwd(), "Subject Wise Testing", new_test_dir)
    os.makedirs(config['output'], exist_ok=True)

    input = config.data.test_data_txt
    # Create logger
    logger = StructuredLogger(Path(config['output']), 'testing')
    logger.info("Starting Subject Wise Testing", checkpoint=checkpoint_path, input=input)
    # Get only the test files
    test_files = list(filter(lambda f: "test" in f, os.listdir(input)))
    no_folds = len(test_files)
    seen_ids = {}
    aggregate_results = {"preds":[], "probs":[], "GT":[]}

    for fold, test_subjects in enumerate(test_files):
        print(f"\n===== Fold {fold + 1}/{no_folds} =====")
        output_per_fold = os.path.join(config['output'], test_subjects)
        os.makedirs(output_per_fold, exist_ok=True)
        try:
            txt_file = os.path.join(input, test_subjects)
            # Create model
            print("Creating a new instance for the model...")
            model = create_model(config)
            print(f"Model: {type(model).__name__}")

            # Create inferencer
            print("Creating inferencer...")
            inferencer = Inferencer(
                model=model, config=config,
                device="cuda" if device_info["cuda_available"] else "cpu"
            )

            # Load checkpoint
            print(f"Loading checkpoint: {checkpoint_path}")
            inferencer.load_checkpoint(checkpoint_path)

            # Run inference
            print("Starting inference...")
            class_names = config.get('data', {}).get('class_names')

            results = inferencer.predict_from_txt(
                    txt_path=str(txt_file),
                    output_path=str(output_per_fold),
                    class_names=class_names,
                    batch_size=config['dataloader']['batch_size'],
                    num_workers=config['dataloader']['num_workers']
                )

            # Print results summary
            print(f"\nInference completed!")
            print(f"Processed one fold of images")
            print(f"Results saved to {output_per_fold}")
            unique_IDS = list(results['results'].keys())
            for id in unique_IDS:

                metrics_instance = MetricsWrapper(config['data'].get('num_classes', 10), config['metrics'].task,
                                                  average=config.get('average', 'macro'),
                                                  threshold=config.get('threshold', 0.5))
                metrics_instance.update(torch.tensor(results['probabilities'][id]), torch.tensor(results['results'][id]['GT']))
                test_metrics = metrics_instance.compute()
                test_metrics = {k: v.float().mean().item() if hasattr(v, 'item') and v.numel() > 1 else (
                    v.item() if hasattr(v, 'item') else v) for k, v in test_metrics.items()}

                # Print sample predictions; actually the first 5 of them
                if len(results['results'][id]) > 0:
                    print("\nSample predictions:")
                    print("=" * 50)
                    sample_results = results['results'][id].head(5)
                    for _, row in sample_results.iterrows():
                        if 'class_name' in row:
                            logger.info(f"{row['image_path']}: {row['class_name']} (confidence: {row['confidence']:.3f})")
                        else:
                            logger.info(f"{row['image_path']}: class {row['prediction']} (confidence: {row['confidence']:.3f})")
                if id not in seen_ids:
                    aggregate_results["GT"].extend(np.array(results['results'][id]['GT']))
                    aggregate_results["probs"].extend(np.array(results['probabilities'][id]))
                    aggregate_results["preds"].extend(np.array(results['predictions'][id]))
                    seen_ids[id] = test_metrics["accuracy"]

                    logger.info(f"Inference completed successfully for the following ID \n {id}",
                                num_images=len(results['results'][id]))
                    print(f"ID - {id}", "=" * 34)
                    print("=============TEST METRICS=============")
                    print(test_metrics)
                    logger.info(test_metrics)
                else:
                    print(f"this id was recycled {id}")

        except Exception as e:
            logger.error(f"Subject Wise testing failed with: \n {e}")
            raise
        print("Mean Final Results")

    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report
    y_true = np.array(aggregate_results["GT"])
    y_prob = np.array(aggregate_results["probs"])
    y_pred = np.array(aggregate_results["preds"])

    global_f1 = f1_score(y_true, y_pred)
    global_auroc = roc_auc_score(y_true, y_prob[:,1])
    global_acc = accuracy_score(y_true, y_pred)

    logger.info(f"Accuracy -> {global_acc}")
    logger.info(f"F1 score -> {global_f1}")
    logger.info(f"AUROC    -> {global_auroc}")
    logger.info(f"Classification report:")
    logger.info(classification_report(y_true, y_pred, zero_division=0))


if __name__ == '__main__':
    print("RUNNING SUBJECT WISE TESTING", "+=" * 34)
    main()
