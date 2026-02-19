"""
TAA-PINN Training Entry Point

Usage:
    python train.py --config configs/AS5_config.yaml
    python train.py --config configs/AS6_config.yaml
    python train.py --config configs/AS5_config.yaml --resume experiments/AS5/best_model.pt
"""

import argparse
import warnings

# Suppress harmless third-party library version warnings that appear at import time
warnings.filterwarnings("ignore", message=".*numexpr.*")
warnings.filterwarnings("ignore", message=".*chardet.*")
warnings.filterwarnings("ignore", message=".*charset_normalizer.*")

from src.training.trainer import TAATrainer


def main():
    parser = argparse.ArgumentParser(description="Train TAA-PINN for single geometry")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to configuration YAML file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    trainer = TAATrainer(args.config, resume_checkpoint=args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
