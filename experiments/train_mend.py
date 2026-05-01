"""
Train a MEND meta-network on ZsRE data.

Usage:
    python train_mend.py --hparams_dir ./hparams/TRAINING/MEND/llama3-8b.yaml
    python train_mend.py --hparams_dir ./hparams/TRAINING/MEND/qwen2.5-7b.yaml
"""

import argparse

from src import EditTrainer, MENDTrainingHparams, ZsreDataset


def main():
    parser = argparse.ArgumentParser(description="Train MEND meta-network on ZsRE")
    parser.add_argument(
        "--hparams_dir",
        required=True,
        help="Path to MENDTrainingHparams YAML (e.g. hparams/TRAINING/MEND/llama3-8b.yaml)",
    )
    parser.add_argument(
        "--train_data",
        default="./data/ZsRE/zsre_train_data.json",
        help="Path to ZsRE training JSON (locality-labeled samples)",
    )
    parser.add_argument(
        "--eval_data",
        default="./data/ZsRE/zsre_edit_data.json",
        help="Path to ZsRE evaluation JSON (edit examples)",
    )
    args = parser.parse_args()

    training_hparams = MENDTrainingHparams.from_hparams(args.hparams_dir)

    train_ds = ZsreDataset(args.train_data, config=training_hparams)
    eval_ds = ZsreDataset(args.eval_data, config=training_hparams)

    trainer = EditTrainer(
        config=training_hparams,
        train_set=train_ds,
        val_set=eval_ds,
    )
    trainer.run()


if __name__ == "__main__":
    main()
