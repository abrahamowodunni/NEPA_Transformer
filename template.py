import os
from pathlib import Path
import logging

# Logging configuration
logging.basicConfig(
    format= "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level= logging.INFO
)

# List of directories and files to create for your project
list_of_files = [
    "artifacts/cleaned_data",  # Cleaned training data
    "artifacts/tokenizers",  # Tokenizer file
    "artifacts/models",  # Saved model checkpoint
    "train_config.yaml",  # Training hyperparameters and paths
    "model_config.yaml",  # Model architecture details
    "model/__init__.py",  # Model module init file
    "model/embeddings.py",  # Embeddings and positional encoding
    "model/layers.py",  # Layers such as LayerNormalization
    "model/attention.py",  # MultiHeadAttentionBlock
    "model/encoder.py",  # EncoderBlock, Encoder classes
    "model/decoder.py",  # DecoderBlock, Decoder classes
    "model/transformer.py",  # Transformer class, projection layer
    "dataset/__init__.py",  # Dataset module init file
    "dataset/bilingual_dataset.py",  # BilingualDataset class
    "training/__init__.py",  # Training module init file
    "training/train.py",  # Training logic
    "training/utils.py",  # Utility functions for training
    "logs/",  # TensorBoard logs directory
    "experiments/",  # Experiment results directory
    "main.py",  # Main entry point for running training
]

# Iterate through the list and create directories and files
for filepath in list_of_files:
    filepath = Path(filepath)  # Convert to the correct format for the OS

    filedir, filename = os.path.split(filepath)  # Split directory and file

    # Create the directory if it doesn't exist
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    # Create the file if it doesn't exist or if it is empty
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        with open(filepath, "w") as f:
            pass  # Create an empty file
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")

logging.info("Project structure has been created successfully.")
