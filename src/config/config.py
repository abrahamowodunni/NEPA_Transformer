import yaml
from pathlib import Path

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def get_config():
    train_config = load_yaml('train.yaml')  # Load training config
    model_config = load_yaml('model.yaml')  # Load model config
    config = {**train_config, **model_config}  # Combine both configs
    return config

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
