from src.config.config import get_config
from src.training.train import train_model
import warnings

def main():
    # Load configuration from the YAML files
    config = get_config()

    # Start training the model with the loaded configuration
    train_model(config)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)