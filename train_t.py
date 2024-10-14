from src.config.config import get_config
from src.training.train import train_model
import warnings

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)