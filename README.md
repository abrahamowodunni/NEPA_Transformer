# English-Yoruba Transformer Model

This project implements a Transformer-based neural network to translate English into Yoruba. The model leverages self-attention mechanisms to capture complex relationships between tokens and handle long-range dependencies. Despite challenges posed by limited training data and computational resources, the project serves as an insightful exploration of transformer architectures in low-resource environments.

## Table of Contents
- [Background](#background)
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Training and Optimization](#training-and-optimization)
- [Evaluation](#evaluation)
- [Results](#results)
- [Challenges and Insights](#challenges-and-insights)
- [Future Work](#future-work)
- [Contributions](#contributions)
- [License](#license)

## Background

Yoruba is a low-resource language with unique tonal and syntactical complexities, making machine translation especially difficult. The project explores the use of Transformer architectures to address these challenges. Transformers, powered by self-attention, have revolutionized NLP by improving how machines understand and generate human language, and this project seeks to apply those advances to English-Yoruba translation.

This project pushes the boundaries of translation models for underrepresented languages and aims to improve translation quality by refining the model architecture and dataset.

## Project Overview

The Transformer model in this project follows a sequence-to-sequence architecture that uses self-attention to process variable-length input sequences. The main components of the model include:

- **Self-attention**: Allows the model to weigh all words in a sentence against each other, rather than processing them sequentially.
- **Positional Encoding**: Adds information about the order of the words, as Transformers do not inherently capture sequence order.
- **Multi-Head Attention**: Enables the model to focus on different parts of the sentence simultaneously, improving its ability to learn from complex structures.
- **Encoder-Decoder Framework**: The encoder processes the English input, and the decoder generates the Yoruba output based on learned mappings between languages.

## Installation

To get started with the project, make sure to clone the repository and install the required dependencies as mentioned in the `requirements.txt` file. This project requires Python 3.7+.

## Usage

Once the repository is set up, the model can be trained or evaluated using the provided dataset. Preprocessing steps like tokenization and handling of diacritics in Yoruba are already implemented, making it easier to get started with your own training experiments.

Additionally, the model can be used for inference to translate new English sentences into Yoruba by loading the pre-trained checkpoint.

## Training and Optimization

Training a transformer model in a low-resource setting brought forth several challenges. Below is a summary of the training setup:

- **Dataset**: A parallel English-Yoruba dataset was preprocessed to ensure tokenization and proper handling of Yoruba's diacritical marks.
  
- **Loss Function**: Cross-entropy loss was applied, with label smoothing to handle overfitting.
  
- **Optimization**: Adam optimizer with weight decay was used, alongside learning rate scheduling (cosine decay) to improve training stability.

- **Regularization**: Dropout layers were introduced to the attention mechanism to prevent overfitting, and data augmentation was employed to maximize the dataset's effectiveness.

### Challenges in Training

- **Erratic Training Loss**: The loss curve exhibited instability, especially in later stages of training, which was largely attributed to high learning rates. This was mitigated through the use of a learning rate scheduler, resulting in more stable training behavior.
  
- **Limited Dataset**: The dataset available was relatively small, limiting the model’s ability to generalize. The performance on longer sentences revealed gaps in understanding and translation accuracy.

- **Long-Range Dependencies**: Handling long-range dependencies in both languages proved challenging. Although the model handled shorter sentences well, longer sentences caused attention patterns to become dispersed and less effective.

### Training Loss Visualization

Training loss was visualized during the experiment, showing early sharp drops, followed by erratic behavior before stabilizing with the implementation of learning rate adjustments.

![Training Loss Graph](![image](https://github.com/user-attachments/assets/7232891f-15cf-47f5-99fc-9317eff30169)
)

## Evaluation

The evaluation phase included both quantitative and qualitative methods:

- **BLEU Score**: BLEU scores were used as the primary evaluation metric, reflecting translation accuracy based on n-gram precision.
  
- **Attention Visualization**: Attention weights were visualized to understand how the model distributed focus across input sequences, revealing insights into its ability to capture correct word mappings.

### Cross-Attention Maps

Cross-attention maps showed that the model learned strong alignments in simple, short sentences but struggled with complex, long-range dependencies.

![Attention Map](path/to/attention_map.png)

## Results

- **Translation Accuracy**: The model produced acceptable translations for basic sentences but struggled with more complex sentence structures, especially in longer sequences.

- **Attention Analysis**: Attention maps revealed that the model effectively captured key token correspondences but displayed erratic behavior when sentences grew in complexity, indicating room for improvement in attention mechanisms.

## Challenges and Insights

Throughout the project, several key challenges were identified, and valuable insights were gained:

1. **Loss Instability**: Initially, the loss fluctuated heavily during training, which was solved using learning rate schedulers and proper weight decay.
   
2. **Overfitting**: The model tended to overfit due to the small dataset size. Techniques like dropout, label smoothing, and data augmentation were applied to help mitigate this.

3. **Long-Range Dependencies**: One of the key limitations was the model’s inability to handle longer sequences effectively. While short sentences were translated well, more complex sentences caused performance to degrade.

4. **Need for More Data**: The model's performance can be significantly enhanced with more diverse data, particularly focusing on longer-range content to improve translations of complex sentences. The current dataset, though functional for basic translations, lacks the variety and richness needed for sophisticated translations.

## Future Work

1. **Architectural Enhancements**: Going forward, I plan to experiment with deeper architectures and more attention heads to capture more nuanced relationships between tokens.
  
2. **Larger Dataset**: Expanding the dataset with long-range and more complex sentence structures will be a priority to improve the model's ability to generalize and perform better on complex translations.

3. **Cloud GPU Resources**: As local resources are limited, I plan to leverage cloud-based solutions like Paperspace or Google Cloud to accelerate training. These platforms will enable faster iterations and exploration of more complex model configurations.

4. **Handling Long-Range Dependencies**: Future work will focus on improving the model's capacity to deal with long-range dependencies, using techniques such as TransformerXL or adding recurrence mechanisms to the architecture.

## Contributions

Contributions to improving the model, expanding the dataset, or enhancing the training pipeline are welcome! Please open a pull request or an issue for discussion.

## License

This project is licensed under the MIT License. See the LICENSE file for more information.
