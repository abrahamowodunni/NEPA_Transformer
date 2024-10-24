import torch
import torch.nn as nn 
import math

class InputEmbeddings(nn.Module):
    """
    A PyTorch module for generating input embeddings for tokens in a sequence.

    Args:
        d_model (int): Dimension of the embedding vectors.
        vocab_size (int): The size of the vocabulary.

    Forward:
        x (Tensor): Input tensor of token indices.
    
    Returns:
        Tensor: Scaled embeddings for the input tokens.
    """
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model) # torch can help us with this 

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """
    A PyTorch module for adding positional encodings to token embeddings.

    Args:
        d_model (int): Dimension of the embedding vectors.
        seq_len (int): Maximum length of the input sequence.
        dropout (float): Dropout rate for regularization.

    Forward:
        x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).

    Returns:
        Tensor: Token embeddings with added positional encodings and dropout applied.
    """
    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # creating a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len,d_model)

        # create a vector of shape (seq_le, 1n)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        denominator = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))

        # apply the sin to even and cos to odd
        pe[:,0::2] = torch.sin(position * denominator)
        pe[:,1::2] = torch.cos(position * denominator)

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer('pe',pe) # so it is not treated as a prameter, meaning no update durring backpropagation. 

    def forward(self,x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # so it doesn't change when training. 
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    """
    Applies Layer Normalization over the last dimension of the input tensor.

    Layer normalization stabilizes the learning process by normalizing the inputs to have zero mean and unit variance,
    followed by scaling and shifting with learnable parameters (alpha and bias).

    Args:
        eps (float): A small value added to the denominator for numerical stability (default: 1e-6).

    Attributes:
        alpha (nn.Parameter): Learnable scaling factor.
        bias (nn.Parameter): Learnable shift parameter.
    
    Forward Input:
        x (Tensor): Input tensor of shape (batch_size, ..., seq_len, d_model).

    Forward Output:
        Tensor: Normalized tensor of the same shape as the input.
    """
    def __init__(self, eps = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # for multiplication
        self.bias = nn.Parameter(torch.zeros(1)) # for addition

    def forward(self,x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x-mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):
    """
    Implements a two-layer feedforward neural network with ReLU activation and dropout, typically used after the attention layer in a transformer.

    Args:
        d_model (int): The input and output dimensionality (model size).
        d_ff (int): The dimensionality of the feedforward layer (hidden size).
        dropout (float): The dropout rate applied after the ReLU activation.

    Forward Input:
        x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).

    Forward Output:
        Tensor: Transformed tensor of shape (batch_size, seq_len, d_model).
    
    Functionality:
        - First transforms input from (d_model) to (d_ff).
        - Applies ReLU for non-linearity.
        - Applies dropout for regularization.
        - Finally transforms back to (d_model).
    """
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # Linear transformation W1 with bias b1
        self.dropout = nn.Dropout(dropout) # Dropout for regularization
        self.linear2 = nn.Linear(d_ff, d_model) # Linear transformation W2 with bias b2

    def forward(self, x):
        # (batch, sq_len, d_model) --> (batch, sq_len, d_ff) --> (batch, sq_len, d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
    
class ResidualConnection(nn.Module): # the skip connection!
    """
    Implements a residual connection with layer normalization and dropout.

    Args:
        dropout (float): Dropout probability to regularize the sublayer output.

    Methods:
        forward(x, sublayer):
            Applies layer normalization to the input 'x', passes it through the 'sublayer',
            applies dropout, and then adds the original input 'x' as a residual connection.

    Returns:
        Tensor: The output with a residual connection added to the transformed input.
    """
    def __init__(self, dropout):

        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class ProjectionLayer(nn.Module):
    """
    A projection layer that maps the hidden states (of dimension d_model) from the transformer decoder 
    to the vocabulary size, providing a probability distribution over the vocabulary for each position 
    in the sequence.

    Args:
        d_model (int): Dimensionality of the model's internal hidden states.
        vocab_size (int): Size of the output vocabulary.

    Methods:
        forward(x):
            Takes the hidden state output from the decoder (shape: batch, seq_len, d_model) 
            and projects it to the vocabulary size (batch, seq_len, vocab_size) using a linear transformation, 
            followed by log_softmax to get the probabilities for each token.

    Input:
        x (Tensor): The hidden state tensor with shape (batch, seq_len, d_model).

    Output:
        Tensor: Log-probabilities of tokens for each position in the sequence with shape 
        (batch, seq_len, vocab_size).
    """
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)