import torch
import torch.nn as nn 
import math

class MultiHeadAttentionBlock(nn.Module):
    """
    Implements the Multi-Head Attention mechanism used in transformers.
    
    Args:
        d_model (int): The dimensionality of the input and output (model size).
        h (int): The number of attention heads.
        dropout (float): The dropout rate applied during attention computation.

    Forward Input:
        q (Tensor): Query tensor of shape (batch_size, seq_len, d_model).
        k (Tensor): Key tensor of shape (batch_size, seq_len, d_model).
        v (Tensor): Value tensor of shape (batch_size, seq_len, d_model).
        mask (Tensor, optional): Mask tensor to avoid attending to padding positions.

    Forward Output:
        Tensor: Output tensor of shape (batch_size, seq_len, d_model) after attention.
    
    Functionality:
        - Splits input into multiple heads (h).
        - Computes scaled dot-product attention for each head.
        - Concatenates the heads' outputs and applies a final linear projection.
    """
    def __init__(self, d_model, h, dropout):
        super().__init__()
        self.d_model =d_model
        self.h = h
        assert d_model % h == 0, "d_modle is not divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1] # d_k = dimension of each head (derived from d_model)

        # Compute the dot product between query and key, scaled by √d_k
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)

        # Apply the mask (if provided) to avoid attending to certain positions
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1)

        # Apply dropout to the attention scores for regularization
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores


    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch sq_len, d_model) --> (batch, sq_len, d_model)
        key = self.w_k(k) # (batch sq_len, d_model) --> (batch, sq_len, d_model)
        value = self.w_v(v) # (batch sq_len, d_model) --> (batch, sq_len, d_model)

        # (batch, sq_len, d_model) --> (batch, sq_len, h, d_k) --> (batch, h, sq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)