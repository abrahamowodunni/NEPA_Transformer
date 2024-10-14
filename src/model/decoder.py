import torch
import torch.nn as nn 
import math

class DecoderBlock(nn.Module):
    """
    Implements a single block of the Transformer decoder, consisting of self-attention, cross-attention, and feed-forward layers with residual connections.

    Attributes:
        self_attention_block (MultiHeadAttentionBlock): Self-attention mechanism for the target sequence.
        cross_attention_block (MultiHeadAttentionBlock): Attention mechanism between the target sequence and the encoder's output (cross-attention).
        feed_forward_block (FeedForwardBlock): Feed-forward layer for further transformations.
        residual_connections (nn.ModuleList): A list containing three residual connections for self-attention, cross-attention, and feed-forward blocks.

    Methods:
        forward(x, encoder_output, src_mask, tgt_mask): Passes the input through self-attention, cross-attention, and feed-forward blocks with residual connections.
    """
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block:MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.cross_attention_block = cross_attention_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
    """
    Decoder module consisting of multiple layers of decoder blocks, each containing self-attention, 
    cross-attention with the encoder output, and feed-forward layers. It includes residual connections 
    and normalization at each step.

    Args:
        layers (nn.Module): A list of DecoderBlock modules that will process the input sequentially.
    
    Methods:
        forward(x, encoder_output, src_mask, tgt_mask):
            Performs the forward pass through each decoder block, applying self-attention and cross-attention
            with the encoder output. Returns the normalized output after all layers are applied.

    Inputs:
        x (Tensor): Input sequence to the decoder (target sequence).
        encoder_output (Tensor): Output from the encoder (used in cross-attention).
        src_mask (Tensor): Mask for the source sequence (used in cross-attention to mask padding or unwanted tokens).
        tgt_mask (Tensor): Mask for the target sequence (used in self-attention to prevent attending to future tokens).

    Output:
        Tensor: The final output from the decoder after processing through all layers and normalization.
    """
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)