class EncoderBlock(nn.Module):
    """
    Encoder block consisting of a self-attention mechanism and a feed-forward network with residual connections.

    Args:
        self_attention_block (MultiHeadAttentionBlock): Self-attention module that allows the input to attend to itself.
        feed_forward_block (FeedForwardBlock): Feed-forward network that transforms the input independently at each position.
        dropout (float): Dropout probability for regularizing the sublayers.

    Methods:
        forward(x, src_mask):
            Applies self-attention with a residual connection, followed by a feed-forward block with another residual connection.
    
    Returns:
        Tensor: The output after passing through the self-attention, feed-forward layers, and residual connections.
    """
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    """
    Implements the encoder stack of the Transformer model.

    Attributes:
        layers (nn.ModuleList): A list of stacked encoder blocks (self-attention + feed-forward + residual connections).
        norm (LayerNormalization): Layer normalization applied after the entire stack of encoder blocks.

    Methods:
        forward(x, mask): Passes the input through each encoder block, applying layer normalization to the final output.
    """
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)