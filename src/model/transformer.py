import torch
import torch.nn as nn 
import math

class Transformer(nn.Module):
    """
    A Transformer model consisting of an encoder and a decoder with input embeddings, 
    positional encodings, and a projection layer to produce vocabulary-level predictions.

    Args:
        encoder (Encoder): The encoder module of the transformer.
        decoder (Decoder): The decoder module of the transformer.
        src_embedding (InputEmbeddings): The embedding layer for the source input.
        tgt_embedding (InputEmbeddings): The embedding layer for the target input.
        src_position (PositionalEncoding): Positional encoding for the source input.
        tgt_position (PositionalEncoding): Positional encoding for the target input.
        projection_layer (ProjectionLayer): A linear layer that projects decoder outputs 
            to a probability distribution over the vocabulary.
    
    Methods:
        encode(src, src_mask):
            Encodes the source input sequence by embedding it, adding positional encodings, 
            and passing it through the encoder.

        decode(encoder_output, src_mask, tgt, tgt_mask):
            Decodes the target sequence by embedding it, adding positional encodings, 
            and passing it through the decoder with the encoder output.

        project(x):
            Projects the decoder output to a vocabulary-level probability distribution.
    
    Inputs:
        - `src`: Source sequence (batch, seq_len) for encoding.
        - `src_mask`: Mask for the source sequence.
        - `tgt`: Target sequence (batch, seq_len) for decoding.
        - `tgt_mask`: Mask for the target sequence.
    
    Outputs:
        - Log-probabilities of tokens for each position in the target sequence.
    """
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embedding: InputEmbeddings, tgt_embedding: InputEmbeddings, src_position: PositionalEncoding, tgt_position: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed= src_embedding
        self.tgt_embed = tgt_embedding
        self.src_pos = src_position
        self.tgt_pos = tgt_position
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src,src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    
def build_tranformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model = 512, N = 6, h = 8, dropout = 0.1, d_ff = 2048) -> Transformer:
    """
    Build a complete Transformer model for sequence-to-sequence tasks such as translation or text generation.

    Args:
        src_vocab_size (int): Size of the source vocabulary.
        tgt_vocab_size (int): Size of the target vocabulary.
        src_seq_len (int): Maximum length of the source sequences.
        tgt_seq_len (int): Maximum length of the target sequences.
        d_model (int, optional): Dimensionality of the embeddings and hidden layers. Default is 512.
        N (int, optional): Number of layers (blocks) in the encoder and decoder. Default is 6.
        h (int, optional): Number of attention heads in the multi-head attention mechanism. Default is 8.
        dropout (float, optional): Dropout probability. Default is 0.1.
        d_ff (int, optional): Dimensionality of the feed-forward network. Default is 2048.

    Returns:
        Transformer: A fully assembled Transformer model.

    """
    # embedding layer
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # positional encoding
    src_pos = PositionalEncoding(d_model, src_seq_len,dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # encoder block
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    # decoder block
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # create the projection LayerNormalization
    project_layer = ProjectionLayer(d_model,tgt_vocab_size)

    # creating the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, project_layer)

    # initialization of parmeters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)
    return transformer
