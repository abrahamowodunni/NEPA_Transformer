import torch
from src.model.transformer import build_tranformer  # Import your transformer architecture
from src.config.config import get_config
import json

def load_model(config, vocab_src_len, vocab_tgt_len, weights_path):
    # Rebuild the model architecture
    model = build_tranformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    
    # Load the saved weights
    model.load_state_dict(torch.load(weights_path)['model_state_dict'])
    
    # Set the model to evaluation mode
    model.eval()
    return model

def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'r') as file:
        vocab = json.load(file)
    return vocab

def tokenize_sentence(sentence, tokenizer):
    return [tokenizer.get(token, tokenizer.get('[UNK]')) for token in sentence.split()]

def detokenize_sentence(tokens, reverse_tokenizer):
    return ' '.join([reverse_tokenizer.get(token, '[UNK]') for token in tokens])

def run_inference(sentence, model, tokenizer_src, tokenizer_tgt, device='cpu'):
    # Tokenize input sentence
    input_tokens = tokenize_sentence(sentence, tokenizer_src)
    input_tensor = torch.tensor(input_tokens).unsqueeze(0).to(device)  # Add batch dimension

    # Generate attention mask (all 1s since no padding)
    input_mask = torch.ones_like(input_tensor).to(device)

    # Encode the input sentence
    with torch.no_grad():
        encoder_output = model.encode(input_tensor, input_mask)
    
    # Initialize the decoder input with the start token
    start_token = tokenizer_tgt.get('[SOS]', 0)  # Assuming '<sos>' is the start token
    decoder_input = torch.tensor([[start_token]]).to(device)

    # List to store the output tokens
    output_tokens = []

    # Generate the translation using greedy decoding (for simplicity)
    for _ in range(config['seq_len']):  # Limit translation to `seq_len` tokens
        decoder_mask = torch.ones_like(decoder_input).to(device)
        decoder_output = model.decode(encoder_output, input_mask, decoder_input, decoder_mask)
        proj_output = model.project(decoder_output)
        
        # Get the token with the highest probability (greedy decoding)
        next_token = torch.argmax(proj_output[:, -1, :], dim=-1).item()

        # Break if the end token is generated
        if next_token == tokenizer_tgt.get('[EOS]', 1):  # Assuming '<eos>' is the end token
            break
        
        # Append the predicted token to the output tokens
        output_tokens.append(next_token)

        # Append the predicted token to the decoder input
        decoder_input = torch.cat([decoder_input, torch.tensor([[next_token]]).to(device)], dim=1)
    
    # Detokenize the output sentence using output_tokens
    reverse_tokenizer = {v: k for k, v in tokenizer_tgt.items()}
    translated_sentence = detokenize_sentence(output_tokens, reverse_tokenizer)  # Skip the start token

    return translated_sentence

if __name__ == '__main__':
    # Load config, tokenizers, and weights
    config = get_config()
    vocab_src = load_tokenizer(config['tokenizer_src_path'])
    vocab_tgt = load_tokenizer(config['tokenizer_tgt_path'])
    
    # Load the model
    weights_path = "artifacts/weights/NEPA_eng_yor_t16.pt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(config, len(vocab_src), len(vocab_tgt), weights_path)
    model.to(device)  # You can switch to 'cuda' if you have GPU support

    # Input sentence in English
    input_sentence = "how are you"
    
    # Run inference
    translated_sentence = run_inference(input_sentence, model, vocab_src, vocab_tgt, device)
    
    print(f"Input: {input_sentence}")
    print(f"Translated: {translated_sentence}")
