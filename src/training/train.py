import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from src.training.utils import get_ds, get_model, run_validation
from src.model.transformer import build_tranformer
from src.config import get_config, get_weights_file_path
from tqdm import tqdm
import json

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_tranformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def save_tokenizers(tokenizer_src, tokenizer_tgt, config):
    with open(config['tokenizer_src_path'], 'w') as f_src:
        json.dump(tokenizer_src.get_vocab(), f_src)
    with open(config['tokenizer_tgt_path'], 'w') as f_tgt:
        json.dump(tokenizer_tgt.get_vocab(), f_tgt)

def train_model(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device {device}')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    save_tokenizers(tokenizer_src, tokenizer_tgt, config)

    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch, global_step = 0, 0
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    validation_interval = config['validation_interval']

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        steps_in_epoch = 0  # Step counter for the current epoch

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            label = batch['label'].to(device)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            steps_in_epoch += 1

            # Run validation after every `validation_interval` steps
            if steps_in_epoch % validation_interval == 0:
                run_validation(
                    model, val_dataloader, tokenizer_src, tokenizer_tgt, 
                    config['seq_len'], device, 
                    lambda msg: batch_iterator.write(msg), global_step, writer
                )

        # Run validation at the end of the epoch as well
        run_validation(
            model, val_dataloader, tokenizer_src, tokenizer_tgt, 
            config['seq_len'], device, 
            lambda msg: batch_iterator.write(msg), global_step, writer
        )

        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
