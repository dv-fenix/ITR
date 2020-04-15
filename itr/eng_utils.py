from config import preEncDec
from data import IndicDataset, PadSequence, TestDataset, TestSequence
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import BertConfig, BertModel, BertForMaskedLM, BertTokenizer
from data import IndicDataset, PadSequence
from easydict import EasyDict as ED
from pathlib import Path
from transformers import WEIGHTS_NAME, CONFIG_NAME
import torch


def gen_model_loaders(config):
    _, _, tokenizers = build_model(config)

    pad_sequence = PadSequence(tokenizers.src.pad_token_id, tokenizers.tgt.pad_token_id)

    train_loader = DataLoader(IndicDataset(tokenizers.src, tokenizers.tgt, config.data, True),
                              batch_size=config.batch_size,
                              shuffle=False,
                              collate_fn=pad_sequence)
    eval_loader = DataLoader(IndicDataset(tokenizers.src, tokenizers.tgt, config.data, False),
                             batch_size=config.eval_size,
                             shuffle=False,
                             collate_fn=pad_sequence)
    return train_loader, eval_loader


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    #print (f'preds: {pred_flat}')
    #print (f'labels: {labels_flat}')
    num = 0.
    num = np.sum(np.equal(pred_flat, labels_flat)) / len(labels_flat)
    return num



def build_model(config):
    src_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    tgt_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    tgt_tokenizer.bos_token = '<s>'
    tgt_tokenizer.eos_token = '</s>'

    # hidden_size and intermediate_size are both wrt all the attention heads.
    # Should be divisible by num_attention_heads
    encoder_config = BertConfig(vocab_size=src_tokenizer.vocab_size,
                                hidden_size=config.hidden_size,
                                num_hidden_layers=config.num_hidden_layers,
                                num_attention_heads=config.num_attention_heads,
                                intermediate_size=config.intermediate_size,
                                hidden_act=config.hidden_act,
                                hidden_dropout_prob=config.dropout_prob,
                                attention_probs_dropout_prob=config.dropout_prob,
                                max_position_embeddings=512,
                                type_vocab_size=2,
                                initializer_range=0.02,
                                layer_norm_eps=1e-12)

    decoder_config = BertConfig(vocab_size=tgt_tokenizer.vocab_size,
                                hidden_size=config.hidden_size,
                                num_hidden_layers=config.num_hidden_layers,
                                num_attention_heads=config.num_attention_heads,
                                intermediate_size=config.intermediate_size,
                                hidden_act=config.hidden_act,
                                hidden_dropout_prob=config.dropout_prob,
                                attention_probs_dropout_prob=config.dropout_prob,
                                max_position_embeddings=512,
                                type_vocab_size=2,
                                initializer_range=0.02,
                                layer_norm_eps=1e-12,
                                is_decoder=True)

    # Create encoder and decoder embedding layers.
    encoder_embeddings = torch.nn.Embedding(src_tokenizer.vocab_size, config.hidden_size,
                                            padding_idx=src_tokenizer.pad_token_id)
    decoder_embeddings = torch.nn.Embedding(tgt_tokenizer.vocab_size, config.hidden_size,
                                            padding_idx=tgt_tokenizer.pad_token_id)

    encoder = BertModel(encoder_config)
    encoder.set_input_embeddings(encoder_embeddings.cuda())

    decoder = BertForMaskedLM(decoder_config)
    decoder.set_input_embeddings(decoder_embeddings.cuda())
    tokenizers = ED({'src': src_tokenizer, 'tgt': tgt_tokenizer})

    return encoder, decoder, tokenizers

def save_model(model, output_dir):

    output_dir = Path(output_dir)
    # Step 1: Save a model, configuration and vocabulary that you have fine-tuned

    # If we have a distributed model, save only the encapsulated model
    # (it was wrapped in PyTorch DistributedDataParallel or DataParallel)
    model_to_save = model.module if hasattr(model, 'module') else model

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = output_dir / WEIGHTS_NAME
    output_config_file = output_dir / CONFIG_NAME

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    #src_tokenizer.save_vocabulary(output_dir)