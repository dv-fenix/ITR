from config import preEncDec
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
from torch import Tensor
import pytorch_lightning as pl
import copy
from eng_utils import gen_model_loaders, flat_accuracy, save_model, build_model


class itr_Net(pl.LightningModule):
    def __init__(self, hparams):
        super(itr_Net, self).__init__()
        self.config = preEncDec
        self.encoder, self.decoder, _ = build_model(self.config)
        self.train_loss = 0
        self.eval_loss = 0
        self.eval_acc = 0
        self.avg_train_loss = 0
        self.avg_val_loss = 0
        self.avg_val_acc = 0
        self.epoch = 0
        self.eval_steps = 0
        self.train_loader, self.eval_loader = gen_model_loaders(self.config)

    def forward(self, encoder_input_ids, decoder_input_ids, testing=False):

        if testing:
            encoder_hidden_states = self.encoder(encoder_input_ids)[0]
            logits_tensor = self.decoder(decoder_input_ids, encoder_hidden_states=encoder_hidden_states)
            return logits_tensor
        '''
        encoder_arr, decoder_arr = self.convert_to_list(encoder_input_ids, decoder_input_ids)
        encoder_hidden_states = self.encoder(encoder_input_ids)[0]
        loss, logits = self.decoder(decoder_input_ids,
                                    encoder_hidden_states=encoder_hidden_states,
                                    masked_lm_labels=decoder_input_ids)

        self.loss = loss
        self.logits = logits
        return loss, logits
        '''

        # iteratively add training loss for each decoder_id
        total_loss = 0

        # array to store word logits
        final_sent_logits = []

        # loops over each word: lm_labels-> lm_tensor -> decoder outputs[loss, logits]
        for j in range(len(decoder_arr[0])):
            label_ids = copy.deepcopy(decoder_arr)  # create a copy of the decoder_arr for masking
            for i in range(len(decoder_arr)):
                if j == 0:
                    continue
                label_ids[i][j:] = [-100] * (len(label_ids[0]) - j)  # generate label_ids
            if j == 0:
                continue

            label_ids = np.array(label_ids)  # convert to array
            label_tensor = torch.tensor(label_ids).to('cuda')  ##update tensor at each step

            loss, logits = self.decoder(label_tensor,
                                        encoder_hidden_states=encoder_hidden_states,
                                        lm_labels=decoder_input_ids)

            logits_arr = logits.cpu().tolist()
            # iteratively adds word logits
            final_sent_logits.append(logits_arr)
            total_loss = total_loss + loss  ##iteratively add loss

        # normalize loss by dividing by total iterations
        norm_loss = total_loss / len(decoder_arr[0])
        # convert to tensor[final sentence in array]
        logits_tensor = torch.FloatTensor(final_sent_logits) #dimension: [sequence length - 1, batch_size, sequence_length, vocab_size]

        #self.loss = norm_loss
        #self.logits = logits_tensor

        return norm_loss, logits_tensor

    def convert_to_list(self, encoder_inp, decoder_inp):
        encoder_arr = encoder_inp.cpu().tolist()
        decoder_arr = decoder_inp.cpu().tolist()

        return encoder_arr, decoder_arr

    def save(self, output):
        save_model(self.encoder, output.encoder)
        save_model(self.decoder, output.decoder)

    def pad_seq(self):
        pad_sequence = PadSequence(self.tokenizers.src.pad_token_id, self.tokenizers.tgt.pad_token_id)

        return pad_sequence

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.eval_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        return optimizer

    def at_epoch_end(self, current_epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader), eta_min=self.config.lr)
        lr_scheduler.step()

    def training_step(self, batch, batch_nb):
        source, target = batch
        loss, _ = self.forward(source, target)
        self.train_loss += loss.item()
        tensorboard_logs = {'loss': loss}

        return {'loss': loss, 'logs': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        source, target = batch
        with torch.no_grad():
            loss, logits = self.forward(source, target)
        self.eval_loss += loss
        logits = logits.cpu().numpy()
        label_ids = target.cpu().numpy()
        '''
        eval_accuracy = 0

        for i, item in enumerate(logits):
            eval_accuracy = eval_accuracy + flat_accuracy(item, label_ids)

        tmp_eval_accuracy = eval_accuracy / logits.shape[0]
        self.eval_acc += tmp_eval_accuracy
        '''

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        self.eval_acc += tmp_eval_accuracy

        tensorboard_logs = {'eval_loss': self.eval_loss, 'eval_acc': self.eval_acc}

        return {'eval_loss': self.eval_loss, 'eval_acc': self.eval_acc, 'logs': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        self.epoch += 1
        current_epoch = self.epoch - 1
        self.avg_val_acc = self.eval_acc / len(eval_loader)
        self.avg_val_loss = self.eval_loss / len(eval_loader)

        self.avg_train_loss = self.train_loss / len(train_loader)
        writer.add_scalar('Avg_Train_loss', self.avg_train_loss, current_epoch)
        writer.add_scalar('Avg_Validation_Acc', self.avg_val_acc, current_epoch)
        writer.add_scalar('Avg_Validation_Loss', self.avg_val_loss, current_epoch)
        for weights in self.parameters():
            writer.add_histogram('Weights', weights, current_epoch)

        self.train_loss = 0
        self.eval_acc = 0
        self.eval_loss = 0

        tensorboard_logs = {'avg_valid_loss': self.avg_val_loss, 'avg_valid_acc': self.avg_val_acc,
                            'avg_train_loss': self.avg_train_loss}
        return {'avg_valid_loss': self.avg_val_loss, 'avg_valid_acc': self.avg_val_acc,
                'avg_train_loss': self.avg_train_loss, 'logs': tensorboard_logs}




