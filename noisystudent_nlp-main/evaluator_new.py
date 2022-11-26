from __future__ import print_function, division
import torch
from torch.nn.functional import threshold
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from util.utils import *
import numpy as np

class Evaluator(object):
    """ Class to evaluate models with given datasets.
    """
    def __init__(self, loss,logger, config):
        self.loss = loss
        self.logger= logger
        self.config = config
        self.batch_size = self.config.valid_batch_size
        self.device = self.config.device
    def calculate_accu(self, big_idx, targets):
        n_correct = (big_idx==targets).sum().item()
        return n_correct
    
    def infer(self,model,data_loader):
        with torch.no_grad():
            for _, batch in enumerate(data_loader):
                ids = batch['input_ids'].to(self.device, dtype=torch.long)
                attention_mask = batch['attention_mask'].to(self.device, dtype=torch.long)
                if self.config.pretrained_model == "bert":
                    token_type_ids = batch['token_type_ids'].to(self.device, dtype=torch.long)
                targets = batch['labels'].to(self.device, dtype=torch.long)
                if self.config.pretrained_model == "bert":
                    outputs = model(ids, attention_mask, token_type_ids, labels=targets)
                else:
                    outputs = model(ids, attention_mask, labels=targets)
                loss, logits = outputs[0], outputs[1]
    
    def evaluate(self, model, data_loader, is_test=False, return_details=False):
        """ Evaluate a model on given dataset and return performance.
        Args:
            model: model to evaluate
            data: dataset to evaluate against
        Returns:
            loss (float): loss of the given model on the given dataset
        """
        loss = self.loss
        model.eval()
        
        eval_loss = 0
        nb_eval_steps = 0
        nb_eval_examples = 0
        n_correct, n_total = 0, 0
        precision_ls, recall_ls, f1_ls, support_ls = [], [], [], []
        pred_ls, target_ls = np.array([]), np.array([])
        ids_ls = np.array([])
        with torch.no_grad():
            for _, (src,noisy_src, targets,inds,ori_targets) in enumerate(data_loader):
                ids = torch.tensor(src['input_ids']).to(self.device, dtype=torch.long)
                attention_mask = torch.tensor(src['attention_mask']).to(self.device, dtype=torch.long)
                if self.config.pretrained_model == "bert":
                    token_type_ids = torch.tensor(src['token_type_ids']).to(self.device, dtype=torch.long)
                targets = ori_targets.to(self.device, dtype=torch.long)

                if self.config.pretrained_model == "bert":
                    outputs = model(ids, attention_mask, token_type_ids, labels=targets)
                else:
                    outputs = model(ids, attention_mask, labels=targets)
                loss, logits = outputs[0], outputs[1]
                
                eval_loss += loss.item()
                big_val, big_idx = torch.max(logits.data, dim=-1) # [bsz,1]
                n_correct += self.calculate_accu(big_idx, targets)
                
                nb_eval_steps += 1
                nb_eval_examples += targets.size(0)
                target_ls = np.append(target_ls, targets.cpu().numpy())
                pred_ls = np.append(pred_ls, big_idx.cpu().numpy())
                ids_ls = np.append(ids_ls, inds.cpu().numpy())
                # if _ % 1000 == 0:
                #     loss_step = eval_loss / nb_eval_steps
                #     accu_step = (n_correct)/nb_eval_examples
                #     if is_test == True:
                #         self.logger.info(f"Test Loss per 1000 steps: {loss_step}")
                #         self.logger.info(f"Test Accuracy per 1000 steps: {accu_step}")
                #     else:
                #         self.logger.info(f"Validation Loss per 1000 steps: {loss_step}")
                #         self.logger.info(f"Validation Accuracy per 1000 steps: {accu_step}")
            
        epoch_loss = eval_loss / nb_eval_steps
        epoch_accu = (n_correct) / nb_eval_examples
        target_arr, pred_arr, ids_arr = target_ls.flatten(), pred_ls.flatten(), ids_ls.flatten()
        # print(target_arr.shape)
        # print(target_arr)
        # print(pred_arr.shape)
        precision, recall, f1_score, support = precision_recall_fscore_support(target_arr, pred_arr, average='weighted')

        if is_test == True:
            self.logger.info(f"Test Loss Epoch: {epoch_loss}")
            self.logger.info(f"Test Accuracy Epoch: {epoch_accu}")
            self.logger.info(f"Test F1-score Epoch: {f1_score}")
            self.logger.info(f"Test Precision Epoch: {precision}")
            self.logger.info(f"Test Recall Epoch: {recall}")
        else:
            self.logger.info(f"Validation Loss Epoch: {epoch_loss}")
            self.logger.info(f"Validation Accuracy Epoch: {epoch_accu}")
            self.logger.info(f"Validation F1-score Epoch: {f1_score}")
            self.logger.info(f"Validation Precision Epoch: {precision}")
            self.logger.info(f"Validation Recall Epoch: {recall}")
        if return_details == True:
            return epoch_loss, epoch_accu, list(zip(pred_arr, target_arr, ids_arr))
        else:
            return epoch_loss, epoch_accu
