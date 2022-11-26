import os
import torch
from evaluator_new import Evaluator
from torch.utils.data import  DataLoader
from torch.nn import CrossEntropyLoss, MSELoss
from util.early_stopping import EarlyStopping
from transformers import BertTokenizer, BertForSequenceClassification
from nltk.stem import WordNetLemmatizer
from util.datasetxl_noisy import *
from util.utils import *
from collections import Counter
import torch.nn.functional as F
import time
import random
import copy
from sklearn.metrics import precision_recall_fscore_support,roc_auc_score,roc_curve

NUMERICAL_EPS = 1e-12

class Trainer(object):
    def __init__(self, config, model,logger, criterion, optimizer, 
                 save_path, train_dataset,dev_dataset,meta_dataset,total_dev_dataset,unlabel_dataset, test_dataset,label_mapping,model_type,noisy_datas):
        self.config = config
        self.logger = logger
        self.loss = criterion
        self.trec = False
        if self.trec:
            self.loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.1,0.1,0.1,0.2,0.15,0.15])).to(self.device)
        self.evaluator = Evaluator(loss=self.loss,logger=self.logger,config=self.config)
        self.optimizer = optimizer
        self.device = self.config.device
        self.model = model.to(self.device)
        self.global_best_model_ls = []
        self.global_best_model = None
        self.best_meta_model = None
        self.best_meta_thresh = None
        self.early_stop_model = None
        self.early_stop_epoch = self.config.hist_len
        self.model_type = model_type
        
        

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.lemmatizer = WordNetLemmatizer()
        self.train_dataset = train_dataset
        self.label_dataset = deepcopy(train_dataset)
        self.dev_dataset = dev_dataset
        self.meta_dataset = meta_dataset
        self.test_dataset = test_dataset
        self.total_dev_dataset = total_dev_dataset
        self.unlabel_dataset = unlabel_dataset
        self.best_dev_loss_ls = []
        self.best_dev_acc_ls = []
        self.best_dev_loss = float('inf')
        self.best_dev_acc = -1
        self.best_test_acc_ls = []
        self.best_test_acc = -1
        self.global_best_epoch_ls = []
        self.global_best_epoch = None
        if self.config.pretrained_model == "bert":
            collate_fn_ = customize_collate_fn
        self.train_loader = DataLoader(
                            dataset=self.train_dataset,
                            batch_sampler=BatchSamplerXL(
                                data_source=self.train_dataset,
                                mode='train',
                                max_bsz=self.config.train_batch_size,
                                max_token_len=self.config.train_max_token_len,
                                shuffle = True),
                            collate_fn=collate_fn_,
                            num_workers=2,
                            pin_memory=True,
                        )
        self.meta_loader = DataLoader(
                            dataset=self.meta_dataset,
                            batch_sampler=BatchSamplerXL(
                                data_source=self.meta_dataset,
                                mode='train',
                                max_bsz=self.config.valid_batch_size,
                                max_token_len=self.config.valid_max_token_len,
                                shuffle = True),
                            collate_fn=collate_fn_,
                            num_workers=2,
                            pin_memory=True,
                        )
        self.unlabel_loader = DataLoader(
                            dataset=self.unlabel_dataset,
                            batch_sampler=BatchSamplerXL(
                                data_source=self.unlabel_dataset,
                                mode='train',
                                max_bsz=self.config.valid_batch_size,
                                max_token_len=self.config.valid_max_token_len,
                                shuffle = True),
                            collate_fn=collate_fn_,
                            num_workers=2,
                            pin_memory=True,
                        )
        self.valid_loader = DataLoader(
                            dataset=self.dev_dataset,
                            batch_sampler=BatchSamplerXL(
                                data_source=self.dev_dataset,
                                mode='dev',
                                max_bsz=self.config.valid_batch_size,
                                max_token_len=self.config.valid_max_token_len,
                                shuffle = False),
                            collate_fn=collate_fn_,
                            num_workers=2,
                            pin_memory=True,
                        )
        self.total_valid_loader = DataLoader(
                            dataset=self.total_dev_dataset,
                            batch_sampler=BatchSamplerXL(
                                data_source=self.total_dev_dataset,
                                mode='dev',
                                max_bsz=self.config.valid_batch_size,
                                max_token_len=self.config.valid_max_token_len,
                                shuffle = False),
                            collate_fn=collate_fn_,
                            num_workers=2,
                            pin_memory=True,
                        )
        self.test_loader = DataLoader(
                            dataset=self.test_dataset,
                            batch_sampler=BatchSamplerXL(
                                data_source=self.test_dataset,
                                mode='dev',
                                max_bsz=self.config.valid_batch_size,
                                max_token_len=self.config.valid_max_token_len,
                                shuffle = False),
                            collate_fn=collate_fn_,
                            num_workers=2,
                            pin_memory=True,
                        )
        self.noisy_datas = noisy_datas
        self.noisy_train = noisy_datas[0]



        self.label_mapping = label_mapping
        self.max_uncertainty = -np.log(1.0/float(len(self.label_mapping)))
        
        self.early_stopping = None
        
        self.save_path = save_path
        self.sup_path = self.save_path +'/sup'
        self.ssl_path = self.save_path +'/ssl'

        if not os.path.isabs(self.sup_path):
            self.sup_path = os.path.join(os.getcwd(), self.sup_path)
        if not os.path.exists(self.sup_path):
            os.makedirs(self.sup_path)
        
        if not os.path.isabs(self.ssl_path):
            self.ssl_path = os.path.join(os.getcwd(), self.ssl_path)
        if not os.path.exists(self.ssl_path):
            os.makedirs(self.ssl_path)

        
    def calculate_accu(self, big_idx, targets):
        n_correct = (big_idx==targets).sum().item()
        return n_correct
    
        
    def train_epoch(self, epoch, noisy=None):
        tr_loss = 0
        n_correct = 0
        nb_tr_steps = 0
        nb_tr_examples = 0
        self.model.train()
        for idx, (src,noisy_src,targets,inds,ori_targets) in enumerate(self.train_loader):
            if noisy:
                ids,attention_mask,token_type_ids = self.get_inputs(noisy_src)
            else:
                ids,attention_mask,token_type_ids = self.get_inputs(src)
            targets = targets.to(self.device, dtype=torch.long)
            outputs = self.model(ids, attention_mask, token_type_ids, labels=targets)
            loss, logits = outputs[0], outputs[1]          
            tr_loss += loss.item()
            scores = torch.softmax(logits, dim=-1)
            big_val, big_idx = torch.max(scores.data, dim=-1)
            n_correct += self.calculate_accu(big_idx, targets)
            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step()

            # if idx % 2 == 0:
            #     dev_loss, dev_acc = self.evaluator.evaluate(self.model, self.total_valid_loader)
            #     self.logger.info(f"Training Epoch {epoch},batch {idx},Dev Loss {dev_loss},Dev Acc is {dev_acc}")

        self.logger.info(f"max # batches are {idx + 1}")
        epoch_loss = tr_loss/nb_tr_steps
        epoch_accu = (n_correct)/nb_tr_examples
        self.logger.info(f"Training Epoch {epoch},Training Loss: {epoch_loss}, Training ACC: {epoch_accu}")


    def train(self,total_epochs,current_outer_epoch,save=True,early_stop=False,need_test = False,need_break = False):
        if self.config.main_target == "acc":
            self.early_stopping = EarlyStopping(patience=self.config.patience, verbose=True,mode='-')
        else:
            self.early_stopping = EarlyStopping(patience=self.config.patience, verbose=True,mode='-')
        self.early_stop_epoch = self.config.hist_len

        for epoch in range(total_epochs):
            self.train_epoch(epoch)
            dev_loss, dev_acc = self.evaluator.evaluate(self.model, self.total_valid_loader) # change
            self.logger.info(f"Training Epoch {epoch},Dev Loss {dev_loss},Dev Acc is {dev_acc}")
            
            if self.config.main_target == "loss" and self.best_dev_loss > dev_loss and not self.early_stopping.early_stop: 
                self.best_dev_loss = dev_loss
                self.best_dev_acc = dev_acc
                self.global_best_model = deepcopy(self.model)
                self.global_best_epoch = (current_outer_epoch,epoch)
                torch.save({'model_state_dict':self.model.state_dict(),
                                    'optimizer_state_dict':self.optimizer.state_dict(),'epoch':epoch},
                                   self.sup_path +f'/global_best.pt')
                if need_test:
                    test_loss, self.best_test_acc, pred_target_ids = self.evaluator.evaluate(self.model, self.test_loader, is_test=True, return_details=True)
            if self.config.main_target == "acc" and self.best_dev_acc < dev_acc and not self.early_stopping.early_stop:
                self.best_dev_loss = dev_loss
                self.best_dev_acc = dev_acc
                self.global_best_model = deepcopy(self.model)
                self.global_best_epoch = (current_outer_epoch,epoch)
                torch.save({'model_state_dict':self.model.state_dict(),
                                    'optimizer_state_dict':self.optimizer.state_dict(),'epoch':epoch},
                                   self.sup_path +f'/global_best.pt')
                                   
                    
                if need_test:
                    test_loss, self.best_test_acc, pred_target_ids = self.evaluator.evaluate(self.model, self.test_loader, is_test=True, return_details=True)
            if need_test and epoch % 5 == 0:
                tmp_test_loss, tmp_test_acc, tmp_pred_target_ids = self.evaluator.evaluate(self.model, self.test_loader, is_test=True, return_details=True)
                self.logger.info(f"Regualarly Test Epoch (outer {current_outer_epoch} inner {epoch}),tmp_test_loss {tmp_test_loss}, tmp_test_ACC is {tmp_test_acc}")
            if save:
                torch.save({'model_state_dict':self.model.state_dict(),
                                    'optimizer_state_dict':self.optimizer.state_dict(),'epoch':epoch},
                                   self.sup_path +f'/checkpoint_{epoch}.pt')
            if early_stop and self.early_stop_model is None:
                if self.config.main_target == "acc":
                    self.early_stopping(dev_loss,self.logger) # dev_acc
                else:
                    self.early_stopping(dev_loss,self.logger)
                # self.early_stopping(dev_loss,self.logger)
                if self.early_stopping.early_stop:
                    self.logger.info("Early Stopping!")
                    self.early_stop_model = deepcopy(self.model)
                    self.early_stop_epoch = epoch + 1
                    if need_break:
                        break
        
        self.best_dev_loss_ls.append(self.best_dev_loss)
        self.best_dev_acc_ls.append(self.best_dev_acc)
        self.best_test_acc_ls.append(self.best_test_acc)
        self.global_best_model_ls.append(self.global_best_model)
        self.global_best_epoch_ls.append(self.global_best_epoch)
        self.logger.info(f"train done, global Best Epoch (outer {self.global_best_epoch[0]} inner {self.global_best_epoch[1]}),global Best Dev Loss is {self.best_dev_loss}, corresponding global Best Dev Acc is {self.best_dev_acc},corresponding global Best Test Acc is {self.best_test_acc}")


    def train_noisy(self,total_epochs,current_outer_epoch,save=True,early_stop=False,need_test = False,need_break = False):
        if self.config.main_target == "acc":
            self.early_stopping = EarlyStopping(patience=self.config.patience, verbose=True,mode='-')
        else:
            self.early_stopping = EarlyStopping(patience=self.config.patience, verbose=True,mode='-')
        self.early_stop_epoch = self.config.hist_len

        for epoch in range(total_epochs):
            self.train_epoch(epoch,noisy=True)
            dev_loss, dev_acc = self.evaluator.evaluate(self.model, self.total_valid_loader) # change
            self.logger.info(f"Training Epoch {epoch},Dev Loss {dev_loss},Dev Acc is {dev_acc}")
            
            if self.config.main_target == "loss" and self.best_dev_loss > dev_loss and not self.early_stopping.early_stop: 
                self.best_dev_loss = dev_loss
                self.best_dev_acc = dev_acc
                self.global_best_model = deepcopy(self.model)
                self.global_best_epoch = (current_outer_epoch,epoch)
                torch.save({'model_state_dict':self.model.state_dict(),
                                    'optimizer_state_dict':self.optimizer.state_dict(),'epoch':epoch},
                                   self.sup_path +f'/global_best.pt')
                if need_test:
                    test_loss, self.best_test_acc, pred_target_ids = self.evaluator.evaluate(self.model, self.test_loader, is_test=True, return_details=True)
            if self.config.main_target == "acc" and self.best_dev_acc < dev_acc and not self.early_stopping.early_stop:
                self.best_dev_loss = dev_loss
                self.best_dev_acc = dev_acc
                self.global_best_model = deepcopy(self.model)
                self.global_best_epoch = (current_outer_epoch,epoch)
                torch.save({'model_state_dict':self.model.state_dict(),
                                    'optimizer_state_dict':self.optimizer.state_dict(),'epoch':epoch},
                                   self.sup_path +f'/global_best.pt')
                                   
                    
                if need_test:
                    test_loss, self.best_test_acc, pred_target_ids = self.evaluator.evaluate(self.model, self.test_loader, is_test=True, return_details=True)
            if need_test and epoch % 5 == 0:
                tmp_test_loss, tmp_test_acc, tmp_pred_target_ids = self.evaluator.evaluate(self.model, self.test_loader, is_test=True, return_details=True)
                self.logger.info(f"Regualarly Test Epoch (outer {current_outer_epoch} inner {epoch}),tmp_test_loss {tmp_test_loss}, tmp_test_ACC is {tmp_test_acc}")
            if save:
                torch.save({'model_state_dict':self.model.state_dict(),
                                    'optimizer_state_dict':self.optimizer.state_dict(),'epoch':epoch},
                                   self.sup_path +f'/checkpoint_{epoch}.pt')
            if early_stop and self.early_stop_model is None:
                if self.config.main_target == "acc":
                    self.early_stopping(dev_loss,self.logger) # dev_acc
                else:
                    self.early_stopping(dev_loss,self.logger)
                # self.early_stopping(dev_loss,self.logger)
                if self.early_stopping.early_stop:
                    self.logger.info("Early Stopping!")
                    self.early_stop_model = deepcopy(self.model)
                    self.early_stop_epoch = epoch + 1
                    if need_break:
                        break
        
        self.best_dev_loss_ls.append(self.best_dev_loss)
        self.best_dev_acc_ls.append(self.best_dev_acc)
        self.best_test_acc_ls.append(self.best_test_acc)
        self.global_best_model_ls.append(self.global_best_model)
        self.global_best_epoch_ls.append(self.global_best_epoch)
        self.logger.info(f"train done, global Best Epoch (outer {self.global_best_epoch[0]} inner {self.global_best_epoch[1]}),global Best Dev Loss is {self.best_dev_loss}, corresponding global Best Dev Acc is {self.best_dev_acc},corresponding global Best Test Acc is {self.best_test_acc}")
              

    def init_main(self,outer_epoch):
        if outer_epoch == 0:
            model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=self.config.class_num,output_hidden_states=True)
        else: # noisy student (model noise)
            # model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=self.config.class_num,output_hidden_states=True)
            model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=self.config.class_num,hidden_dropout_prob=self.config.dropout,output_hidden_states=True)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,self.config.hist_len * (len(self.train_dataset) // self.config.train_batch_size), eta_min=self.config.learning_rate * 0.01,verbose=False)
        self.loss = torch.nn.CrossEntropyLoss().to(self.device)
        self.best_dev_loss = float('inf')
        self.best_dev_acc = -1
        self.best_test_acc = -1
        self.global_best_epoch = None
        self.early_stopping = None
        self.global_best_model = None
        self.best_meta_model = None
        self.best_meta_thresh = None
        self.early_stop_model = None
        self.early_stop_epoch = self.config.hist_len


    def naive_self_train(self,noisy_type=None,early_stop=True):
        best_accuracy = -1
        selected_LD = None
        for outer_epoch in range(self.config.epochs):
            if outer_epoch >= 1 and noisy_type == 'data':
                self.model.train()
                combined_dataset,_ = self.add_dataset(self.label_dataset,selected_LD,0,None)
                self.train_loader.dataset._data = deepcopy(combined_dataset._data)
                if early_stop and self.early_stop_model is not None: #从early stop model开始续训练, 前面一个earlystop是全局的参数，后面一个是训练过程中得到的earlystop模型
                    self.model = deepcopy(self.early_stop_model)
                if self.config.training_from_scratch == True:
                    self.init_main(outer_epoch) 
                self.train_noisy(self.config.hist_len,outer_epoch,save=True,early_stop=early_stop,need_test=True,need_break=True)
                self.logger.info("######### pseudo labeling on UNLABEL dataset ############")
                self.pseudo_labeling(self.unlabel_loader,None)
                selected_LD = self.select(threshold=self.config.confidence_thres,type="naive")
            else:
                self.model.train()
                #combine selected unlabel and labeled            
                combined_dataset,_ = self.add_dataset(self.label_dataset,selected_LD,0,None)
                self.train_loader.dataset._data = deepcopy(combined_dataset._data)
                #train on combined dataset, in first epoch, this is only the label dataset
                if early_stop and self.early_stop_model is not None: #从early stop model开始续训练, 前面一个earlystop是全局的参数，后面一个是训练过程中得到的earlystop模型
                    self.model = deepcopy(self.early_stop_model)
                if self.config.training_from_scratch == True:
                    self.init_main(outer_epoch) 
                self.train(self.config.hist_len,outer_epoch,save=True,early_stop=early_stop,need_test=True,need_break=True)
                # pseudo-labeling，打伪标签，在unlabeldataset
                self.logger.info("######### pseudo labeling on UNLABEL dataset ############")
                self.pseudo_labeling(self.unlabel_loader,None)
                selected_LD = self.select(threshold=self.config.confidence_thres,type="naive")
        if self.config.training_from_scratch == True:
            if self.config.main_target == "loss":
                all_info = list(zip(self.global_best_model_ls,self.global_best_epoch_ls,
                                    self.best_dev_loss_ls,self.best_test_acc_ls))
                all_info = sorted(all_info, key=lambda x:x[2])
                self.global_best_model = all_info[0][0]
                self.global_best_epoch = all_info[0][1]
                self.best_dev_loss = all_info[0][2]
                self.best_test_acc = all_info[0][3]
                self.logger.info(f'Global best epoch: outer epoch:{self.global_best_epoch[0]} inner epoch:{self.global_best_epoch[1]} best_dev_loss:{self.best_dev_loss} best_test_acc:{self.best_test_acc}')
                torch.save({'model_state_dict':self.model.state_dict()},
                                    self.sup_path +f'/checkpoint_best.pt')
            else:
                all_info = list(zip(self.global_best_model_ls,self.global_best_epoch_ls,self.best_dev_loss_ls,
                self.best_dev_acc_ls,self.best_test_acc_ls))
                all_info = sorted(all_info, key=lambda x:(x[3],-x[2]))
                self.global_best_model = all_info[-1][0]
                self.global_best_epoch = all_info[-1][1]
                self.best_dev_acc = all_info[-1][3]
                self.best_test_acc = all_info[-1][4]
                self.logger.info(f'Global best epoch: outer epoch:{self.global_best_epoch[0]} inner epoch:{self.global_best_epoch[1]} best_dev_acc:{self.best_dev_acc} best_test_acc:{self.best_test_acc}')
                torch.save({'model_state_dict':self.model.state_dict()},
                            self.sup_path +f'/checkpoint_best.pt')
        else:
            self.logger.info('Best accuracy {}'.format(best_accuracy))
    

    def cross_entropy_sample(self, logits, labels):
        num_class = self.config.class_num
        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(logits.view(-1, num_class), labels.view(-1))
        return loss

    def pseudo_labeling(self,dataloader, guide_type=None):
        self.model.eval()
        # new_dataset = {label:[] for label in range(self.config.class_num)}
        pseudo_labeling_chancge_num = 0
        total = 0
        origin_target_crct_num = 0
        current_crct_num = 0
        new_total = 0
        after_change_correct_verify = 0
        with torch.no_grad():
            for idx,(src,noisy_src,targets,inds,ori_targets) in enumerate(dataloader):
                total += len(targets)
                if self.config.pretrained_model == "bert":
                    ids,attention_mask,token_type_ids = self.get_inputs(src)
                else:
                    ids,attention_mask = self.get_inputs(src)
                targets = targets.to(self.device, dtype=torch.long)
                ori_targets = ori_targets.to(self.device, dtype=torch.long)
                # outputs = self.model(ids, attention_mask, token_type_ids, labels=targets)
                if self.config.pretrained_model == "bert":
                    outputs = self.global_best_model(ids, attention_mask, token_type_ids, labels=targets)
                loss, logits = outputs[0], outputs[1]
                confidences = torch.softmax(logits, dim=-1)
                big_val, big_idx = torch.max(confidences.data, dim=-1)
                chancge_num = targets != big_idx
                pseudo_labeling_chancge_num += chancge_num.sum().item()
                origin_target_crct_num += (targets == ori_targets).sum().item()
                current_crct_num += (big_idx == ori_targets).sum().item()
                if guide_type is None:
                    dataloader.dataset._update_target(inds,big_idx.cpu().tolist())

        self.logger.info(f"total samples {total}, pseudo_labeling_chancge_num {pseudo_labeling_chancge_num}, origin_target_crct_num {origin_target_crct_num}, current_crct_num {current_crct_num}")
        self.logger.info(f"original accuracy {(origin_target_crct_num*100 / total):.4f}, current accuracy {(current_crct_num*100 / total):.4f}")
        

    def select(self,threshold=0.5,type=None,epoch='None'):
        self.model.eval()
        select_data = {}
        n_correct, n_total = 0, 0 
        unlabel_preds = []
        unlabel_gt = []
        with torch.no_grad():
            for idx, (src,noisy_src,targets,inds,ori_targets) in enumerate(self.unlabel_loader):
                targets = targets.to(self.device,dtype=torch.long)
                if self.config.pretrained_model == "bert":
                    ids,attention_mask,token_type_ids = self.get_inputs(src)
                    outputs = self.model(ids, attention_mask, token_type_ids, labels=targets)
                unlabel_loss, logits = outputs[0], outputs[1]
                bsz = logits.shape[0]
                unlabel_prob = F.softmax(logits,dim=-1)
                argmax_ids = torch.argmax(unlabel_prob,dim=-1)
                sift_pos = unlabel_prob[torch.arange(bsz),argmax_ids] >= threshold
                inds_list = inds[sift_pos].cpu().tolist()
                selected_num = sift_pos.sum().item()
                unlabel_preds.extend(sift_pos.long().cpu().tolist())
                    
                n_total += targets.size(0)
                for i,ind in enumerate(inds_list):
                    select_data[ind] = deepcopy(self.unlabel_loader.dataset._data[ind])
            acc = n_correct/n_total
            # precision, recall, f1_score, support = precision_recall_fscore_support(unlabel_gt, unlabel_preds,average='binary')
            # beta = 0.0625
            # fx_score = (1+beta**2)*(precision*recall) / (beta**2*precision+recall)
        return select_data


    def add_dataset(self, labeled_dataset, new_dataset=None, add_meta_ratio = 0,meta_dataset = None):
        #generat a deepcopy of label dataset, and update that copy, do not change original dataset
        if new_dataset is None:
            return labeled_dataset,None
        res_data = deepcopy(labeled_dataset._data)
        leng = len(labeled_dataset)
        add_meta_num = 0
        if add_meta_ratio > 0:
            add_meta_num = max(1,int(add_meta_ratio) * len(new_dataset)) #incase zero meta num
            meta_data = deepcopy(meta_dataset._data)
            leng_meta = len(meta_dataset)
        pop_list = []
        for i,(k,v) in enumerate(new_dataset.items()): # add meta
            if add_meta_num == 0:
                break
            else:
                add_meta_num -= 1
                v['index'] = leng_meta+i
                meta_data[leng_meta+i] = deepcopy(v)
                pop_list.append(k)
        for k in pop_list:
            new_dataset.pop(k)
        for i,(k,v) in enumerate(new_dataset.items()):
            assert res_data.get(leng+i) is None
            v['index'] = leng+i
            res_data[leng+i] = deepcopy(v)

        if add_meta_num > 0:
            return DatasetXL(res_data,self.label_mapping),DatasetXL(meta_data,self.label_mapping)
        else:
            return DatasetXL(res_data,self.label_mapping),None
    
    
    def remove_dataset(self, unlabeled_dataset, new_dataset):
        unlabeled_texts = [data[0] for data in unlabeled_dataset]
        unlabeled_labels = [data[1] for data in unlabeled_dataset]
        
        new_texts = [data[0] for data in new_dataset]
        new_labels = [data[1] for data in new_dataset]
        
        # remove pseudo-labeled from unlabeled dataset
        for text in new_texts:
            idx = unlabeled_texts.index(text)
            unlabeled_texts.pop(idx)
            unlabeled_labels.pop(idx)
                    
        return list(zip(unlabeled_texts, unlabeled_labels))
    
        
    def encode_dataset(self, texts, labels):
        encodings = self.tokenizer(texts, truncation=True, padding=True)
        dataset = Dataset(encodings, labels)
        return dataset
    
    
    def decode_dataset(self, dataset):
        decoded_texts = []
        labels = []
        for idx in range(len(dataset)):
            text_id = dataset[idx]['input_ids']
            label = dataset[idx]['labels'].item()
            decoded_text = self.tokenizer.decode(text_id, skip_special_tokens=True)
            decoded_texts.append(decoded_text)
            labels.append(label)
        return decoded_texts, labels

    def min_max_normalize(self,target):
        min_val = torch.min(target)
        max_val = torch.max(target)
        normalized_target = (target - min_val.detach()) / ((max_val - min_val) + torch.finfo(target.dtype).eps).detach()
        return normalized_target
    
    def get_inputs(self,src):
        ids = torch.tensor(src['input_ids']).to(self.device, dtype=torch.long)
        attention_mask = torch.tensor(src['attention_mask']).to(self.device, dtype=torch.long)
        if self.config.pretrained_model == "bert":
            token_type_ids = torch.tensor(src['token_type_ids']).to(self.device, dtype=torch.long)
            return ids,attention_mask,token_type_ids
        return ids,attention_mask
    

    def load(self,path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    




