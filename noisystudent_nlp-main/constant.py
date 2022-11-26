import torch
class Config():
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.seed = 2021
        self.training_from_scratch = True # suitable for both baseline and ours
        self.baseline = "naive"
        self.pretrained_model = "bert"
        self.learning_rate = 2e-5 # 2e-5
        self.epochs = 5 # 10
        self.inner_epochs = 1 # 10
        self.meta_epochs = 200 # 200
        self.hist_len = 100 # 25
        self.multiple_len = 1
        self.main_target='acc'
        self.meta_target='precision'
        self.beta = 0.0625
        self.confidence_thres = 0.0
        self.add_meta_ratio = 0
        self.block_entropy = True
        self.model_noise = False
        self.model_noise_type = "dropout"
        self.dropout = 0.2

        self.meta_cfd_threshold = 0.8
        self.relative = False

        self.split_num_groups = 5
        self.train_max_token_len = 15000
        self.train_batch_size = 16
        self.valid_batch_size = 16
        self.valid_max_token_len = 15000
        self.test_batch_size = 128
        self.unlabeled_batch_size = 128
        self.class_num = 2
        self.sample_num = 125 # 500 <-> 1:12.5
        self.label_sample_ratio_imdb = 0.02
        self.dev_sample_ratio_imdb = 0.5
        self.meta_dev_sample_ratio_imdb = 0.5

        self.label_sample_ratio_sms = 0.02
        self.dev_sample_ratio_sms = 0.5
        self.meta_dev_sample_ratio_sms = 0.5

        self.label_sample_ratio_trec = 0.02
        self.dev_sample_ratio_trec = 0.5
        self.meta_dev_sample_ratio_trec = 0.5

        self.label_sample_ratio_youtube = 0.1
        self.dev_sample_ratio_youtube = 0.5
        self.meta_dev_sample_ratio_youtube = 0.5


        self.bidirectional = True

        #meta model
        self.hidden_size = 128
        self.act = 'elu'
        self.patience = 10

