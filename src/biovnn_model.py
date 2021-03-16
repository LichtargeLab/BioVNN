#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Community Convolutional Layer
Created on Sep 19, 2018.
Last edited on Oct 11, 2018.
@author: Chih-Hsu Lin
"""
import numpy as np
import os
from IPython import embed
from collections import OrderedDict
from collections import defaultdict as ddict
import subprocess
import pandas as pd
import sys
import logging
import h5py

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.metrics import (average_precision_score, roc_auc_score)
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from pytorch_layer import BioVNN, DepMapDataset, SequentialResumeSampler, FullyNet
from pytorch_layer import Ranger as Ranger_class
from utils import compute_AUC_bootstrap, plot_pred_true_r_by_gene_MAD, plot_pred_true_r_by_gene_mean, gene_level_cor, \
    individual_auc, plot_hist_cor, plot_hist_auc

cudnn.benchmark = True


class BioVNNmodel(object):
    def __init__(self, model_name, params={}, output_dir='./', layer_names={}, community_filter=None,
                 class_weight_neg=None, class_weight_pos=None, sample_class_weight_neg=None,
                 sample_class_weight_pos=None, group_level_dict=None, level_group_dict=None):
        self.model_name = model_name
        self.output_dir = output_dir
        self.layer_names = layer_names
        self.community_filter = community_filter
        self.group_level_dict = group_level_dict
        self.level_group_dict = level_group_dict
        self.params = params
        self.run_mode = params.get('run_mode', 'ref')
        self.batch_size = params.get('batch_size', 2000)
        self.epochs = params.get('epochs', 200)
        self.epochs_min = params.get('epochs_min', 0)
        self.goal = params.get('goal', None)
        self.model_v = params.get('model_v', 'clh_v1')
        self.seed = params['seed']
        self.patience = params.get('patience', 2)
        self.use_EarlyStopping = params.get('use_EarlyStopping', True)
        self.learning_rate = float(params.get('learning_rate', 1e-3))
        self.use_sample_weight = params.get('use_sample_weight', False)
        self.sample_weight_factor = params.get('sample_weight_factor', 0.5)
        self.reg_factor_l1 = float(params.get('reg_factor_l1', 0))
        self.reg_factor_l2 = float(params.get('reg_factor_l2', 1))
        self.com_layer_n = params.get('com_layer_n', 1)
        self.loss2_ratio = params.get('loss2_ratio', 0.3)
        self.average_loss2 = params.get('average_loss2', False)
        self.test_run = params.get('test_run', False)
        self.dropout_p1 = params.get('dropout_p1', 0)
        self.dropout_p2 = params.get('dropout_p2', 0.5)
        self.only_combine_child_gene_group = params.get('only_combine_child_gene_group', True)
        self.neuron_ratio = params.get('neuron_ratio', 0.2)
        self.neuron_min = params.get('neuron_min', 10)
        self.use_average_neuron_n = params.get('use_average_neuron_n', True)
        os.makedirs(self.output_dir, exist_ok=True)
        self.model = None
        self.model_encoder = None
        self.model_decoder = None
        self.optimizer = None
        self.optimizer_name = params.get('optimizer', 'Ranger')
        self.history = None
        self.use_cuda = params.get('use_cuda', True)
        self.use_classification = params.get('use_classification', True)
        self.loss_name = params.get('loss_name', None)
        self.use_class_weights = params.get('use_class_weights', True)
        self.class_weight_neg = class_weight_neg
        self.class_weight_pos = class_weight_pos
        self.use_sample_class_weights = params.get('use_sample_class_weights', False)
        self.sample_class_weight_neg = sample_class_weight_neg
        self.sample_class_weight_pos = sample_class_weight_pos

        self.criterion = torch.nn.BCELoss()
        if self.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.precision = params.get('precision', 32)
        self.act_func = params.get('act_func', 'Mish')

        self.log = pd.DataFrame()
        self.ckpt_path = os.path.join(self.output_dir,
                                      '{}_{}_{}.tar'.format(self.model_name, self.model_v, self.seed))
        self.use_sigmoid_output = self.params.get('use_sigmoid_output', True)
        self.use_genomic_info = self.params.get('use_genomic_info', True)
        self.use_deletion_vector = self.params.get('use_deletion_vector', True)
        os.makedirs(os.path.join(self.output_dir, 'hdf5'), exist_ok=True)
        self.report_gene_level = self.params.get('report_gene_level', False)
        self.use_lr_finder = self.params.get('use_lr_finder', False)
        self.lr_finder_iteration = self.params.get('lr_finder_iteration', 2000)
        self.use_cyclic_lr = self.params.get('use_cyclic_lr', False)
        self.cyclic_lr_step_size_up = int(self.params.get('cyclic_lr_step_size_up', 800))
        self.cyclic_gamma = float(self.params.get('cyclic_gamma', 1))
        self.cyclic_lr_base = float(self.params.get('cyclic_lr_base', 0.01))
        self.cyclic_lr_max = float(self.params.get('cyclic_lr_max', 0.1))
        self.iteration_count = 0

    def _save_hdf5(self, x, output_prefix):
        # 0.436 sec
        f = os.path.join(self.output_dir, 'hdf5', output_prefix + '.h5')
        with h5py.File(f, 'w') as hf:
            hf.create_dataset(output_prefix, data=x)

    def _load_hdf5(self, output_prefix, load_dir=None):
        # 0.0234 sec
        if load_dir is None:
            load_dir = self.output_dir
        f = os.path.join(load_dir, 'hdf5', output_prefix + '.h5')
        with h5py.File(f, 'r') as hf:
            try:
                x = hf[output_prefix][:, :]
            except TypeError:
                x = hf[output_prefix][:]
        return x

    def _save_progress(self, finished_batch, batch_size, output_prefix):
        # 0.436 sec
        f = os.path.join(self.output_dir, 'hdf5', 'progress_{}.tsv'.format(output_prefix))
        df = pd.DataFrame()
        df['finished_batch'] = [finished_batch]
        df['batch_size'] = [batch_size]
        df.to_csv(f, sep='\t')

    def _load_progress(self, resume_dir, output_prefix):
        f = os.path.join(resume_dir, 'hdf5', 'progress_{}.tsv'.format(output_prefix))
        df = pd.read_csv(f, sep='\t', index_col=0)
        finished_batch = int(df.loc[0, 'finished_batch'])
        batch_size = int(df.loc[0, 'batch_size'])
        return finished_batch, batch_size

    def _epoch(self, dataloader, criterion, epoch_n, mode='pred', output_prefix=None, finished_batch=0,
               y_index=None, y_col=None):
        with torch.autograd.profiler.emit_nvtx(enabled=False, record_shapes=True) as prof:
            running_loss = 0.0
            running_loss2 = 0.0

            if mode == 'train':
                self.model.train()
                set_grad_enabled = True
            elif mode == 'val' or mode == 'test' or 'pred' in mode:
                self.model.eval()
                set_grad_enabled = False

            labels_all = []
            pred_prob_root = []
            for X, labels, weights in tqdm(dataloader):
                if mode == 'train':
                    # zero the parameter gradients
                    self.optimizer.zero_grad()
                labels = labels.view(-1)
                labels_all.append(labels.cpu().numpy())
                X_input = [X[:, i].reshape(X.shape[0], -1) for i in range(X.shape[1])]
                with torch.set_grad_enabled(set_grad_enabled):
                    pred_prob, output = self.model(X_input)
                    # The loss of root
                    if self.use_classification:
                        # Weight the class differently to deal with imbalanced classes
                        if self.use_sample_class_weights:
                            if self.use_cuda:
                                weight = weights.cuda().float()
                            else:
                                weight = weights.float()
                            criterion = torch.nn.BCELoss(weight=weight)
                        elif self.use_class_weights:
                            weight = torch.Tensor(
                                [self.class_weight_neg if x < 0.5 else self.class_weight_pos for x in labels])
                            if self.use_cuda:
                                weight = weight.cuda().float()
                            else:
                                weight = weight.float()
                            criterion = torch.nn.BCELoss(weight=weight)
                        # loss = criterion(pred_prob[-1][:, 1], labels)
                        # pred_prob_root.append(pred_prob[-1][:, 1].cpu().detach().numpy())
                        # The loss of other non-root communities
                        # loss2 = criterion(pred_prob[0][:, 1], labels)
                        loss = criterion(pred_prob[-1].view(-1), labels)
                        pred_prob_root.append(pred_prob[-1].view(-1).cpu().detach().numpy())
                        # The loss of other non-root communities
                        loss2 = criterion(pred_prob[0].view(-1), labels)
                        for p in pred_prob[1:-1]:
                            loss2 += criterion(p.view(-1), labels)
                    else:
                        loss = criterion(pred_prob[-1].view(-1), labels)
                        pred_prob_root.append(pred_prob[-1].view(-1).cpu().detach().numpy())
                        # The loss of other non-root communities
                        loss2 = criterion(pred_prob[0].view(-1), labels)
                        for p in pred_prob[1:-1]:
                            loss2 += criterion(p.view(-1), labels)
                    if self.average_loss2:
                        loss2 /= (len(pred_prob) - 1.0)
                    loss2 *= self.loss2_ratio
                    if 'pred' in mode:
                        output = [x.cpu().detach().numpy() for x in output]
                        # output_all.append(output)

                    if mode == 'train':
                        # backward + optimize only if in training phase
                        # L1 regularization
                        if self.reg_factor_l1 != 0:
                            reg_loss = 0
                            for param in self.parameters():
                                reg_loss += F.l1_loss(param, target=torch.zeros_like(param), size_average=False)
                            reg_loss *= self.reg_factor_l1
                            total_loss = loss + loss2 + reg_loss
                        else:
                            total_loss = loss + loss2
                        total_loss.backward()
                        self.optimizer.step()
                        self.iteration_count += 1

                if 'pred' in mode and self.run_mode == 'ref':
                    for i, o in enumerate(output):
                        self._save_hdf5(o, '{}_intermediate_{}_{}'.format(output_prefix, i, finished_batch))
                    self._save_progress(finished_batch, dataloader.batch_size, output_prefix)
                    self._save_hdf5(pred_prob_root[-1], '{}_{}_prediction'.format(output_prefix, finished_batch))
                    finished_batch += 1
                running_loss += loss.item() * labels.size(0)
                running_loss2 += loss2.item() * labels.size(0)

        epoch_loss1 = running_loss / len(dataloader.dataset)
        epoch_loss2 = running_loss2 / len(dataloader.dataset)
        labels_all = np.concatenate(labels_all)
        pred_prob_root = np.concatenate(pred_prob_root)
        epoch_r = np.corrcoef(pred_prob_root, labels_all)[0, 1]
        labels_all_binary = (labels_all >= 0.5) + 0
        if np.sum(labels_all_binary) == 0 or np.sum(labels_all_binary) == len(labels_all_binary):
            epoch_auroc, epoch_auprc, epoch_f1, epoch_f1_weighted = np.nan, np.nan, np.nan, np.nan
        else:
            epoch_auroc = roc_auc_score(labels_all_binary, pred_prob_root)
            epoch_auprc = average_precision_score(labels_all_binary, pred_prob_root)
            pred_binary = (pred_prob_root >= 0.5) + 0
            epoch_f1 = np.nan
            epoch_f1_weighted = np.nan
            # epoch_f1 = f1_score(labels_all_binary, pred_binary)
            # epoch_f1_weighted = f1_score(labels_all_binary, pred_binary, average='weighted')
        if 'val' == mode and self.report_gene_level:
            labels_all = labels_all.reshape((len(y_index), len(y_col)))
            labels_all = pd.DataFrame(labels_all, index=y_index, columns=y_col)
            pred = pred_prob_root.copy()
            pred = pred.reshape((len(y_index), len(y_col)))
            pred = pd.DataFrame(pred, index=y_index, columns=y_col)
            self.compute_gene_level_cor(pred, labels_all, output_prefix + '_val_epoch_{}'.format(epoch_n))
            self.compute_gene_level_auc(pred, labels_all, output_prefix + '_val_epoch_{}'.format(epoch_n))

        if 'pred' in mode:
            epoch_n = 'pred_{}'.format(output_prefix)
        self.log.loc[epoch_n, 'loss1_{}'.format(mode)] = epoch_loss1
        self.log.loc[epoch_n, 'loss2_{}'.format(mode)] = epoch_loss2
        self.log.loc[epoch_n, 'loss_{}'.format(mode)] = epoch_loss1 + epoch_loss2
        self.log.loc[epoch_n, 'pearson_r_{}'.format(mode)] = epoch_r
        self.log.loc[epoch_n, 'AUROC_{}'.format(mode)] = epoch_auroc
        self.log.loc[epoch_n, 'AUPRC_{}'.format(mode)] = epoch_auprc
        self.log.loc[epoch_n, 'F1_{}'.format(mode)] = epoch_f1
        self.log.loc[epoch_n, 'F1_w_{}'.format(mode)] = epoch_f1_weighted
        f = os.path.join(self.output_dir, '{}_log.tsv'.format(output_prefix))
        self.log.to_csv(f, sep='\t')
        if 'val' == mode or 'pred' in mode:
            logging.info(self.log.loc[epoch_n])

        if 'pred' in mode:
            # output_all2 = []
            # for i in range(len(output_all[0])):
            #     output_i = np.concatenate([output_all[x][i] for x in range(len(output_all))])
            #     output_all2.append(output_i)
            # return pred_prob_root, output_all2
            return pred_prob_root
        else:
            return self.log.loc[epoch_n, 'loss1_{}'.format(mode)]

    def compute_gene_level_cor(self, pred, labels_all, output_prefix):
        df_gene_cor_var = gene_level_cor(pred, labels_all)
        logging.info(output_prefix)
        logging.info(df_gene_cor_var.head(10))
        self.output_pred(df_gene_cor_var, 'by_gene_' + output_prefix)
        plot_pred_true_r_by_gene_MAD(df_gene_cor_var, self.output_dir, output_prefix)
        plot_pred_true_r_by_gene_mean(df_gene_cor_var, self.output_dir, output_prefix)
        plot_hist_cor(df_gene_cor_var['Pearson_r'], self.output_dir, output_prefix)

    def compute_gene_level_auc(self, pred, labels_all, output_prefix):
        labels_all_binary = labels_all.loc[pred.index]
        labels_all_binary = (labels_all_binary > 0.5) + 0
        df_gene_auc, df_gene_bootstrap = individual_auc(pred, labels_all_binary, labels_all)
        logging.info(output_prefix)
        df_auc = df_gene_auc.loc[df_gene_auc['GS_positive_n'] >= 6]
        df_auc = df_auc.loc[df_auc['GS_negative_n'] >= 6]
        df_auc = df_auc.sort_values('AUROC', ascending=False)
        logging.info(df_auc.head(10))
        self.output_pred(df_gene_auc, 'by_gene_auc_' + output_prefix)
        for x, y in zip(df_gene_bootstrap, ['AUROCs', 'AUPRCs', 'pAUROCs', 'pAUPRCs']):
            self.output_pred(x, 'by_gene_{}_{}'.format(y, output_prefix))
        plot_pred_true_r_by_gene_MAD(df_gene_auc, self.output_dir, output_prefix, mode='auc')
        plot_pred_true_r_by_gene_mean(df_gene_auc, self.output_dir, output_prefix, mode='auc')
        plot_hist_auc(df_gene_auc['AUROC'], self.output_dir, output_prefix)
        plot_hist_auc(df_gene_auc['AUPRC'], self.output_dir, output_prefix)

    def output_pred(self, pred, output_prefix):
        pred.to_csv(
            os.path.join(self.output_dir, 'pred_{}.tsv'.format(output_prefix)), sep="\t")

    def extract_intermediate_output(self, X):
        _, output = self.model(X)
        return output

    def set_model(self, input_dim, mask, output_dim, feature_dim, mask_fully=None):
        if self.run_mode == 'full':
            model_class = FullyNet
        else:
            model_class = BioVNN
        self.model = model_class(input_dim, mask, output_dim, feature_dim, act_func=self.act_func,
                                 use_sigmoid_output=self.use_sigmoid_output,
                                 dropout_p=self.dropout_p2, layer_names=self.layer_names,
                                 only_combine_child_gene_group=self.only_combine_child_gene_group,
                                 neuron_min=self.neuron_min, neuron_ratio=self.neuron_ratio,
                                 use_classification=self.use_classification, child_map_fully=mask_fully,
                                 group_level_dict=self.group_level_dict,
                                 use_average_neuron_n=self.use_average_neuron_n, for_lr_finder=self.use_lr_finder)

        if self.use_sigmoid_output:
            logging.info("PyTorch Model {} with {} was set.".format(self.model_v, self.act_func))
        else:
            logging.info("PyTorch Model {} with {} without sigmoid output was set.".format(self.model_v, self.act_func))

    def set_optimizer(self, mask=None, mask2=None):
        Adam = optim.Adam(self.model.parameters(), lr=self.learning_rate,
                          weight_decay=self.reg_factor_l2)
        if self.use_lr_finder:
            Ranger = Ranger_class(self.model.parameters(), lr=1e-7,
                                  weight_decay=1e-2)
        else:
            Ranger = Ranger_class(self.model.parameters(), lr=self.learning_rate,
                                  weight_decay=self.reg_factor_l2)

        self.optimizer = locals().get(self.optimizer_name)
        logging.info("Optimizer {} was set.".format(self.optimizer_name))

    def compile(self):
        if self.use_cuda:
            self.model = self.model.to(self.device)
        if self.precision == 64:
            self.model.double()
        elif self.precision == 32:
            self.model.float()
        elif self.precision == 16:
            self.model.half()
        else:
            raise Exception('Precision %d is not in (16, 32, 64)' % self.precision)

    def _load_ckpt(self, path=None):
        if path is None:
            path = self.ckpt_path

        if os.path.isfile(path):
            model_ckpt = torch.load(path)
            try:
                self.model.load_state_dict(model_ckpt['model'])
            except RuntimeError:
                weights = model_ckpt['model']
                new_weights = OrderedDict()
                for k, v in weights.items():
                    new_weights['model.' + k] = v
                self.model.load_state_dict(new_weights)
                self.optimizer.load_state_dict(model_ckpt['optimizer'])
            self.start_epoch = model_ckpt['epoch'] + 1
            self.ckpt_epoch = model_ckpt['epoch']
            self.log = model_ckpt['log']
            logging.info('Check point %s loaded', path)

    def _save_ckpt(self, epoch):
        torch.save({
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'log': self.log,
        }, self.ckpt_path)

    def _rm_ckpt(self):
        os.remove(self.ckpt_path)

    def report_memory_usage(self, msg):
        if self.use_cuda:
            output = subprocess.run('nvidia-smi', stdout=subprocess.PIPE)
            logging.info("{}\n{}".format(msg, str(output).replace('\\n', '\n')))

    def _set_dtype(self, x):
        if self.precision == 64:
            dtype = np.float64
        elif self.precision == 32:
            dtype = np.float32
        elif self.precision == 16:
            dtype = np.float16
        else:
            raise Exception('Precision %d is not in (16, 32, 64)' % self.precision)
        return [y.astype(dtype) for y in x]

    def train(self, X_train, mask, y_train, sample_weight=None, X_val=None, y_val=None, load_weight_dir=None,
              y_train2=None, y_val2=None, mask2=None, output_prefix=None, y_val_index=None, y_col=None,
              mask_fully=None):
        # if mask.shape[1] < 50 and X_train.shape[1]<500:
        #     self.plot_net_structure()
        np.random.seed(self.seed)
        # if self.use_classification:
        #     output_dim = 1
        # else:
        output_dim = 1
        if self.use_deletion_vector:
            input_dim = X_train.shape[1] + y_train.shape[1]
        else:
            input_dim = X_train.shape[1]
        logging.info("Input dim: {}".format(input_dim))
        self.set_model(input_dim, mask, output_dim, X_train.shape[1], mask_fully)
        self.set_optimizer(mask, mask2)
        self.compile()
        if X_val is not None and y_val is not None:
            self.report_memory_usage("Before training:")
            X_train, y_train, X_val, y_val = self._set_dtype([X_train, y_train, X_val, y_val])
            if self.community_filter is not None:
                self.community_filter = self._set_dtype(self.community_filter)

            train_ds = DepMapDataset(X_train, y_train, self.community_filter, self.use_genomic_info,
                                     self.dropout_p1, self.use_deletion_vector, self.sample_class_weight_neg,
                                     self.sample_class_weight_pos, use_cuda=self.use_cuda)
            train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0)
            val_ds = DepMapDataset(X_val, y_val, self.community_filter, self.use_genomic_info,
                                   use_deletion_vector=self.use_deletion_vector, use_cuda=self.use_cuda)
            val_dl = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=0)
            best_loss = np.inf
            best_epoch = 0
            for epoch in range(self.epochs):
                self._epoch(train_dl, self.criterion, epoch, mode='train', output_prefix=output_prefix)
                loss_val = self._epoch(val_dl, self.criterion, epoch, mode='val', output_prefix=output_prefix,
                                       y_index=y_val_index, y_col=y_col)
                if loss_val < best_loss:
                    best_epoch = epoch
                    best_loss = loss_val
                    self._save_ckpt(epoch)
                if self.use_EarlyStopping:
                    if (epoch >= self.epochs_min) and (epoch - best_epoch >= self.patience):
                        logging.info('Validation loss does not improve for %d epochs! Break.', epoch - best_epoch)
                        break
            self.report_memory_usage("After training:")
        else:
            self._load_ckpt(load_weight_dir)

    def predict(self, X_test, y_test=None, batch_size=None, output_prefix=None, resume_dir=None, wt=False,
                y_index=None, y_col=None):
        if batch_size is None:
            batch_size = self.batch_size
        X_test, y_test = self._set_dtype([X_test, y_test])
        if self.community_filter is not None:
            self.community_filter = self._set_dtype(self.community_filter)
        test_ds = DepMapDataset(X_test, y_test, self.community_filter, self.use_genomic_info,
                                use_deletion_vector=self.use_deletion_vector, use_cuda=self.use_cuda)
        if resume_dir is not None:
            finished_batch, batch_size = self._load_progress(resume_dir, output_prefix)
            resume_idx = finished_batch * batch_size
            sampler = SequentialResumeSampler(test_ds, resume_idx)
            test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0,
                                 sampler=sampler)
            self.output_dir = resume_dir
        else:
            finished_batch = 0
            test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        pred = self._epoch(test_dl, self.criterion, None, mode='pred', output_prefix=output_prefix,
                           y_index=y_index, y_col=y_col, finished_batch=finished_batch)
        return pred

    def load_intermediate_output(self, output_prefix, load_dir, sub_neuron_list):
        finished_batch, _batch_size = self._load_progress(load_dir, output_prefix)
        result_list = []
        for neuron in sub_neuron_list:
            data_all = []
            for batch in range(finished_batch + 1):
                data_prefix = '{}_intermediate_{}_{}'.format(output_prefix, neuron, batch)
                data = self._load_hdf5(data_prefix, load_dir)
                data_all.append(data)
            data_all = np.concatenate(data_all)
            result_list.append(data_all)
        return result_list

    def load_prediction(self, output_prefix, load_dir):
        finished_batch, _batch_size = self._load_progress(load_dir, output_prefix)
        data_all = []
        for batch in range(finished_batch + 1):
            data_prefix = '{}_{}_prediction'.format(output_prefix, batch)
            data = self._load_hdf5(data_prefix, load_dir)
            data_all.append(data)
        data_all = np.concatenate(data_all)
        return data_all

    def plot_net_structure(self, X, y):
        X, y = self._set_dtype([X, y])
        self.community_filter = self._set_dtype(self.community_filter)
        ds = DepMapDataset(X[:2, :], y[:2], self.community_filter, self.use_genomic_info, self.dropout_p1)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=0)
        for x, _ in dl:
            X_input = [x[:, i].reshape(x.shape[0], -1) for i in range(x.shape[1])]
        y1, y2 = self.model(X_input)
        dot_obj = make_dot(y1[-1], params=dict(list(self.model.named_parameters()) + [('x', x)]))
        dot_obj.render(os.path.join(self.output_dir, 'net_structure'))

    def plot_grad_flow(self, named_parameters, output_prefix):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backwards() as
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads = []
        layers = []
        for n, p in named_parameters:
            if (p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        fout = os.path.join(self.output_dir, 'grad_{}.png'.format(output_prefix))
        plt.savefig(fout, bbox_inches='tight')
        plt.close()


