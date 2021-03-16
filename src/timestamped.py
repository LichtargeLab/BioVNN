#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Timestamped
----------------------------------

Timestamped prediction class

Created on Feb 8, 2018
Last edited on Feb 8, 2018
@author: Chih-Hsu Lin
"""
import os
import sys
import datetime
import numpy as np
from IPython import embed
import pandas as pd
import logging
import matplotlib.pyplot as plt

plt.switch_backend('agg')
from glob import glob

from collections import defaultdict as ddict

import paths
from dependency import Dependency, load_params, save_params
import scipy
from scipy.stats.mstats import gmean
from biovnn_model import BioVNNmodel


class Timestamped(Dependency):
    def __init__(self, cancer_type, data_dir, result_dir, run_name, params,
                 depmap_ver='19Q3', use_hierarchy=True):
        super().__init__(cancer_type, data_dir, result_dir, run_name, params, depmap_ver, use_hierarchy)
        self.predict_target = params.get('predict_target', ['BT474', 'T47D', 'MCF10A'])
        self.depmap_ver_target = params.get('depmap_ver_target', '20Q2')
        self.hdf5_df_file = os.path.join(self.temp_path, 'df_{}_depmap_{}_target_{}.hdf5'.format(self.data_types[0],
                                                                                                 self.depmap_ver,
                                                                                                 self.depmap_ver_target))
        self.pred = ddict()
        self.pred['test'] = ddict(list)
        self.save_load_data = ['rna', 'dependency', 'dependency_target', 'rna_target']

    def prepare_data(self):
        self.load_selected_gene_list()
        self.load_known_genes(self.depmap_ver_target)
        if not self.load_data():
            self.load_dependency(self.depmap_ver_target)
            if 'rna' in self.data_types:
                self.load_rna(self.depmap_ver_target)
                self.rna_target = self.rna.copy()
                self.rna_target_all = self.rna_all.copy()
                self.load_rna(self.depmap_ver)
            self.cancer_type_to_patients_target = self.cancer_type_to_patients.copy()
            self.dependency_target = self.dependency.copy()
            self.load_dependency(self.depmap_ver)
            self.save_data()
        self.load_communities()
        self.load_known_genes(self.depmap_ver)
        self.align_data()
        self.compare_data_versions()

    def compare_data_versions(self):
        overlap_idx = self.dependency.index & self.dependency_target.index
        overlap_col = self.dependency.columns & self.dependency_target.columns
        dp_old = self.dependency.loc[overlap_idx, overlap_col].values.flatten()
        dp_new = self.dependency_target.loc[overlap_idx, overlap_col].values.flatten()
        pearson_r = np.corrcoef(dp_old, dp_new)[0, 1]
        spearman_rho = scipy.stats.spearmanr(dp_old, dp_new)[0]

        logging.info("{} and {} dependency Pearson's r {}".format(self.depmap_ver, self.depmap_ver_target, pearson_r))

        overlap_idx = self.rna.index & self.rna_target.index
        overlap_col = self.rna.columns & self.rna_target.columns
        rna_old = self.rna.loc[overlap_idx, overlap_col].values.flatten()
        rna_new = self.rna_target.loc[overlap_idx, overlap_col].values.flatten()
        pearson_r = np.corrcoef(rna_old, rna_new)[0, 1]
        spearman_rho = scipy.stats.spearmanr(rna_old, rna_new)[0]

        logging.info("{} and {} RNA Pearson's r {}".format(self.depmap_ver, self.depmap_ver_target, pearson_r))

    def run_exp(self, model_name, model_suffix, params, com_mat, repeat, fold, community_filter=None,
                com_mat_fully=None):
        if 'random_predictor' in model_name:
            self.compute_metric(None, 'test', model_name, model_suffix, self.y_train, self.y_test, com_mat, repeat,
                                self.y_test2)
        elif 'mean_control' in model_name:
            self.compute_metric(None, 'test', model_name, model_suffix, self.y_train, self.y_test, com_mat, repeat,
                                self.y_test2)
        else:
            if self.use_community_filter:
                cm = BioVNNmodel(model_name + model_suffix, params, self.result_path,
                                 self.community_hierarchy_dicts_all, community_filter,
                                 class_weight_neg=self.class_weight_neg,
                                 class_weight_pos=self.class_weight_pos,
                                 sample_class_weight_neg=self.sample_class_weight_neg,
                                 sample_class_weight_pos=self.sample_class_weight_pos,
                                 group_level_dict=self.community_level_dict,
                                 level_group_dict=self.level_community_dict)
            else:
                cm = BioVNNmodel(model_name + model_suffix, params, self.result_path,
                                 self.community_hierarchy_dicts_all,
                                 class_weight_neg=self.class_weight_neg,
                                 class_weight_pos=self.class_weight_pos,
                                 sample_class_weight_neg=self.sample_class_weight_neg,
                                 sample_class_weight_pos=self.sample_class_weight_pos,
                                 group_level_dict=self.community_level_dict,
                                 level_group_dict=self.level_community_dict)

            load_ckpt = os.path.join(self.load_result_dir,
                                     '{}_{}_{}.tar'.format(model_name + model_suffix, self.params['model_v'],
                                                           self.params['seed']))
            cm.train(self.X_train, com_mat, self.y_train, load_weight_dir=load_ckpt, mask_fully=com_mat_fully)

            self.compute_metric(cm, 'test', model_name, model_suffix, self.X_test, self.y_test, com_mat, repeat,
                                self.y_test2)
            self.output_pred_mean()
            self._clear_gpu(model_name, model_suffix)
        model_suffix = str(params['seed']) + 'repeat' + str(repeat)
        self.compute_metric_all_test('test', model_name, model_suffix, self.X_test, self.y_test, repeat)
        self.output_metric()

    def compute_metric(self, cm, data_type, model_name, model_suffix, X, y_true, com_mat, repeat, y_true2=None,
                       pred=None):
        output_prefix = model_name + model_suffix + '_' + data_type
        y_index = self.idx[data_type]
        y_col = self.y.columns
        if 'random_predictor' in model_name:
            pred = []
            for i in range(y_true.shape[1]):
                pred.append(np.random.rand(y_true.shape[0], 1))
            pred = np.concatenate(pred, axis=1).flatten()
        elif 'mean_control' in model_name:
            pred = np.tile(X.mean(axis=0), y_true.shape[0])
        else:
            pred = cm.predict(X, y_true, 5000, output_prefix, y_index=y_index, y_col=y_col)

        if np.isnan(y_true).sum(axis=0).max() == y_true.shape[0]:
            nan_ct = np.isnan(y_true).sum(axis=0)
            nan_idx = np.where((nan_ct == y_true.shape[0]))
            nan_gene = list(y_col[nan_idx])
            logging.info("Gene {} in the target data is nan and was removed from computing metric".format(nan_gene))
            y_true = np.delete(y_true, nan_idx, axis=1)
            pred = pred.reshape([len(y_index), len(y_col)])
            pred = np.delete(pred, nan_idx, axis=1).flatten()
            y_col = np.delete(y_col, nan_idx)

        pred = pred.reshape([len(y_index), len(y_col)])
        pred = pd.DataFrame(pred, index=y_index, columns=y_col)
        y_true = pd.DataFrame(y_true, index=y_index, columns=y_col)
        self.compute_overall_cor(pred, y_true, repeat, data_type, model_name, model_suffix, output_prefix)
        for x in pred.index:
            self.pred['test'][x + '_{}_{}'.format(model_name, repeat)].append(pred.loc[x])
            self.pred['test'][x + '_{}'.format(model_name)].append(pred.loc[x])

    def output_pred_mean(self):
        data_type = 'test'
        for cell_line in self.pred[data_type]:
            df = pd.DataFrame(index=self.pred[data_type][cell_line][0].index)
            pred_n = len(self.pred[data_type][cell_line])
            df['pred_mean_{}'.format(pred_n)] = np.mean(np.array(self.pred[data_type][cell_line]), axis=0)
            df['pred_median_{}'.format(pred_n)] = np.median(np.array(self.pred[data_type][cell_line]), axis=0)
            df['pred_geomean_{}'.format(pred_n)] = gmean(np.array(self.pred[data_type][cell_line]), axis=0)
            df['pred_std_{}'.format(pred_n)] = np.std(np.array(self.pred[data_type][cell_line]), axis=0)
            rank_dfs = np.array([x.rank(ascending=False) for x in self.pred[data_type][cell_line]])
            df['pred_rank_mean_{}'.format(pred_n)] = np.mean(rank_dfs, axis=0)
            df['pred_rank_median_{}'.format(pred_n)] = np.median(rank_dfs, axis=0)
            df['pred_rank_geomean_{}'.format(pred_n)] = gmean(rank_dfs, axis=0)
            df['pred_rank_std_{}'.format(pred_n)] = np.std(rank_dfs, axis=0)
            df.to_csv(os.path.join(self.result_path, 'pred_mean_{}_{}.tsv'.format(data_type, cell_line)),
                      sep="\t")

    def load_prediction_all(self, data_type, model_name, model_suffix):
        load_dir = self.result_path
        pred_files = glob('{}/pred_{}*fold*{}.tsv'.format(load_dir, model_name + model_suffix, data_type))
        pred = pd.read_csv(pred_files[0], sep='\t', index_col=0)
        for p in pred_files[1:]:
            p = pd.read_csv(p, sep='\t', index_col=0)
            pred = pd.concat([pred, p])
        if len(pred.index) != len(set(pred.index)):
            logging.info("Found duplicated index and took their mean")
            pred_unique = pd.DataFrame(columns=pred.columns)
            for idx in set(pred.index):
                pred_unique.loc[idx] = pred.loc[idx].mean()
            pred = pred_unique
        return pred


def main():
    from set_logging import set_logging
    # Set up parameters of the model
    param_f = sys.argv[1]
    params = load_params(param_f=param_f)

    params['load_result_dir_suffix'] = 'Dependency/' + params['load_result_dir_name']

    load_result_dir_name = params['load_result_dir_name']

    model_name = 'Reactome_filter'

    if '19Q3' in load_result_dir_name:
        depmap_ver = '19Q3'
    else:
        depmap_ver = '18Q4'

    if 'random' in load_result_dir_name:
        params['run_mode'] = 'random'

    params['analysis_source'] = 'PANC'
    run_suffix = 'timestamped' + params['load_result_dir_name']
    if 'run_mode' in params:
        run_suffix += '_' + params['run_mode']

    start_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    start_time += '_' + run_suffix
    data_dir = paths.DATA_DIR
    result_dir = paths.RESULTS_DIR

    log_dir = os.path.join(result_dir, 'Timestamped', start_time)
    set_logging('Timestamped', log_dir)

    ts = Timestamped(params['analysis_source'], data_dir, result_dir, start_time, params, depmap_ver)
    ts.perform(model_name, params)


if __name__ == '__main__':
    main()
