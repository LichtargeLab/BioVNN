#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dependency
----------------------------------

Dependency analysis class

Created on Nov 8, 2018
Last edited on Nov 8, 2018
@author: Chih-Hsu Lin
"""
import os
import io
import sys
import datetime
import numpy as np
from IPython import embed
import pandas as pd
import logging
import matplotlib.pyplot as plt

plt.switch_backend('agg')
import seaborn as sns
import re
import subprocess
import yaml
from glob import glob

import scipy
from statsmodels.robust.scale import mad
from collections import Counter
from collections import defaultdict as ddict
from sklearn.metrics import roc_curve, average_precision_score, f1_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection._search import ParameterGrid
import torch

import paths
from biovnn_model import BioVNNmodel
from utils import compute_AUC_bootstrap, plot_pred_true_r_by_gene_MAD, plot_pred_true_r_by_gene_mean, gene_level_cor, \
    individual_auc, plot_ROC, plot_top_ROC, plot_hist_cor, plot_hist_auc

disease_mapping = {'Bladder Cancer': 'BLCA',
                   'Breast Cancer': 'BRCA',
                   'breast': 'BRCA',
                   'Cervical Cancer': 'CESC',
                   'Colon Cancer': 'COAD',
                   'Colon/Colorectal Cancer': 'COAD',
                   'colorectal': 'COAD',
                   'GBM/Brain Cancer': 'GBM',
                   'glioblastoma': 'GBM',
                   'Head and Neck Cancer': 'HNSC',
                   'upper_aerodigestive': 'HNSC',
                   'Liver Cancer': 'LIHC',
                   'liver': 'LIHC',
                   'Ovarian Cancer': 'OV',
                   'ovary': 'OV',
                   'Skin Cancer': 'SKCM',
                   'skin': 'SKCM',
                   'Gastric Cancer': 'STAD',
                   'Soft Tissue/ Thyroid Cancer': 'THCA',
                   'Thyroid Cancer': 'THCA',
                   'Endometrial Cancer': 'UCEC',
                   'Endometrial/Uterine Cancer': 'UCEC',
                   'uterus': 'UCEC',
                   'Esophageal Cancer': 'ESCA',
                   'esophagus': 'ESCA',
                   'Pancreatic Cancer': 'PAAD',
                   'pancreas': 'PAAD',
                   'Non-Small Cell Lung Cancer (NSCLC), Adenocarcinoma': 'LUAD',
                   'Non-Small Cell Lung Cancer (NSCLC), Squamous Cell Carcinoma': 'LUSC',
                   'Renal Carcinoma, clear cell': 'KIRC',
                   'Glioblastoma': 'GBM',
                   'Acute Myelogenous Leukemia (AML)': 'LAML',
                   'AML': 'LAML'}


def load_params(output_dir=None, param_f=None):
    if param_f is None:
        param_f = os.path.join(output_dir, 'param.yaml')
    with open(param_f, 'r') as stream:
        params = yaml.safe_load(stream)
    return params


def save_params(output_dir, params):
    with io.open(os.path.join(output_dir, 'param.yaml'), 'w', encoding='utf8') as outfile:
        yaml.dump(params, outfile, default_flow_style=False, allow_unicode=True)
    assert params == load_params(output_dir)


class Dependency(object):
    def __init__(self, cancer_type, data_dir, result_dir, run_name, params,
                 depmap_ver='19Q3', use_hierarchy=True):
        self.method = 'BioVNN'
        self.cancer_type = cancer_type
        self.n_cluster = None
        self.run_name = run_name
        self.patient_list = []
        self.cancer_type_to_patients = ddict(list)
        self.rna_dir = os.path.join(data_dir, 'DepMap', depmap_ver)
        self.data_dir = data_dir
        if 'ref_groups' in params and params['ref_groups'] == 'GO':
            self.community_file = os.path.join(data_dir, 'GO', 'goa_human_20201212.gmt')
            self.community_hierarchy_file = os.path.join(self.data_dir, 'GO', 'go_20201212_relation.txt')
        else:
            self.community_file = os.path.join(data_dir, 'Reactome', 'ReactomePathways.gmt')
            self.community_hierarchy_file = os.path.join(self.data_dir, 'Reactome', 'ReactomePathwaysRelation.txt')
        self.gene_id_file = os.path.join(self.data_dir, 'Reactome', 'Homo_sapiens_9606.gene_info')
        self.gene_id_dict = pd.read_csv(self.gene_id_file, sep='\t', index_col=1)['Symbol'].to_dict()
        self.Reactome_name_file = os.path.join(data_dir, 'Reactome', 'ReactomePathways.txt')
        self.Reactome_name_dict = pd.read_csv(self.Reactome_name_file, sep='\t', index_col=0, header=None)[1].to_dict()
        self.Reactome_reaction_file = os.path.join(self.data_dir, 'Reactome', 'NCBI2Reactome_PE_Reactions_human.txt')
        self.Reactome_reaction_df = pd.read_csv(self.Reactome_reaction_file, sep='\t', index_col=None, header=None)
        self.Reactome_gene_reaction_dict = ddict(list)
        self.Reactome_reaction_gene_dict = ddict(list)
        for i, row in self.Reactome_reaction_df.iterrows():
            if 'HSA' in row[1] and 'HSA' in row[3]:  # Make sure they are from human
                if row[0] in self.gene_id_dict:
                    symbol = self.gene_id_dict[row[0]]
                else:
                    symbol = row[2].split(' [')[0]
                self.Reactome_gene_reaction_dict[symbol].append(row[3])
                self.Reactome_reaction_gene_dict[row[3]].append(symbol)
        self.community_dict = {}
        self.community_hierarchy = []
        self.community_hierarchy_all = None
        self.community_hierarchy_random = []
        self.community_hierarchy_random_all = None
        self.community_hierarchy_ones = []
        self.community_hierarchy_ones_all = None
        self.community_hierarchy_dicts_all = {}
        self.use_hierarchy = use_hierarchy
        self.community_matrix = None
        self.result_path = os.path.join(result_dir, self.__class__.__name__, run_name)
        self.temp_path = os.path.join(result_dir, self.__class__.__name__, 'temp')
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(self.temp_path, exist_ok=True)
        self._dependency_classes = ['Dependency', 'Transfer', 'Postanalysis', 'Postanalysis_ts',
                                    'Postanalysis_transfer', 'Prospective',
                                    'Timestamped', 'Interpret', 'Interpret_ts']
        self._dependency_classes_plot = ['Dependency', 'Transfer', 'Postanalysis', 'Postanalysis_ts',
                                         'Postanalysis_transfer', 'Timestamped']

        self.params = params
        self.load_result = params.get('load_result', False)
        self.load_result_dir_name = params.get('load_result_dir_name', False)
        if self.load_result and self.load_result_dir_name:
            if 'load_result_dir_suffix' in params:
                if 'load_result_dir_full' in params:
                    if params['load_result_dir_full']:
                        self.load_result_dir = params['load_result_dir_suffix']
                    else:
                        self.load_result_dir = os.path.join(result_dir, params['load_result_dir_suffix'])
                else:
                    self.load_result_dir = os.path.join(result_dir, params['load_result_dir_suffix'])
            else:
                self.load_result_dir = '/'.join(self.result_path.split('/')[:-1] + [self.load_result_dir_name])
            params = load_params(self.load_result_dir)
            if 'run_mode' in self.params:
                run_mode = self.params['run_mode']
            else:
                run_mode = None
            self.params.update(params)
            params = self.params
            if run_mode:
                params['run_mode'] = run_mode
                self.params['run_mode'] = run_mode
        self.use_cuda = params.get('use_cuda', True)
        self.data_types = params.get('data_types', ['rna'])
        self.use_all_gene = params.get('use_all_gene', True)
        self.exp_ratio_min = params.get('exp_ratio_min', 0.01)
        self.feature_max = params.get('feature_max', 99999)
        self.feature_per_group_max = params.get('feature_per_group_max', 100)
        self.repeat_n = params.get('repeat_n', 1)
        self.fold_n = params.get('fold_n', 5)
        self.cv_fold = params.get('cv_fold', 0)
        self.model_v = params.get('model_v', 'clh_v1')
        self.cv_fold_only_run = params.get('cv_fold_only_run', 1)
        self.other_cancer_types = params.get('other_cancer_types', [])
        self.rna_top_n_std = params.get('rna_top_n_std', 10000)
        self.community_affected_size_min = params.get('community_affected_size_min', 5)
        self.community_affected_size_max = params.get('community_affected_size_max', 999999)
        self.require_label_gene_in_gene_group = params.get('require_label_gene_in_gene_group', True)
        self.clip_Xval_Xtest = params.get('clip_Xval_Xtest', [-1, 1])
        self.use_MinMaxScaler = params.get('use_MinMaxScaler', False)
        self.use_StandardScaler = params.get('use_StandardScaler', True)
        self.use_tanh_feature = params.get('use_tanh_feature', False)
        self.use_sigmoid_feature = params.get('use_sigmoid_feature', False)
        self.use_community_filter = params.get('use_community_filter', True)
        self.test_run = params.get('test_run', False)
        self.select_genes_in_label = params.get('select_genes_in_label', 'dgidb_w_interaction')
        self.use_classification = params.get('use_classification', True)
        self.use_binary_dependency = params.get('use_binary_dependency', True)
        self.use_class_weights = params.get('use_class_weights', True)
        self.use_normalized_class_weights = params.get('use_normalized_class_weights', False)
        self.use_sample_class_weights = params.get('use_sample_class_weights', False)
        self.use_normalized_sample_class_weights = params.get('use_normalized_sample_class_weights', True)
        self.use_all_dependency_gene = params.get('use_all_dependency_gene', True)
        self.use_all_feature_for_random_group = params.get('use_all_feature_for_random_group', False)
        self.use_all_feature_for_fully_net = params.get('use_all_feature_for_fully_net', False)
        self.use_deletion_vector = params.get('use_deletion_vector', True)
        self.use_consistant_groups_for_labels = params.get('use_consistant_groups_for_labels', False)
        self.run_mode = params.get('run_mode',
                                   'ref')  # Could be ref, random_predictor, random, expression_control or full
        self.random_group_permutation_ratio = params.get('random_group_permutation_ratio', 1)
        self.random_group_hierarchy_permutation_ratio = params.get('random_group_hierarchy_permutation_ratio', 1)
        self.random_group_permutation_seed = params.get('random_group_permutation_seed', 9527)
        self.leaf_group_gene_in_label_max = params.get('leaf_group_gene_in_label_max', 50)
        self.split_by_cancer_type = params.get('split_by_cancer_type', True)
        self.save_model_ckpt = params.get('save_model_ckpt', True)
        self.output_pred_small = ['RPS20', 'MYC', 'MYCN', 'PIK3CA']
        self.GSP_min = params.get('GSP_min', 6)
        self.GSN_min = params.get('GSN_min', 6)
        self.gene_list = None
        self.gene_list_name = None
        self.accuracy = None
        self.f1 = None
        self.confusion_mat = None
        self.mcc = None
        self.pearson_r = None
        self.spearman_rho = None
        self.mse = None
        self.feature_importance = []
        metrics = ['accuracy', 'confusion_mat', 'f1', 'mcc', 'pearson_r', 'spearman_rho', 'mse', 'pearson_r2',
                   'AUC', 'PR']
        data_splits = ['train', 'val', 'test']
        for x in metrics:
            self.__dict__[x] = ddict(dict)
            for z in range(self.repeat_n + 1):
                self.__dict__[x][z] = ddict(dict)
                for y in data_splits:
                    self.__dict__[x][z][y] = ddict(list)
        for x in ['pred', 'idx']:
            self.__dict__[x] = ddict(dict)
            for y in data_splits:
                self.__dict__[x][y] = ddict(list)

        self.metric_output = {}
        for y in data_splits:
            self.metric_output[y] = pd.DataFrame()

        self.save_load_data = ['rna']

        self.depmap_ver = depmap_ver
        os.makedirs(self.rna_dir, exist_ok=True)
        self.save_load_data = ['rna', 'dependency']
        self.hdf5_df_file = os.path.join(self.temp_path,
                                         'df_{}_depmap_{}.hdf5'.format('_'.join(sorted(self.data_types)),
                                                                       self.depmap_ver))

    def prepare_data(self):
        self.load_communities()
        self.load_known_genes()
        self.load_selected_gene_list()
        if not self.load_data():
            self.load_dependency()
            if 'rna' in self.data_types:
                self.load_rna()
            self.save_data()
        self.align_data()

    def load_selected_gene_list(self):
        if isinstance(self.select_genes_in_label, str):
            if self.select_genes_in_label.lower() == 'dgidb_w_interaction':
                dgidb_file = os.path.join(self.data_dir, 'DGIdb_genes_w_interactions.txt')
            else:
                raise ValueError("Cannot recongnize select_genes_in_label {}".format(self.select_genes_in_label))
            self.select_genes_in_label = pd.read_csv(dgidb_file, header=None)[0].tolist()
        elif 'ref_leaf_group' in self.run_mode:
            if isinstance(self.select_genes_in_label, list):
                leaf_communities, df = self.load_leaf_communities()
                initial_select = set(self.select_genes_in_label)
                initial_n = len(initial_select)
                logging.info("Selected genes {} were used to find additional genes in the same leaf gene groups".format(
                    self.select_genes_in_label))
                leaf_communities_with_genes = {}
                for group in leaf_communities:
                    if len(initial_select.intersection(self.community_dict[group])) > 0:
                        leaf_communities_with_genes[group] = len(self.community_dict[group])
                # Select leaf groups from small to large groups until it reaches the self.leaf_group_gene_in_label_max
                for group, size in sorted(leaf_communities_with_genes.items(), key=lambda x: x[1]):
                    if len(initial_select | set(self.community_dict[group])) < self.leaf_group_gene_in_label_max:
                        initial_select |= set(self.community_dict[group])
                        logging.info("{} gene group was added as genes in labels".format(group))
                self.select_genes_in_label = sorted(list(initial_select))
                logging.info(
                    "Additional {} genes in the same leaf gene groups with selected genes were added".format(
                        len(self.select_genes_in_label) - initial_n))

    def save_label_genes(self, genes):
        """Save label genes to file."""
        fout = open(os.path.join(self.result_path, 'dependency_genes.tsv'), 'w')
        for x in genes:
            fout.write('{}\n'.format(x))
        fout.close()

    def save_communities(self, d=None):
        """Save community genes to file."""
        if d is None:
            fout = open(os.path.join(self.result_path, 'community_list.tsv'), 'w')
            d = self.community_dict
            s = ''
        else:
            fout = open(os.path.join(self.result_path, 'community_random_list.tsv'), 'w')
            s = '_random'
        for k, v in d.items():
            fout.write('{}\n'.format('\t'.join([k + s] + v)))
        fout.close()

    def save_data(self):
        hf = pd.HDFStore(self.hdf5_df_file)
        for x in self.save_load_data:
            if x in self.__dict__:
                hf[x] = self.__dict__[x]
        hf['data_types'] = pd.DataFrame(self.data_types)
        # hf['other_cancer_types'] = pd.DataFrame(self.other_cancer_types)
        # hf['cancer_type'] = pd.DataFrame([self.cancer_type])
        # for ct in set(self.cancer_type_to_patients.keys()) | set(self.other_cancer_types) | set([self.cancer_type]):
        #     hf[ct] = pd.DataFrame(self.cancer_type_to_patients[ct])
        # if 'cancer_type_to_patients_target' in self.__dict__:
        #     for ct in set(self.cancer_type_to_patients_target.keys()) | set(self.other_cancer_types) | set(
        #             [self.cancer_type]):
        #         hf[ct + '_target'] = pd.DataFrame(self.cancer_type_to_patients_target[ct])
        hf.close()

    def load_data(self):
        if os.path.isfile(self.hdf5_df_file):
            hf = pd.HDFStore(self.hdf5_df_file)
            try:
                for x in self.save_load_data:
                    if x in hf:
                        self.__dict__[x] = hf[x]
                        self.__dict__[x + '_all'] = self.__dict__[x].copy()
                logging.info("Loaded data from existing hdf5 file.")
                hf.close()
                return True
            except:
                logging.info(
                    "Current Data types, Cancer type or Other cancer types do not match that of existing hdf5 file.")
                hf.close()
                return False
        else:
            return False

    def load_communities(self, load_original=True):
        """Parses out a geneset from file."""
        if self.load_result and not load_original:
            lines = open('{}/community_list.tsv'.format(self.load_result_dir)).readlines()
            ind_key = 0
            ind_gene = 1
        else:
            lines = open('{}'.format(self.community_file)).readlines()
            if 'pathway' in self.community_file.lower():
                ind_key = 1
                ind_gene = 3
            elif self.community_file.lower().endswith('.gmt'):
                ind_key = 1
                ind_gene = 3
            else:
                ind_key = 0
                ind_gene = 1

        self.community_genes = set()
        self.community_dict = {}
        self.gene_community_dict = ddict(list)
        self.community_size_dict = {}

        for line in lines:
            line = line.strip().split('\t')
            self.community_dict[line[ind_key]] = line[ind_gene:]
            self.community_size_dict[line[ind_key]] = len(line[ind_gene:])
            self.community_genes |= set(line[ind_gene:])
            for g in line[ind_gene:]:
                self.gene_community_dict[g].append(line[ind_key])

    def load_random_communities(self, load_original=True):
        """Parses out a geneset from file."""
        lines = open('{}/community_random_list.tsv'.format(self.load_result_dir)).readlines()
        ind_key = 0
        ind_gene = 1
        self.random_community_genes = set()
        self.community_dict_random = {}
        self.random_community_size_dict = {}

        for line in lines:
            line = line.strip().split('\t')
            group = line[ind_key].split('_')[0]
            self.community_dict_random[group] = line[ind_gene:]
            self.random_community_size_dict[group] = len(line[ind_gene:])
            self.random_community_genes |= set(line[ind_gene:])

    def load_leaf_communities(self):
        f = self.community_hierarchy_file
        # The first column 0 is the parent and the second column 1 is the child
        df = pd.read_csv(f, sep='\t', header=None)
        if 'Reactome' in f:
            df = df.loc[df[0].str.contains('HSA')]  # Get human-only pathways
        # Make root as the parent of those gene groups without parents
        df_root = pd.DataFrame(columns=df.columns)
        for x in set(df[0]) - set(df[1]):
            if x in self.community_dict or 'GO:' in x:
                df_root = pd.concat([df_root, pd.DataFrame(['root', x]).T])
        # Remove those relationship of groups not in the analysis
        df = df.loc[df[1].isin(self.community_dict.keys()) & df[0].isin(self.community_dict.keys())]
        df = pd.concat([df, df_root])
        leaf_communities = sorted(list((set(df[1]) - set(df[0])) & set(self.community_dict.keys())))

        return leaf_communities, df

    def load_random_hierarchy(self):
        f = '{}/random_group_hierarchy.tsv'.format(self.load_result_dir)
        df = pd.read_csv(f, sep='\t', header=None)
        return df

    def load_known_genes(self, depmap_ver=None):
        if depmap_ver is None:
            depmap_ver = self.depmap_ver
        regex = re.compile(r"\d\dQ\d", re.IGNORECASE)
        depmap_dir = os.environ.get('DEPMAP_DIR')

        if depmap_ver not in depmap_dir:
            depmap_dir = regex.sub(depmap_ver, depmap_dir)
        if '19Q2' == depmap_ver or '19Q3' == depmap_ver or '19Q4' == depmap_ver or '20Q' in depmap_ver:
            depmap_cell_line_file = os.path.join(depmap_dir, 'sample_info.csv')
        else:
            depmap_cell_line_file = os.path.join(depmap_dir, 'DepMap-20{}-celllines.csv'.format(depmap_ver.lower()))

        self.cell_line_metadata = pd.read_csv(depmap_cell_line_file)
        self.cell_line_metadata = self.cell_line_metadata.set_index('DepMap_ID')
        try:
            self.cell_line_id_mapping = self.cell_line_metadata['CCLE_Name'].to_dict()
            self.cell_line_id_pri_dis = self.cell_line_metadata.set_index('CCLE_Name')
        except:
            self.cell_line_id_mapping = self.cell_line_metadata['CCLE Name'].to_dict()
            self.cell_line_id_pri_dis = self.cell_line_metadata.set_index('CCLE Name')
        try:
            self.cell_line_id_pri_dis = self.cell_line_id_pri_dis['Primary Disease'].to_dict()
        except:
            self.cell_line_id_pri_dis = self.cell_line_id_pri_dis['lineage'].to_dict()
        try:
            self.cell_line_id_sub_dis = self.cell_line_metadata.set_index('CCLE_Name')
        except:
            self.cell_line_id_sub_dis = self.cell_line_metadata.set_index('CCLE Name')
        try:
            self.cell_line_id_sub_dis = self.cell_line_id_sub_dis['Subtype Disease'].to_dict()
        except:
            self.cell_line_id_sub_dis = self.cell_line_id_sub_dis['lineage_subtype'].to_dict()
        self.cell_line_id_mapping = ddict(lambda: None, self.cell_line_id_mapping)
        self.cell_line_id_pri_dis = ddict(lambda: None, self.cell_line_id_pri_dis)
        self.cell_line_id_sub_dis = ddict(lambda: None, self.cell_line_id_sub_dis)

    def load_dependency(self, depmap_ver=None, dep_data_type='Dependency'):
        depmap_genetic_vulnerabilities_dir = os.environ.get('DEPMAP_DIR')
        regex = re.compile(r"\d\dQ\d", re.IGNORECASE)
        if depmap_ver is None:
            depmap_ver = self.depmap_ver
        if '19Q2' == depmap_ver or '19Q3' == depmap_ver or '19Q4' == depmap_ver or '20Q' in depmap_ver:
            if depmap_ver not in depmap_genetic_vulnerabilities_dir:
                depmap_genetic_vulnerabilities_dir = regex.sub(depmap_ver, depmap_genetic_vulnerabilities_dir)
            if dep_data_type == 'CERES':
                depmap_file = 'Achilles_gene_effect.csv'
            elif dep_data_type == 'Dependency':
                depmap_file = 'Achilles_gene_dependency.csv'
        self.dependency = pd.read_csv(os.path.join(depmap_genetic_vulnerabilities_dir, depmap_file), header=0,
                                      index_col=0)
        self.dependency.columns = [x.split(' (')[0] for x in self.dependency.columns]
        self.dependency = self.dependency[sorted(self.dependency.columns)]
        # Map cell line id to name
        self.dependency.index = [self.cell_line_id_mapping[x] if x in self.cell_line_id_mapping else x for x in
                                 self.dependency.index]
        self.dependency = self.dependency.loc[sorted(self.dependency.index)]
        self.dependency = self.dependency.fillna(0)

    def load_rna(self, depmap_ver=None):
        depmap_genomic_characterization_dir = os.environ.get('DEPMAP_DIR')
        regex = re.compile(r"\d\dQ\d", re.IGNORECASE)
        if depmap_ver is None:
            depmap_ver = self.depmap_ver
        if '19Q2' == depmap_ver or '19Q3' == depmap_ver or '19Q4' == depmap_ver or '20Q' in depmap_ver:
            if depmap_ver not in depmap_genomic_characterization_dir:
                depmap_genomic_characterization_dir = regex.sub(depmap_ver, depmap_genomic_characterization_dir)
            depmap_file = 'CCLE_expression.csv'
        if '20Q2' in depmap_ver:
            sep_str = '\t'
        else:
            sep_str = ','
        self.rna = pd.read_csv(os.path.join(depmap_genomic_characterization_dir, depmap_file), header=0,
                               index_col=0, sep=sep_str)
        self.rna.columns = [x.split(' (')[0] for x in self.rna.columns]
        # Merge columns with the same gene symbol
        dup_genes = [item for item, count in Counter(self.rna.columns).items() if count > 1]
        unique_genes = list(set(self.rna.columns).difference(dup_genes))
        RNAseq_gene = self.rna[unique_genes]
        for col in set(dup_genes):
            RNAseq_gene[col] = self.rna[col].sum(axis=1)

        # Map cell line id to name
        RNAseq_gene.index = [self.cell_line_id_mapping[x] if x in self.cell_line_id_mapping else x for x in
                             RNAseq_gene.index]
        for cell in set(self.dependency.index).intersection(RNAseq_gene.index):
            cell_type = self.cell_line_id_pri_dis[cell]
            cell_subtype = self.cell_line_id_sub_dis[cell]
            if cell_type in disease_mapping:
                if cell not in self.cancer_type_to_patients[disease_mapping[cell_type]]:
                    self.cancer_type_to_patients[disease_mapping[cell_type]].append(cell)
            elif cell_subtype in disease_mapping:
                if cell not in self.cancer_type_to_patients[disease_mapping[cell_subtype]]:
                    self.cancer_type_to_patients[disease_mapping[cell_subtype]].append(cell)
            if cell not in self.cancer_type_to_patients[cell_type]:
                self.cancer_type_to_patients[cell_type].append(cell)

        self.rna = RNAseq_gene
        self.rna = self.rna[sorted(self.rna.columns)]
        self.rna = self.rna.loc[sorted(self.rna.index)]
        self.rna_all = self.rna.copy()

    def _subset_samples(self):
        # Get overlapping patients among data types
        overlapping_patients = set(self.dependency.index)
        for x in self.data_types:
            # Get patient ID
            overlapping_patients &= set(self.__dict__[x].index)
        if self.cancer_type == 'PANC':
            selected_samples = sorted(list(overlapping_patients))
        else:
            selected_samples = sorted(list(set(self.cancer_type_to_patients[self.cancer_type])))
        overlapping_patients &= set(selected_samples)
        overlapping_patients = sorted(list(overlapping_patients))
        for x in self.data_types:
            self.__dict__[x] = self.__dict__[x].loc[overlapping_patients]
        self.dependency = self.dependency.loc[overlapping_patients]

        logging.info("Total {} samples have {} and dependency data".format(
            len(overlapping_patients), " ".join(self.data_types)))

    def _subset_target_genes(self):
        try:
            self.genes_in_label = pd.read_csv(self.load_result_dir + '/dependency_genes.tsv', sep='\t', header=None)
            self.genes_in_label = list(self.genes_in_label.values.T[0])
        except:
            if self.use_all_dependency_gene:
                self.genes_in_label = sorted(list(set(self.community_genes).intersection(self.dependency.columns)))
            else:
                self.genes_in_label = sorted(list(set(self.genes).intersection(self.dependency.columns)))
        if len(self.select_genes_in_label) > 0:
            self.genes_in_label = sorted(list(set(self.genes_in_label).intersection(self.select_genes_in_label)))
            genes_not_found = set(self.select_genes_in_label).difference(self.genes_in_label)
            logging.debug("Genes not found: {}".format(genes_not_found))
            if 'Timestamped' not in self.__class__.__name__:
                logging.info("{} out of {} selected genes are in dependency data.".format(
                    len(self.genes_in_label) - len(genes_not_found),
                    len(self.select_genes_in_label)))
        gsp_total = (self.dependency[self.genes_in_label] >= 0.5).sum()
        cond = (gsp_total >= self.GSP_min) & (self.dependency.shape[0] - gsp_total >= self.GSN_min)
        cond_col = sorted([y for x, y in zip(cond, cond.index) if x])
        logging.info("{} genes have at least {} gold standard positives and {} negatives".format(len(cond_col),
                                                                                                 self.GSP_min,
                                                                                                 self.GSN_min))
        self.dependency = self.dependency[cond_col]
        self.genes_in_label = cond_col
        self.gsp_n = (self.dependency >= 0.5).sum().sum()
        self.gsn_n = (self.dependency < 0.5).sum().sum()
        if self.use_classification:
            logging.info("Positive:negative samples = {}:{}".format(self.gsp_n, self.gsn_n))

    def _select_feature_genes(self):
        overlapping_genes = set(self.community_genes)
        try:
            self.rna_mad = pd.read_csv(self.load_result_dir + '/RNA_mad.tsv', sep='\t', index_col=0)
            self.rna_mad.columns = [0]
        except:
            overlapping_genes &= set(self.rna.columns)
            self.rna = self.rna[sorted(list(overlapping_genes))]
            expressed_genes = ((self.rna >= 1).sum() > (self.rna.shape[0]) * self.exp_ratio_min)
            self.rna_mad = self.rna.apply(mad)
            self.rna_mad = pd.DataFrame(self.rna_mad, index=self.rna.columns)
            self.rna_mad = self.rna_mad.loc[expressed_genes]
            self.rna_mad = self.rna_mad.sort_values(by=0, ascending=False)

        self.rna_mad.to_csv(os.path.join(self.result_path, 'RNA_mad.tsv'), sep='\t')
        top_mad_genes = self.rna_mad.head(min(self.rna_top_n_std, self.rna_mad.shape[0])).index
        self.output_pred_small += list(top_mad_genes)[0:20]
        self.output_pred_small += list(top_mad_genes)[
                                  int(self.rna_top_n_std / 2 - 10):int(self.rna_top_n_std / 2 + 10)]
        self.output_pred_small += list(top_mad_genes)[-20:]

        self.rna = self.rna[top_mad_genes]
        overlapping_genes &= set(self.rna.columns)
        self.rna = self.rna[sorted(list(overlapping_genes))]
        logging.info("Total {} genes have top {} mad and gene group data".format(
            len(overlapping_genes), self.rna.shape[1]))

    def _filter_community(self):
        com_to_drop = []
        modeled_com_genes = set()
        modeled_genes = set()
        for data_type in self.data_types:
            modeled_genes |= set(self.__dict__[data_type].columns)
        for com, members in self.community_dict.items():
            if self.use_all_dependency_gene:
                self.community_dict[com] = sorted(
                    list((set(modeled_genes) & set(members)) | (set(members) & set(self.genes_in_label))))
            else:
                self.community_dict[com] = sorted(list(set(modeled_genes).intersection(members)))
            if len(self.community_dict[com]) < self.community_affected_size_min:
                com_to_drop.append(com)
            elif len(self.community_dict[com]) > self.community_affected_size_max:
                com_to_drop.append(com)
            elif len(set(members) & set(self.genes_in_label)) < 1:
                if self.require_label_gene_in_gene_group:
                    com_to_drop.append(com)
                else:
                    modeled_com_genes |= set(self.community_dict[com])
            else:
                modeled_com_genes |= set(self.community_dict[com])
        for com in com_to_drop:
            self.community_dict.pop(com, None)

    def _run_create_filter(self):
        self.feature_genes = set()
        self.genes_in_label_idx = {}
        self.idx_genes_in_label = {}
        self.community_filter = self.__create_filter(self.gene_community_dict, self.community_dict,
                                                     self.community_size_dict, random=False)

    def __create_filter(self, gene_community_dict, community_dict, community_size_dict, random=False):
        community_filter = ddict(set)
        if not random:
            self.genes_in_label_idx = {}
            self.idx_genes_in_label = {}
        i = 0
        for g in self.genes_in_label:
            coms = gene_community_dict[g]
            coms = list(set(coms) & (community_dict.keys()))
            com_size = [community_size_dict[x] for x in coms]
            community_filter[g] |= set([g])
            for s, com in sorted(zip(com_size, coms)):
                genes = set(community_dict[com])
                # Choose top n genes so that not too many features were used per gene group
                if 'ref' not in self.run_mode and self.use_all_feature_for_random_group:
                    if len(self.data_types) > 1:
                        added_genes = set(genes - community_filter[g]) & (set(self.mut.columns) | set(self.rna.columns))
                    elif 'rna' in self.data_types:
                        added_genes = set(genes - community_filter[g]) & set(self.rna.columns)
                    elif 'mut' in self.data_types:
                        added_genes = set(genes - community_filter[g]) & set(self.mut.columns)

                    if len(added_genes) == 0:
                        continue
                    if isinstance(self.feature_per_group_max, int):
                        choose_n = min(self.feature_per_group_max, len(added_genes))
                        top_genes = list(np.random.choice(list(added_genes), choose_n, replace=False))
                    elif isinstance(self.feature_per_group_max, float) and self.feature_per_group_max < 1:
                        top_n = np.ceil(len(genes) * self.feature_per_group_max)
                        choose_n = min(top_n, len(added_genes))
                        top_genes = list(np.random.choice(list(added_genes), choose_n, replace=False))
                    else:
                        raise ValueError("feature_per_group_max {} should be integer or between 0 and 1".format(
                            self.feature_per_group_max))
                else:
                    if len(self.data_types) > 1:
                        added_genes = set(genes - community_filter[g]) & (set(self.mut.columns) | set(self.rna.columns))
                        variable_genes = self.rna_mad.loc[list(added_genes)].sort_values(0, ascending=False)
                    elif 'rna' in self.data_types:
                        added_genes = set(genes - community_filter[g]) & set(self.rna.columns)
                        variable_genes = self.rna_mad.loc[list(added_genes)].sort_values(0, ascending=False)
                    elif 'mut' in self.data_types:
                        added_genes = set(genes - community_filter[g]) & set(self.mut.columns)
                        variable_genes = self.mut_freq.loc[list(added_genes)].sort_values(0, ascending=False)
                    if isinstance(self.feature_per_group_max, int):
                        top_genes = variable_genes.head(self.feature_per_group_max).index
                    elif isinstance(self.feature_per_group_max, float) and self.feature_per_group_max < 1:
                        top_n = np.ceil(len(genes) * self.feature_per_group_max)
                        top_genes = variable_genes.head(top_n).index
                    else:
                        raise ValueError("feature_per_group_max {} should be integer or between 0 and 1".format(
                            self.feature_per_group_max))
                community_filter[g] |= set(top_genes)
                if len(community_filter[g]) >= self.feature_max:
                    break
            if not random:
                if len(community_filter[g]) > 0:
                    self.genes_in_label_idx[g] = i
                    self.idx_genes_in_label[i] = g
                    i += 1
                else:
                    logging.info("Gene {} could not find feature genes".format(g))
        if not random:
            logging.info(
                "The dependency of total {} genes will be predicted".format(len(self.genes_in_label_idx.keys())))
        return community_filter

    def _build_hierarchy(self):
        leaf_communities, df = self.load_leaf_communities()
        child = leaf_communities
        # The layer having only gene children
        level = 1
        self.community_level_dict = dict()
        self.level_community_dict = dict()
        count_dict = ddict(int)
        for x in child:
            self.community_level_dict[x] = level
            count_dict[x] += 1
        self.level_community_dict[level] = child
        # logging.info("Layer {} has {} gene groups".format(level, len(child)))
        while 1:
            df_level = df.loc[df[1].isin(child)]
            if df_level.shape[0] == 0:
                break
            level += 1
            parent = sorted(list(set(df_level[0])))
            for parent_group in parent:
                self.community_level_dict[parent_group] = level
                count_dict[parent_group] += 1
            self.level_community_dict[level] = parent
            child = parent
        # Make the layer number of each community unique
        self.level_community_dict = ddict(list)
        for g, level in self.community_level_dict.items():
            self.level_community_dict[level].append(g)
        for level, groups in sorted(self.level_community_dict.items()):
            logging.info("Layer {} has {} gene groups".format(level, len(groups)))

        gene_groups_all = sorted(list(self.community_dict.keys())) + ['root']
        logging.info(
            "Total {} layers of {} gene groups in the hierarchy including the root".format(level, len(gene_groups_all)))

        feature_genes_all = []
        self.feature_n = []
        np.random.RandomState(self.params['seeds'][0])
        for data_type in self.data_types:
            feat_n = len(self.__dict__[data_type].columns)
            self.feature_n.append(feat_n)
            # Randomly reselect features for each feature matrix
            if 'full' in self.run_mode and self.use_all_feature_for_fully_net:
                feat_pool = sorted(list(self.__dict__[data_type + '_all'].columns))
                feature_genes_all += feat_pool
                cell_idx = self.__dict__[data_type].index
                self.__dict__[data_type] = self.__dict__[data_type + '_all'].loc[cell_idx, feat_pool]
                logging.info(
                    "Use all {} genes from {} as features to form fully connected networks".format(feat_n, data_type))
            elif 'ref' not in self.run_mode and self.use_all_feature_for_random_group:
                feat_pool = list(self.__dict__[data_type + '_all'].columns)
                # Require gene labels in the features
                pre_select = set(feat_pool) & set(self.genes_in_label)
                feat_pool = sorted(list(set(feat_pool) - set(self.genes_in_label)))
                random_feat = sorted(list(np.random.choice(feat_pool, feat_n - len(pre_select), replace=False)))
                feature_genes_all += random_feat + list(pre_select)
                feature_genes_all = sorted(feature_genes_all)
                cell_idx = self.__dict__[data_type].index
                self.__dict__[data_type] = self.__dict__[data_type + '_all'].loc[cell_idx, random_feat]
                logging.info(
                    "Randomly select {} genes including {} gene of prediction from {} as features to form random gene groups".format(
                        feat_n, len(self.genes_in_label), data_type))
            else:
                feature_genes_all += sorted(list(self.__dict__[data_type].columns))

        del_genes_all = sorted(list(self.genes_in_label_idx.keys()))
        self.feature_n.append(len(del_genes_all))
        self.genes_in_label = del_genes_all
        self.save_label_genes(self.genes_in_label)
        self.y = self.dependency[self.genes_in_label]
        self.y_binary = ((self.y >= 0.5) + 0).astype(int)
        # The order of indexed genes and gen groups:
        if self.use_deletion_vector:
            entity_all = feature_genes_all + del_genes_all + gene_groups_all
        else:
            entity_all = feature_genes_all + gene_groups_all
        self.idx_name = {i: k for i, k in enumerate(entity_all)}
        name_idx = ddict(list)
        for k, v in self.idx_name.items():
            name_idx[v].append(k)
        if len(self.data_types) > 1:
            self.mut_genes_idx = {}
            self.rna_genes_idx = {}
            for k, v in name_idx.items():
                for idx in v:
                    if idx < self.feature_n[0]:
                        self.mut_genes_idx[k] = idx
                    elif self.feature_n[0] <= idx < self.feature_n[0] + self.feature_n[1]:
                        self.rna_genes_idx[k] = idx
        self.feature_genes_idx = {x: min(name_idx[x]) for x in feature_genes_all}
        self.del_genes_idx = {x: max(name_idx[x]) for x in del_genes_all}
        self.gene_group_idx = {x: name_idx[x][0] for x in gene_groups_all}
        self.community_hierarchy_dicts_all = {'idx_name': self.idx_name,
                                              'feature_genes_idx': self.feature_genes_idx,
                                              'del_genes_idx': self.del_genes_idx,
                                              'gene_group_idx': self.gene_group_idx}
        self.child_map_all = []
        self.child_map_all_random = []
        self.child_map_all_ones = []
        feature_only_genes = set(feature_genes_all) - set(del_genes_all)
        dep_only_genes = set(del_genes_all) - set(feature_genes_all)
        feature_dep_both_genes = set(feature_genes_all) & set(del_genes_all)
        gene_pool = sorted(list(set(feature_genes_all) | set(del_genes_all)))
        self.community_filter_random = ddict(list)
        if 'Timestamped' in self.__class__.__name__ or 'Sensitivity' in self.__class__.__name__:
            self.load_random_communities()
            random_hierarchy = self.load_random_hierarchy()
        else:
            self.community_dict_random = {}
            random_hierarchy = pd.DataFrame()

        self.gene_community_dict_random = ddict(list)
        self.community_size_dict_random = {}
        prng = np.random.RandomState(self.params['seeds'][0])
        logging.info("Building gene group hierarchy")
        if self.run_mode == 'random':
            idx_gene_pool = {i: g for i, g in enumerate(gene_pool)}
            gene_pool_idx = {g: i for i, g in enumerate(gene_pool)}
            partially_shuffled_membership = self.__partially_shuffle_gene_group(gene_pool, gene_pool_idx)
            idx_gene_group = {i: g for g, i in self.gene_group_idx.items()}
            partially_shuffled_relation = self.__partially_shuffle_gene_group_hierarchy(df, idx_gene_group)
        else:
            partially_shuffled_membership = None
            partially_shuffled_relation = None
            idx_gene_group = None
            idx_gene_pool = None

        min_group_idx = min(self.gene_group_idx.values())
        for group, idx in sorted(self.gene_group_idx.items()):
            if group in self.community_dict:
                genes = self.community_dict[group]
                gene_idx = self._genes_to_feat_del_idx(genes)
                if 'Timestamped' in self.__class__.__name__ or 'Sensitivity' in self.__class__.__name__:
                    genes_random = self.community_dict_random[group]
                else:
                    if partially_shuffled_membership is not None:
                        genes_random_idx = partially_shuffled_membership[idx - min_group_idx].nonzero()[0]
                        genes_random = sorted([idx_gene_pool[x] for x in genes_random_idx])
                    else:
                        if self.use_consistant_groups_for_labels:
                            gene_pool = sorted(list(set(gene_pool) - set(self.genes_in_label)))
                            pre_select = set(genes) & set(self.genes_in_label)
                            if len(set(genes) & set(self.genes_in_label)) > 0:
                                random_feat = list(prng.choice(gene_pool, len(genes) - len(pre_select), replace=False))
                                genes_random = sorted(random_feat + list(pre_select))
                            else:
                                genes_random = sorted(
                                    list(prng.choice(gene_pool, len(genes) - len(pre_select), replace=False)))
                        else:
                            genes_random = sorted(list(prng.choice(gene_pool, len(genes), replace=False)))
                    self.community_dict_random[group] = genes_random
                for g in genes_random:
                    self.gene_community_dict_random[g].append(group)
                self.community_size_dict_random[group] = len(genes_random)

                feat_genes = set(genes_random) & set(self.feature_genes_idx.keys())
                del_genes = set(genes_random) & set(self.del_genes_idx.keys())
                if len(self.data_types) > 1:
                    feat_gene_idx = []
                    for g in feat_genes:
                        if g in self.mut_genes_idx:
                            feat_gene_idx.append(self.mut_genes_idx[g])
                        if g in self.rna_genes_idx:
                            feat_gene_idx.append(self.rna_genes_idx[g])
                else:
                    feat_gene_idx = [self.feature_genes_idx[x] for x in feat_genes]
                if self.use_deletion_vector:
                    del_gene_idx = [self.del_genes_idx[x] for x in del_genes]
                else:
                    del_gene_idx = []
                gene_idx_random = feat_gene_idx + del_gene_idx
            else:
                gene_idx = []
                gene_idx_random = []

            child = sorted(df.loc[df[0] == group, 1].tolist())
            child_idx = sorted([self.gene_group_idx[x] for x in child if x in self.gene_group_idx])
            self.child_map_all.append(sorted(gene_idx + child_idx))
            if len(self.child_map_all[-1]) == 0:
                logging.info("Gene group {} does not have children".format(group))
            # Build random group hierarchy
            if 'Timestamped' in self.__class__.__name__ or 'Sensitivity' in self.__class__.__name__:
                child_random = sorted(random_hierarchy.loc[random_hierarchy[0] == group, 1].tolist())
                child_idx_random = sorted([self.gene_group_idx[x] for x in child_random if x in self.gene_group_idx])
            else:
                if partially_shuffled_relation is not None:
                    child_idx_random = partially_shuffled_relation[idx - min_group_idx, :].nonzero()[0]
                    child_idx_random = [x + min_group_idx for x in child_idx_random]
                    child_random = sorted([idx_gene_group[x] for x in child_idx_random])
                else:
                    child_idx_random = []
                    child_random = []
                    for c in child:
                        child_level = self.community_level_dict[c]
                        random_child = prng.choice(self.level_community_dict[child_level], 1, replace=False)[0]
                        child_random.append(random_child)
                        random_c_idx = self.gene_group_idx[random_child]
                        child_idx_random.append(random_c_idx)

                for rc in sorted(child_random):
                    random_hierarchy = pd.concat([random_hierarchy, pd.DataFrame([group, rc]).T], axis=0)
            self.child_map_all_random.append(sorted(gene_idx_random + child_idx_random))
            try:
                assert len(gene_idx) == len(gene_idx_random), "Random gene number does not match"
            except AssertionError:
                pass

            # Children for fully connected neural networks
            if group in leaf_communities:
                gene_idx_ones = list(self.feature_genes_idx.values())
            else:
                gene_idx_ones = []
            parent_level = self.community_level_dict[group]
            child_level = parent_level - 1
            if child_level in self.level_community_dict:
                child_ones = self.level_community_dict[child_level]
            else:
                child_ones = []
            child_idx_ones = [self.gene_group_idx[x] for x in child_ones if x in self.gene_group_idx]
            self.child_map_all_ones.append(sorted(gene_idx_ones + child_idx_ones))

        self.save_communities(self.community_dict_random)
        # Save random hierarchy as file
        random_hierarchy.to_csv(os.path.join(self.result_path, 'random_group_hierarchy.tsv'),
                                index=None, sep='\t', header=None)
        self.community_filter_random = self.__create_filter(self.gene_community_dict_random, self.community_dict_random,
                                                            self.community_size_dict_random, random=True)

        self.community_filter_map = []
        self.community_filter_map_random = []
        feature_n = len(feature_genes_all)
        for g in del_genes_all:
            feat_genes = set(self.community_filter[g])
            if len(self.data_types) > 1:
                feat_gene_idx = []
                for g in feat_genes:
                    if g in self.mut_genes_idx:
                        feat_gene_idx.append(self.mut_genes_idx[g])
                    if g in self.rna_genes_idx:
                        feat_gene_idx.append(self.rna_genes_idx[g])
                feat_gene_idx = sorted(feat_gene_idx)
            else:
                feat_gene_idx = sorted([self.feature_genes_idx[x] for x in feat_genes if x in self.feature_genes_idx])
            feat_genes_array = np.zeros(feature_n)
            feat_genes_array[feat_gene_idx] = 1
            self.community_filter_map.append(feat_genes_array)
            feat_genes_random = set(self.community_filter_random[g])
            if len(self.data_types) > 1:
                feat_genes_random_idx = []
                for g in feat_genes:
                    if g in self.mut_genes_idx:
                        feat_genes_random_idx.append(self.mut_genes_idx[g])
                    if g in self.rna_genes_idx:
                        feat_genes_random_idx.append(self.rna_genes_idx[g])
                feat_genes_random_idx = sorted(feat_genes_random_idx)
            else:
                feat_genes_random_idx = sorted(
                    [self.feature_genes_idx[x] for x in feat_genes_random if x in self.feature_genes_idx])
            feat_genes_array = np.zeros(feature_n)
            feat_genes_array[feat_genes_random_idx] = 1
            self.community_filter_map_random.append(feat_genes_array)

    def __partially_shuffle_gene_group(self, gene_pool, gene_pool_idx):
        group_gene_membership_matrix = np.zeros([len(self.gene_group_idx), len(gene_pool)])
        min_group_idx = min(self.gene_group_idx.values())
        for group, idx in sorted(self.gene_group_idx.items()):
            if group in self.community_dict:
                idx -= min_group_idx
                genes = self.community_dict[group]
                gene_idx = [gene_pool_idx[gene] for gene in genes]
                group_gene_membership_matrix[idx, gene_idx] = 1
        all_idx = group_gene_membership_matrix.nonzero()
        prng = np.random.RandomState(self.random_group_permutation_seed)
        shuffled_number = int(self.random_group_permutation_ratio * len(all_idx[0]))
        shuffled_relationship_idx = prng.choice(range(len(all_idx[0])), shuffled_number, replace=False)
        logging.info(
            f"{self.random_group_permutation_ratio*100}% ({shuffled_number}) of gene membership was randomly shuffled")
        # No shuffling
        if self.random_group_permutation_ratio == 0:
            return group_gene_membership_matrix
        connections_to_shuffled = np.zeros([len(self.gene_group_idx), len(gene_pool)])
        connections_to_shuffled[all_idx[0][shuffled_relationship_idx], all_idx[1][shuffled_relationship_idx]] = 1
        partially_shuffled_membership = np.zeros([len(self.gene_group_idx), len(gene_pool)])
        for i in range(group_gene_membership_matrix.shape[0]):
            original = group_gene_membership_matrix[i].nonzero()[0]
            to_shuffled = connections_to_shuffled[i].nonzero()[0]
            if len(to_shuffled) > 0:
                keep = list(set(original) - set(to_shuffled))
                pool = sorted(list(set(range(len(group_gene_membership_matrix[i]))) - set(keep)))
                after_shuffled = list(prng.choice(pool, len(to_shuffled), replace=False))
                partially_shuffled_membership[i][keep + after_shuffled] = 1
            else:
                partially_shuffled_membership[i][original] = 1

        return partially_shuffled_membership

    def __partially_shuffle_gene_group_hierarchy(self, df, idx_gene_group):
        gene_group_relation_matrix = np.zeros([len(self.gene_group_idx), len(self.gene_group_idx)])
        min_group_idx = min(self.gene_group_idx.values())
        for _, row in df.iterrows():
            parent = self.gene_group_idx[row[0]] - min_group_idx
            child = self.gene_group_idx[row[1]] - min_group_idx
            gene_group_relation_matrix[parent, child] = 1

        all_idx = gene_group_relation_matrix.nonzero()
        prng = np.random.RandomState(self.random_group_permutation_seed)
        shuffled_number = int(self.random_group_hierarchy_permutation_ratio * len(all_idx[0]))
        shuffled_relationship_idx = prng.choice(range(len(all_idx[0])), shuffled_number, replace=False)
        logging.info(
            f"{self.random_group_hierarchy_permutation_ratio*100}% ({shuffled_number}) of gene group hierarchy was randomly shuffled")
        connections_to_shuffled = np.zeros(gene_group_relation_matrix.shape)
        connections_to_shuffled[all_idx[0][shuffled_relationship_idx], all_idx[1][shuffled_relationship_idx]] = 1
        partially_shuffled_relation = np.zeros(gene_group_relation_matrix.shape)
        # No shuffling
        if self.random_group_hierarchy_permutation_ratio == 0:
            return gene_group_relation_matrix
        # Shuffle child group for each parent
        for i in range(gene_group_relation_matrix.shape[0]):
            original = gene_group_relation_matrix[i].nonzero()[0]
            to_shuffled = connections_to_shuffled[i].nonzero()[0]
            if len(to_shuffled) > 0:
                keep = list(set(original) - set(to_shuffled))
                children = [idx_gene_group[x + min_group_idx] for x in to_shuffled]
                child_levels = [self.community_level_dict[child] for child in children]
                after_shuffled = []
                for child_level in child_levels:
                    random_child = prng.choice(self.level_community_dict[child_level], 1, replace=False)[0]
                    random_child_idx = self.gene_group_idx[random_child] - min_group_idx
                    after_shuffled.append(random_child_idx)
                after_shuffled = list(set(after_shuffled))
                partially_shuffled_relation[i][keep + after_shuffled] = 1
            else:
                partially_shuffled_relation[i][original] = 1
        return partially_shuffled_relation

    def _genes_to_feat_del_idx(self, genes):
        feat_genes = set(genes) & set(self.feature_genes_idx.keys())
        del_genes = set(genes) & set(self.del_genes_idx.keys())
        if len(self.data_types) > 1:
            feat_gene_idx = []
            for g in feat_genes:
                if g in self.mut_genes_idx:
                    feat_gene_idx.append(self.mut_genes_idx[g])
                if g in self.rna_genes_idx:
                    feat_gene_idx.append(self.rna_genes_idx[g])
        else:
            feat_gene_idx = [self.feature_genes_idx[x] for x in feat_genes]
        if self.use_deletion_vector:
            del_gene_idx = [self.del_genes_idx[x] for x in del_genes]
        else:
            del_gene_idx = []
        gene_idx = feat_gene_idx + del_gene_idx
        return gene_idx

    def _get_genes_in_child_group(self, group, genes_in_child_gene_group=set()):
        _, df = self.load_leaf_communities()
        children = df.loc[df[0] == group, 1].tolist()
        for child in children:
            if child in self.community_dict:
                genes = self.community_dict[child]
                genes_in_child_gene_group |= set(genes)
                self._get_genes_in_child_group(child, genes_in_child_gene_group)
        return genes_in_child_gene_group

    def align_data(self):
        self._subset_samples()
        self._subset_target_genes()
        self._select_feature_genes()
        self._filter_community()
        self._run_create_filter()
        if len(self.data_types) > 1:
            self.X = pd.concat([self.mut, self.rna], axis=1)
        else:
            self.X = self.__dict__[self.data_types[0]]
        self.X_all = self.X
        self._build_hierarchy()
        # self._refine_community()
        logging.info("Generating data splits for {} repeats and {} folds".format(self.repeat_n, self.fold_n))
        self.split_data()

    def split_data(self):
        self.split_idx = dict()
        for repeat in range(self.repeat_n):
            seed = self.params['seeds'][repeat]
            if self.split_by_cancer_type and self.cancer_type == 'PANC':
                cancer_type_id = ddict(list)
                for x in self.X.index:
                    t = '_'.join(x.split('_')[1:])
                    cancer_type_id[t].append(x)
                self.split_idx[repeat] = [ddict(list) for _ in range(self.fold_n)]
                for j, (cancer_type, idx) in enumerate(cancer_type_id.items()):
                    logging.debug("{} has {} cell lines".format(cancer_type, len(idx)))
                    if len(idx) >= self.fold_n + 1:
                        logging.debug("{} has {} cell lines splitting".format(cancer_type, len(idx)))
                        split_subidx = self._split_data(self.X.loc[idx], self.y.loc[idx], seed)
                        for fold, split_dict in enumerate(split_subidx):
                            for split_type in split_dict.keys():
                                self.split_idx[repeat][fold][split_type] += list(split_dict[split_type])
                if 'Timestamped' in self.__class__.__name__ or 'Sensitivity' in self.__class__.__name__:
                    target_idx = set(self.dependency_target.index) & set(self.rna_target_all.index)
                    target_idx_only = target_idx - set(self.dependency.index)
                    target_idx_only = sorted(list(target_idx_only))
                    for fold in range(len(self.split_idx[repeat])):
                        self.split_idx[repeat][fold]['test'] = target_idx_only
                    self.X_all = pd.concat([self.X_all, self.rna_target.loc[target_idx_only, self.X_all.columns]])
                    self.y = pd.concat([self.y, self.dependency_target.loc[target_idx_only, self.y.columns]])
                    y_binary_target = ((self.y.loc[target_idx_only] >= 0.5) + 0).astype(int)
                    self.y_binary = pd.concat([self.y_binary, y_binary_target])
            else:
                self.split_idx[repeat] = self._split_data(self.X, self.y, seed)

    def _split_data(self, X, y, seed):
        kf1 = KFold(n_splits=self.fold_n, random_state=seed)
        split_idx = []
        for fold, (train_index, test_index) in enumerate(kf1.split(X, y)):
            split_dict = dict()
            split_dict['test'] = list(X.index[test_index])
            # Generate validation data by splitting part of training data
            X_train, y_train = X.loc[X.index[train_index]], y.loc[X.index[train_index]]
            if X_train.shape[0] < self.fold_n:
                return []
            kf = KFold(n_splits=self.fold_n, random_state=seed)
            for fold_2, (train_index, test_index) in enumerate(kf.split(X_train, y_train)):
                split_dict['train'] = list(X_train.index[train_index])
                split_dict['val'] = list(X_train.index[test_index])
                if fold_2 == fold:  # Taking the different splits to differentiate it
                    break
            split_idx.append(split_dict)
        return split_idx

    def get_split_data(self, i, j):
        self.idx['train'] = self.split_idx[i][j]['train']
        self.idx['val'] = self.split_idx[i][j]['val']
        self.idx['test'] = self.split_idx[i][j]['test']
        if self.use_binary_dependency:
            y = self.y_binary
        else:
            y = self.y
        self.X_train, self.y_train = self.X_all.loc[self.idx['train']].values, y.loc[self.idx['train']].values
        self.X_val, self.y_val = self.X_all.loc[self.idx['val']].values, y.loc[self.idx['val']].values
        self.X_test, self.y_test = self.X_all.loc[self.idx['test']].values, y.loc[self.idx['test']].values
        if 'cl3_' in self.model_v or 'cl5_' in self.model_v:
            scaler = StandardScaler()
            self.y_train2 = scaler.fit_transform(self.y_train)
            self.y_val2 = scaler.transform(self.y_val)
            self.y_test2 = scaler.transform(self.y_test)
        elif 'clh_' in self.model_v:
            self.y_train2 = self.y_train
            self.y_val2 = self.y_val
            self.y_test2 = self.y_test
        else:
            self.y_train2 = None
            self.y_val2 = None
            self.y_test2 = None
        logging.info("Repeat {}, fold {}".format(i, j))
        logging.info("Training data shape X: {}, y: {}".format(self.X_train.shape, self.y_train.shape))
        logging.info("y label counts: {}".format(Counter(np.argmax(self.y_train, axis=1))))
        logging.info("Validation data shape X: {}, y: {}".format(self.X_val.shape, self.y_val.shape))
        logging.info("y label counts: {}".format(Counter(np.argmax(self.y_val, axis=1))))
        logging.info("Test data shape X: {}, y: {}".format(self.X_test.shape, self.y_test.shape))
        logging.info("y label counts: {}".format(Counter(np.argmax(self.y_test, axis=1))))

    def perform(self, model_name, params=None):
        if params is None:
            params = self.params
        save_params(self.result_path, params)
        if self.cv_fold != 0:
            if 'models' in params and 'random_forest' in self.run_mode:
                self.perform_cv('random_forest', params)
            else:
                self.perform_cv(model_name, params)
        else:
            self.prepare_data()
            # self.community_filter_ones = np.ones(self.community_filter.shape)
            model_name_base = model_name
            for repeat in range(self.repeat_n):
                params['seed'] = params['seeds'][repeat]
                # self.community_matrix_random = lil_matrix(self.community_matrix.shape)
                np.random.seed(params['seed'])

                if 'clh_v' in self.model_v:
                    mask = self.child_map_all
                    mask_random = self.child_map_all_random
                    mask_ones = self.child_map_all_ones
                else:
                    mask = self.community_hierarchy
                    mask_random = self.community_hierarchy_random
                    mask_ones = self.community_hierarchy_ones

                for fold in range(len(self.split_idx[repeat])):
                    model_suffix = str(params['seed']) + 'repeat' + str(repeat) + 'fold' + str(fold)
                    self.get_split_data(repeat, fold)
                    self.calculate_weights()
                    self.normalize_data()
                    if 'ref' in self.run_mode:
                        self.run_exp(model_name_base, model_suffix,
                                     params, mask, repeat, fold, self.community_filter_map)
                    elif 'random_forest' in self.run_mode.lower():
                        self.run_exp('random_forest', model_suffix,
                                     params, mask, repeat, fold, None, mask_ones)
                    elif 'random_predictor' in self.run_mode:
                        self.run_exp('random_predictor', model_suffix,
                                     params, mask_random, repeat, fold, self.community_filter_map_random)
                    elif 'random' in self.run_mode:
                        self.run_exp('random_control', model_suffix,
                                     params, mask_random, repeat, fold, self.community_filter_map_random)
                    elif 'expression_control' in self.run_mode:
                        self.run_exp('expression_control', model_suffix,
                                     params, mask_random, repeat, fold, self.community_filter_map_random)
                    elif 'full' in self.run_mode:
                        self.run_exp('gene_control', model_suffix,
                                     params, mask, repeat, fold, None, mask_ones)

    def calculate_weights(self):
        if self.use_class_weights:
            gsp_n = (self.y_train >= 0.5).sum().sum()
            gsn_n = (self.y_train < 0.5).sum().sum()
            if self.use_normalized_class_weights:
                self.class_weight_neg = (gsp_n + gsn_n) / (2.0 * (gsn_n))
                self.class_weight_pos = (gsp_n + gsn_n) / (2.0 * (gsp_n))
            else:
                self.class_weight_neg = gsp_n / gsn_n
                self.class_weight_pos = 1
        else:
            self.class_weight_neg = None
            self.class_weight_pos = None
        if self.use_sample_class_weights:
            gsp_n = (self.y_train >= 0.5).sum(axis=0)
            gsn_n = (self.y_train < 0.5).sum(axis=0)
            if self.use_normalized_sample_class_weights:
                self.sample_class_weight_neg = (gsp_n + gsn_n) / (2.0 * (gsn_n))
                self.sample_class_weight_pos = (gsp_n + gsn_n) / (2.0 * (gsp_n))
            else:
                self.sample_class_weight_neg = gsp_n / gsn_n
                self.sample_class_weight_pos = np.array([1] * len(gsn_n))
        else:
            self.sample_class_weight_neg = None
            self.sample_class_weight_pos = None

    def split_data_cv(self):
        self.split_idx_cv = ddict(list)
        for repeat in range(self.repeat_n):
            seed = self.params['seeds'][repeat]
            kf1 = KFold(n_splits=self.cv_fold, random_state=seed)
            idx = sorted(list(self.idx['train']) + list(self.idx['val']))
            X_train_val = self.X_all.loc[idx]
            y_train_val = self.y.loc[idx]
            for train_index, val_index in kf1.split(X_train_val, y_train_val):
                split_dict = {}
                split_dict['train'] = X_train_val.index[train_index]
                split_dict['val'] = X_train_val.index[val_index]
                self.split_idx_cv[repeat].append(split_dict)

    def get_split_data_cv(self, i, j):
        self.idx['train'] = self.split_idx_cv[i][j]['train']
        self.idx['val'] = self.split_idx_cv[i][j]['val']
        if self.use_binary_dependency:
            y = self.y_binary
        else:
            y = self.y
        self.X_train, self.y_train = self.X_all.loc[self.idx['train']].values, y.loc[self.idx['train']].values
        self.X_val, self.y_val = self.X_all.loc[self.idx['val']].values, y.loc[self.idx['val']].values
        if 'cl3_' in self.model_v or 'cl5_' in self.model_v:
            scaler = StandardScaler()
            self.y_train2 = scaler.fit_transform(self.y_train)
            self.y_val2 = scaler.transform(self.y_val)
            self.y_test2 = scaler.transform(self.y_test)
        elif 'clh_' in self.model_v:
            self.y_train2 = self.y_train
            self.y_val2 = self.y_val
            self.y_test2 = self.y_test
        else:
            self.y_train2 = None
            self.y_val2 = None
            self.y_test2 = None
        logging.info("Repeat {}, cv_fold {}".format(i, j))
        logging.info("Training data shape X: {}, y: {}".format(self.X_train.shape, self.y_train.shape))
        logging.info("y label counts: {}".format(Counter(np.argmax(self.y_train, axis=1))))
        logging.info("Validation data shape X: {}, y: {}".format(self.X_val.shape, self.y_val.shape))
        logging.info("y label counts: {}".format(Counter(np.argmax(self.y_val, axis=1))))

    def _normalize_rna(self, X_train, X_val, X_test):
        # scaler = MinMaxScaler()
        # self.X_train = scaler.fit_transform(self.X_train)
        # self.X_val = scaler.transform(self.X_val)
        # self.X_test = scaler.transform(self.X_test)
        # self.X_train = np.log2(self.X_train + 1)
        # self.X_val = np.log2(self.X_val + 1)
        # self.X_test = np.log2(self.X_test + 1)
        if self.use_StandardScaler:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            # feature_no_info = ((self.X_train.sum(axis=0) == 0) + 0).nonzero()[0]
            X_val = scaler.transform(X_val)
            # self.X_val[self.X_val > self.X_train.max()] = self.X_train.max()
            # self.X_val[:, feature_no_info] = 0
            X_test = scaler.transform(X_test)

        if self.use_sigmoid_feature:
            X_train = 1 / (1 + np.exp(-X_train))
            X_val = 1 / (1 + np.exp(-X_val))
            X_test = 1 / (1 + np.exp(-X_test))

        if self.use_tanh_feature:
            X_train = np.tanh(X_train)
            X_val = np.tanh(X_val)
            X_test = np.tanh(X_test)

        if self.use_MinMaxScaler:
            scaler = MinMaxScaler(feature_range=(0, 1))
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
            if self.clip_Xval_Xtest is not None:
                logging.info("Before cliping,\n"
                             "Val data (min,max) = ({}, {})\n"
                             "Test data (min,max) = ({}, {})".format(
                    X_val.min(),
                    X_val.max(),
                    X_test.min(),
                    X_test.max(),
                ))
                X_val = np.clip(X_val, self.clip_Xval_Xtest[0], self.clip_Xval_Xtest[1])
                X_test = np.clip(X_test, self.clip_Xval_Xtest[0], self.clip_Xval_Xtest[1])
        return X_train, X_val, X_test

    def normalize_data(self):
        self.X_train = np.nan_to_num(self.X_train)
        self.X_val = np.nan_to_num(self.X_val)
        self.X_test = np.nan_to_num(self.X_test)
        self.X_train, self.X_val, self.X_test = self._normalize_rna(self.X_train, self.X_val, self.X_test)

    def run_exp(self, model_name, model_suffix, params, com_mat, repeat, fold,
                community_filter=None, com_mat_fully=None):
        logging.info("Running {} repeat {} fold {}".format(model_name, repeat, fold))
        output_prefix = model_name + model_suffix
        if 'random_predictor' in model_name:
            self.compute_metric(None, 'test', model_name, model_suffix, self.y_train, self.y_test, com_mat, repeat,
                                self.y_test2)
        elif 'mean_control' in model_name:
            # self.compute_metric(cm, 'train', model_name, model_suffix, self.y_train, self.y_train, com_mat, repeat,
            #                     self.y_train2)
            # self.compute_metric(cm, 'val', model_name, model_suffix, self.y_train, self.y_val, com_mat, repeat,
            #                     self.y_val2)
            self.compute_metric(None, 'test', model_name, model_suffix, self.y_train, self.y_test, com_mat, repeat,
                                self.y_test2)
        elif 'expression_control' in model_name:
            self.compute_metric(None, 'test', model_name, model_suffix, self.X_test, self.y_test, com_mat, repeat,
                                self.y_test2)
        elif 'random_forest' in model_name:
            sk_all = []
            params['n_jobs'] = -1
            for i in range(self.y_train.shape[1]):
                sk = SklearnModel(model_name + model_suffix, params)
                sk.train(self.X_train, self.y_train[:, i])
                sk_all.append(sk)

            self.compute_metric(sk_all, 'train', model_name, model_suffix, self.X_train, self.y_train, com_mat, repeat,
                                self.y_train2)
            self.compute_metric(sk_all, 'val', model_name, model_suffix, self.X_val, self.y_val, com_mat, repeat,
                                self.y_val2)
            self.compute_metric(sk_all, 'test', model_name, model_suffix, self.X_test, self.y_test, com_mat, repeat,
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
            if hasattr(self, 'load_result_dir'):
                load_ckpt = os.path.join(self.load_result_dir,
                                         '{}_{}_{}.tar'.format(model_name + model_suffix, self.model_v,
                                                               params['seed']))
                cm.train(self.X_train, com_mat, self.y_train, load_weight_dir=load_ckpt, mask_fully=com_mat_fully)
            else:
                y_val_index = self.idx['val']
                y_col = self.y.columns
                cm.train(self.X_train, com_mat, self.y_train, None, self.X_val, self.y_val, y_train2=self.y_train2,
                         y_val2=self.y_val2, output_prefix=output_prefix, y_val_index=y_val_index, y_col=y_col,
                         mask_fully=com_mat_fully)
                self._clear_gpu(model_name, model_suffix)
                cm.train(self.X_train, com_mat, self.y_train, mask_fully=com_mat_fully)
            # self.analyze_weights(cm, model_name, model_suffix)
            self._clear_gpu(model_name, model_suffix)
            self.compute_metric(cm, 'train', model_name, model_suffix, self.X_train, self.y_train, com_mat, repeat,
                                self.y_train2)
            self._clear_gpu(model_name, model_suffix)
            self.compute_metric(cm, 'val', model_name, model_suffix, self.X_val, self.y_val, com_mat, repeat,
                                self.y_val2)
            self._clear_gpu(model_name, model_suffix)
            self.compute_metric(cm, 'test', model_name, model_suffix, self.X_test, self.y_test, com_mat, repeat,
                                self.y_test2)
            self._clear_gpu(model_name, model_suffix)
        model_suffix = str(params['seed']) + 'repeat' + str(repeat)
        self.compute_metric_all_test('test', model_name, model_suffix, self.X_test, self.y_test, repeat)

        self.output_metric()

    def run_exp_cv(self, model_name, model_suffix, params, com_mat, repeat, fold,
                   community_filter=None, com_mat_fully=None, grid_name=None):
        logging.info("Running {}".format(model_suffix))
        if 'random_forest' in self.run_mode:
            embed()
            sys.exit(0)
            params['n_jobs'] = -1
            sk = SklearnModel(model_name + model_suffix, params)
            sk.train(self.X_train, self.y_train)
            self.compute_metric(sk, 'train', model_name, model_suffix, self.X_train, self.y_train, com_mat, repeat)
            self.compute_metric(sk, 'val', model_name, model_suffix, self.X_val, self.y_val, com_mat, repeat)

            # sk_all = []
            # for i in range(self.y_train.shape[1]):
            #     sk = SklearnModel(model_name + model_suffix, params)
            #     sk.train(self.X_train, self.y_train[:, i])
            #     sk_all.append(sk)
            #
            # self.compute_metric(sk_all, 'train', model_name, model_suffix, self.X_train, self.y_train, com_mat, repeat)
            # self.compute_metric(sk_all, 'val', model_name, model_suffix, self.X_val, self.y_val, com_mat, repeat)
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

            y_val_index = self.idx['val']
            y_col = self.y.columns
            output_prefix = model_name + model_suffix
            cm.train(self.X_train, com_mat, self.y_train, None, self.X_val, self.y_val, y_train2=self.y_train2,
                     y_val2=self.y_val2, output_prefix=output_prefix, y_val_index=y_val_index, y_col=y_col,
                     mask_fully=com_mat_fully)
            self._clear_gpu(model_name, model_suffix)
            cm.train(self.X_train, com_mat, self.y_train)
            self.compute_metric(cm, 'val', model_name, model_suffix, self.X_val, self.y_val, com_mat, repeat,
                                self.y_val2)
            self._clear_gpu(model_name, model_suffix)
            if not self.save_model_ckpt:
                cm._rm_ckpt()
        self.output_metric()

        model_suffix = str(params['seed']) + 'repeat' + str(repeat) + '_' + grid_name
        self.compute_metric_all_test('val', model_name, model_suffix, self.X_test, self.y_test, repeat)
        self.output_metric()
        metric_output = {}
        for x in self.metric_output:
            if self.metric_output[x].shape[0] > 0:
                df = self.metric_output[x].copy()
                df = df.loc[['fold' not in y for y in df.index]]
                if df.shape[0] > 0:
                    grid_df = self.grid_df.copy().T
                    grid_df.index = df.index
                    metric_output[x] = pd.concat([df, grid_df], axis=1)
        self.output_metric(metric_output, '_all')

    def perform_cv(self, model_name, params):
        grid = ParameterGrid(params['grid_search'])
        params_backbone = params.copy()
        self.grid_df = pd.DataFrame()
        logging.info("{} points are searching in grid".format(len(grid)))
        for i, param_grid in enumerate(grid):
            self.grid_df = pd.concat([self.grid_df, pd.Series(param_grid, name=i)], axis=1)
            self.grid_df.to_csv(os.path.join(self.result_path, 'grid_cv.tsv'), sep="\t")
            grid_name = 'grid_{}'.format(i)
            logging.info("Running cross-validation for \n{}".format(self.grid_df[i]))
            params.update(param_grid)
            self.__dict__.update(param_grid)
            self.prepare_data()
            for repeat in range(self.repeat_n):
                params['seed'] = params['seeds'][repeat]
                # self.community_matrix_random = lil_matrix(self.community_matrix.shape)
                np.random.seed(params['seed'])
                if 'clh_v' in self.model_v:
                    mask = self.child_map_all
                    mask_random = self.child_map_all_random
                    mask_ones = self.child_map_all_ones
                else:
                    mask = self.community_hierarchy
                    mask_random = self.community_hierarchy_random
                    mask_ones = self.community_hierarchy_ones
                self.split_data()
                for cv_fold in range(self.fold_n):
                    model_suffix = str(params['seed']) + 'repeat' + str(
                        repeat) + '_' + grid_name + '_' + 'cv_fold' + str(cv_fold)
                    self.get_split_data(repeat, cv_fold)
                    self.calculate_weights()
                    self.normalize_data()
                    if 'ref' in self.run_mode:
                        self.run_exp_cv(model_name, model_suffix,
                                        params, mask, repeat, cv_fold, self.community_filter_map,
                                        grid_name=grid_name)
                    elif 'random_forest' in self.run_mode:
                        self.run_exp_cv('random_forest', model_suffix,
                                        params, mask_random, repeat, cv_fold, None,
                                        grid_name=grid_name)
                    elif 'random_predictor' in self.run_mode:
                        self.run_exp_cv('random_predictor', model_suffix,
                                        params, mask_random, repeat, cv_fold, self.community_filter_map_random,
                                        grid_name=grid_name)
                    elif 'random' in self.run_mode:
                        self.run_exp_cv('random_control', model_suffix,
                                        params, mask_random, repeat, cv_fold, self.community_filter_map_random,
                                        grid_name=grid_name)
                    elif 'full' in self.run_mode:
                        self.run_exp_cv('gene_control', model_suffix,
                                        params, mask, repeat, cv_fold, None, mask_ones, grid_name=grid_name)
                    if self.cv_fold_only_run == (cv_fold + 1):
                        break

    def _clear_gpu(self, model_name, model_suffix):
        if self.use_cuda:
            logging.debug("Clearing session {}".format(model_name + model_suffix))
            msg = 'before clean'
            output = subprocess.run('nvidia-smi', stdout=subprocess.PIPE)
            logging.debug("{}\n{}".format(msg, str(output).replace('\\n', '\n')))
            torch.cuda.empty_cache()
            msg = 'after clean'
            output = subprocess.run('nvidia-smi', stdout=subprocess.PIPE)
            logging.debug("{}\n{}".format(msg, str(output).replace('\\n', '\n')))
            logging.debug("Cleared session {}".format(model_name + model_suffix))

    def compute_metric(self, cm, data_type, model_name, model_suffix, X, y_true, com_mat, repeat, y_true2=None,
                       pred=None):
        output_prefix = model_name + model_suffix + '_' + data_type
        y_index = self.idx[data_type]
        y_col = self.y.columns
        pred2 = None
        if pred is None:
            if 'random_predictor' in model_name:
                pred = []
                prng = np.random.RandomState()
                for i in range(y_true.shape[1]):
                    pred.append(prng.rand(y_true.shape[0], 1))
                pred = np.concatenate(pred, axis=1).flatten()
            elif 'mean_control' in model_name:
                pred = np.tile(X.mean(axis=0), y_true.shape[0])
            elif 'expression_control' in model_name:
                x_df = pd.DataFrame(X, columns=self.rna.columns, index=y_index)
                y_df = pd.DataFrame(y_true, columns=self.dependency.columns, index=y_index)
                genes_without_expression = sorted(list(set(y_df.columns) - set(x_df.columns)))
                logging.info(f'{len(genes_without_expression)} genes do not have expression for prediction')
                genes_with_expression = sorted(list(set(y_df.columns) & set(x_df.columns)))
                logging.info(f'{len(genes_with_expression)} genes having expression were used to predict dependency')
                pred = x_df[genes_with_expression]
                y_true = y_df[genes_with_expression]

            elif 'LogisticRegression' in model_name or 'xgb' in model_name or 'random_forest' in model_name:
                if isinstance(cm, list):
                    pred = []
                    for cm_i in cm:
                        pred.append(cm_i.predict(X))
                    pred = np.array(pred).T.flatten()
                else:
                    pred = cm.predict(X)
            else:
                pred = cm.predict(X, y_true, 10000, output_prefix, y_index=y_index, y_col=y_col)
        self.compute_overall_cor(pred, y_true, repeat, data_type, model_name, model_suffix, output_prefix,
                                 pred2=None, y_true2=None)

    def compute_overall_cor(self, pred, y_true, repeat, data_type, model_name, model_suffix, output_prefix,
                            pred2=None, y_true2=None):
        if isinstance(pred, pd.DataFrame):
            y_true_flatten = y_true.values.flatten()
            pred_flatten = pred.values.flatten()
        else:
            y_true_flatten = y_true.flatten()
            pred_flatten = pred
        pearson_r = np.corrcoef(pred_flatten, y_true_flatten)[0, 1]
        spearman_rho = scipy.stats.spearmanr(pred_flatten, y_true_flatten)[0]
        y_true_flatten_binary = (y_true_flatten >= 0.5) + 0

        if np.sum(y_true_flatten_binary) == 0 or np.sum(y_true_flatten_binary) == len(y_true_flatten_binary):
            auroc, auprc, f1, f1_weighted = np.nan, np.nan, np.nan, np.nan
        else:
            auroc = roc_auc_score(y_true_flatten_binary, pred_flatten)
            auprc = average_precision_score(y_true_flatten_binary, pred_flatten)
            pred_binary = (pred_flatten >= 0.5) + 0
            f1 = f1_score(y_true_flatten_binary, pred_binary)
            f1_weighted = f1_score(y_true_flatten_binary, pred_binary, average='weighted')
            fpr, tpr, _thresholds = roc_curve(y_true_flatten_binary, pred_flatten)
            prediction_n = len(y_true_flatten_binary)
            gs_positive_n = np.sum(y_true_flatten_binary)
            plot_ROC(fpr, tpr, prediction_n, gs_positive_n, auroc, self.result_path, output_prefix)

        self.pearson_r[repeat][data_type][model_name].append(pearson_r)
        self.spearman_rho[repeat][data_type][model_name].append(spearman_rho)
        if pred2 is not None:
            pearson_r2 = np.corrcoef(pred2, y_true2.flatten())[0, 1]
            self.pearson_r2[repeat][data_type][model_name].append(pearson_r2)
        else:
            pearson_r2 = None
        if pearson_r2 is not None:
            metric_df = pd.DataFrame([pearson_r, spearman_rho, pearson_r2]).T
            metric_df.columns = ['Pearson_r', 'Spearman_rho', 'Pearson_r2']
            logging.info(
                "{} {} Pearson r {}; Spearman rho {}; Pearson r 2 {};".format(model_name + model_suffix, data_type,
                                                                              pearson_r, spearman_rho, pearson_r2))
        else:
            metric_df = pd.DataFrame([pearson_r, spearman_rho, auroc, auprc, f1, f1_weighted]).T
            metric_df.columns = ['Pearson_r', 'Spearman_rho', 'AUROC', 'AUPRC', 'F1', 'F1_weighted']
            logging.info(
                "{} {} Pearson r {}; Spearman rho {}; AUROC {}; AUPRC {}; F1 {}; F1_weighted {}".format(
                    model_name + model_suffix, data_type, pearson_r,
                    spearman_rho, auroc, auprc, f1, f1_weighted))
        metric_df.index = [model_name + model_suffix]
        self.metric_output[data_type] = pd.concat([self.metric_output[data_type], metric_df])
        if isinstance(pred, pd.DataFrame):
            self.plot_pred_true(pred.values.flatten(), y_true.values, output_prefix)
        else:
            self.plot_pred_true(pred, y_true, output_prefix)
            pred = pred.reshape(y_true.shape)
            pred = pd.DataFrame(pred, index=self.idx[data_type], columns=self.y.columns)
        self.output_pred(pred, output_prefix)
        self.output_pred(pred[list(set(self.output_pred_small) & set(pred.columns))], 'small_' + output_prefix)

        if model_name != 'expression_control' and 'random_forest' not in model_name:
            if 'Timestamped' in self.__class__.__name__:
                self.compute_gene_level_cor(pred, output_prefix, y_true)
                self.compute_gene_level_auc(pred, output_prefix, y_true)
            else:
                self.compute_gene_level_cor(pred, output_prefix)
                self.compute_gene_level_auc(pred, output_prefix)

        if pred2 is not None:
            output_prefix = model_name + model_suffix + '_' + data_type + '_2'
            self.plot_pred_true2(pred2, y_true2, output_prefix)
            pred2 = pred2.reshape(y_true.shape)
            pred2 = pd.DataFrame(pred2, index=self.idx[data_type], columns=self.y.columns)
            self.output_pred(pred2, output_prefix)
            self.output_pred(pred2[list(set(self.output_pred_small) & set(pred2.columns))], 'small_' + output_prefix)

    def compute_gene_level_cor(self, pred, output_prefix, y=None):
        if y is None:
            if 'Timestamped' in self.__class__.__name__ or 'Postanalysis_ts' in self.__class__.__name__:
                y = self.y_target
            else:
                y = self.y
        df_gene_cor_var = gene_level_cor(pred, y)
        logging.info(df_gene_cor_var.loc[self.select_genes_in_label])
        self.output_pred(df_gene_cor_var, 'by_gene_' + output_prefix)
        plot_pred_true_r_by_gene_MAD(df_gene_cor_var, self.result_path, output_prefix, 'r')
        plot_pred_true_r_by_gene_mean(df_gene_cor_var, self.result_path, output_prefix, 'r')
        plot_hist_cor(df_gene_cor_var['Pearson_r'], self.result_path, output_prefix)

    def compute_gene_level_auc(self, pred, output_prefix, labels_all=None):
        if labels_all is None:
            labels_all_binary = self.y_binary.loc[pred.index]
            labels_all = self.y.loc[pred.index]
        else:
            labels_all_binary = ((labels_all >= 0.5) + 0).astype(int)

        df_gene_auc, df_gene_bootstrap = individual_auc(pred, labels_all_binary, labels_all)
        logging.info(output_prefix)
        plot_top_ROC(df_gene_auc, labels_all_binary, pred, self.result_path, 'by_gene_' + output_prefix)
        self.output_pred(df_gene_auc, 'by_gene_auc_' + output_prefix)
        for x, y in zip(df_gene_bootstrap, ['AUROCs', 'AUPRCs', 'pAUROCs', 'pAUPRCs']):
            self.output_pred(x, 'by_gene_{}_{}'.format(y, output_prefix))
        plot_pred_true_r_by_gene_MAD(df_gene_auc, self.result_path, output_prefix, mode='auc')
        plot_pred_true_r_by_gene_mean(df_gene_auc, self.result_path, output_prefix, mode='auc')
        plot_hist_auc(df_gene_auc['AUROC'], self.result_path, 'by_gene_' + output_prefix)
        plot_hist_auc(df_gene_auc['AUPRC'], self.result_path, 'by_gene_' + output_prefix, mode='AUPRC')

        if pred.shape[1] >= 10:
            df_cell_auc, df_cell_bootstrap = individual_auc(pred.T, labels_all_binary.T, labels_all.T)
            plot_top_ROC(df_cell_auc, labels_all_binary.T, pred.T, self.result_path, 'by_cell_' + output_prefix)
            self.output_pred(df_cell_auc, 'by_cell_auc_' + output_prefix)
            for x, y in zip(df_cell_bootstrap, ['AUROCs', 'AUPRCs', 'pAUROCs', 'pAUPRCs']):
                self.output_pred(x, 'by_cell_{}_{}'.format(y, output_prefix))
            plot_hist_auc(df_cell_auc['AUROC'], self.result_path, 'by_cell_' + output_prefix)
            plot_hist_auc(df_cell_auc['AUPRC'], self.result_path, 'by_cell_' + output_prefix, mode='AUPRC')

    def compute_metric_all_test(self, data_type, model_name, model_suffix, X, y_true, repeat, y_true2=None):
        pred = self.load_prediction_all(data_type, model_name, model_suffix)
        y_true = self.y.loc[pred.index, pred.columns]
        output_prefix = model_name + model_suffix + '_' + data_type
        self.compute_overall_cor(pred, y_true, repeat, data_type, model_name, model_suffix, output_prefix,
                                 pred2=None, y_true2=None)
        gene = self.genes_in_label[0]
        output_prefix = output_prefix + '_' + gene
        self.plot_pred_true_scatter(pred[gene], self.y.loc[pred.index, gene], output_prefix)

        df_pred = pd.DataFrame(index=pred.index)
        df_pred['Predicted'] = pred[gene]
        df_pred['Measured'] = self.y.loc[pred.index, gene]
        self.output_pred(df_pred, '' + output_prefix)

    def load_prediction_all(self, data_type, model_name, model_suffix):
        if hasattr(self, 'load_result_dir'):
            load_dir = self.load_result_dir
        else:
            load_dir = self.result_path
        pred_files = glob('{}/pred_{}*fold*{}.tsv'.format(load_dir, model_name + model_suffix, data_type))
        pred = pd.read_csv(pred_files[0], sep='\t', index_col=0)
        for p in pred_files[1:]:
            p = pd.read_csv(p, sep='\t', index_col=0)
            pred = pd.concat([pred, p])
        if len(pred.columns) != len(set(pred.columns)):
            logging.info("Found duplicated columns and only kept the first one")
            pred = pred[~pred.columns.duplicated(keep='first')]
        if len(pred.index) != len(set(pred.index)):
            logging.info("Found duplicated index and only kept the first one")
            pred = pred[~pred.index.duplicated(keep='first')]
        return pred

    def _get_full_X_y(self, X, y):
        X_full = np.repeat(X, [y.shape[1]] * y.shape[0], axis=0)
        del_array = np.zeros([y.shape[1], y.shape[1]])
        for i in range(y.shape[1]):
            del_array[i, i] = 1
        del_full = np.tile(del_array, [y.shape[0], 1])
        X_full = np.concatenate([X_full, del_full], axis=1)
        y_full = y.flatten()
        return X_full, y_full

    def plot_pred_true_scatter(self, x, y, output_prefix):
        ax = sns.scatterplot(x=x, y=y)
        ax.set_xlabel('Predicted dependency')
        ax.set_ylabel('Measured dependency')
        plt.savefig('{}/pred_true_scatter_{}.pdf'.format(self.result_path, output_prefix),
                    bbox_inches='tight')
        plt.close()

    def calculate_RLIPP(self, X, y, data_type, output_prefix, cm, load_dir=None, select_gene=None, metric='auprc'):
        if self.use_classification:
            if metric == 'auprc':
                df = pd.DataFrame(columns=['auprc_parent', 'auprc_child', 'RLIPP', 'c_parent', 'c_child'])
            else:
                df = pd.DataFrame(columns=['auroc_parent', 'auroc_child', 'RLIPP', 'c_parent', 'c_child'])
        else:
            df = pd.DataFrame(columns=['rho_parent', 'rho_child', 'RLIPP', 'alpha_parent', 'alpha_child'])
        if self.use_deletion_vector:
            input_dim = X.shape[1] + y.shape[1]
        else:
            input_dim = X.shape[1]
        y_flat = y.flatten()
        if load_dir is None:
            if hasattr(self, 'load_result_dir'):
                load_dir = self.load_result_dir
            else:
                load_dir = self.result_path
        if select_gene is not None:
            select_gene_idx = list(self.y.columns).index(select_gene)
        else:
            select_gene_idx = None
        for com in self.community_dict.keys():
            com_idx = self.community_hierarchy_dicts_all['gene_group_idx'][com]
            child_idx = self.child_map_all[com_idx - input_dim]
            # feat_idx = [x for x in child_idx if x < input_dim]
            child_com_idx = [x - input_dim for x in child_idx if x >= input_dim]
            if len(child_com_idx) == 0:
                continue
            com_idx -= input_dim
            X_parent = cm.load_intermediate_output(output_prefix, load_dir, [com_idx])[0]
            X_child = np.concatenate(cm.load_intermediate_output(output_prefix, load_dir, child_com_idx), axis=1)
            X_child = X_child[:X_parent.shape[0]]
            if select_gene_idx is None:
                y2 = y_flat[:X_parent.shape[0]]
            else:
                y2 = y[:, select_gene_idx].flatten()
                X_parent = X_parent.reshape(y.shape[0], y.shape[1], X_parent.shape[1])
                X_parent = X_parent[:, select_gene_idx, :]
                X_child = X_child.reshape(y.shape[0], y.shape[1], X_child.shape[1])
                X_child = X_child[:, select_gene_idx, :]

            # ## To include child genes
            # X_child2 = np.concatenate([ds[(x, feat_idx)] for x in range(X_child.shape[0])], axis=1)
            # X_child = np.concatenate([X_child, X_child2], axis=1)
            metric_p, metric_c, RLIPP, reg_p, reg_c = self._RLIPP(X_parent, X_child, y2, metric)
            com2 = '{}_n{}'.format(com, X_parent.shape[0])
            df.loc[com2] = [metric_p, metric_c, RLIPP, reg_p, reg_c]
            logging.info("{}".format(df.loc[com2]))
        df = df.sort_values('RLIPP', ascending=False)
        if select_gene is None:
            df.to_csv(os.path.join(self.result_path, 'RLIPP_{}.tsv'.format(output_prefix)), sep="\t")
        else:
            df.to_csv(os.path.join(self.result_path, 'RLIPP_{}_{}.tsv'.format(output_prefix, select_gene)), sep="\t")
        return df

    def output_pred(self, pred, output_prefix):
        pred.to_csv(
            os.path.join(self.result_path, 'pred_{}.tsv'.format(output_prefix)), sep="\t")

    def output_metric(self, metric_output=None, suffix=''):
        if metric_output is None:
            metric_output = self.metric_output
        for x in metric_output:
            if metric_output[x].shape[1] == 2:
                metric_output[x].columns = ['Pearson_r', 'Spearman_rho']
            elif metric_output[x].shape[1] == 3:
                metric_output[x].columns = ['Pearson_r', 'Spearman_rho', 'Pearson_r2']
            elif metric_output[x].shape[1] == 6:
                metric_output[x].columns = ['Pearson_r', 'Spearman_rho', 'AUROC', 'AUPRC', 'F1', 'F1_weighted']
            if metric_output[x].shape[1] > 0:
                metric_output[x].to_csv(
                    os.path.join(self.result_path, 'metric_{}{}.tsv'.format(x, suffix)), sep="\t")

    # def _umap_2d(self, data):
    #     reducer = umap.UMAP()
    #     embedding = reducer.fit_transform(data)
    #     return embedding

    def _pca_2d(self, data):
        pca = PCA(n_components=2)
        embedding = pca.fit_transform(data)
        return embedding

    def plot_intermediate_output_2d(self, X, y, data_type, output_prefix, cm, com, load_dir=None):
        input_dim = X.shape[1] + y.shape[1]
        com_idx = self.community_hierarchy_dicts_all['gene_group_idx'][com]
        com_idx = com_idx - input_dim

        if load_dir is None:
            if hasattr(self, 'load_result_dir'):
                load_dir = self.load_result_dir
            else:
                load_dir = self.result_path
        X_intermediate = cm.load_intermediate_output(data_type, load_dir, [com_idx])[0]
        pca_data = self._pca_2d(X_intermediate)
        sc = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=y.flatten(), cmap='bwr', s=0.1)
        plt.colorbar(sc)
        f = os.path.join(self.result_path, output_prefix)
        plt.savefig('{}_{}.png'.format(f, com), bbox_inches='tight')
        plt.close()

    def plot_pred_true(self, pred, y_true, output_prefix):
        df = pd.DataFrame([pred, y_true.ravel()]).T
        df.columns = ['Predicted dependency', 'Measured dependency']
        interval_order = []
        for x in np.linspace(0, 0.9, num=10):
            interval = df.loc[(df['Predicted dependency'] >= x) & (df['Predicted dependency'] < x + 0.1)].index
            if x == 0:
                interval_str = '[0,0.1)'
            elif x == 0.9:
                interval_str = '[0.9,1]'
            else:
                interval_str = '[{:.1f},{:.1f})'.format(x, x + 0.1)
            interval_str += '\nn={}'.format(len(interval))
            df.loc[interval, 'interval'] = interval_str
            interval_order.append(interval_str)
        sns.boxplot(x='interval', y='Measured dependency', data=df, fliersize=0.05, order=interval_order)
        plt.xticks(rotation=45)
        plt.savefig('{}/boxplot_{}.pdf'.format(self.result_path, output_prefix), bbox_inches='tight')
        plt.close()

    def plot_pred_true2(self, pred, y_true, output_prefix):
        df = pd.DataFrame([pred, y_true.ravel()]).T
        df.columns = ['Predicted dependency z', 'Measured dependency z']
        interval_order = []
        ranges = sorted(list(set(-np.linspace(0, 5, num=11)) | set(np.linspace(0, 5, num=11))))
        for x in ranges:
            if x == ranges[0]:
                interval = df.loc[(df['Predicted dependency z'] <= x)].index
                interval_str = '<={}'.format(x)
            elif x == ranges[-1]:
                interval = df.loc[(df['Predicted dependency z'] > x)].index
                interval_str = '>{}'.format(x)
            else:
                interval = df.loc[(df['Predicted dependency z'] >= x) & (df['Predicted dependency z'] < x + 0.5)].index
                interval_str = '[{:.1f},{:.1f})'.format(x, x + 0.5)
            interval_str += '\nn={}'.format(len(interval))
            df.loc[interval, 'interval'] = interval_str
            interval_order.append(interval_str)
        sns.boxplot(x='interval', y='Measured dependency z', data=df, fliersize=0.05, order=interval_order)
        plt.xticks(rotation=45)
        plt.savefig('{}/boxplot_{}.pdf'.format(self.result_path, output_prefix), bbox_inches='tight')
        plt.close()


def main():
    from set_logging import set_logging
    param_f = sys.argv[1]
    params = load_params(param_f=param_f)
    depmap_ver = params.get('depmap_ver', '19Q3')
    data_dir = paths.DATA_DIR
    result_dir = paths.RESULTS_DIR

    model_name = 'BioVNN'
    if 'ref_groups' in params and params['ref_groups'].lower() == 'go':
        run_suffix = 'GO'
    else:
        run_suffix = 'Reactome'
    run_suffix += '_' + params['run_mode']
    cancer_type = 'PANC'
    run_suffix_2 = run_suffix + '_' + cancer_type
    params['cancer_type'] = cancer_type
    if 'cv_fold' in params and params['cv_fold'] > 0:
        run_suffix_2 += '_cv'
    # Set up parameters of the model
    start_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    start_time += '_' + run_suffix_2

    log_dir = os.path.join(result_dir, 'Dependency', start_time)
    set_logging('Dependency', log_dir)
    logging.debug("Running cancer type {}.".format(cancer_type))
    logging.info("DepMap {} was used.".format(depmap_ver))
    dp = Dependency(params['cancer_type'], data_dir, result_dir, start_time, params,
                    depmap_ver)
    dp.perform(model_name, params)


if __name__ == '__main__':
    main()
