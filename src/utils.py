#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility functions for running cancer community project
Created on Sep 7, 2018
Last edited on Sep 10, 2018
@author: Chih-Hsu Lin
"""
import numpy as np
import itertools, os, time, copy, bisect, re
import multiprocessing as mp
from multiprocessing import Manager
from collections import defaultdict as ddict
import glob
from IPython import embed
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, f1_score, roc_auc_score
from statsmodels.robust.scale import mad
import matplotlib.pyplot as plt
import seaborn as sns

# Import path and sys.path information
import paths
import logging
# from bioClasses import network
# from bioClasses.algorithms.recursive_louvain import RecursiveLouvain


####################################################################################################################
# 1. Detect communities based on a network
####################################################################################################################

def commToGmt(comms, output_prefix):
    file_name = "{}.gmt".format(output_prefix)
    f = open(file_name, 'w')
    for com in comms:
        tofile = ''
        for item in comms[com]:
            tofile += '{}\t'.format(item)
        f.write(str(com) + '\t' + tofile.strip() + '\n')


def output_hierarchy(hierarchy, output_prefix):
    file_name = "{}_hierarchy.tsv".format(output_prefix)
    f = open(file_name, 'w')
    for x in hierarchy:
        for y in hierarchy[x]:
            f.write(x + '\t' + y + '\n')


def detect_community(network_path, network_name, output_path):
    """Run Recursive Louvain detection methods for given network

    Args:
        network_path (str): The path to the network edge list file
        network_name (str): The name of the network, which will be part of output name
        output_path (str): The path to output community file

    Returns:
        str: The full path to the output community file

    Examples:
    >>> network_path = os.path.join(paths.DATA_DIR, 'Networks', 'experimental_9.1.txt')
    >>> network_name = 'experimental_9.1'
    >>> output_path = os.path.join(paths.RESULTS_DIR, 'communities', 'Raw')
    >>> detect_community(network_path, network_name, output_path)
    """
    net = network.network()
    net.from_file(network_path)
    output_prefix = output_path + '/{}_recursive_Louvain_{}_{}'.format(network_name, len(net.nodes), len(net.edges))
    logging.info('{} network: {} nodes and {} edges were loaded.'.format(network_name, len(net.nodes), len(net.edges)))
    if os.path.isfile(output_prefix + '.gmt'):
        logging.info('Communities have been detected.')
    else:
        rl = RecursiveLouvain()
        rl.fit(net)
        comms = rl.community
        hierarchy = rl.hierarchy
        os.makedirs(output_path, exist_ok=True)
        commToGmt(comms, output_prefix)
        output_hierarchy(hierarchy, output_prefix)
        logging.info('{} communities were detected.'.format(len(comms)))
        logging.info('Community file was saved as {}.gmt'.format(output_prefix))
    return '{}.gmt'.format(output_prefix)


####################################################################################################################
# 2. Collapse communities to reduce redundancy
####################################################################################################################


def Jmatrix(arg):
    pair, pathways, keys, threshold, q = arg[0], arg[1], arg[2], arg[3], arg[4]
    mat = {}
    denom = float(len(set(pathways[pair[0]] + pathways[pair[1]])))
    if denom == 0:
        jaccard = 0
    else:
        numer = float(len(set(pathways[pair[0]]).intersection(set(pathways[pair[1]]))))
        jaccard = numer / denom
    if jaccard >= threshold:
        x = keys.index(pair[0])
        y = keys.index(pair[1])
        mat[(x, y)] = jaccard
    q.put(mat)


def getsimmat(pathways, threshold):
    keys = pathways.keys()
    pairs = list(itertools.combinations(keys, 2))
    manager = Manager()
    q = manager.Queue()
    args = list(zip(pairs, itertools.repeat(pathways), itertools.repeat(keys), itertools.repeat(threshold)))
    args = [list(x).append(q) for x in args]
    simmat_lst = {}
    pool = mp.Pool(processes=5)
    pool.map_async(Jmatrix, args)
    while not q.empty():
        i = q.get()
        simmat_lst.update(i)
    profile = list(simmat_lst.values())
    return profile, simmat_lst, keys


def getsimmat_loop(newpathways, newkeys, newpath, samepath, simmat_lst, threshold):
    pairs1 = list(itertools.combinations(newpath, 2))
    pairs2 = list(itertools.product(newpath, samepath))
    pairs = pairs1 + pairs2
    manager = Manager()
    q = manager.Queue()
    args = list(zip(pairs, itertools.repeat(newpathways), itertools.repeat(newkeys), itertools.repeat(threshold)))
    args = [list(x).append(q) for x in args]
    pool = mp.Pool(processes=5)
    pool.map_async(Jmatrix, args)
    while not q.empty():
        i = q.get()
        simmat_lst.update(i)
    profile = list(simmat_lst.values())
    return profile, simmat_lst


def collapse_gene_set(fl, threshold=0.9, size_min=3):
    '''
    Collapses a set of genes based on Jaccard Similarity.
    '''
    tic = time.time()
    path = fl.split('/')
    path[-2] = 'Collapsed'
    os.makedirs('/'.join(path[:-1]), exist_ok=True)
    if os.path.isfile('/'.join(path)):
        logging.info('Communities have been collapsed.')
        path[-2] = 'Filtered'
    else:
        OUT = open('/'.join(path), 'w')
        path[-2] = 'Filtered'
        os.makedirs('/'.join(path[:-1]), exist_ok=True)
        OUT_filtered = open('/'.join(path), 'w')
        pathways = [x.strip().split('\t') for x in open(fl).readlines()]
        pathways = {x[0]: x[1:] for x in pathways}
        t1 = time.time()
        logging.debug('Time to finish reading {} sec'.format(t1 - tic))
        profile, simmat_lst, keys = getsimmat(pathways, threshold)
        t2 = time.time()
        logging.debug('similarity matrix {} sec'.format(t2 - t1))
        newpathways = {key: val for key, val in pathways.items()}
        newpathways.update({'blank': []})
        newkeys = [x for x in keys]
        c = 0
        deleted = []
        t3 = time.time()
        logging.debug('Before while loop {} sec'.format(t3 - t2))
        while len(profile) != 0:
            logging.info(profile)
            newpath = []
            c += 1
            ind = [k for (k, v) in simmat_lst.items() if v >= threshold]
            tt1 = time.time()
            delete_item = []
            for pair in ind:
                # pairs of indicies
                l1 = newkeys[pair[0]]  # get the name corresponding to that index
                l2 = newkeys[pair[1]]
                # logging.info pair[0], pair[1], l1, l2
                if l1 in deleted or l2 in deleted or l1 == 'blank' or l2 == 'blank' or l1 in delete_item or l2 in delete_item:
                    continue
                geneset1 = newpathways[l1]  # get the genes corresponding to that name of a pathway
                geneset2 = newpathways[l2]
                del newpathways[l1]
                deleted.append(l1)
                delete_item.append(l1)
                delete_item.append(l2)
                del newpathways[l2]
                deleted.append(l2)
                newlabel = "{}{}{}".format(l1, '-.', l2)
                delete_item.append(newlabel)
                # newlabel="{}{}{}".format(l1,''.join(['-.' for i in range(c)]),l2)
                logging.info(newlabel)
                newpath.append(newlabel)
                newpathways[newlabel] = list(set(geneset1 + geneset2))
                # logging.info newlabel in newpathways.keys()
                newkeys[pair[0]] = newlabel
                newkeys[pair[1]] = 'blank'
                simmat_lst_copy = copy.deepcopy(simmat_lst)
                for pair_ind in simmat_lst_copy:
                    if pair[1] in pair_ind:
                        del simmat_lst[pair_ind]

            samepath = list(set(newpathways.keys()) - set(newpath))
            try:
                samepath.remove('blank')
            except:
                pass
            tt2 = time.time()
            logging.info('For loop {} sec'.format(tt2 - tt1))
            profile, simmat_lst = getsimmat_loop(newpathways, newkeys, newpath, samepath, simmat_lst, threshold)
            tt3 = time.time()
            logging.info('similarity matrix {} sec'.format(tt3 - tt1))
        t4 = time.time()
        logging.info('After while loop {} sec'.format(t4 - t3))
        del newpathways['blank']
        for key in newpathways.keys():
            OUT.write('{}\t{}\n'.format(key, '\t'.join(newpathways[key])))
            if len(newpathways[key]) >= size_min:
                OUT_filtered.write('{}\t{}\n'.format(key, '\t'.join(newpathways[key])))

        size = [len(v) for v in newpathways.values() if len(v) >= size_min]
        logging.info("Comm #\tComm # (collapsed)\tComm # (collapsed, size>3)\tMedian of size\tMax of size\n")
        logging.info(
            "{}\t{}\t{}\t{}\t{}\n".format(len(pathways.keys()), len(newpathways.keys()), len(size), np.median(size),
                                          np.max(size)))
    return '/'.join(path)


####################################################################################################################
# 3. Map communities to gene symbol and filter
####################################################################################################################

def readANNOVAR(ANNOVAR_refGene):
    f = open(ANNOVAR_refGene, 'r').readlines()
    gene_names = set()
    p_NRNM = re.compile('^N[MR]_\d+')
    for line in f:
        line = line.strip('\n').split('\t')
        m = p_NRNM.match(line[1])
        if m:
            gene_names |= {line[12]}
        else:
            gene_names |= {line[1], line[12]}
    return gene_names


# Getting mapping for string genes
def entrezString(fl, string_entrez={}):
    for line in open(fl).readlines():
        if '#' not in line:
            line = line.strip('\n').split()
            if '9606.' in line[1] and line[0]:
                string_entrez[line[1].replace('9606.', '')] = line[0]
    return string_entrez


def mergeDict(dict_list, ANNOVAR_set):
    conflicts = 0
    merged = ddict(list)
    print([len(d) for d in dict_list])
    for d in dict_list:
        for k, v in d.items():
            merged[k].append(v)

    for k, v in merged.items():
        if len(set(v)) > 1:
            v_intersect = [x for x in v if x in ANNOVAR_set]
            if len(v_intersect) > 0:
                merged[k] = v_intersect
            if len(set(merged[k])) > 1:
                conflicts += 1
                print(k, merged[k])
        merged[k] = merged[k][0]
    logging.info('Conflicting mapping from ID to symbol: {}'.format(conflicts))
    return merged


# Mapping
def getMappingENSPtoSymbol(mapping_data_dir, ANNOVAR_refGene):
    ANNOVAR_set = readANNOVAR(ANNOVAR_refGene)
    string_entrez = entrezString(mapping_data_dir + '/entrez_gene_id.vs.string.v10.28042015.tsv')
    string_entrez = entrezString(mapping_data_dir + '/entrez_gene_id.vs.string.v9.05.28122012.txt',
                                 string_entrez)

    entrez_name = {}
    for line in open(mapping_data_dir + '/HGNC_3-20-18.txt').readlines():
        if 'Symbol' not in line:
            line = line.strip('\n').split()
            if len(line) >= 2:
                entrez_name[line[1]] = line[0]

    id_name3 = {}
    for k, v in string_entrez.items():
        if v in entrez_name:
            if k not in id_name3:
                if len(entrez_name[v]) > 0:
                    id_name3[k] = entrez_name[v]
            else:
                print(k)

    p_ENSP = re.compile('ENSP\d+')
    p_HUMAN = re.compile('^9606')
    mapping_fl = mapping_data_dir + '/all_go_knowledge_explicit.tsv'
    id_name = {}
    for line in open(mapping_fl).readlines():
        m = p_HUMAN.match(line)
        if m:
            line = line.strip('\n').split()
            m = p_ENSP.match(line[1])
            if m:
                m = p_ENSP.match(line[2])
                if not m:
                    id_name[line[1]] = line[2]

    mapping_fl = mapping_data_dir + '/mart_export_07122018_Ensembl92.txt'
    id_name2 = {}
    for line in open(mapping_fl).readlines():
        line = line.strip('\n').split('\t')
        if line[0]:
            if line[1] == line[3]:
                id_name2[line[0]] = line[1]
            elif line[3]:
                # print(line)
                id_name2[line[0]] = line[3]
            elif line[1]:
                # print(line)
                id_name2[line[0]] = line[1]
            # else:
            #     print(line)

    merge = mergeDict([id_name, id_name2, id_name3], ANNOVAR_set)
    return merge, ANNOVAR_set


def map_communities(fl, mapping_data_dir):
    path = fl.split('/')
    path[-2] = 'Mapped'
    Mapped_dir = '/'.join(path[:-1])
    os.makedirs(Mapped_dir, exist_ok=True)
    ANNOVAR_refGene = paths.CEDAR_DIR + '/home/katsonis/Archive/Codes/Software/annovar/humandb/hg38_refGene.txt'
    name_list = fl.split('/')[-1].split('.')
    name = name_list[0]
    if len(name_list) > 2:
        name = name + '.' + name_list[1]

    merge, ANNOVAR_set = getMappingENSPtoSymbol(mapping_data_dir, ANNOVAR_refGene)
    mapped_file = Mapped_dir + '/{}.gmt'.format(name)
    if os.path.isfile(mapped_file):
        logging.info('{} file exists. Pass mapping.'.format(mapped_file))
        return mapped_file
    else:
        out = open(mapped_file, 'w')

    ENSP_count = set()
    mapped_count = ddict(set)
    total_node = set()
    p_ENSP = re.compile('ENSP\d+')
    for line in open(fl).readlines():
        line = line.strip('\n').split('\t')
        line1 = []
        for i in line[1:]:
            total_node |= {i}
            i = mapped = i.replace('9606.', '')
            if mapped in merge:
                mapped = merge[mapped]
            # mapped = mapGene(i,id_name,entrez_name,string_entrez,id_name2,mapped_count)
            line1.append(mapped)
            if p_ENSP.match(mapped):
                ENSP_count |= {mapped}
            else:
                mapped_count[mapped] |= {i}
        line1 = sorted(list(set(line1)))
        line1.insert(0, line[0])
        line1_txt = '\t'.join(str(x) for x in line1)
        out.write('{}\n'.format(line1_txt))
    out.close()
    logging.info("Total node #: {}".format(len(total_node)))
    logging.info("Total node # after mapping: {}".format(len(ENSP_count) + len(mapped_count)))
    logging.info("Unmapped ENSP: {}".format(len(ENSP_count)))
    logging.info("Mapped id: {}".format(len(mapped_count)))
    logging.info("Mapped id in ANNOVAR hg38: {}".format(len(set(mapped_count.keys()).intersection(ANNOVAR_set))))
    out = open(Mapped_dir + '/{}_unmapped.tsv'.format(name), 'w')

    # release 77 uses human reference genome GRCh38
    release = 92
    # data = EnsemblRelease(release)
    not_found = 0
    for x in ENSP_count:
        # try:
        #     gene = data.gene_by_protein_id(x)
        # except:
        #     not_found += 1
        out.write('{}\n'.format(x))
    logging.info('# of ID not found in Ensembl release {}: {}'.format(release, not_found))
    out.close()
    out = open(Mapped_dir + '/{}_mapped_count.tsv'.format(name), 'w')
    merged_id = 0
    for k, v in mapped_count.items():
        out.write('{}\t{}\n'.format(k, ';'.join(list(v))))
        if len(v) > 1:
            # print(k,v)
            merged_id += 1
    out.close()
    logging.info("Id merged by symbols: {}".format(merged_id))
    return mapped_file


####################################################################################################################
# Metric calculation
####################################################################################################################

def gene_level_cor(pred, y):
    df_gene_cor_var = pd.DataFrame(index=y.columns)
    for gene in pred.columns:
        df_gene_cor_var.loc[gene, 'Pearson_r'] = np.corrcoef(y.loc[pred.index, gene], pred[gene])[0, 1]
    df_gene_cor_var['Dep_mad'] = mad(y.loc[pred.index])
    df_gene_cor_var['Dep_std'] = np.std(y.loc[pred.index])
    df_gene_cor_var['Dep_mean'] = np.mean(y.loc[pred.index])
    df_gene_cor_var['Dep_min'] = np.min(y.loc[pred.index])
    df_gene_cor_var['Dep_max'] = np.max(y.loc[pred.index])
    df_gene_cor_var['Dep_min_max_diff'] = df_gene_cor_var['Dep_max'] - df_gene_cor_var['Dep_min']
    df_gene_cor_var['Pred_mad'] = mad(pred)
    df_gene_cor_var['Pred_std'] = pred.std()
    df_gene_cor_var['Pred_mean'] = pred.mean()
    df_gene_cor_var['Pred_min'] = pred.min()
    df_gene_cor_var['Pred_max'] = pred.max()
    df_gene_cor_var['Pred_min_max_diff'] = df_gene_cor_var['Pred_max'] - df_gene_cor_var['Pred_min']
    df_gene_cor_var = df_gene_cor_var.sort_values('Pearson_r', ascending=False)
    return df_gene_cor_var


def individual_auc(pred, labels_all_binary, labels_all):
    df_auc = pd.DataFrame(index=pred.columns)
    df_bootstrap = [pd.DataFrame(columns=pred.columns), pd.DataFrame(columns=pred.columns),
                    pd.DataFrame(columns=pred.columns), pd.DataFrame(columns=pred.columns)]
    for col in pred.columns:
        y_true = labels_all_binary[col].values.astype(int)
        y_score = pred[col].values
        AUC_bootstrap, AUC_PR_bootstrap, pAUC_bootstrap, pAUC_PR_bootstrap, AUC_bootstrapMedian, AUC_PR_bootstrapMedian, pAUC_bootstrapMedian, pAUC_PR_bootstrapMedian, AUC_bootstrapStd, AUC_PR_bootstrapStd, pAUC_bootstrapStd, pAUC_PR_bootstrapStd = compute_AUC_bootstrap(
            y_true, y_score, bootstrapRatio=0.5)
        if np.sum(y_true) == 0 or np.sum(y_true) == len(y_true):
            auroc, auprc, f1, f1_weighted = np.nan, np.nan, np.nan, np.nan
            logging.info("AUCs cannot be computed for {}".format(col))
        else:
            try:
                auroc = roc_auc_score(y_true, y_score)
                auprc = average_precision_score(y_true, y_score)
                pred_binary = (y_score >= 0.5) + 0
                f1 = np.nan
                f1_weighted = np.nan
                # f1 = f1_score(y_true, pred_binary)
                # f1_weighted = f1_score(y_true, pred_binary, average='weighted')
            except:
                auroc, auprc, f1, f1_weighted = np.nan, np.nan, np.nan, np.nan
                logging.info("AUCs cannot be computed for {}".format(col))
        df_auc.loc[col, 'AUROC'] = auroc
        df_auc.loc[col, 'AUPRC'] = auprc
        df_auc.loc[col, 'F1'] = f1
        df_auc.loc[col, 'F1_weighted'] = f1_weighted
        df_bootstrap[0][col] = AUC_bootstrap
        df_bootstrap[1][col] = AUC_PR_bootstrap
        df_bootstrap[2][col] = pAUC_bootstrap
        df_bootstrap[3][col] = pAUC_PR_bootstrap
        df_auc.loc[col, 'AUROC_bootstrapMedian'] = AUC_bootstrapMedian
        df_auc.loc[col, 'AUPRC_bootstrapMedian'] = AUC_PR_bootstrapMedian
        df_auc.loc[col, 'pAUROC_colbootstrapMedian'] = pAUC_bootstrapMedian
        df_auc.loc[col, 'pAUPRC_bootstrapMedian'] = pAUC_PR_bootstrapMedian
        df_auc.loc[col, 'AUROC_bootstrapStd'] = AUC_bootstrapStd
        df_auc.loc[col, 'AUPRC_bootstrapStd'] = AUC_PR_bootstrapStd
        df_auc.loc[col, 'pAUROC_bootstrapStd'] = pAUC_bootstrapStd
        df_auc.loc[col, 'pAUPRC_bootstrapStd'] = pAUC_PR_bootstrapStd
        df_auc.loc[col, 'GS_positive_n'] = np.sum(y_true)
        df_auc.loc[col, 'GS_negative_n'] = len(y_true) - df_auc.loc[col, 'GS_positive_n']
    df_auc['Dep_mad'] = mad(labels_all.loc[pred.index])
    df_auc['Dep_mean'] = np.mean(labels_all.loc[pred.index])
    df_auc = df_auc.sort_values('AUROC', ascending=False)
    return df_auc, df_bootstrap


def compute_pAUC(fprGiven, tprGiven, trapezoid=False, pAUC_thrshold=0.1):
    '''
    Compute partial auc
    Use the FPR and TPR to compute the partial AUC.  This code has been
    scribbed from the following link:
    http://stackoverflow.com/questions/39537443/how-to-calculate-a-partial-area-under-the-curve-auc
    Parameters
    ----------
    fprGiven: list
        A list of fpr values for a given experiment for which to calculate
        a partial AUC.
    tprGiven list
        A list of tpr values for a given experiment for which to calculate
        a partial AUC.
    trapezoid: bool
        Whether or not to use the trapezoid rule when calculating the
        partial AUC.  Set to False by default.
    Returns
    -------
    float
        The partial AUC given for the first section of the AUC curve as
        defined by the pAUC_thrshold variable.
    '''
    p = bisect.bisect_left(fprGiven, pAUC_thrshold)
    fprGiven[p] = pAUC_thrshold
    pFpr = fprGiven[: p + 1]
    pTpr = tprGiven[: p + 1]
    area = 0
    ft = list(zip(pFpr, pTpr))
    for p0, p1 in zip(ft[: -1], ft[1:]):
        area += (p1[0] - p0[0]) * \
                ((p1[1] + p0[1]) / 2 if trapezoid else p0[1])
    return area


def compute_AUC_bootstrap(y_true, y_score, bootstrapRatio=0.5, min_positive_n=3, bootstrap_n=50):
    '''
    Compute AUC Bootstrap
    Computes the data to fill in AUC_bootstrapMedian,
    AUC_PR_bootstrapMedian, pAUC_bootstrapMedian,
    and pAUC_PR_bootstrapMedian.
    '''
    truePositiveN = sum(y_true)
    if truePositiveN < min_positive_n / bootstrapRatio:
        return [np.nan] * bootstrap_n, [np.nan] * bootstrap_n, [np.nan] * bootstrap_n, [
            np.nan] * bootstrap_n, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    idx_TP = y_true.nonzero()[0]
    idx_TN = np.array(list(set(range(len(y_true))).difference(set(idx_TP))))
    if len(idx_TP) == 0 or len(idx_TN) == 0:
        return [np.nan] * bootstrap_n, [np.nan] * bootstrap_n, [np.nan] * bootstrap_n, [
            np.nan] * bootstrap_n, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    sampleN = int(truePositiveN * bootstrapRatio)
    # print('Computing bootstraped AUC for {} samples per run and {} iterations'.format(
    #     sampleN * 2, bootstrap_n))
    AUC_bootstrap = []
    AUC_PR_bootstrap = []
    pAUC_bootstrap = []
    pAUC_PR_bootstrap = []
    for j in range(bootstrap_n):
        idx_TP_selected = np.random.choice(idx_TP, sampleN, replace=False)
        if len(idx_TN) >= sampleN:
            idx_TN_selected = np.random.choice(idx_TN, sampleN, replace=False)
        else:
            idx_TN_selected = np.random.choice(idx_TN, sampleN, replace=True)
        idx = np.concatenate([idx_TP_selected, idx_TN_selected])
        fpr, tpr, _threshold = roc_curve(y_true[idx], y_score[idx])
        precision, recall, _ = precision_recall_curve(y_true[idx], y_score[idx])
        AUC_PR_bootstrap.append(average_precision_score(y_true[idx], y_score[idx]))
        AUC_bootstrap.append(auc(fpr, tpr))
        pAUC_bootstrap.append(compute_pAUC(fpr, tpr))
        # recall_sorted = sorted(recall)
        # precision_sorted = [x for y,x in sorted(zip(recall,precision))]
        # The reversed array is the same as the sorted array from small to
        # large
        recall_sorted = recall[::-1]
        precision_sorted = precision[::-1]
        pAUC_PR_bootstrap.append(compute_pAUC(recall_sorted, precision_sorted))
    AUC_bootstrapMedian = np.median(AUC_bootstrap)
    AUC_PR_bootstrapMedian = np.median(AUC_PR_bootstrap)
    pAUC_bootstrapMedian = np.median(pAUC_bootstrap)
    pAUC_PR_bootstrapMedian = np.median(pAUC_PR_bootstrap)
    AUC_bootstrapStd = np.std(AUC_bootstrap)
    AUC_PR_bootstrapStd = np.std(AUC_PR_bootstrap)
    pAUC_bootstrapStd = np.std(pAUC_bootstrap)
    pAUC_PR_bootstrapStd = np.std(pAUC_PR_bootstrap)
    return AUC_bootstrap, AUC_PR_bootstrap, pAUC_bootstrap, pAUC_PR_bootstrap, AUC_bootstrapMedian, AUC_PR_bootstrapMedian, pAUC_bootstrapMedian, pAUC_PR_bootstrapMedian, AUC_bootstrapStd, AUC_PR_bootstrapStd, pAUC_bootstrapStd, pAUC_PR_bootstrapStd


def plot_top_ROC(df_auc, labels_all_binary, pred, result_path, output_prefix, top_n=10):
    df_auc = df_auc.loc[df_auc['GS_positive_n'] >= 6]
    df_auc = df_auc.loc[df_auc['GS_negative_n'] >= 6]
    df_auc = df_auc.sort_values('AUROC', ascending=False)
    logging.info(df_auc.head(10))
    for idx in df_auc.head(top_n).index:
        y_true = labels_all_binary[idx].values.astype(int)
        y_score = pred[idx].values
        fpr, tpr, _thresholds = roc_curve(y_true, y_score)
        AUC = auc(fpr, tpr)
        prediction_n = len(y_true)
        gs_positive_n = np.sum(y_true)
        plot_ROC(fpr, tpr, prediction_n, gs_positive_n, AUC, result_path, '{}_{}'.format(output_prefix, idx))


def plot_ROC(fpr, tpr, prediction_n, gs_positive_n, AUC, result_path, output_prefix, randomline=True):
    plt.figure(figsize=(8, 8), dpi=100)
    plt.xlabel("FPR", fontsize=30)
    plt.ylabel("TPR", fontsize=30)
    # plt.title("ROC Curve", fontsize=30)
    plt.plot(fpr, tpr, linewidth=2,
             label='Prediction #= {0} (+:-={1}:{2}), AUC = {3:.3f}'.format(prediction_n, gs_positive_n,
                                                                           prediction_n - gs_positive_n, AUC))
    # plt.plot(fpr, tpr, linewidth=2,
    #          label='Reactome, AUROC={:.3f}'.format(auroc))
    # plt.plot(fpr_r, tpr_r, linewidth=2,
    #          label='Random gene group, AUROC={:.3f}'.format(auroc_r))
    if randomline:
        x = [0.0, 1.0]
        plt.plot(x, x, linestyle='dashed', color='red', linewidth=2, label='Random, AUROC=0.5')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16, loc='best')
    plt.tight_layout()
    # output_name = '{}/ROC_{}_with_random.pdf'.format(result_path, output_prefix)
    output_name = '{}/ROC_{}.pdf'.format(result_path, output_prefix)
    plt.savefig(output_name, bbox_inches='tight')
    plt.close()


def plot_pred_true_r_by_gene_MAD(df, result_path, output_prefix, mode='r'):
    interval_order = []
    for x in np.linspace(0, 0.4, num=5):
        if x == 0:
            interval = df.loc[(df['Dep_mad'] < x + 0.1)].index
            interval_str = '<0.1'
        elif x == 0.4:
            interval = df.loc[(df['Dep_mad'] >= x)].index
            interval_str = '>=0.4'
        else:
            interval = df.loc[(df['Dep_mad'] >= x) & (df['Dep_mad'] < x + 0.1)].index
            interval_str = '[{:.1f},{:.1f})'.format(x, x + 0.1)
        interval_str += '\nn={}'.format(len(interval))
        df.loc[interval, 'Measured MAD'] = interval_str
        interval_order.append(interval_str)
    if mode == 'r':
        y_label = 'Pearson_r'
        ylim = [-1, 1]
        output_name = '{}/boxplot_cor_by_gene_MAD_{}.pdf'.format(result_path, output_prefix)
    elif mode == 'auc':
        y_label = 'AUROC'
        ylim = [0, 1]
        output_name = '{}/boxplot_auroc_by_gene_MAD_{}.pdf'.format(result_path, output_prefix)

    sns.boxplot(x='Measured MAD', y=y_label, data=df, fliersize=0.05, order=interval_order)
    plt.xticks(rotation=45)
    plt.ylim(ylim)
    plt.savefig(output_name, bbox_inches='tight')
    plt.close()


def plot_pred_true_r_by_gene_mean(df, result_path, output_prefix, mode='r'):
    interval_order = []
    for x in np.linspace(0, 0.9, num=10):
        interval = df.loc[(df['Dep_mean'] >= x) & (df['Dep_mean'] < x + 0.1)].index
        if x == 0:
            interval_str = '[0,0.1)'
        elif x == 0.4:
            interval_str = '[0.9,1]'
        else:
            interval_str = '[{:.1f},{:.1f})'.format(x, x + 0.1)
        interval_str += '\nn={}'.format(len(interval))
        df.loc[interval, 'Measured mean'] = interval_str
        interval_order.append(interval_str)
    if mode == 'r':
        y_label = 'Pearson_r'
        ylim = [-1, 1]
        output_name = '{}/boxplot_cor_by_gene_mean_{}.pdf'.format(result_path, output_prefix)
    elif mode == 'auc':
        y_label = 'AUROC'
        ylim = [0, 1]
        output_name = '{}/boxplot_auc_by_gene_mean_{}.pdf'.format(result_path, output_prefix)

    sns.boxplot(x='Measured mean', y=y_label, data=df, fliersize=0.05, order=interval_order)
    plt.xticks(rotation=45)
    plt.ylim(ylim)
    plt.savefig(output_name, bbox_inches='tight')
    plt.close()


def plot_hist_auc(df, result_path, output_prefix, mode='AUROC'):
    data = df.loc[~df.isnull()].values
    bins = np.linspace(0, 1, 11)
    weights = np.zeros_like(data) + 1. / data.size
    output_name = '{}/{}_hist_{}.pdf'.format(result_path, mode, output_prefix)
    xticks = np.arange(0, 1.5, 0.5)
    xticklabels = ['0', '0.5', '1']
    __hist(data, bins, weights, output_name, xticks, xticklabels)
    weights = None
    output_name = '{}/{}_hist_ct_{}.pdf'.format(result_path, mode, output_prefix)
    __hist(data, bins, weights, output_name, xticks, xticklabels)


def plot_hist_cor(df, result_path, output_prefix):
    # ax = sns.distplot(df.loc[~df.isnull()].values,
    #                   hist=False, kde=True)
    # # ax.get_legend().remove()
    # # ax.figure.colorbar(sm)
    # # colorbar_ax = ax.figure.axes[1]
    # # colorbar_ax.set_title(hue)
    # plt.savefig('{}/cor_hist_{}.pdf'.format(self.result_path, output_prefix),
    #             bbox_inches='tight')
    # plt.close()
    data = df.loc[~df.isnull()].values
    bins = np.linspace(-1, 1, 21)
    if data.size == 0:
        return
    weights = np.zeros_like(data) + 1. / data.size
    output_name = '{}/cor_hist_{}.pdf'.format(result_path, output_prefix)
    xticks = np.arange(-1, 1.5, 0.5)
    xticklabels = ['-1', '-0.5', '0', '0.5', '1']
    __hist(data, bins, weights, output_name, xticks, xticklabels)
    weights = None
    output_name = '{}/cor_hist_ct_{}.pdf'.format(result_path, output_prefix)
    __hist(data, bins, weights, output_name, xticks, xticklabels)


def __hist(data, bins, weights, output_name, xticks, xticklabels):
    if len(data) == 0:
        return
    fig = plt.figure()
    # plt.xticks(rotation=30)
    ax = fig.add_subplot(111)
    # colors = ("red", "black", "green", "orange", "darkviolet", "yellowgreen",
    #           "blue", "slateblue", "darkred", "skyblue", "magenta", "pink", "steelblue")
    ax.hist(data, bins=bins, histtype='bar', color='black',
            rwidth=0.9, weights=weights)
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    if 'cor' in output_name:
        ax.set_xlabel('Pearson r', fontsize=30)
    else:
        ax.set_xlabel('AUROC', fontsize=30)
    ax.set_ylabel('Frequency', fontsize=30)
    # ax.set_ylim([0, 0.2])
    ax.xaxis.set_ticks(xticks)
    ax.set_xticklabels(xticklabels)
    plt.savefig(output_name, bbox_inches='tight')
    plt.close()
