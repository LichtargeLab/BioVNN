#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The script for logging setting
Created on Sep 22, 2018
Last edited on Sep 22, 2018
@author: Chih-Hsu Lin
"""
import logging
import datetime
import os
import paths

def set_logging(exp_type, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename='{}/log'.format(log_dir),
                        filemode='w')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    return