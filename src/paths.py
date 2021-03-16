#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The script for loading environment variables.
Created on Sep 7, 2018
Last edited on Sep 7, 2018
@author: Chih-Hsu Lin
"""
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
REPO_DIR = os.getenv("REPO_DIR")
RESULTS_DIR = os.getenv("RESULTS_DIR")
DATA_DIR = os.getenv("DATA_DIR")
