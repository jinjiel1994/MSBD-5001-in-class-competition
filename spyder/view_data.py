#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 16:03:28 2019

@author: jimlee
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('../data/train.csv', index_col='id')
test = pd.read_csv('../data/test.csv', index_col='id')