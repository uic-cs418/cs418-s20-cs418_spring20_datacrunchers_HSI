# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:51:26 2020

@author: Varun
"""
import os
from os import listdir
from os.path import isfile, join
import struct
import numpy as np
import random
import operator
import matplotlib.pyplot as plt
import gzip
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import os
from os import listdir
from os.path import isfile, join


codepath=r'C:\Users\Varun\Desktop\IDS Project\Codes'
datapath=r'C:\Users\Varun\Desktop\IDS Project\Dataset'
os.chdir(datapath)
onlyfiles = [f for f in listdir(datapath) if isfile(join(datapath, f))]
onlyfiles

pd.set_option("display.max_rows", None)
DD=pd.read_csv('DATAELEMENTDESCRIPTION.csv')
DD=DD[['PAGE_NAME','COLUMN_NAME','DESCRIPTION','IS_PERCENT_DATA']]

ColstoUse=pd.read_csv('ColstoUse.csv')
DD=DD[DD['COLUMN_NAME'].isin(list(ColstoUse['Cols']))]
#print(DD)