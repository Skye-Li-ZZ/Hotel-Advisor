# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 19:09:23 2021

@author: Skye Li
"""
import pandas as pd
from data_wrangling_func import data_wrangling

files = ['Berckshires.csv','Bristol.csv','CapeCod.csv','Franklin.csv','GreaterSpringfield.csv','GreaterMerrimackValley.csv','Hampshire.csv','MarthasVineyard.csv','Nantucket.csv','NorthBoston.csv','Plymouth.csv']

df_all = data_wrangling('GreaterBoston.csv')
for file in files:
    df = data_wrangling(file)
    df_all = pd.concat([df_all, df], axis=0)

df_all.to_csv('all_hotels.csv', encoding='utf-8')
