#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 17:02:53 2021

@author: Tiago Minuzzi
"""
import numpy as np
import pandas as pd
# Assign files to vars
COORDS = 'db_te_coords_beta.csv'
TABOLD = 'db_te_classified_beta.csv'
# Read files as dataframes
coords = pd.read_csv(COORDS, usecols=('score','id','start','end','family'))
tabold = pd.read_csv(TABOLD,usecols=('id','class','order','family'))
# Merge dataframes
merged = pd.merge(tabold,coords,on=['id','family'],how='outer')
# Slice dataframes to select columns
## class, family dataframe
f_classes = merged[['class','family']].loc[~merged['order'].isnull()]
## order, family dataframe
f_ordens = merged[['order','family']].loc[~merged['order'].isnull()]
# Create dictionary from sliced dataframes
## classes dicitionary
d_classes = f_classes.set_index('family').T.to_dict('list')
## orders dictionary
d_ordens = f_ordens.set_index('family').T.to_dict('list')
# Map dictionaries values to dataframe columns
## map values to class column
merged['class'] = merged['family'].map(d_classes).str[0]
## map values to order column
merged['order'] = merged['family'].map(d_ordens).str[0]
# Remove duplicates
merged.drop_duplicates(inplace=True,ignore_index=True)
# Save dataframe to file
merged.to_csv('db_te_class_beta_1.csv',index=False)
# Get value counts for class, order, family
# merged['class'].value_counts()
# merged['order'].value_counts()
# merged['family'].value_counts()