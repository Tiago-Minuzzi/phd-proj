#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Filter and organize RepeatMasker classification output table

import pandas as pd

TABELA='RepeatMasker_output.tabular'

df = pd.read_csv(TABELA,sep='\t',usecols=['SW score','query sequence','repeat','pos in query: begin','end','class/family','pos in repeat: begin'])

dfc=df[~df['pos in repeat: begin'].str.isnumeric()]
dfc=dfc[['SW score','query sequence','pos in query: begin','end','pos in repeat: begin']]
dfc.columns=['score','query','start','end','class/family']
dfc.reset_index(drop=True,inplace=True)

dfn=df[df['pos in repeat: begin'].str.isnumeric()]
dfn=dfn[['SW score','query sequence','pos in query: begin','end','class/family']]
dfn.columns=['score','query','start','end','class/family']
dfn.reset_index(drop=True,inplace=True)

# Get max score for the 2 best
# dfn.sort_values(by=['query','score'],ascending=[True,False]).groupby('query').head(2)
# # Get max score for each id
# dfn = dfn.loc[dfn.groupby('query')['score'].idxmax()]

# remove duplicates
dfn = dfn[~dfn.duplicated()]
# dfn = dfn[['query','class/family','score']]
dfc = dfc[~dfc.duplicated()]
# dfc = dfc[['query','class/family','score']]

# drop parameter to avoid the old index being added as a column:
dfn.reset_index(drop=True,inplace=True)
dfc.reset_index(drop=True,inplace=True)
# print(dfn.head())
# print(dfc.head())

# concatenate dfn and dfc
dfa=pd.concat([dfn,dfc],ignore_index=True)
# dfa=dfa[['query','class/family','score']]

# sort values
dfa.sort_values(['query','score'],ascending=[True,False])

# remove possible duplicates
dfa.drop_duplicates(inplace=True)

# remove unwanted characters
dfa['class/family'].str.replace('-.*$','')
dfa['class/family']=dfa['class/family'].str.replace('-.*$','')
dfa['class/family']=dfa['class/family'].str.replace('\?$','')

# select query with best score
# dfa = dfa.loc[dfa.groupby('query')['score'].idxmax()]
# dfa.reset_index(drop=True,inplace=True)

# remove not TE
dfa = dfa[(~dfa['class/family'].str.contains('Other')) & (dfa['class/family']!='Simple_repeat') & (~dfa['class/family'].str.contains('Unkno')) & (~dfa['class/family'].str.contains('Satel'))]
# drop duplicates
dfa.drop_duplicates(ignore_index = True,inplace = True)

# select TEs classified with class and family
# dfa[dfa['class/family'].str.contains('/')]
fam = dfa[dfa['class/family'].str.contains('/')]

# change column names
fam.columns=['score','id','start','end','class']
fam.reset_index(drop=True,inplace=True)

# split column in two
fam[['class','family']]=fam['class'].str.split('/', expand=True)

# save to file
fam.to_csv('db_TE_coords_beta.csv',index=False)


# # group classication by ID 
# grouped = pd.DataFrame(fam.groupby('id')['class'].agg(';'.join),columns=['class'])
# # save grouped
# grouped.to_csv('db_TE_clfam_grouped.csv')

# Classified without family
# less = dfa[~dfa['class/family'].str.contains('/')]
# less[(less['class/family'].str.contains('Low_com')) | (less['class/family'].str.contains('scRNA')) | (less['class/family'].str.contains('rRNA'))]
# less['class/family'].unique()
# less = less[(less['class/family']!='Low_complexity') & (less['class/family']!='snRNA') & (less['class/family']!='Segmental') & (less['class/family']!='rRNA') & (less['class/family']!='scRNA') & (less['class/family']!='tRNA') & (less['class/family']!='Unspecified') & (less['class/family']!='Other')]
# less['class/family']=less['class/family'].str.replace('?','')
# less.reset_index(drop=True,inplace=True)
# less.head()
# less.columns=['score','id','start','end','class']
# less.head()
# less.to_csv('db_TE_famless.csv',index=False)