# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 20:46:34 2024

@author: KeliDu
"""
import pandas as pd
from scipy import stats


######################################################################################################################
#get distinctive topics using Welch's T-test


cols = ['id', 'doc_name']
topic_num = 50
n = 0
while n < topic_num:
    cols.append('topic_' + str(n))
    n+=1

#load MALLET topics file
topics = pd.read_csv(r'topics\output_keys_50topics.txt', sep='\t', names=['id', 'alpla', 'topic_words'])

#load MALLET topic document distribution file
doc_topic = pd.read_csv(r'topic_doc_distribution\output_composition_50topics.txt', sep='\t', names=cols)

doc_names = doc_topic['doc_name'].tolist()

subgenres = []
for i in doc_names:
    subgenres.append(i.split('_')[0])
    
doc_topic['subgenre'] = subgenres
    
def split_df (doc_topic, subgenre):
    true_false_list = doc_topic['subgenre'] == subgenre
    target, comparison = doc_topic[true_false_list], doc_topic[~true_false_list]
    target_distributions = target[target.columns[~target.columns.isin(['id', 'doc_name', 'subgenre'])]]
    target_distributions = target_distributions.reset_index(drop=True)
    comparison_distributions = comparison[comparison.columns[~comparison.columns.isin(['id', 'doc_name', 'subgenre'])]]
    comparison_distributions = comparison_distributions.reset_index(drop=True)
    return target_distributions, comparison_distributions

def t_test (subgenre):
    target_distributions, comparison_distributions = split_df (doc_topic, subgenre)
    t_test = []
    for col in target_distributions.columns:
        test = stats.ttest_ind(target_distributions[col], comparison_distributions[col], equal_var = False)
        t_test.append((col, test[0], test[1], ''.join(topics[topics['id'] == int(col.split('_')[1])]['topic_words'].tolist())))
    
    t_test_df = pd.DataFrame(t_test, columns=['topic_no', 'test_statistic', 'p-value', 'topic_words'])
    t_test_df = t_test_df.sort_values(by='test_statistic', ascending=False)
    t_test_df.to_csv('distinctive_' + subgenre + '_' + str(n) + 'topics.csv', sep='\t', index=False)
    #return t_test_df

for subgenre in set(subgenres):
    t_test(subgenre)






















    
    
    