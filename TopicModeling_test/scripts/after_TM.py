# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 20:46:34 2024

@author: KeliDu
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


cols = ['id', 'doc_name']
n = 0
while n < 20:
    cols.append('topic_' + str(n))
    n+=1

doc_topic = pd.read_csv(r'C:\Workstation\Trier\Gutenberg\4subgenres_1900-1969\TM_resutls\output_composition_20topics.txt', sep='\t', names=cols)

group = []
x = 0
while x < len(doc_topic):
    group.append(doc_topic['doc_name'][x].split('_')[0])
    x+=1
doc_topic['subgenre'] = group




######################################################################################################################
#get mean topic doc distribution from topic segment distribution
plaintextfolder = r'C:\Workstation\Trier\Gutenberg\4subgenres_1900-1969\lemma_renamed'
txt_files = sorted([os.path.join(plaintextfolder, fn) for fn in os.listdir(plaintextfolder)])

segmentfolder = r'C:\Workstation\Trier\Gutenberg\4subgenres_1900-1969\lemma_segments'
segment_files = sorted([os.path.join(segmentfolder, fn) for fn in os.listdir(segmentfolder)]) 

for_avg = doc_topic[doc_topic.columns[~doc_topic.columns.isin(['id','doc_name', 'subgenre', 'pca-one', 'pca-two'])]]

file_nr = 0
files_topic = []
while file_nr < len(txt_files):
    idxs = []
    for segment in segment_files:
        segment_name = os.path.basename(segment)[:-8]
        if segment_name == os.path.basename(txt_files[file_nr])[:-4]:
            idxs.append(segment_files.index(segment))
    file_topic = list(for_avg.loc[idxs].mean())
    files_topic.append((os.path.basename(txt_files[file_nr])[:-4], file_topic))        
    file_nr += 1
    
topic_doc_avg = pd.read_csv(r'C:\Workstation\Trier\Gutenberg\4subgenres_1900-1969\TM_resutls\topic_doc_avg_20topics.csv', sep='\t')    

group = []
x = 0
while x < len(topic_doc_avg):
    group.append(topic_doc_avg['doc_name'][x].split('_')[0])
    x+=1
topic_doc_avg['subgenre'] = group

#########################################################################################################################
#PCA
from sklearn.decomposition import PCA

def make_pca (df, alpha=0.5):
    pca = PCA(n_components=2)
    for_PCA = df[df.columns[~df.columns.isin(['id','doc_name', 'subgenre'])]]
    pca_result = pca.fit_transform(for_PCA)
    
    
    df['pca-one'] = pca_result[:,0]
    df['pca-two'] = pca_result[:,1] 
    #print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
    
    #plt.figure(figsize=(25,15))
    sns.jointplot(
        x="pca-one", y="pca-two",
        hue="subgenre",
        palette=sns.color_palette("tab10", len(set(group))),
        data=df,
        legend="full",
        #s=30,
        alpha=alpha,
        height=15, 
        kind="kde"
    )
    plt.xlabel('pca-one: ' + str(pca.explained_variance_ratio_[0]))
    plt.ylabel('pca-two: ' + str(pca.explained_variance_ratio_[1]))

make_pca(topic_doc_avg, 1) 

make_pca(doc_topic, 0.5)   

   
    
#########################################################################################################################
#t-sne
from sklearn.manifold import TSNE

t_sne = TSNE(n_components=2, verbose=2, perplexity=5, metric ='cosine', n_iter=300, method='exact', random_state=0)#, n_iter_without_progress=100)

for_tsne = topic_doc_avg[topic_doc_avg.columns[~topic_doc_avg.columns.isin(['id','doc_name', 'subgenre'])]]
tsne_results = t_sne.fit_transform(for_tsne)

topic_doc_avg['tsne-2d-one'] = tsne_results[:,0]
topic_doc_avg['tsne-2d-two'] = tsne_results[:,1]

plt.figure(figsize=(25,15))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="subgenre",
    palette=sns.color_palette("colorblind", len(set(group))),
    data=topic_doc_avg,
    legend="full",
    s=70,
    alpha=0.8
)


######################################################################################################################
#classification der segmente, binär!

from sklearn import svm
from sklearn.model_selection import cross_val_score


cols = ['id', 'doc_name']
n = 0
while n < 100:
    cols.append('topic_' + str(n))
    n+=1


doc_topic = pd.read_csv(r'C:\Workstation\Trier\Gutenberg\4subgenres_1900-1969\TM_resutls\output_composition_100topics.txt', sep='\t', names=cols)
#doc_topic = doc_topic.fillna(0)

segmentfolder = r'C:\Workstation\Trier\Gutenberg\4subgenres_1900-1969\lemma_segments'
segment_files = sorted([os.path.join(segmentfolder, fn) for fn in os.listdir(segmentfolder)]) 

segment_group = []
for file in segment_files:
    segment_group.append(os.path.basename(file).split('_')[0])

def plot_coefficients(classifier, feature_names, group, top_features=10):
 coef = classifier.coef_.ravel()
 top_positive_coefficients = np.argsort(coef)[-top_features:]
 top_negative_coefficients = np.argsort(coef)[:top_features]
 top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
 # create plot
 plt.figure(figsize=(15, 5))
 colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
 plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
 feature_names = np.array(feature_names)
 plt.xticks(np.arange(0, 2 * top_features), feature_names[top_coefficients], rotation=90, ha='right')
 plt.title(group)
 plt.show()
    
def classification (binary_group_true):
    df_classification = doc_topic[doc_topic.columns[~doc_topic.columns.isin(['id','doc_name'])]]
    labels = []
    for group in segment_group:
        if group == binary_group_true:
            labels.append(1)
        else:
            labels.append(0)
    #print(len(labels))
    if segment_group.count(binary_group_true) != labels.count(1):
        print('Something went wrong!')
    else:    
        clf = svm.LinearSVC()
        accuracy = cross_val_score(clf, df_classification, labels, cv=5, scoring='accuracy', n_jobs = 2, error_score='raise')
        f1 = cross_val_score(clf, df_classification, labels, cv=5, scoring='f1_macro', n_jobs = 2, error_score='raise')
        clf.fit(df_classification, labels)
        coef = clf.coef_
    features_names = df_classification.columns
    clf.fit(df_classification, labels)
    plot_coefficients(clf, features_names, binary_group_true+'_f1: '+str(round(f1.mean(), 3)))
    
    return binary_group_true, accuracy.mean(), f1.mean(), coef

classification_results = []
group_list = sorted(list(set(segment_group)))
for group in group_list:
    print(group)
    result = classification(group)
    classification_results.append(result)


results_df = pd.read_csv(r'C:\Workstation\Trier\Gutenberg\4subgenres_1900-1969\TM_resutls\binary_classification_seg_20topics.csv', sep='\t')
results_df_melt = pd.melt(results_df, id_vars=['subgenre','accuracy', 'f1-macro'], value_vars=['topic_0','topic_1','topic_2','topic_3','topic_4','topic_5','topic_6','topic_7','topic_8','topic_9','topic_10','topic_11','topic_12','topic_13','topic_14','topic_15','topic_16','topic_17','topic_18','topic_19'], var_name='topics', value_name='contribution')

sns.set(font_scale=2)
sns.set_style("whitegrid")
g = sns.FacetGrid(results_df_melt, col = 'topics', col_wrap=4, height=5, aspect = 1.4)#)
g.map(sns.barplot, "subgenre", "contribution", palette="colorblind")
for axes in g.axes.flat:
    _ = axes.set_xticklabels(axes.get_xticklabels())#, rotation=90)
plt.tight_layout()

######################################################################################################################
#get distinctive topics using Welch's T-test
from scipy import stats

cols = ['id', 'doc_name']
n = 0
while n < 50:
    cols.append('topic_' + str(n))
    n+=1

topics = pd.read_csv(r'C:\Workstation\Trier\Gutenberg\4subgenres_1900-1969\TM_resutls\output_keys_50topics.txt', sep='\t', names=['id', 'alpla', 'topic_words'])

doc_topic = pd.read_csv(r'C:\Workstation\Trier\Gutenberg\4subgenres_1900-1969\TM_resutls\output_composition_50topics.txt', sep='\t', names=cols)

doc_names = doc_topic['doc_name'].tolist()
#txt_names = doc_topic['doc_name'].unique().tolist()

subgenres = []
for i in doc_names:
    subgenres.append(i.split('_')[0])
    
doc_topic['subgenre'] = subgenres
    
subgenres = list(set(subgenres))

def split_df (doc_topic, subgenre):
    true_false_list = doc_topic['subgenre'] == subgenre
    target, comparison = doc_topic[true_false_list], doc_topic[~true_false_list]
    target = target[target.columns[~target.columns.isin(['id', 'doc_name', 'subgenre'])]]
    target = target.reset_index(drop=True)
    comparison = comparison[comparison.columns[~comparison.columns.isin(['id', 'doc_name', 'subgenre'])]]
    comparison = comparison.reset_index(drop=True)
    return target, comparison

def t_test (subgenre):
    target, comparison = split_df (doc_topic, subgenre)
    t_test = []
    for col in target.columns:
        test = stats.ttest_ind(target[col], comparison[col], equal_var = False)
        t_test.append((col, test[0], test[1], ''.join(topics[topics['id'] == int(col.split('_')[1])]['topic_words'].tolist())))
    
    t_test_df = pd.DataFrame(t_test, columns=['topic_no', 'test_statistic', 'p-value', 'topic_words'])
    t_test_df = t_test_df.sort_values(by='test_statistic', ascending=False)
    t_test_df.to_csv('distinctive_' + subgenre + '_' + str(n) + 'topics.csv', sep='\t', index=False)
    #return t_test_df

os.chdir(r'C:\Workstation\Trier\Gutenberg\4subgenres_1900-1969\TM_resutls')

for i in subgenres:
    t_test(i)






















    
    
    