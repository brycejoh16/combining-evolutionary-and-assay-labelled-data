from metl_predictors import OnehotRidgePredictor,spearman
from metl_evaluate import random_get_train_test
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

AA = ['Q', 'V', 'K', 'T', 'R', 'M', 'S', 'A', 'C', 'L', 'N', 'I', 'P', 'W', 'E', 'H', 'F', 'D', 'Y', 'G']
AA_MAPPING = {c: i for i, c in enumerate(AA)}
GB1_WT = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"


def get_heatmap_df(seqs):
    length_wt = len(seqs.iloc[0])
    length_AAs = len(AA)

    # np.zeros((length_AAs, length_wt))
    tot = set()
    for seq in list(seqs):
        for i, aa in enumerate(seq):
            tot.add(aa)
    assert np.array(
        [t in AA for t in tot]).all(), 'making the assumption their are 20 AAs and they are these ones, no "*"'
    # https://github.com/samgelman/RosettaTL/blob/master/code/constants.py#L48
    # without the '*' of course
    count = np.zeros((length_AAs, length_wt))
    for seq in list(seqs):
        for i, aa in enumerate(seq):
            count[AA_MAPPING[aa]][i] += 1

    df = pd.DataFrame(data=count,
                      index=AA,
                      columns=[str(i) for i in np.arange(0, length_wt)])

    for i, aa in enumerate(GB1_WT):
        df.loc[aa][str(i)] = np.nan

    return df


def hist_with_heatmaps(outdir, data, run_name, steps=[-np.inf, -6, -4, -2, 0, 2, np.inf],
                       bins=100,column='log_fitness'):
    '''
    makes heatmaps for different splits of a histogram
    :param outdir: top directory
    :param data:
    :param run_name: sub-directory
    :param steps: make sure to include -np.inf and np.inf to get complete closure.
    :return: None

    '''
    outdir = os.path.join(outdir, run_name)
    if not os.path.exists(outdir):
        os.mkdir(f"{outdir}")
    spread = data[column].max() - data[column].min()
    print(f"max of {column} {data[column].max()}")
    print(f"min of {column} {data[column].min()}")
    # step=spread/nb_splits
    # step=spread/nb_splits

    for i in range(len(steps) - 1):
        begin = steps[i]
        end = steps[i + 1]
        interval = data[((data[column] >= begin) & (data[column] < end))].copy()
        if len(interval)==0:
            continue
        # assert len(interval) > 0, 'must have more than one value'
        assert ((interval[column] >= begin) & (interval[column] < end)).all(), 'wierd'
        print(f'column : {column}, interval [{begin} , {end} ] , num_seqs : {len(interval)}')
        df = get_heatmap_df(interval['seq'])
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax = sns.heatmap(df, ax=ax, cmap='plasma')
        ax.set_title(f'section {i} \n interval [{begin} , {end} ], num_seqs : {len(interval)} \n {column}')
        fig.savefig(os.path.join(outdir, f"heatmap_{i}.png"))

    fig, ax = plt.subplots(1, 1)
    data[column].hist(ax=ax, bins=bins)
    for i,step in enumerate(steps):
        ax.axvline(step, color='orange')
        ax.text(step,1,str(i),fontsize='large')
    fig.savefig(os.path.join(outdir, "hist.png"))


def ridge_regression_analysis(dataset_name):
    assert 'gb1' in dataset_name, 'this can only work for gb1 right now, b/c of wildtype in heatmap'
    train, test, data = random_get_train_test(dataset_name=dataset_name)
    outdir = os.path.join('results', dataset_name, 'ridge_analysis')
    if not os.path.exists(outdir):
        os.mkdir(f"{outdir}")
    hist_with_heatmaps(outdir, data, 'full_dataset')


    train_50=train.sample(n=50)

    hist_with_heatmaps(outdir,train_50,'train_50',steps=[-np.inf,-4,0,np.inf],bins=25)
    hist_with_heatmaps(outdir,test,'test_actual_output')


    onehot = OnehotRidgePredictor(dataset_name=dataset_name)
    onehot.train(train_50.seq.values, train_50.log_fitness.values)
    test['predicted'] = onehot.predict(test.seq.values)

    hist_with_heatmaps(outdir,test,'test_predicted_output_onehot_ridge_test_50',column='predicted')
    print(f'spearman : {spearman(test["predicted"],test["log_fitness"])}')



if __name__ == '__main__':
    ridge_regression_analysis('gb1_double_single_40')
