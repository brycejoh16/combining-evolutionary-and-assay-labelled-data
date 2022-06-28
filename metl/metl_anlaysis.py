from metl_predictors import OnehotRidgePredictor,spearman,int_to_aa,REG_COEF_LIST
from metl_evaluate import random_get_train_test
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression


AA = ['Q', 'V', 'K', 'T', 'R', 'M', 'S', 'A', 'C', 'L', 'N', 'I', 'P', 'W', 'E', 'H', 'F', 'D', 'Y', 'G']
AA_MAPPING = {c: i for i, c in enumerate(AA)}
GB1_WT = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"

def W_heatmap(outdir,model):
    '''
    this is a function to make a heatmap of the weights from a ridge regression function
    it will ignore all AA's which aren't in the AA one above
    :param outdir:
    :param run_name:
    :param model:
    :return:
    '''
    W=model.coef_.reshape(57,24).T[:,1:]
    print(f'sum of all weights : {W.sum()}')
    assert W.shape==(24,56),'shape for gb1 weights'
    assert W.shape[0]< W.shape[1], "shape should be AA x length"
    print('this only works for GB1 !! ')
    W_star=np.zeros((len(AA),len(GB1_WT)))
    tot=[]
    for i in np.arange(W.shape[0]):
        aa=int_to_aa[i]
        if aa not in AA :
            print(f'{aa} not in AA')
            assert W[i,:].sum()<1e-10, 'this value shouldnt even be present'
            continue
        else:
            assert aa in AA , 'hmmmm'
            idx=AA_MAPPING[aa]
            new_row=W[i, :].copy()
            assert len(new_row)==56,'gotcha'
            W_star[idx,:]=new_row
            tot.append(aa)
    assert set(tot)==set(AA), ' some AAs were missed? '
    df=pd.DataFrame(data=W_star,
                    index=AA,
                    columns=[str(i) for i in np.arange(0, len(GB1_WT))])
    return df,W.sum()




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
        ax = sns.heatmap(np.isnan(df), ax=ax,cmap='Greys',linewidths=.1,linecolor='k',cbar=False)
        ax = sns.heatmap(df, ax=ax, cmap='Oranges',linewidths=.1,linecolor='k')

        #ax.scatter(*np.isnan(df), marker="x", color="black", s=20)
        ax.set_title(f'section {i} \n interval [{begin} , {end} ], num_seqs : {len(interval)} \n {column}')
        fig.savefig(os.path.join(outdir, f"heatmap_{i}.png"))
        plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    data[column].hist(ax=ax, bins=bins)
    for i,step in enumerate(steps):
        ax.axvline(step, color='orange')
        ax.text(step,1,str(i),fontsize='large')
    fig.savefig(os.path.join(outdir, "hist.png"))
    plt.close(fig)

def single_model_data_analysis(dataset_name,outdir,train_down,test,n,reg_coef):

    onehot = OnehotRidgePredictor(dataset_name=dataset_name,reg_coef=reg_coef)

    print(f'starting traning, for model {onehot}')
    onehot.train(train_down.seq.values, train_down.log_fitness.values)

    df,sumW=W_heatmap(outdir, onehot.model)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax = sns.heatmap(df, ax=ax, cmap='bwr', linewidths=.1, linecolor='k')
    ax.set_title(f'weights for run train_{n}, sum:{sumW}')
    fig.savefig(os.path.join(outdir, f"weights_{onehot}.png"))
    plt.close(fig)


    test['predicted'] = onehot.predict(test.seq.values)
    hist_with_heatmaps(outdir, test, f'test_predicted_{onehot}_test_{n}', column='predicted')
    print(f'spearman : {spearman(test["predicted"], test["log_fitness"])}')
    print(f'done training for {onehot} model')

    return df


def individual_train_ridge_dataset_analysis(dataset_name='gb1_double_single_40',
                                      n=50):

    train, test, data = random_get_train_test(dataset_name=dataset_name)
    outdir = os.path.join('results', dataset_name, 'ridge_analysis')
    if not os.path.exists(outdir):
        os.mkdir(f"{outdir}")


    outdir = os.path.join(outdir, f"train_{n}")
    if not os.path.exists(outdir):
        os.mkdir(f"{outdir}")

    # hist_with_heatmaps(outdir, data, 'full_dataset')

    train_down = train.sample(n=n, random_state=0)

    hist_with_heatmaps(outdir, train_down, f'train_{n}', steps=[-np.inf, -4, 0, np.inf], bins=25)
    print('========================================\nlinear model ')
    df0=single_model_data_analysis(dataset_name, outdir, train_down, test, n, reg_coef=0)
    print('========================================\nridge alpha=1 ')
    df1=single_model_data_analysis(dataset_name, outdir, train_down, test, n, reg_coef=1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax = sns.heatmap(df1-df0, ax=ax, cmap='bwr', linewidths=.1, linecolor='k')
    ax.set_title(f'ridge_w - linear_w')
    fig.savefig(os.path.join(outdir, f"weights_difference.png"))
    plt.close(fig)
    # print('ridge w/ cross validation')
    # single_model_data_analysis(dataset_name, outdir, train_down, test, n, reg_coef=REG_COEF_LIST)


    # hist_with_heatmaps(outdir,test,'test_actual_output')

def analysis_predicting_most_popular_number():
    pg_dir=os.path.join('databases','ProteinGym_substitutions')
    dms_ids=['GFP_AEQVI_Sarkisyan_2016','HIS7_YEAST_Pokusaeva_2019','SPG1_STRSG_Olson_2014']

    for dms_id in dms_ids:
        df=pd.read_csv(os.path.join(pg_dir,f'{dms_id}.csv')).set_index('mutant')
        counts=df['DMS_score'].value_counts(bins=100)
        dms_score2predict=counts.index[0].mid
        df['predicted']=np.ones((len(df['DMS_score']),1)).reshape(-1)*dms_score2predict
        df['predicted_random_noise']=df['predicted'].apply(lambda x: x + np.random.normal(scale=.05))

        spear=spearman(df['predicted'],df['DMS_score'])
        print(f'spearman most frequent value( {dms_score2predict}  for dataset {dms_id}'
              f'\n -----> {spear}')

        spear=spearman(df['predicted_random_noise'],df['DMS_score'])
        print(f'spearman most frequent value with guassian noise ( {dms_score2predict}  for dataset {dms_id}'
              f'\n -----> {spear}')





def ridge_regression_analysis(dataset_name):
    assert 'gb1' in dataset_name, 'this can only work for gb1 right now, b/c of wildtype in heatmap'

    individual_train_ridge_dataset_analysis(dataset_name)

    # test['predicted'] = onehot.predict(test.seq.values)
    # hist_with_heatmaps(outdir,test,'test_predicted_output_onehot_ridge_test_50',column='predicted')
    # print(f'spearman : {spearman(test["predicted"],test["log_fitness"])}')
    #
    # train_1000 = train.sample(n=1000,random_state=0)
    #
    # hist_with_heatmaps(outdir,train_1000,'train_1000')
    # onehot = OnehotRidgePredictor(dataset_name=dataset_name)
    # onehot.train(train_1000.seq.values, train_1000.log_fitness.values)
    # test['predicted_1000'] = onehot.predict(test.seq.values)
    #
    # hist_with_heatmaps(outdir, test, 'test_predicted_output_onehot_ridge_test_1000', column='predicted_1000')



if __name__ == '__main__':
    # ridge_regression_analysis('gb1_double_single_40')
    # individual_train_ridge_dataset_analysis(n=50)
    # individual_train_ridge_dataset_analysis(n=1000)
    analysis_predicting_most_popular_number()

