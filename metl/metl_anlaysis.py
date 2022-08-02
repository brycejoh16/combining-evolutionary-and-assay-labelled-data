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



def raw2clean_BLAT_ECOLX_comparision():
    filename1 = os.path.join('databases', 'ProteinGym_substitutions', f'BLAT_ECOLX_Deng_2012.csv')
    filename2 = os.path.join('databases', 'substitutions_raw_DMS', f'BLAT_ECOLX_Stiffler_2015.csv')
    dataset1 = pd.read_csv(filename1).set_index('mutant')
    dataset2 = pd.read_csv(filename2).set_index('mutant')

    dataset1['Stiffler625']=dataset2['625']

    dataset1['Stiffler2500']=dataset2['2500']

    dataset1.plot.scatter(x='DMS_score',y='Stiffler625',s=0.2)
    plt.title(f'spearman : {spearman(dataset1["DMS_score"],dataset1["Stiffler625"])}')
    plt.savefig(os.path.join('databases','database_comparison','BLAT_ECOLX_Deng_2012_vs_Stiffler625.png'))

    dataset1.plot.scatter(x='DMS_score', y='Stiffler2500', s=0.2)
    plt.title(f'spearman : {spearman(dataset1["DMS_score"], dataset1["Stiffler2500"])}')
    plt.savefig(os.path.join('databases', 'database_comparison', 'BLAT_ECOLX_Deng_2012_vs_Stiffler2500.png'))

    min_values=[]
    for row in dataset1.itertuples():
        min_values.append(min(row.Stiffler2500 ,row.Stiffler625))

    dataset1['min625_2500']=min_values

    dataset1.plot.scatter(x='DMS_score', y='min625_2500', s=0.2)
    plt.title(f'minimum value of 625 and 2500 ,spearman : {spearman(dataset1["DMS_score"], dataset1["min625_2500"])}')
    plt.savefig(os.path.join('databases', 'database_comparison', 'BLAT_ECOLX_Deng_2012_vs_min625_2500Stiffler.png'))

def competing_landscape(files=['BLAT_ECOLX_Stiffler_2015','BLAT_ECOLX_Deng_2012']):



    filename1=os.path.join('databases','ProteinGym_substitutions',f'{files[0]}.csv')
    filename2=os.path.join('databases', 'ProteinGym_substitutions', f'{files[1]}.csv')

    dataset1=pd.read_csv(filename1).set_index('mutant')
    dataset2 = pd.read_csv(filename2).set_index('mutant')


    # assert (dataset1.index==dataset2.index).all(), 'your assuming equality'
    set_dataset_2=set(dataset2.index)

    isin=[i in set_dataset_2 for i in dataset1.index]
    mystring=f'{sum(isin)} mutations are in each dataset'
    mystring+=f'\n{len(dataset1)}:{files[0]}, {len(dataset2)}:{files[1]} '
    myindex=dataset1.index[isin]

    dms_score_dataset_1=dataset1['DMS_score'][myindex]
    dms_score_dataset_2 = dataset2['DMS_score'][myindex]
    mystring+=f"\n spearman : {spearman(dms_score_dataset_1,dms_score_dataset_2)}"
    fig,ax=plt.subplots(1,1)
    ax.scatter(dms_score_dataset_1,dms_score_dataset_2,s=0.2,c='g')
    ax.set_xlabel(files[0])
    ax.set_ylabel(files[1])
    ax.set_title(mystring)
    plt.tight_layout()
    fig.savefig(os.path.join('databases','database_comparison',
                             f"{files[0]}_vs_{files[1]}.png"))
def make_a_fasta_file():
    filename=os.path.join('..','data','gb1_double_single_40','data.csv')
    df=pd.read_csv(filename)
    with open(os.path.join('..','data','gb1_double_single_40','data.fasta'),'w') as fout:
        for row in df.itertuples():
            fout.write(f'>{row.mutant}\n{row.seq}\n')

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

def get_mutation(mutants,offset=0):
    ans=[]
    for mutant in mutants.split(','):
        idx=int(mutant[1:-1]) - offset
        assert idx>=0,'why is offset greater than the mutation position?'
        ans.append((str(mutant[0]), idx, str(mutant[-1])))
    return ans

def get_mutation_helper(mutants,offset=0):
    mutation_triples=get_mutation(mutants,offset)
    count=0
    mutant_name=''
    for mutant_triple in mutation_triples:
        if count:
            mutant_name+=','
        assert len(mutant_triple)==3
        mutant_name+=mutant_triple[0]+str(mutant_triple[1])+mutant_triple[2]
        count=1
    return mutant_name

def unit_test_get_mutation():
    assert get_mutation('W394P')==[('W',394,'P')]
    assert get_mutation('W3P') == [('W', 3, 'P')]
    assert get_mutation('W3P',2)==[('W',1,'P')]
    assert get_mutation('W3P,T54Y')==[('W',3,'P'),('T',54,'Y')]

    print(get_mutation('Y239P'))
    print(get_mutation('W3P,T54Y,T56R',offset=2))
    print(get_mutation_helper('W3P,T54Y,T56R',offset=2))

def gb1_dataset_enrich2_vs_lnW():
    nn4dms=os.path.join('..','data','gb1_double_single_40','data.csv')
    protein_gym=os.path.join('databases','ProteinGym_substitutions','SPG1_STRSG_Olson_2014.csv')

    enrich2=pd.read_csv(nn4dms).set_index('mutant')
    lnW=pd.read_csv(protein_gym)
    lnW['mutant']=lnW['mutant'].apply(lambda x:get_mutation_helper(str(x.replace(":",",")),offset=227))
    lnW=lnW.set_index('mutant')
    assert len(pd.merge(lnW,enrich2,how='inner',on='mutant'))>5e5, 'should have majority of variants'

    lnW['enrich2']=enrich2['log_fitness']

    ax=lnW.plot.scatter(x='DMS_score',y='enrich2',s=0.2,alpha=0.3)
    ax.set_xlabel('lnW')
    ax.set_title('number of data points in lnW with score \n'
                 'equal to -4.60517018598809\n'
                 f'=>number {len(lnW[lnW["DMS_score"]==-4.60517018598809])}')
    plt.show()
    new_mutants_with_offset=[]
    # for mutant in lnW['mutant']:
    # #
    # lnW=lnW['mutant'].apply(lambda x: )

def spearman_on_all(lnW):
    print(f'length of dataset {len(lnW)}')
    df=pd.DataFrame()
    for col in lnW.columns:
        if col not in ['mutant', 'DMS_score', 'DMS_score_bin']:
            score=spearman(lnW["DMS_score"], lnW[col])
            print(f'{col}: {score:0.2f}')
          #  df[col]=score
        #todo : i need to go
    return df

def spearman_correlation_toy_around():
    protein_gym = os.path.join('databases', 'protein_gym_benchmark', 'SPG1_STRSG_Olson_2014.csv')
    lnW = pd.read_csv(protein_gym)
    lnW['mutant'] = lnW['mutant'].apply(lambda x: get_mutation_helper(str(x.replace(":", ",")), offset=227))
    lnW = lnW.set_index('mutant')
    spearman_same_number=spearman(np.array([-4.60517018598809]*len(lnW)),lnW['DMS_score'])

    print(f'spearman same number {spearman_same_number}')
    print(f'filtering out two nan values')
    lnW=lnW[~np.isnan(lnW['EVE_single'])]
    df1=spearman_on_all(lnW)
    lnW=lnW[lnW["DMS_score"]!=-4.60517018598809]
    print('\n==========filter out all the values of same score =============')
    df2=spearman_on_all(lnW)

    df=pd.concat([df1,df2])
    df.plot.bar()

    plt.show()





def gb1_dataset_confusion():
    filename_gb1=os.path.join('databases','ProteinGym_substitutions','SPG1_STRSG_Olson_2014.csv')
    df=pd.read_csv(filename_gb1)
    df['mutant']=df['mutant'].apply(lambda x:x.replace(':',','))
    df=df.set_index('mutant')
    #
    count=0
    count_double=0
    for mutant in df.index:
        if "," in str(mutant):
            count_double+=1
        else:
            count+=1

    print(f'found {count} single mutants, {count_double} double')


    # filename_nn4dms=os.path.join('..','data','gb1_double_single_40','data.csv')
    filename_unsup='gb1_unsupervised.tsv'
    # filename_raw=os.path.join('databases','substitutions_raw_DMS','SPG1_STRSG_Olson_2014.csv')
    dfraw=pd.read_csv(filename_unsup,sep='\t').set_index('variant')
    # dfnn4dms=pd.read_csv(filename_nn4dms).set_index('mutant')
    count2=0
    count_double2=0
    for mutant in dfraw.index:
        if ',' in mutant:
            count_double2+=1
        else:
            count2+=1


    print(f"found {count2} single mutants in dfraw, {count_double2} double")

    # mystery solved!! so its based on the input and sel counts. And they don't remove ones which are bad.

    full_df=pd.concat([df,dfraw]).index.drop_duplicates(keep=False)

if __name__ == '__main__':
#     # ridge_regression_analysis('gb1_double_single_40')
#     # individual_train_ridge_dataset_analysis(n=50)
#     # individual_train_ridge_dataset_analysis(n=1000)
#     # analysis_predicting_most_popular_number()
#
    # competing_landscape(['SPIKE_SARS2_Starr_bind_2020','SPIKE_SARS2_Starr_expr_2020'])
    # competing_landscape(['VKOR1_HUMAN_Chiasson_abundance_2020','VKOR1_HUMAN_Chiasson_activity_2020'])
    # gb1_dataset_confusion()
    # make_a_fasta_file()
    # competing_landscape(['PTEN_HUMAN_Matreyek_2021','PTEN_HUMAN_Mighell_2018'])
    # competing_landscape(['CP2C9_HUMAN_Amorosi_abundance_2021','CP2C9_HUMAN_Amorosi_activity_2021'])
#     raw2clean_BLAT_ECOLX_comparision()
#     competing_landscape(['DLG4_HUMAN_Faure_2021','DLG4_RAT_McLaughlin_2012'])
#     competing_landscape(['HSP82_YEAST_Flynn_2019','HSP82_YEAST_Mishra_2016'])
#     competing_landscape(['A0A2Z5U3Z0_9INFA_Doud_2016','A0A2Z5U3Z0_9INFA_Wu_2014'])
#     unit_test_get_mutation()
#     gb1_dataset_enrich2_vs_lnW()
    spearman_correlation_toy_around()