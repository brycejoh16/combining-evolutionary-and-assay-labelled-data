import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import metl_predictors as mtlp
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

METL_SPLIT_DIRECTORY=os.path.join('..','RosettaTL','data','dms_data','gb1','splits','dev')
def unit_test_on_predict_unsupervised():
    filepath = os.path.join('..', 'data', 'BLAT_ECOLX_Ranganathan2015-2500', 'data.csv')
    df = pd.read_csv(filepath)
    ev = mtlp.EVPredictor('BLAT_ECOLX_Ranganathan2015-2500')
    df = df[:5].copy()
    df['seq'] = df['seq'].apply(lambda x: x.lower())
    print(ev.predict_unsupervised(df.seq.values))

    ### so I guess this is failing which is good.
    ### when I set ignore_gaps to true it works which makes sense.

def unit_test_for_nan_values(dataset_name):
    # first thing we need to do is load the data
    filepath = os.path.join('..', 'data', dataset_name, 'data.csv')
    data = pd.read_csv(filepath)[:10]
    ev = mtlp.EVPredictor(dataset_name)
    # test = data.sample(frac=0.2, random_state=0)
    # test = test.copy()
    # train = data.drop(test.index)
    predictions = ev.predict_unsupervised(data.seq.values)
    data['pred'] = predictions
    my_spearman = mtlp.spearman(data.pred.values, data.log_fitness.values)
    ### so that solves the mystery, if its zero then it doesn't throw an error it just returns zero,
    #### it says the gaps here are two much so I'm not going to predict at all.
    ## I don't want to do a hard replace that would be gross.
    print(f'num data points {len(data)}, protein : {dataset_name}  , spearman: {my_spearman}')


def save_bitscore_models(dataset_name):
    filepath = os.path.join('..', 'data', dataset_name, 'data.csv')
    ev = mtlp.EVPredictor(dataset_name)
    data = pd.read_csv(filepath)
    predictions = ev.predict_unsupervised(data.seq.values)
    data['pred'] = predictions
    data.to_csv(f'models/{dataset_name}_scores.csv')


def metl_learning_curve(dataset_name,data,split_name, point_eval,df):
    assert split_name=='reduced_s3' or split_name=='reduced_s2','s1 doesnt work b/c of tune data'
    split_dir=os.path.join(METL_SPLIT_DIRECTORY,split_name)
    Y,Yerr,X=[],[],[]
    for split in tqdm(os.listdir(split_dir)):
        assert 'tu0' in split and 'te0.1' in split, 'must have 0 tune data'
        single_split_dir=os.path.join(split_dir,split)
        y,x=[],[]
        for s in os.listdir(single_split_dir):
            filename=os.path.join(single_split_dir,s)
            train_idx=np.loadtxt(os.path.join(filename,'train.txt'),dtype=int)
            test_idx= np.loadtxt(os.path.join(filename,'test.txt'),dtype=int)
            train=data.loc[train_idx].copy()
            test=data.loc[test_idx].copy()
            assert (train.index==train_idx).all() and (test.index==test_idx).all()
            spearman=point_eval(dataset_name,train,test,df=df)
            y.append(spearman)
            x.append(len(train))
        Y.append(np.array(y).mean())
        Yerr.append(np.array(y).std())
        assert (np.array(x)==len(train)).all()
        X.append(x[0])
    return X,Y,Yerr

def metl_make_learning_curve():

    dataset_name='gb1_double_single_40'
    outdir = os.path.join('results', dataset_name,'metl')
    if not os.path.exists(outdir):
        os.mkdir(f"{outdir}")
    split_name='reduced_s3'
    filepath = os.path.join('..', 'data', dataset_name, 'data.csv')
    data = pd.read_csv(filepath)


    funcs2include = [one_hot_single_data_point, joint_pretrained_ev_onehot_single_data_point,
                     joint_pretrained_rosetta_onehot_single_data_point]
    path = os.path.join('models', f"gb1_pretrained_features.csv")
    df = pd.read_csv(path).set_index('seq')

    # todo: add the unsupervised
    # for unsup in ['predictions_rosetta', 'evmutation_epistatis_MSA_40']:
    #     unsup_scores = df.loc[test.set_index('seq').index][unsup]
    #     assert (unsup_scores.index == test.set_index('seq').index).all(), 'these indexes must be the same'
    #     spearman_unsup = mtlp.spearman(unsup_scores, test.log_fitness.values)
    #     plt.axhline(spearman_unsup, label=unsup)


    fig, ax=plt.subplots(1,1)
    for point_eval in funcs2include:
        print(f'doing function {point_eval.__name__}')
        X,Y,Yerr=metl_learning_curve(dataset_name,data,split_name,point_eval,df=df)
        ax.errorbar(X, Y, Yerr, label=point_eval.__name__, marker='^', capsize=5, ls='')

    ax.set_xscale("log")
    fig.legend()
    ax.set_title(f'spearman correlation vs nb of data points \n {split_name}')
    ax.set_xlabel('nb of training points')
    ax.set_ylabel('spearman correlation')
    fig.savefig(os.path.join(outdir, f'{split_name}_learning_curve_reg_ceof_{mtlp.REG_COEF_LIST}.png'))







def learning_curve(dataset_name, train, test, point_eval, x, df):
    assert (train.sample(n=20).index==train.sample(n=20).index).sum()  < 20, 'make sure your not using random seed'
    y = []
    for nb_train in x:
        print(f'nb_train {nb_train} , {point_eval.__name__}')
        spearman=point_eval(dataset_name, train.sample(n=nb_train), test, df=df)
        y.append(abs(spearman))
        print(f'spearman {spearman:0.2f}')

    print('leaving learning curve')
    return y


def random_get_train_test(dataset_name, frac=0.2,random_state=0):
    filepath = os.path.join('..', 'data', dataset_name, 'data.csv')
    data = pd.read_csv(filepath)
    test = data.sample(frac=frac,random_state=random_state)  # so im hard coding my test state,
    # but not my split on which points to sample from.
    test = test.copy()
    train = data.drop(test.index)
    assert len(train) > len(test)

    print(f'train length  {len(train)}, ')
    return train, test, data


def random_full_learning_curve(dataset_name):
    outdir = os.path.join('results', dataset_name)
    if not os.path.exists(outdir):
        os.mkdir(f"{outdir}")


    func_names=['ridge+onehot','linear+onehot','rosetta+onehot+ev',
                'ev+onehot','rosetta+onehot']

    funcs2include=[one_hot_single_data_point, linear_model_onehot_single_data_point,
                   joint_pretrained_rosetta_onehot_evmutation_single_data_point,
                   joint_pretrained_ev_onehot_single_data_point,
                       joint_pretrained_rosetta_onehot_single_data_point]

    #


    train, test, data = random_get_train_test(dataset_name)
    melt_df = get_unsupervised(data)
    fig,ax=plt.subplots(1,1)

    # todo: this all kind of needs to be re-abstracted to allow for variable
    #      -> protein of interest
    #      -> variable unsupervised methods
    #      -> variable supervised models

    for i,unsup in enumerate(['predictions_rosetta', 'gb1_double_single_40']):
        unsup_scores = melt_df.loc[test.set_index('seq').index][unsup]
        assert (unsup_scores.index == test.set_index('seq').index).all(), 'these indexes must be the same'
        spearman_unsup = abs(mtlp.spearman(unsup_scores, test.log_fitness.values))
        ax.axhline(spearman_unsup, label=['rosetta','ev'][i],color=['red','green'][i])

    x = np.arange(10, 210 + 50, 50)
    nb_of_seeds = 5

    df_spearman=pd.DataFrame(data=x,columns=['nb_training_points']).set_index('nb_training_points')
    df_spearman['seeds']=np.ones((x.shape[0],1)).reshape(-1)*nb_of_seeds
    for name,point_eval in zip(func_names,funcs2include):
        Y = []
        for _ in range(nb_of_seeds):
            y = learning_curve(dataset_name, train, test, point_eval, x,df=melt_df)
            Y.append(y)

        yerr = np.array(Y).std(axis=0)
        y = np.array(Y).mean(axis=0)

        df_spearman[f"{name}_mean"]=y
        df_spearman[f"{name}_err"]=yerr

        ax.errorbar(x, y, yerr, label=name,marker='^',capsize=5,ls='')

    fig.legend(loc='lower right')
    ax.set_title(f'spearman correlation vs nb of training points\n for {dataset_name}')
    ax.set_xlabel('nb of training points')
    ax.set_ylabel('spearman correlation')
    fig.savefig(os.path.join(outdir, f'random_learning_curve_nb_seeds_{nb_of_seeds}.png'))
    df_spearman.to_csv(os.path.join(outdir,'results.csv'))

def eUniRep_reg_single_data_point(dataset_name,train,test,**kwargs):
    eUniRep=mtlp.EUniRepRegressionPredictor(dataset_name)
    eUniRep.train(train.seq.values, train.log_fitness.values)
    predicted=eUniRep.predict(test.seq.values)
    return mtlp.spearman(predicted,test.log_fitness.values)
def eUniRep_LL_single_data_point(dataset_name,train,test,**kwargs):
    #ohhh i see, it needs to be a joint predictor. Its always just predicting
    # as if it was a
    eUniRepLL=mtlp.EUniRepLLPredictor(dataset_name)
    eUniRepLL.train(train.seq.values,train.log_fitness.values)
    predicted=eUniRepLL.predict(test.seq.values)
    return mtlp.spearman(predicted,test.log_fitness.values)


def unit_test_eUniRep_reg_single_data_point():
    dataset='BLAT_ECOLX_Ranganathan2015-2500'
    print(dataset)
    train,test,data=random_get_train_test(dataset)
    Nb=[48,  96, 144, 192, 240, 288]
    for i in Nb:
        sample=train.sample(n=i,random_state=1)
        s=eUniRep_reg_single_data_point(dataset,sample,test)
        s_ll=eUniRep_LL_single_data_point(dataset,sample,test)
        s_ll_onehot=joint_eUniRep_LL_onehot_single_data_point(dataset,sample,test)
        print(f'nb : {i} spearman-reg {s:0.2f} spearman-ll {s_ll:0.2f} spearman-ll+onehot {s_ll_onehot:0.2f}')
def one_hot_single_data_point(dataset_name, train, test, **kwargs):
    onehot = mtlp.OnehotRidgePredictor(dataset_name=dataset_name)
    onehot.train(train.seq.values, train.log_fitness.values)
    predicted = onehot.predict(test.seq.values)
    return mtlp.spearman(predicted, test.log_fitness.values)


def joint_ev_onehot_single_data_point(dataset_name, train, test):
    return joint_predictor_dp(dataset_name, [mtlp.EVPredictor, mtlp.OnehotRidgePredictor], train, test)


def joint_pretrained_ev_onehot_single_data_point(dataset_name, train, test, **kwargs):
    assert 'df' in kwargs.keys(), 'need to pass in df to use this function'
    return joint_predictor_dp(dataset_name, [mtlp.PretrainedFeature, mtlp.OnehotRidgePredictor],
                              train=train, test=test, df=kwargs['df'], feature=['gb1_double_single_40'])

def joint_eUniRep_LL_onehot_single_data_point(dataset_name,train,test,**kwargs):
    return joint_predictor_dp(dataset_name,[mtlp.EUniRepLLPredictor,mtlp.OnehotRidgePredictor],train=train,
                              test=test,**kwargs)
def joint_pretrained_rosetta_onehot_single_data_point(dataset_name, train, test, **kwargs):
    assert 'df' in kwargs.keys(), 'need to pass in df to use this function'
    return joint_predictor_dp(dataset_name, [mtlp.PretrainedFeature, mtlp.OnehotRidgePredictor],
                              train=train, test=test, df=kwargs['df'], feature=['predictions_rosetta'])

def get_unsupervised(data):
    path = 'gb1_unsupervised.tsv'
    metl_data = pd.read_csv(path, sep='\t').set_index('variant')
    data2=data.set_index('mutant').copy()
    metl_data['seq']=data2['seq'].copy()
    return metl_data.set_index('seq')
def unit_test_rosetta_onehot_evmutation():
    dataset_name = 'gb1_double_single_40'
    train, test, data = random_get_train_test(dataset_name)
    df=get_unsupervised(data)
    for n in [48,96,142,196,250]:
        out=joint_pretrained_rosetta_onehot_evmutation_single_data_point(dataset_name, train.sample(n=n), test,df=df)
        print(f"spearman {out:0.3f} for n={n} datapoints")


def joint_pretrained_rosetta_onehot_evmutation_single_data_point(dataset_name, train, test, **kwargs):
    assert 'df' in kwargs.keys(), 'need to pass in df to use this function'
    return joint_predictor_dp(dataset_name, [mtlp.PretrainedFeature, mtlp.PretrainedFeature,
                                             mtlp.OnehotRidgePredictor],
                              train=train, test=test, df=kwargs['df'], feature=['predictions_rosetta',
                                                                                'gb1_double_single_40'])

def playing_around_with_rosetta_and_evmutation():
    dataset_name = 'gb1_double_single_40'
    train, test, data = random_get_train_test(dataset_name)
    df = get_unsupervised(data)
    df.plot.scatter(x='predictions_rosetta',y='gb1_double_single_40',alpha=0.005)
    plt.savefig(os.path.join('results','gb1','predictions_rosettaVgb1_double_single_40.png'))

    df.plot.hist('predictions_rosetta',bins=1000,alpha=0.3)
    plt.savefig(os.path.join('results', 'gb1', 'predictions_rosetta_hist.png'))

    df.plot.hist('gb1_double_single_40',bins=1000,alpha=0.3)
    plt.savefig(os.path.join('results', 'gb1', 'predictions_rosetta_hist.png'))
def linear_model_onehot_single_data_point(dataset_name,train,test,**kwargs):
    onehot = mtlp.OnehotRidgePredictor(dataset_name=dataset_name,reg_coef=0)
    onehot.train(train.seq.values, train.log_fitness.values)
    predicted = onehot.predict(test.seq.values)
    return mtlp.spearman(predicted, test.log_fitness.values)

def unit_test_linear_model_onehot_single_data_point():
    dataset_name = 'gb1_double_single_40'
    train, test, data = random_get_train_test(dataset_name)
    spear= linear_model_onehot_single_data_point(dataset_name,
                                                 train.sample(n=48),
                                                 test)
    print(f"spearman correlation: {spear}, linear model")


def onehot_split_single_data_point(dataset_name,train,test,**kwargs):
    splits=10
    train=train.copy()
    test=test.copy()
    print(f'using {splits} splits')
    print(f'train length {len(train)}')
    rows=[]
    rows_binary=[]
    for i in tqdm(range(splits)):
        start=i*10
        end=start+10
        print(f"start {start}, end {end}")
        train_i=train[start:end].copy()
        assert len(train_i)==10, 'this is optimized to work with 100 datapoints right now'
        onehot = mtlp.OnehotRidgePredictor(dataset_name=dataset_name)
        onehot.train(train_i.seq.values, train_i.log_fitness.values)
        row=f'predicted_{i}'
        row_binary=f"{row}_bin"
        train[row]=onehot.predict(train.seq.values)
        test[row]=onehot.predict(test.seq.values)


        rows_binary.append(row_binary)
        rows.append(row)

    # try CV next before you give up .
    threshold = -.75



    ridge_top=Ridge(alpha=0)
    ridge_top.fit(train[rows],train.log_fitness.values)
    predicted_test=ridge_top.predict(test[rows])
    ridge_top_B = Ridge(alpha=0)
    ridge_top_B.fit(train[rows]>threshold, train.log_fitness.values)
    predicted_test_b = ridge_top_B.predict(test[rows]>threshold)
    plt.hist(predicted_test_b,label='test_predicted_b',alpha=0.4,bins=100)
    plt.hist(predicted_test,label='test predicted',alpha=0.4,bins=100)
    plt.hist(test.log_fitness.values,label='test actual',alpha=0.4,bins=100)
    plt.title('test predicted vs actual')
    plt.legend()
    plt.show()



    predicted_train=ridge_top.predict(train[rows])
    predicted_train_b=ridge_top_B.predict(train[rows]>threshold)
    plt.hist(predicted_train_b,label='train predicted b',alpha=0.4,bins=25)
    plt.hist(predicted_train, label='train predicted', alpha=0.4,bins=25)
    plt.hist(train.log_fitness.values, label='train actual', alpha=0.4,bins=25)
    plt.title('train predicted vs actual')
    plt.legend()
    plt.show()




    sp=mtlp.spearman(predicted_test, test.log_fitness.values)

    print(f"spearman correlation for {len(train)} datapoints ,{splits} splits : {sp}")

def unit_test_onehot_single_split():
    dataset_name='gb1_double_single_40'
    train, test, data =random_get_train_test(dataset_name)
    onehot_split_single_data_point(dataset_name,train.sample(n=100),test)

def unit_test_pretrained_model_data_point(dataset_name):
    path = os.path.join('models', f"gb1_pretrained_features.csv")
    df = pd.read_csv(path).set_index('seq')
    train, test, data = random_get_train_test(dataset_name)
    spearman = pretrained_model_data_point(dataset_name, train, test, df, 'evmutation_epistatis_MSA_40')
    print(f'spearman for evmutation_epistatis_MSA_40 with {len(train)} training (random) is '
          f'{spearman}')



def joint_predictor_dp(dataset_name, predictor_classes, train, test, **kwargs):
    predictor = mtlp.JointPredictor(dataset_name=dataset_name,
                                    predictor_classes=predictor_classes, **kwargs)
    predictor.train(train.seq.values, train.log_fitness.values)
    predicted = predictor.predict(test.seq.values)
    return mtlp.spearman(predicted, test.log_fitness.values)


def pretrained_model_data_point(dataset_name, train, test, df, feature):
    predictor = mtlp.PretrainedFeature(dataset_name=dataset_name, df=df, feature=feature)
    predictor.train(train.seq.values, train.log_fitness.values)
    predicted = predictor.predict(test.seq.values)
    return mtlp.spearman(predicted, test.log_fitness.values)

# archiving this function its just silly. makes no sense.
# def add_columns_unsupervised():
#     path = os.path.join('models', f"gb1_unsupervised.tsv")
#     metl_data = pd.read_csv(path, sep='\t').set_index('variant')
#     augmented_data = pd.read_csv(os.path.join('models', 'gb1_double_single_40_scores.csv')).set_index('mutant')
#     metl_data['evmutation_epistatis_MSA_40'] = augmented_data['pred']
#     metl_data['seq'] = augmented_data['seq']
#     assert metl_data['seq'].isna().sum() == 0 \
#            and len(metl_data['seq']) == len(augmented_data['seq'])
#
#     metl_data.to_csv(os.path.join('models', 'gb1_pretrained_features.csv'))

    # augmented_data
def parity_plot_ev_mutation():
    path = 'gb1_unsupervised.tsv'
    metl_data = pd.read_csv(path, sep='\t').set_index('variant')
    fig,ax=plt.subplots(1,1)

    metl_data=metl_data[~np.isnan(metl_data['predictions_evmutation_epistatic'])].copy()

    ax=metl_data.plot.scatter(x='predictions_evmutation_epistatic',y='gb1_nn4dms',alpha=.005,ax=ax)
    res_dir=os.path.join('results','gb1')
    fig.savefig(os.path.join(res_dir,"gb1_nn4dmsVpredictions_evmutation_epistatic_filtered.png"))


    print(f"spearman correlation evmutation to rosetta "
          f"{mtlp.spearman(metl_data.gb1_double_single_40.values,metl_data.predictions_rosetta.values)}")


    # for  i in metl_data.keys():
    #     for j in metl_data.keys():
    #         if i!=j:
    #             temp=metl_data[[np.isnan(metl_data[i]) & np.isnan(metl_data[j])]].copy()




def ev_predictor(dataset_name):
    # first thing we need to do is load the data
    assert 'gb1' in dataset_name, 'this only works for gb1 right now, few edits and it will work for all'
    path = 'gb1_unsupervised.tsv'
    metl_data = pd.read_csv(path, sep='\t').set_index('variant')[:5000]

    filepath = os.path.join('..', 'data', dataset_name, 'data.csv')
    data = pd.read_csv(filepath).set_index('mutant')[:5000]

    data=data.loc[['Q1W,V38Q']].copy()

    # assert (data.index==metl_data.index).all(), 'these should be the same'
    ev = mtlp.EVPredictor(dataset_name)

    print(f'running evmutation for {dataset_name}')
    predictions = ev.predict_unsupervised(data.seq.values)
    predictions[predictions==0]=np.nan
    total_predicted_nans=np.isnan(predictions).sum()
    print(f'{dataset_name} couldnt predict on {total_predicted_nans} values')

    # data[dataset_name] = predictions
    # metl_data[dataset_name]=data[dataset_name]
    # metl_data.to_csv(path,sep='\t')
    print('saving csv file ')





    # todo : make unit test here for where the allowed values are.
    # my_spearman = mtlp.spearman(data.dataset_name.values, data.log_fitness.values)

    # print(f'num data points {len(data)}, protein : {dataset_name}  , spearman: {my_spearman}')


def parity_plot_filter_out_chloes_model():
    '''
    parity plots to compare new EVMutation run to sam's old run... some evaluations for meeting with tony...
    :return:
    '''

    dataset = 'gb1_double_single_30_scores'
    assert 'gb1_double_single' in dataset, 'this code only works for gb1 for now'
    outdir = os.path.join('results', dataset)
    # outdir=outdir.replace('-','_')
    # print(f"pwd:{os.getcwd()}")
    # print(f"outdir; {outdir}")
    if not os.path.exists(outdir):
        os.mkdir(f"{outdir}")

    df1 = pd.read_csv(f'models/{dataset}.csv').set_index('mutant')
    df2 = pd.read_csv('models/gb1_unsupervised.tsv', sep='\t').set_index('variant')
    assert (df1.index == df2.index).all(), 'why are indexs different'

    df1_no_zeros = df1[df1['pred'] != 0].copy()
    assert len(df1_no_zeros) == (df1['pred'] != 0).sum()
    # if len(df1_new)<len(df1):
    print(f"========================================\n"
          f"len of new df {len(df1_no_zeros)} compared to before : {len(df1)}\n")

    # true correlation
    filtered_zeros_corr = mtlp.spearman(df1_no_zeros['log_fitness'], df1_no_zeros['pred'])
    print(
        f'spearman correlation for DMS and chloes (with no zeros)\n for dataset {dataset} : {filtered_zeros_corr:0.3f}')

    df2['pred'] = df1_no_zeros['pred']
    y1 = 'predictions_evmutation_independent'
    y2 = 'predictions_evmutation_epistatic'
    assert len(df2['pred']) == len(df2[y1]), ' these columns should be the same length.'
    df2['log_fitness'] = df1['log_fitness']
    assert len(df2['pred']) == len(df2['log_fitness']), ' these columns should be the same length.'

    # filtering out all
    assert (np.isnan(df2['predictions_evmutation_independent']) == np.isnan(
        df2['predictions_evmutation_epistatic'])).all(), \
        'im making this assumption so if its not true we have a problem'

    # filtering out the dataset to only include non-nan values
    df2_filtered = df2[~np.isnan(df2[y1]) & ~np.isnan(df2[y2])
                       & ~np.isnan(df2['pred'])].copy()
    print(f"found {len(df2_filtered)} sequences out of {len(df2)} sequences\n")

    assert (np.isnan(df2_filtered)).sum().sum() == 0, 'why are their nans in here'

    # making spearman correlations here.
    spearman_dict = {}
    for y in [y1, y2, 'pred']:
        s = mtlp.spearman(df2_filtered[y], df2_filtered['log_fitness'])
        print(f'spearman correlation for {y} vs log_fitness : {s}')
        if y.startswith('predictions_evmutation_'):
            y = y[len('predictions_evmutation_'):]

        spearman_dict[y] = [s]
        if y == 'pred':
            spearman_dict[y].append(filtered_zeros_corr)
        else:
            spearman_dict[y].append(np.nan)

    assert len(
        spearman_dict) == 3, 'you are hard coding above for a specific case, if you want to change this change above'

    fig, ax = plt.subplots(1, 1)
    ax = pd.DataFrame(spearman_dict, index=[len(df2_filtered), len(df1_no_zeros)]).T.plot.bar(ax=ax, alpha=0.3, rot=0)
    ax.set_title('spearman correlation\n after removing chloe and sam nan points')
    ax.set_ylabel('spearman')

    fig.savefig(os.path.join(outdir, "spearman_evmutation.png"))

    assert (df2['pred'] != 0).all(), 'why are values equal to zero '

    # df2[(nndf2['pred'])]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    nb_nan_indep, nb_nan_epi, nb_tot = (np.isnan(df2[y1]).sum(), np.isnan(df2[y2]).sum(), len(df2))
    chloeNans = np.isnan(df2["pred"]).sum()

    ax1.set_title(f'sams model found {nb_nan_indep} \n invalid seqs ({nb_nan_indep / nb_tot:0.2f})')
    ax2.set_title(f'sams model found {nb_nan_epi} invalid seqs ({nb_nan_epi / nb_tot:0.2f})')

    fig.suptitle(f'parity plots for {dataset} ,'
                 f'Total datapoints:  {nb_tot} \n '
                 f'total nans in chloes dataset: {chloeNans} ({chloeNans / nb_tot:0.2f})\n'
                 f'total data points for plot {len(df2_filtered)}({len(df2_filtered) / nb_tot:0.2f})')
    ax1 = df2_filtered.plot.scatter(x='pred', y=y1, ax=ax1, s=.03, alpha=0.05)
    ax2 = df2_filtered.plot.scatter(x='pred', y=y2, ax=ax2, s=.03, alpha=0.05)
    plt.gcf().subplots_adjust(top=.8)
    fig.savefig(os.path.join(outdir, f"filter_parity_plot_{len(df2_filtered)}_tot_seqs.png"))

if __name__ == '__main__':
    random_full_learning_curve('gb1_double_single_40')
    unit_test_eUniRep_reg_single_data_point()
    # metl_make_learning_curve()
    # unit_test_onehot_single_split()
    # ev_predictor('gb1_double_single_40')
    # ev_predictor('gb1_double_single_30')
    # ev_predictor('gb1_nn4dms')

    # parity_plot_ev_mutation()
    # unit_test_rosetta_onehot_evmutation()
    # playing_around_with_rosetta_and_evmutation()
    # unit_test_linear_model_onehot_single_data_point()
