import os

import matplotlib.pyplot as plt
import pandas as pd
import metl_predictors as mtlp
import numpy as np


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


def metl_learning_curve(data, point_eval):
    pass


def learning_curve(dataset_name, train, test, point_eval, x, df):
    assert (train.sample(n=20).index==train.sample(n=20).index).sum()  < 20, 'make sure your not using random seed'
    y = []
    for nb_train in x:
        print(f'nb_train {nb_train} , {point_eval.__name__}')
        spearman=point_eval(dataset_name, train.sample(n=nb_train), test, df=df)
        y.append(spearman)
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

    path = os.path.join('models', f"gb1_pretrained_features.csv")
    df = pd.read_csv(path).set_index('seq')

    funcs2include=[one_hot_single_data_point,joint_pretrained_ev_onehot_single_data_point,
                       joint_pretrained_rosetta_onehot_single_data_point]
    # funcs2include = [one_hot_single_data_point]
    train, test, data = random_get_train_test(dataset_name)

    # df.loc[test.set_index('seq')]['evmutation_epistatis_MSA_40']
    for unsup in ['predictions_rosetta', 'evmutation_epistatis_MSA_40']:
        unsup_scores = df.loc[test.set_index('seq').index][unsup]
        assert (unsup_scores.index == test.set_index('seq').index).all(), 'these indexes must be the same'
        spearman_unsup = mtlp.spearman(unsup_scores, test.log_fitness.values)
        plt.axhline(spearman_unsup, label=unsup)

    x = np.arange(10, 210 + 50, 50)
    nb_of_seeds = 5
    for point_eval in funcs2include:
        Y = []
        for _ in range(nb_of_seeds):
            y = learning_curve(dataset_name, train, test, point_eval, x, df=df)
            Y.append(y)

        yerr = np.array(Y).std(axis=0)
        y = np.array(Y).mean(axis=0)
        plt.errorbar(x, y, yerr, label=point_eval.__name__,marker='^',capsize=5,ls='')

    # ev = mtlp.EVPredictor(dataset_name)
    # predictions = ev.predict_unsupervised(data.seq.values)
    # plt.axhline(mtlp.spearman(predictions,data.log_fitness.values),label='ev_unsupervised')
    plt.legend()
    plt.title(f'spearman correlation vs nb of training points\n for {dataset_name}')
    plt.xlabel('nb of training points')
    plt.ylabel('spearman correlation')
    plt.savefig(os.path.join(outdir, f'random_learning_curve_nb_seeds_{nb_of_seeds}_reg_ceof_{mtlp.REG_COEF_LIST}.png'))


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
                              train=train, test=test, df=kwargs['df'], feature='evmutation_epistatis_MSA_40')


def joint_pretrained_rosetta_onehot_single_data_point(dataset_name, train, test, **kwargs):
    assert 'df' in kwargs.keys(), 'need to pass in df to use this function'
    return joint_predictor_dp(dataset_name, [mtlp.PretrainedFeature, mtlp.OnehotRidgePredictor],
                              train=train, test=test, df=kwargs['df'], feature='predictions_rosetta')


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


def add_columns_unsupervised():
    path = os.path.join('models', f"gb1_unsupervised.tsv")
    metl_data = pd.read_csv(path, sep='\t').set_index('variant')
    augmented_data = pd.read_csv(os.path.join('models', 'gb1_double_single_40_scores.csv')).set_index('mutant')
    metl_data['evmutation_epistatis_MSA_40'] = augmented_data['pred']
    metl_data['seq'] = augmented_data['seq']
    assert metl_data['seq'].isna().sum() == 0 \
           and len(metl_data['seq']) == len(augmented_data['seq'])

    metl_data.to_csv(os.path.join('models', 'gb1_pretrained_features.csv'))

    # augmented_data


def ev_predictor(dataset_name):
    # first thing we need to do is load the data
    filepath = os.path.join('..', 'data', dataset_name, 'data.csv')
    data = pd.read_csv(filepath)
    ev = mtlp.EVPredictor(dataset_name)
    # test = data.sample(frac=0.2, random_state=0)
    # test = test.copy()
    # train = data.drop(test.index)
    predictions = ev.predict_unsupervised(data.seq.values)
    data['pred'] = predictions

    # todo : make unit test here for where the allowed values are.
    my_spearman = mtlp.spearman(data.pred.values, data.log_fitness.values)

    print(f'num data points {len(data)}, protein : {dataset_name}  , spearman: {my_spearman}')
    return my_spearman, predictions


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

