import numpy as np
import os
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge,LinearRegression,ElasticNet
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import make_scorer
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.autograd import Variable
import math



from os.path import isdir, join, basename

chars = ["*", "A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
aa_map = {c: i for i, c in enumerate(chars)}
num_categories = len(chars)
REG_COEF_LIST = [1e-1, 1e0, 1e1, 1e2, 1e3]

def spearman(y_pred, y_true):
    if np.var(y_pred) < 1e-6 or np.var(y_true) < 1e-6:
        return 0.0
    return spearmanr(y_pred, y_true).correlation

def enc_one_hot(int_seqs, num_categories):
    one_hot = np.eye(num_categories)[int_seqs]
    return one_hot.astype(np.float32)

def enc_int_seqs_from_variants(variants, wild_type_seq, aa_map):

    # convert wild type seq to integer encoding
    wild_type_int = np.zeros(len(wild_type_seq), dtype=np.uint8)
    for i, c in enumerate(wild_type_seq):
        wild_type_int[i] = aa_map[c]

    # tile the wild-type seq
    seq_ints = np.tile(wild_type_int, (len(variants), 1))

    for i, variant in enumerate(variants):
        # variants are a list of mutations [mutation1, mutation2, ....]
        variant = variant.split(",")
        for mutation in variant:
            # mutations are in the form <original char><position><replacement char>
            position = int(mutation[1:-1])
            replacement = aa_map[mutation[-1]]
            seq_ints[i, position] = replacement

    return seq_ints.astype(int)
class pytorch_linear_regression(torch.nn.Module):
    def __init__(self,inputsize,outputsize):
        super(pytorch_linear_regression,self).__init__()
        self.linear=torch.nn.Linear(inputsize,outputsize)
    def forward(self,x):
        out=self.linear(x)
        return out

class helper_pytorch_linear:
    def __init__(self,learning_rate,epochs,batch_size=None):
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.batch_size=batch_size
    def fit(self,X,y):
        if self.batch_size is None:
            self.batch_size=len(y)
        self.model = pytorch_linear_regression(X.shape[1], 1)
        criterion = torch.nn.MSELoss()
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=self.learning_rate)
        for epoch in range(self.epochs):
            n_batches=math.floor(len(y)/self.batch_size)
            for i in range(n_batches):
                batch_X=X[i*self.batch_size:(i+1)*self.batch_size]
                batch_y=y[i*self.batch_size:(i+1)*self.batch_size]
                assert len(batch_y)==self.batch_size,'batch_szie different from found length'
                inputs =Variable(torch.from_numpy(batch_X)).float()
                labels = Variable(torch.from_numpy(batch_y.reshape(-1,1))).float()
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                # print(loss)
                # get gradients w.r.t to parameters
                loss.backward()
                self.optimizer.step()

            if len(y)> (n_batches)*self.batch_size:
                batch_X=X[(n_batches)*self.batch_size:]
                batch_y=y[(n_batches)*self.batch_size:]
                assert len(batch_y)< self.batch_size ,'batch of y should be less than batch size'
                inputs = Variable(torch.from_numpy(batch_X)).float()
                labels = Variable(torch.from_numpy(batch_y.reshape(-1, 1))).float()
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                # print(loss)
                # get gradients w.r.t to parameters
                loss.backward()
                self.optimizer.step()

    def predict(self,X):
        return self.model(torch.from_numpy(X)).data.numpy()
    def __repr__(self):
        return f"pytorchLinear(lr={self.learning_rate:0.2f},epochs={self.epochs}"

class AugmentedModel():
    def __init__(self ,WT,reg_coef=1,offset=0,split_character=':',model=Ridge(alpha=1)):
        self.reg_coef =reg_coef
        self.WT =WT
        self.offset=offset
        self.split_character=split_character
        self.model=model
    def train(self, mutants, DMS_score, density_feature=None, reg_coef_density_feature=1e-8, reg_coef_onehot=1):
        '''
        training function for augmented model
        :param mutants: pd.Series of all mutants e.g. pd.Series(['W24P',...])
        :param DMS_score: pd.Series of associated DMS_scores
        :param density_feature: optional -- pd.Series of associated density features
        :param reg_coef_density_feature: regularization coefficient for the density specific feature
        :param reg_coef_onehot: regularization coefficient for the onehot values
        :return:
        '''
        if density_feature is not None:
            assert sum(np.isnan(density_feature))==0, 'accidentally passed in some density features which where nan'
            assert type(density_feature) == pd.Series, 'density feature must be a pandas series'
            assert len(mutants)==len(density_feature),'density feature must be same length as mutants'
        assert sum(np.isnan(DMS_score))==0,'accidentally passed in some DMS_scores which where nan'
        assert len(mutants) == len(DMS_score), 'mutants must be the same lenght as DMS_score'
        assert type(DMS_score) == pd.Series,'dms score must be a pandas series'
        assert type(mutants)==pd.Series,'mutants must be a pandas series'

        # set feature specific regularization coefficients
        self.reg_coef_onehot = reg_coef_onehot
        self.reg_coef_density_feature = reg_coef_density_feature

        # generate features using the regularization coefficients as defined above
        features=self.features(mutants,density_feature)

        # cross validation
        if self.reg_coef == 'CV':
            self.find_optimal_reg_coef(features,DMS_score)

        # set the model
        self.model.fit(features, DMS_score.to_numpy())


    def predict(self ,mutants ,density_feature=None):
        features=self.features(mutants,density_feature)
        return self.model.predict(features).reshape(-1)

    def features(self,mutants,density_feature):
        int_seqs = enc_int_seqs_from_variants(mutants, self.WT, aa_map)
        one_hot_seqs = enc_one_hot(int_seqs, num_categories)
        one_hot_seqs_flat = one_hot_seqs.reshape(one_hot_seqs.shape[0], -1)
        features = one_hot_seqs_flat * np.sqrt(1.0 / self.reg_coef_onehot)
        # todo: makes sure onehot encoding works for multiple sequences
        if density_feature is not None:
            assert self.reg_coef_density_feature is not None, 'Why is reg_coef of the density feature None'
            density_feature_numpy = density_feature.to_numpy().reshape(-1,1)* np.sqrt(1.0 / self.reg_coef_density_feature)
            features = np.concatenate([features,density_feature_numpy],axis=1)
        return features
    def find_optimal_reg_coef(self,features,DMS_scores):
        best_rc, best_score = None, -np.inf
        print('doing CV on ridge')
        for rc in REG_COEF_LIST:
            model = Ridge(alpha=rc)
            score = cross_val_score(
                model, features, DMS_scores.to_numpy(), cv=5,
                scoring=make_scorer(spearman)).mean()
            if score > best_score:
                best_rc = rc
                best_score = score
        self.reg_coef = best_rc
    def __repr__(self):
        return f"ridge reg_coef {self.reg_coef},onehot reg_ceof {self.reg_coef_onehot},density reg_coef {self.reg_coef_density_feature}"

def learning_curve(ag,train,test,nb_train,nb_simulations,density_features,evaluation=spearman):

  df_spearman=pd.DataFrame()
  df_spearman['nb_train']=nb_train
  df_spearman=df_spearman.set_index('nb_train')

  for density_col_name in tqdm(density_features):
      S=[]
      for i in nb_train:
          s=[]
          for _ in range(nb_simulations):
              train_i=train.sample(n=i)
              if density_col_name is 'onehot':
                  df_train = None
                  df_test = None
              else:
                  df_train = train_i[density_col_name]
                  df_test = test[density_col_name]

              # if you want to mess with the reg_coef of density features individually here is
              # where you would do it.
              ag.train(mutants=train_i['mutant'],DMS_score=train_i['DMS_score'],density_feature=df_train)
              predicted=ag.predict(mutants=test['mutant'],density_feature=df_test)
              s.append(evaluation(predicted,test['DMS_score']))
          S.append(np.array(s).mean())

      df_spearman[density_col_name]=S
  return df_spearman


def nn4dms_evaluation():
    dataset='SAM_ONEHOT_gb1_enrich2'
    wt_seq = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
    assert wt_seq=='MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE', 'whyyyyyy'

    nn4dms_filename = os.path.join('..', 'data', 'gb1_double_single_40', 'data.csv')
    df = pd.read_csv(nn4dms_filename)
    df['DMS_score']=df['log_fitness']

    ag = AugmentedModel(wt_seq, offset=0, split_character=",", reg_coef=1)

    frac = 0.2
    test = df.sample(frac=frac, random_state=1)
    test = test.copy()
    train = df.drop(test.index)
    print(f"len of train {len(train)}")
    print(f"len of test {len(test)}")
    # nb_train = np.arange(1, 6, 1) * 48
    nb_train=np.arange(1,6,1)*100
    nb_simulations = 1

    assert len(pd.concat([test, train]).drop_duplicates(keep=False)) == len(df)



    df_spearman = learning_curve(ag, train, test, nb_train, nb_simulations, ['onehot'])

    ax = df_spearman.plot.line(marker='x')
    ax.set_xlabel('nb training points')
    ax.set_xticks(nb_train)
    ax.set_title(f'{dataset}\n'
                 f'{ag}')
    # ax.set_yticks([0,.25,.5,.75,1])
    plt.legend()
    plt.savefig(
        os.path.join('results', 'learning_curves', f'{dataset}_frac_{frac}_reg_coef_{ag.reg_coef}_pearson.png'))
def show_multiple_models_plot():
    df_spearman=pd.read_csv(os.path.join('results', 'learning_curves', f'nn4dms_reduced3_splits_multiple_models_prodcution_run.csv'))
    df_spearman=df_spearman.set_index('nb_train')
    ax = df_spearman.plot.line(marker='x')
    ax.set_xlabel('nb training points')
    ax.set_xscale('log')
    plt.legend()
    plt.savefig(
        os.path.join('results', 'learning_curves', f'nn4dms_reduced3_multiple_models_Aug17.png'))

def multiple_models_reduced3_splits_analysis():
    dataset = 'nn4dms_reduced3_splits_gb1'
    df, all_splits = load_splits()
    df_ros = pd.read_csv('gb1_unsupervised.tsv', sep='\t').set_index('variant')
    df = df.set_index('variant')
    df['rosetta'] = df_ros['predictions_rosetta']
    # df['log_rosetta'] = np.log(df['rosetta'] + abs(df['rosetta'].min()) + .1)
    # Z= np.sum(np.exp(-df['rosetta']))
    # df['exp_rosetta'] = np.exp(-df['rosetta'])/Z
    df = df.reset_index()
    wt_seq = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
    density_feature = 'log_rosetta'
    # model=helper_pytorch_linear(learning_rate=0.0001,epochs=500)
    models={'rosetta':(Ridge(alpha=1),'rosetta'),
            'exp_rosetta':(Ridge(alpha=1),'exp_rosetta')}
    #         'onehot_ridge':(Ridge(alpha=1),None),
    #         'pytorch_onehot_lr1e-4':(helper_pytorch_linear(learning_rate=0.0001,epochs=500),None),
    #         'onehot_Ridge(alpha=0)':(Ridge(alpha=0),None),
    #         'onehot_Linear_Regression':(LinearRegression(),None),
    models={  'pytorch_onehot_1r1e-4_batch_size_128':(helper_pytorch_linear(learning_rate=1e-4,epochs=500,batch_size=128),None)}
    for train_size, splits in all_splits.items():
        print(f"Train size: {train_size:>5} Replicates: {len(splits)}")

    df_spearman=pd.DataFrame()
    for dataset in models:
        print(f"\n =============== starting a new model ===================")
        print(f"{models[dataset]}")
        S,N= nn4dms_evaluation_reduced3_splits(dataset,all_splits,df,wt_seq,models[dataset][0],models[dataset][1])
        df_spearman[dataset]=S

    df_spearman['nb_train']=N
    df_spearman=df_spearman.set_index('nb_train')
    df_spearman.to_csv(os.path.join('results', 'learning_curves', f'nn4dms_reduced3_splits_multiple_models_prodcution_run.csv'),mode='a')
    # ax = df_spearman.plot.line(marker='x')
    # ax.set_xlabel('nb training points')
    # ax.set_xscale('log')
    # plt.legend()
    # plt.savefig(
    #     os.path.join('results', 'learning_curves', f'nn4dms_reduced3_rosetta_exp.png'))
def nn4dms_evaluation_reduced3_splits(dataset,all_splits,df,wt_seq,model,density_feature):

    S,P,N=[],[],[]
    for train_size, splits in all_splits.items():
        print(f'now on train size {train_size} with {len(splits)} models')
        s,p=[],[]
        for split in splits:
            train=df.iloc[split['train']]
            # print(f'length of train {len(train)}')
            test=df.iloc[split['test']]
            if density_feature is None:
                dens_train = None
                dens_test = None
            else:
                dens_test=test[density_feature]
                dens_train=train[density_feature]
            # print(f'length of test {len(test)}')
            ag = AugmentedModel(wt_seq, offset=0, split_character=",", reg_coef=1,model=model)
            ag.train(train['variant'],train['score'],dens_train)
            predictions=ag.predict(test['variant'],dens_test)
            s.append(abs(spearman(predictions,test['score'])))
            p.append(abs(pearsonr(predictions,test['score'])[0]))
        print(np.array(s).mean())
        S.append(np.array(s).mean())
        P.append(np.array(p).mean())
        N.append(train_size)
    return S,N
    # df_spearman['onehot_spearman']=S
    # df_spearman['onehot_pearson']=P
    # df_spearman['nb_train']=N
    # df_spearman=df_spearman.set_index('nb_train')
    #
    # ax = df_spearman.plot.line(marker='x')
    # ax.set_xlabel('nb training points')
    # ax.set_title(f'{dataset}\n')
    # ax.set_xscale('log')
    # plt.legend()
    # plt.savefig(
    #     os.path.join('results', 'learning_curves', f'{dataset}_reduced3_splits_{model}_density_feature_{density_feature}.png'))


def unit_test_linear_regression():
    # model=ElasticNet(alpha=1,l1_ratio=0.5)
    model=Ridge(alpha=0)
    idx=1
    nb_train=2000
    save_dir=os.path.join('results','learning_curves','linear_regression_paradox')

    df,all_splits=load_splits()
    train=df.iloc[all_splits[nb_train][idx]['train']]
    test=df.iloc[all_splits[nb_train][idx]['test']]
    print(f'{test["score"].max()}, min : {test["score"].min()}')
    print(f'{train["score"].max()}, min : {train["score"].min()}')

    WT="MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
    int_seqs = enc_int_seqs_from_variants(train['variant'],WT, aa_map)
    one_hot_seqs = enc_one_hot(int_seqs, num_categories)
    one_hot_seqs_flat = one_hot_seqs.reshape(one_hot_seqs.shape[0], -1)

    model.fit(one_hot_seqs_flat,train['score'])
    train_predictions=model.predict(one_hot_seqs_flat)
    plt.hist(train_predictions,color='red',label='predictions train',alpha=0.3)
    plt.hist(train['score'],color='green',label='training set',alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(save_dir,f'{model}_idx{idx}_nb_train_{nb_train}_hist_train_predictions.png'))
    plt.show()


    df_weights=pd.DataFrame(data=model.coef_.reshape(56, 21).T,index=chars,columns=np.arange(0,56,1))

    sns.heatmap(df_weights,linecolor='b',cmap='RdBu')
    plt.title(f'weights for {model} ')
    plt.savefig(os.path.join(save_dir,f'{model}_idx{idx}_nb_train_{nb_train}__heatmap_weights.png'))
    plt.show()

    plt.hist(model.coef_,color='r',alpha=0.3,bins=20)
    plt.title('model weights histogram')
    plt.savefig(os.path.join(save_dir,f'{model}_idx{idx}_nb_train_{nb_train}__model_weights_histogram.png'))
    plt.show()
    WT = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
    int_seqs = enc_int_seqs_from_variants(test['variant'], WT, aa_map)
    one_hot_seqs = enc_one_hot(int_seqs, num_categories)
    one_hot_seqs_flat_test = one_hot_seqs.reshape(one_hot_seqs.shape[0], -1)


    prediction=model.predict(one_hot_seqs_flat_test)
    plt.hist(prediction,alpha=0.3,color='r',label='prediciton',bins=20)
    plt.hist(test['score'],alpha=0.3,color='g',label='actual labels',bins=20)
    plt.legend()
    plt.savefig(os.path.join(save_dir,f'{model}_idx{idx}_nb_train_{nb_train}__test_hist_predictions.png'))
    plt.show()

    plt.scatter(prediction,test['score'],alpha=0.2)
    plt.xlabel('prediction')
    plt.ylabel('test score')
    plt.savefig(os.path.join(save_dir,f'{model}_idx{idx}_nb_train_{nb_train}__test_vs_predict.png'))
    plt.show()

    print(f"spearman , {spearman(prediction,test['score'])}")

def load_single_split_dir(split_dir):
    fns = [join(split_dir, f) for f in os.listdir(split_dir) if not f.startswith(".")]
    split = {basename(fn)[:-4]: pd.read_csv(fn, header=None)[0].to_numpy() for fn in fns}
    return split


def load_split_dir(split_dir):
    # get all the files in the given split_dir
    fns = [join(split_dir, f) for f in os.listdir(split_dir) if not f.startswith(".")]
    if isdir(fns[0]):
        # reduced train size split dir with multiple split replicates
        # sort the directories by the rep number to ensuyre list is in correct order
        # alternatively, use a dictionary with replicate number as key
        fns = sorted(fns, key=lambda fn: fn.split("_")[-1])
        splits = [load_single_split_dir(fn) for fn in fns]
        return splits
    else:
        split = load_single_split_dir(split_dir)
        return split


def load_reduced_splits(reduced_split_dir):
    splits = {}
    for sd in [f for f in os.listdir(reduced_split_dir) if not f.startswith(".")]:
        train_size = int(sd.split("_")[1][2:])
        splits[train_size] = load_split_dir(join(reduced_split_dir, sd))
    # sort by train size to make things cleaner
    splits = {k: splits[k] for k in sorted(splits)}
    return splits

def unit_load_splits():
    parent_directory=os.path.join('..','..','RosettaTL','notebooks','load_gb1')
    gb1_fn = "gb1.tsv"
    gb1 = pd.read_csv(os.path.join( parent_directory,gb1_fn), sep="\t")
    # gb1.head()
    reduced_s3_dir = os.path.join(parent_directory,"splits","reduced_s3")
    all_splits = load_reduced_splits(reduced_s3_dir)
    for train_size, splits in all_splits.items():
        print(f"Train size: {train_size:>5} Replicates: {len(splits)}")

    train_size = 25
    print(f"Train size: {train_size}")
    for i, split_replicate in enumerate(all_splits[train_size]):
        print(f"Replicate: {i}")
        print(f"\tTrain size: {len(split_replicate['train'])} Test size: {len(split_replicate['test'])}")
        print(f"\tFirst train index (should be different): {split_replicate['train'][0]}")
        print(f"\tFirst test index (should be the same): {split_replicate['test'][0]}")
def load_splits():
    parent_directory = os.path.join('..', '..', 'RosettaTL', 'notebooks', 'load_gb1')
    gb1_fn = "gb1.tsv"
    gb1 = pd.read_csv(os.path.join(parent_directory, gb1_fn), sep="\t")
    # gb1.head()
    reduced_s3_dir = os.path.join(parent_directory, "splits", "reduced_s3")
    all_splits = load_reduced_splits(reduced_s3_dir)
    return gb1,all_splits


if __name__ == '__main__':
    # nn4dms_evaluation()
    # unit_load_splits()
    # nn4dms_evaluation_reduced3_splits()
    # unit_test_linear_regression()
    # multiple_models_reduced3_splits_analysis()
    show_multiple_models_plot()