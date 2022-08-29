'''
Author: Bryce Johnson
Gitter Lab ~ August 5th , 2022
code augmented from https://github.com/chloechsu/combining-evolutionary-and-assay-labelled-data
to implement augmented models from precomputed density features and manually define test and train splits
'''
import nn4dms_onehot as nn4dms
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
from sklearn.metrics import make_scorer
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_decoupled_mutants(mutants ,offset=0 ,split_character=':'):
    ans =[]
    for mutant in mutants.split(split_character):
        idx =int(mutant[1:-1]) - offset
        assert idx >= 0 ,'why is offset greater than the mutation position?'
        ans.append((str(mutant[0]), idx, str(mutant[-1])))
    return ans

def mutant2sequence(mutant,WT,offset, split_character=':'):
    # todo: check this works for multiple sequences
    decouple_mutants=get_decoupled_mutants(mutant,offset,split_character)
    new_seq=WT
    for dc_mutant in decouple_mutants:
        original,idx,new=dc_mutant[0],dc_mutant[1],dc_mutant[2]
        assert new_seq[idx]==original, f'original mutant ({original}) at index {idx} not the same as that in wildtype'
        temp = list(new_seq)
        temp[idx] = new
        new_seq = "".join(temp)
    return new_seq

def seqs_to_onehot(seqs):
    seqs = format_batch_seqs(seqs)
    # key to get higher dimension:  seqs.shape[1]*24
    X = np.zeros((seqs.shape[0], seqs.shape[1]*24), dtype=int)
    for i in range(seqs.shape[1]):
        for j in range(24):

            X[:, i*24+j] = (seqs[:, i] == j)
    return X
def format_batch_seqs(seqs):
    maxlen = -1
    for s in seqs:
        if len(s) > maxlen:
            maxlen = len(s)
    formatted = []
    for seq in seqs:
        pad_len = maxlen - len(seq)
        # its one sequence longer b/c she's adding in a start value
        padded = np.pad(format_seq(seq), (0, pad_len), 'constant', constant_values=0)
        formatted.append(padded)
    return np.stack(formatted)

def format_seq(seq,stop=False):
    """
    Takes an amino acid sequence, returns a list of integers in the codex of the babbler.
    Here, the default is to strip the stop symbol (stop=False) which would have
    otherwise been added to the end of the sequence. If you are trying to generate
    a rep, do not include the stop. It is probably best to ignore the stop if you are
    co-tuning the babbler and a top model as well.
    """
    if stop:
        int_seq = aa_seq_to_int(seq.strip())
    else:
        int_seq = aa_seq_to_int(seq.strip())[:-1]
    return int_seq

def aa_seq_to_int(s):
    """
    Return the int sequence as a list for a given string of amino acids
    """
    return [0] + [aa_to_int[a] for a in s] + [25]

# Lookup tables
# edit bcj 6/14/22: this was incorrect , it missed AAs which had
#    index >23 , so I made the gap character have value 0 instead of 24.
def spearman(y_pred, y_true):
    if np.var(y_pred) < 1e-6 or np.var(y_true) < 1e-6:
        return 0.0
    return spearmanr(y_pred, y_true).correlation

aa_to_int = {
    'M':1,
    'R':2,
    'H':3,
    'K':4,
    'D':5,
    'E':6,
    'S':7,
    'T':8,
    'N':9,
    'Q':10,
    'C':11,
    'U':12,
    'G':13,
    'P':14,
    'A':15,
    'V':16,
    'I':17,
    'F':18,
    'Y':19,
    'W':20,
    'L':21,
    'O':22, #Pyrrolysine
    'X':23, # Unknown
    'Z':23, # Glutamic acid or GLutamine
    'B':23, # Asparagine or aspartic acid
    'J':23, # Leucine or isoleucine
    'start':0,
    'stop':25,
    '-':26,
}

REG_COEF_LIST = [1e-1, 1e0, 1e1, 1e2, 1e3]

class AugmentedModel():
    def __init__(self ,WT,reg_coef=1,offset=0,split_character=':'):
        self.reg_coef =reg_coef
        self.WT =WT
        self.offset=offset
        self.split_character=split_character
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
        self.model = Ridge(alpha=self.reg_coef)
        self.model.fit(features, DMS_score.to_numpy())


    def predict(self ,mutants ,density_feature=None):
        features=self.features(mutants,density_feature)
        return self.model.predict(features)

    def features(self,mutants,density_feature):

        sequences = mutants.apply(
            lambda x: mutant2sequence(x, WT=self.WT, offset=self.offset, split_character=self.split_character))
        features = seqs_to_onehot(sequences) * np.sqrt(1.0 / self.reg_coef_onehot)
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


def learning_curve(ag,train,test,nb_train,nb_simulations,density_features):
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
              s.append(spearman(predicted,test['DMS_score']))
          S.append(np.array(s).mean())

      df_spearman[density_col_name]=S
  return df_spearman

def enrich2_evaluation():
    dataset='SPG1_STRSG_Olson_2014_enrich2'
    nn4dms_filename = os.path.join('..', 'data', 'gb1_double_single_40', 'data.csv')
    df = pd.read_csv(nn4dms_filename)
    df['DMS_score']=df['log_fitness']
    WT = 'MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE'
    ag = AugmentedModel(WT, offset=0, split_character=",", reg_coef=1)

    #### ------------ problem set up code -----------------------------;
    frac = 0.1
    test = df.sample(frac=frac, random_state=1)
    test = test.copy()
    train = df.drop(test.index)
    nb_train = np.arange(1, 6, 1) * 48
    nb_simulations = 1

    df_spearman=learning_curve(ag,train,test,nb_train,nb_simulations,['onehot'])

    ax = df_spearman.plot.line(marker='x')
    ax.set_xlabel('nb training points')
    ax.set_xticks(nb_train)
    ax.set_title(f'{dataset}\n'
                 f'{ag}')
    # ax.set_yticks([0,.25,.5,.75,1])
    plt.legend()
    plt.savefig(os.path.join('results', 'learning_curves', f'{dataset}_frac_{frac}_enrich2_scores_reg_coef_{ag.reg_coef}.png'))


# def protein_gym():
#     dataset = 'SPG1_STRSG_Olson_2014'
#     beta_filename = os.path.join('databases', 'protein_gym_benchmark', f"{dataset}.csv")
#     df = pd.read_csv(beta_filename)
#     WT = 'MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE'
#     ag = AugmentedModel(WT, offset=227, split_character=":", reg_coef=1)
#     # print(f"can keep {sum(~np.isnan(df['EVmutation']))} sequences out of {len(df)}")
#     # if following Tranception evaluation must remove all points EVmutation can't predict on for all datasets
#     df = df[~np.isnan(df['EVE_single'])]
#     frac = 0.1
#     test = df.sample(frac=frac, random_state=1)  # so im hard coding my test state,
#     # but not my split on which points to sample from.
#     test = test.copy()
#     train = df.drop(test.index)
#
#     ##### ==========problem set-up complete ========================
#
#     # make a learning curve for one density feature
#     nb_train = np.arange(1, 6, 1) * 48
#     nb_simulations = 1
#
#     ax = df_spearman.plot.line(marker='x')
#     ax.set_xlabel('nb training points')
#     ax.set_xticks(nb_train)
#     ax.set_title(f'{dataset}\n'
#                  f'{ag}')
#     # ax.set_yticks([0,.25,.5,.75,1])
#     plt.legend()
#     plt.savefig(os.path.join('results', 'learning_curves', f'{dataset}_frac_{frac}.png'))
#
#


if __name__=='__main__':
    enrich2_evaluation()