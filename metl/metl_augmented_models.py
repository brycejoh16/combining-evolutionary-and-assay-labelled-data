'''
code augmented from https://github.com/chloechsu/combining-evolutionary-and-assay-labelled-data
to implement
'''

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
from sklearn.metrics import make_scorer

def get_decoupled_mutants(mutants ,offset=0 ,split_character=':'):
    ans =[]
    for mutant in mutants.split(split_character):
        idx =int(mutant[1:-1]) - offset
        assert idx >= 0 ,'why is offset greater than the mutation position?'
        ans.append((str(mutant[0]), idx, str(mutant[-1])))
    return ans

def mutant2sequence(mutant,WT,offset, split_character=':'):
    decouple_mutants=get_decoupled_mutants(mutant,offset,split_character)
    new_seq=WT
    for dc_mutant in decouple_mutants:
        original,idx,new=dc_mutant[0],dc_mutant[1],dc_mutant[2]
        assert new_seq[idx]==original, f'original mutant ({original}) at index {idx} not the same as that in wildtype'
        new_seq[idx]=new
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
    def __init__(self ,WT ,reg_coef=1,offset=0,split_character=':'):
        self.reg_coef =reg_coef
        self.WT =WT
        self.offset=offset
        self.split_character=split_character
    def train(self, mutants, DMS_scores, density_feature=None, reg_coef_density_feature=None, reg_coef_onehot=1):
        # set feature specific regularization coefficients
        self.reg_coef_onehot = reg_coef_onehot
        self.reg_coef_density_feature = reg_coef_density_feature

        # generate features using the regularization coefficients as defined above
        features=self.features(mutants,density_feature)

        # cross validation
        if self.reg_coef is None or self.reg_coef == 'CV':
            self.find_optimal_reg_coef(features,DMS_scores)

        # set the model
        self.model = Ridge(alpha=self.reg_coef)
        self.model.fit(features, DMS_scores.to_numpy())


    def predict(self ,mutants ,density_feature=None):
        features=self.features(mutants,density_feature)
        return self.model.predict(features)

    def features(self,mutants,density_feature):
        sequences = mutants.apply(
            lambda x: mutant2sequence(x, WT=self.WT, offset=self.offset, split_character=self.split_character))
        features = seqs_to_onehot(sequences) * np.sqrt(1.0 / self.reg_coef_onehot)
        if density_feature is not None:
            assert self.reg_coef_density_feature is not None, 'Why is reg_coef of the density feature None'
            density_feature_numpy = density_feature.to_numpy() * np.sqrt(1.0 / self.reg_coef_density_feature)
            # todo: check this again
            features = np.concatenate([features, density_feature_numpy], axis=1)
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



        # if density_features is not None:
        #     density_columns=density_features.columns()
        #     if reg_coef_density_features is None:
        #         reg_coef_density_features = [1]*len(density_columns)
        #     assert len(density_columns) ==len(reg_coef_density_features), 'number of density columns must equal the number of regression coefficients'
        #     for col in density_columns:
        #         density_features[]



        # multiply by the factor for the augmented factors (maybe make a dictionary for this)

        #