import numpy as np
import os
import glob
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from scipy.stats import spearmanr
from metl_couplings_model import CouplingsModel
from Bio import SeqIO

REG_COEF_LIST = [1e-1, 1e0, 1e1, 1e2, 1e3]
# REG_COEF_LIST = [ 1e0]
def spearman(y_pred, y_true):
    if np.var(y_pred) < 1e-6 or np.var(y_true) < 1e-6:
        return 0.0
    return spearmanr(y_pred, y_true).correlation

def read_fasta(filename, return_ids=False):
    records = SeqIO.parse(filename, 'fasta')
    seqs = list()
    ids = list()
    for record in records:
        seqs.append(str(record.seq))
        ids.append(str(record.id))
    if return_ids:
        return seqs, ids
    else:
        return seqs
def seq2effect(seqs, model, offset=1, ignore_gaps=False):
    effects = np.zeros(len(seqs))
    for i in range(len(seqs)):
        true_mutations=seq2mutation(seqs[i], model,
                     ignore_gaps=False, offset=offset)
        mutations = seq2mutation(seqs[i], model,
                ignore_gaps=ignore_gaps, offset=offset)
        dE, _, _ = model.delta_hamiltonian(mutations)
        effects[i] = dE
    return effects

def seq2mutation(seq, model, return_str=False, ignore_gaps=False,
        sep=":", offset=1):
    '''
    function is responsible for returning mutations as a list of tuples,
    it finds the index map of the alignment and that of the sequence. compares the sequences,
    and if they are different adds to a list.
    '''
    mutations = []
    for pf, pm in model.index_map.items():
        if seq[pf-offset] != model.target_seq[pm]:
            if ignore_gaps and (
                    seq[pf-offset] == '-' or seq[pf-offset] not in model.alphabet):
                continue
            mutations.append((pf, model.target_seq[pm], seq[pf-offset]))
    if return_str:
        return sep.join([m[1] + str(m[0]) + m[2] for m in mutations])
    return mutations


class BasePredictor():
    """Abstract class for predictors."""

    def __init__(self, dataset_name, **kwargs):
        self.dataset_name = dataset_name

    def select_training_data(self, data, n_train):
        return data.sample(n=n_train)

    def train(self, train_seqs, train_labels):
        """Trains the model.
        Args:
            - train_seqs: a list of sequences
            - train_labels: a list of numerical fitness labels
        """
        raise NotImplementedError

    def predict(self, predict_seqs):
        """Gets model predictions.
        Args:
            - predict_seqs: a list of sequences
        Returns:
            A list of numerical fitness predictions.
        """
        raise NotImplementedError

    def predict_unsupervised(self, predict_seqs):
        """Gets model predictions before training.
        Args:
            - predict_seqs: a list of sequences
        Returns:
            A list of numerical fitness predictions.
        """
        return np.random.randn(len(predict_seqs))

class BaseRegressionPredictor(BasePredictor):
    def __init__(self, dataset_name, reg_coef=None, linear_model_cls=Ridge, **kwargs):
        # self.dataset_name = dataset_name
        # print(linear_model_cls.__name__)
        self.reg_coef = reg_coef
        self.linear_model_cls = linear_model_cls
        self.model = None
        super().__init__(dataset_name, **kwargs)

    def seq2feat(self, seqs):
        raise NotImplementedError

    def train(self, train_seqs, train_labels):
        # so here it is, the seq2feat , first thing,
        X = self.seq2feat(train_seqs)
        if self.reg_coef is None or self.reg_coef == 'CV':
            best_rc, best_score = None, -np.inf
            print('doing CV on ridge')
            for rc in REG_COEF_LIST:
                model = self.linear_model_cls(alpha=rc)
                score = cross_val_score(
                    model, X, train_labels, cv=5,
                    scoring=make_scorer(spearman)).mean()
                if score > best_score:
                    best_rc = rc
                    best_score = score
            self.reg_coef = best_rc
            # print(f'Cross validated reg coef {best_rc}')

        if self.reg_coef==0 :
            print('using linear regrssion')
            self.model=LinearRegression()
        else:
            self.model = self.linear_model_cls(alpha=self.reg_coef)
        self.model.fit(X, train_labels)

    def predict(self, predict_seqs):
        if self.model is None:
            return np.random.randn(len(predict_seqs))
        # here it is on the test seq2feat again.
        X = self.seq2feat(predict_seqs)
        return self.model.predict(X)




class PretrainedFeature(BaseRegressionPredictor):
    def __init__(self, dataset_name,df,feature,idx, reg_coef=1, **kwargs):
        super().__init__(dataset_name, reg_coef, **kwargs)
        self.feature_of_interest=df[feature[idx]]
        self.reg_coef=reg_coef
    def seq2feat(self, seqs):
        # filter based off what sequences are their and return.
        # return format  [n x 1]
        feat=self.feature_of_interest[seqs]
        assert (feat.index.to_numpy()==seqs).all(),'it looks like the odering changed?'
        feat_numpy=feat.to_numpy()[:,None]
        assert (~np.isnan(feat_numpy)).all(),' their shouldnt be any nans'
        assert feat_numpy.shape[1]==1 ,'the shape of feat_numpy should be n x 1'
        return feat_numpy



class JointPredictor(BaseRegressionPredictor):
    """Combining regression predictors by training jointly."""

    def  __init__(self, dataset_name, predictor_classes, predictor_name=None,
        reg_coef='CV', **kwargs):
        # todo : reg_coef is CV right their, that will need to change...

        assert predictor_name is None , 'why are defining a predictor name'
        super(JointPredictor, self).__init__(dataset_name, reg_coef, **kwargs)
        self.predictors = []
        assert reg_coef =='CV', 'you made this assumption'
        # commenting out b/c i'm not going to be specifying any specific regularization parameters for
        # any of these models. - BCJ - 5/17/22
        # for c, name in zip(predictor_classes, predictor_name):
            # if f'reg_coef_{name}' in kwargs:
            #     self.predictors.append(c(dataset_name,
            #         reg_coef=float(kwargs[f'reg_coef_{name}']), **kwargs))
            # else:
            #     self.predictors.append(c(dataset_name, **kwargs))

        for i,c in enumerate(predictor_classes):
            kwargs['idx']=i
            self.predictors.append(c(dataset_name,**kwargs))

    def seq2feat(self, seqs):
        # To apply different regularziation coefficients we scale the features
        # by a multiplier in Ridge regression
        features = [p.seq2feat(seqs) * np.sqrt(1.0 / p.reg_coef)
            for p in self.predictors]
        return np.concatenate(features, axis=1)

    # def __repr__(self):
    #     return f"{[p.__name__ for p in self.predictors]}_{self.reg_coef}"



class EVPredictor(BaseRegressionPredictor):
    """plmc mutation effect prediction."""
    def __init__(self, dataset_name, model_name='uniref100',
        reg_coef=1e-8, ignore_gaps=False, **kwargs):
        '''
        okay we will have to abstract away in understanding ignore gaps...
        :param dataset_name:
        :param model_name: where the model parameters are saved
        :param reg_coef: for the ridge regression so that makes sense
        :param ignore_gaps: ?
        :param kwargs:
        '''
        super(EVPredictor, self).__init__(dataset_name, reg_coef=reg_coef,
                **kwargs)
        self.ignore_gaps = ignore_gaps
        self.couplings_model_path = os.path.join('..','inference', dataset_name,
                'plmc', model_name + '.model_params')
        self.couplings_model = CouplingsModel(self.couplings_model_path)
        wtseqs, wtids = read_fasta(os.path.join('..','data', dataset_name,
            'wt.fasta'), return_ids=True)
        if '/' in wtids[0]:
            self.offset = int(wtids[0].split('/')[-1].split('-')[0])
        else:
            self.offset = 1
        expected_wt = wtseqs[0]
        for pf, pm in self.couplings_model.index_map.items():
            if expected_wt[pf-self.offset] != self.couplings_model.target_seq[pm]:
                print(f'WT and model target seq mismatch at {pf}')

    def seq2score(self, seqs):
        return seq2effect(seqs, self.couplings_model, self.offset,
                ignore_gaps=self.ignore_gaps)

    def seq2feat(self, seqs):
        # todo : why is this different than predict unsupervised.
        #   but here is where you need to look up in some defined way like get the saved score.
        #   so yeah idk. let me think.
        return self.seq2score(seqs)[:, None]

    def predict_unsupervised(self, seqs):
        return self.seq2score(seqs)


class OnehotRidgePredictor(BaseRegressionPredictor):
    """Simple one hot encoding + ridge regression."""

    def __init__(self, dataset_name, reg_coef=1.0, **kwargs):

        print(f'one hot predictor {dataset_name},reg_coef: {reg_coef}')
        super(OnehotRidgePredictor, self).__init__(
                dataset_name, reg_coef, Ridge, **kwargs)

    def seq2feat(self, seqs):
        return seqs_to_onehot(seqs)

    def __repr__(self):
        return f"onehot_reg_coef_{self.reg_coef:0.2f}"


def load_rows_by_numbers(filename_glob_pattern, line_numbers):
    lns_sorted = sorted(line_numbers)
    lns_idx = np.argsort(line_numbers)
    n_rows = len(line_numbers)
    current_ln = 0   # current (accumulated) line number in opened file
    j = 0            # index in lns
    rows = None
    for f in sorted(glob.glob(filename_glob_pattern)):
        with open(f) as fp:
            for line in fp:
                while j < n_rows and lns_sorted[j] == current_ln:
                    thisrow = np.array([float(x) for x in line.split(' ')])
                    if rows is None:
                        rows = np.full((n_rows, len(thisrow)), np.nan)
                    rows[lns_idx[j], :] = thisrow
                    j += 1
                current_ln += 1
    assert j == n_rows, (f"Expected {n_rows} rows, found {j}. "
    f"Scanned {current_ln} lines from {filename_glob_pattern}.")
    return rows


class BaseUniRepPredictor(BaseRegressionPredictor):
    """UniRep representation + regression."""

    def __init__(self, dataset_name, rep_name, reg_coef=1.0, **kwargs):
        super(BaseUniRepPredictor, self).__init__(
            dataset_name, reg_coef, Ridge)
        self.load_rep(dataset_name, rep_name)

    def load_rep(self, dataset_name, rep_name):
        self.rep_path = os.path.join('..','inference', dataset_name,
                'unirep', rep_name, f'avg_hidden.npy*')
        self.seq_path = os.path.join('..','inference', dataset_name,
                'unirep', rep_name, f'seqs.npy')
        #self.features = load(self.rep_path)
        self.seqs = np.loadtxt(self.seq_path, dtype=str, delimiter=' ')
        self.seq2id = dict(zip(self.seqs, range(len(self.seqs))))

    def seq2feat(self, seqs):
        """Look up representation by sequence."""
        ids = [self.seq2id[s] for s in seqs]
        return load_rows_by_numbers(self.rep_path, ids)


class EUniRepRegressionPredictor(BaseUniRepPredictor):
    """Evotuned UniRep + Ridge regression."""

    def __init__(self, dataset_name, rep_name='uniref100', **kwargs):
        super(EUniRepRegressionPredictor, self).__init__(
                dataset_name, rep_name, **kwargs)

class UniRepLLPredictor(BaseUniRepPredictor):
    """UniRep log likelihood."""

    def __init__(self, dataset_name, rep_name, reg_coef=1e-8, **kwargs):
        super(UniRepLLPredictor, self).__init__(
                dataset_name, rep_name, reg_coef=reg_coef, **kwargs)
        self.loss_path = os.path.join('..','inference', dataset_name,
                'unirep', rep_name, f'loss.npy*')

    def seq2feat(self, seqs):
        """Look up log likelihood by sequence."""
        ids = [self.seq2id[s] for s in seqs]
        return -load_rows_by_numbers(self.loss_path, ids)

    def predict_unsupervised(self, seqs):
        return self.seq2feat(seqs).ravel()

class EUniRepLLPredictor(UniRepLLPredictor):
    def __init__(self, dataset_name, rep_name='uniref100', **kwargs):
        super(EUniRepLLPredictor, self).__init__(dataset_name,
                rep_name, **kwargs)

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

int_to_aa = {value:key for key, value in aa_to_int.items()}