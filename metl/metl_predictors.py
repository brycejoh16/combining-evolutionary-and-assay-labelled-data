import numpy as np
import os
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from scipy.stats import spearmanr
from metl_couplings_model import CouplingsModel
from Bio import SeqIO

REG_COEF_LIST = [1e-1, 1e0, 1e1, 1e2, 1e3]

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
        mutations = seq2mutation(seqs[i], model,
                ignore_gaps=ignore_gaps, offset=offset)
        dE, _, _ = model.delta_hamiltonian(mutations)
        effects[i] = dE
    return effects

def seq2mutation(seq, model, return_str=False, ignore_gaps=False,
        sep=":", offset=1):
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
        self.reg_coef = reg_coef
        self.linear_model_cls = linear_model_cls
        self.model = None
        super().__init__(dataset_name, **kwargs)

    def seq2feat(self, seqs):
        raise NotImplementedError

    def train(self, train_seqs, train_labels):
        X = self.seq2feat(train_seqs)
        if self.reg_coef is None or self.reg_coef == 'CV':
            best_rc, best_score = None, -np.inf
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
        self.model = self.linear_model_cls(alpha=self.reg_coef)
        self.model.fit(X, train_labels)

    def predict(self, predict_seqs):
        if self.model is None:
            return np.random.randn(len(predict_seqs))
        X = self.seq2feat(predict_seqs)
        return self.model.predict(X)




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
        return self.seq2score(seqs)[:, None]

    def predict_unsupervised(self, seqs):
        return self.seq2score(seqs)


