import warnings
from copy import copy, deepcopy

import numpy as np
from poap.strategy import EvalRecord
from pySOT.experimental_design import SymmetricLatinHypercube
from pySOT.optimization_problems import OptimizationProblem
from pySOT.strategy import SRBFStrategy
from pySOT.surrogate import CubicKernel, LinearTail, RBFInterpolant, SurrogateUnitBox

import scipy.stats as ss
from turbo import Turbo1
from turbo.utils import from_unit_cube, latin_hypercube, to_unit_cube

from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main
from bayesmark.space import JointSpace


def order_stats(X):
    _, idx, cnt = np.unique(X, return_inverse=True, return_counts=True)
    obs = np.cumsum(cnt)
    o_stats = obs[idx]
    return o_stats


def copula_standardize(X):
    X = np.nan_to_num(np.asarray(X))
    assert X.ndim == 1 and np.all(np.isfinite(X))
    o_stats = order_stats(X)
    quantile = np.true_divide(o_stats, len(X) + 1)
    X_ss = ss.norm.ppf(quantile)
    return X_ss


class tuSOTOptimizer(AbstractOptimizer):
    primary_import = "pysot"

    def __init__(self, api_config):
        """Build wrapper class to use an optimizer in benchmark.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """
        AbstractOptimizer.__init__(self, api_config)

        self.space_x = JointSpace(api_config)
        self.bounds = self.space_x.get_bounds()
        self.create_opt_prob()  # Sets up the optimization problem (needs self.bounds)
        self.max_evals = np.iinfo(np.int32).max  # NOTE: Largest possible int
        self.turbo_batch_size = None
        self.pysot_batch_size = None
        self.history = []
        self.proposals = []
        self.lb, self.ub = self.bounds[:, 0], self.bounds[:, 1]
        self.dim = len(self.bounds)

        self.turbo = Turbo1(
            f=None,
            lb=self.bounds[:, 0],
            ub=self.bounds[:, 1],
            n_init=2 * self.dim + 1,
            max_evals=self.max_evals,
            batch_size=4,  # We need to update this later
            verbose=False,
        )

    def restart(self):
        self.turbo._restart()
        self.turbo._X = np.zeros((0, self.turbo.dim))
        self.turbo._fX = np.zeros((0, 1))
        X_init = latin_hypercube(self.turbo.n_init, self.dim)
        self.X_init = from_unit_cube(X_init, self.lb, self.ub)

    def create_opt_prob(self):
        """Create an optimization problem object."""
        opt = OptimizationProblem()
        opt.lb = self.bounds[:, 0]  # In warped space
        opt.ub = self.bounds[:, 1]  # In warped space
        opt.dim = len(self.bounds)
        opt.cont_var = np.arange(len(self.bounds))
        opt.int_var = []
        assert len(opt.cont_var) + len(opt.int_var) == opt.dim
        opt.objfun = None
        self.opt = opt

    def start(self):
        """Starts a new pySOT run."""
        self.history = []
        self.proposals = []

        # Symmetric Latin hypercube design
        des_pts = max([self.pysot_batch_size, 2 * (self.opt.dim + 1)])
        slhd = SymmetricLatinHypercube(dim=self.opt.dim, num_pts=des_pts)

        # Warped RBF interpolant
        rbf = RBFInterpolant(dim=self.opt.dim, kernel=CubicKernel(), tail=LinearTail(self.opt.dim), eta=1e-4)
        rbf = SurrogateUnitBox(rbf, lb=self.opt.lb, ub=self.opt.ub)

        # Optimization strategy
        self.strategy = SRBFStrategy(
            max_evals=self.max_evals,
            opt_prob=self.opt,
            exp_design=slhd,
            surrogate=rbf,
            asynchronous=True,
            batch_size=1,
            use_restarts=True,
        )

    def pysot_suggest(self, turbo_suggest, n_suggestions=1):
        if self.pysot_batch_size is None:  # First call to suggest
            self.pysot_batch_size = n_suggestions
            self.start()

        # Set the tolerances pretending like we are running batch
        d, p = float(self.opt.dim), float(n_suggestions)
        self.strategy.failtol = p * int(max(np.ceil(d / p), np.ceil(4 / p)))

        # Now we can make suggestions
        x_w = []
        self.proposals = []
        for _ in range(n_suggestions):
            proposal = self.strategy.propose_action()
            record = EvalRecord(proposal.args, status="pending")
            proposal.record = record
            proposal.accept()  # This triggers all the callbacks

            # It is possible that pySOT proposes a previously evaluated point
            # when all variables are integers, so we just abort in this case
            # since we have likely converged anyway. See PySOT issue #30.
            x = list(proposal.record.params)  # From tuple to list
            x_unwarped, = self.space_x.unwarp(x)
            if x_unwarped in self.history:
                warnings.warn("pySOT proposed the same point twice")
                self.start()
                return self.suggest(n_suggestions=n_suggestions)

            # NOTE: Append unwarped to avoid rounding issues
            self.history.append(copy(x_unwarped))
            self.proposals.append(proposal)
            x_w.append(copy(x_unwarped))

        turbo_suggest_warps = self.space_x.warp(turbo_suggest)
        for i, warps in enumerate(turbo_suggest_warps):
            proposal = self.strategy.make_proposal(warps)
            proposal.add_callback(self.strategy.on_initial_proposal)
            record = EvalRecord(proposal.args, status="pending")
            proposal.record = record
            proposal.accept()

            self.history.append(copy(turbo_suggest[i]))
            self.proposals.append(proposal)

        return x_w

    def turbo_suggest(self, n_suggestions=1):
        if self.turbo_batch_size is None:  # Remember the batch size on the first call to suggest
            self.turbo_batch_size = n_suggestions
            self.turbo.batch_size = n_suggestions
            self.turbo.failtol = np.ceil(np.max([4.0 / self.turbo_batch_size, self.dim / self.turbo_batch_size]))
            self.turbo.n_init = max([self.turbo.n_init, self.turbo_batch_size])
            self.restart()

        X_next = np.zeros((n_suggestions, self.dim))

        # Pick from the initial points
        n_init = min(len(self.X_init), n_suggestions)
        if n_init > 0:
            X_next[:n_init] = deepcopy(self.X_init[:n_init, :])
            self.X_init = self.X_init[n_init:, :]  # Remove these pending points

        # Get remaining points from TuRBO
        n_adapt = n_suggestions - n_init
        if n_adapt > 0:
            if len(self.turbo._X) > 0:  # Use random points if we can't fit a GP
                X = to_unit_cube(deepcopy(self.turbo._X), self.lb, self.ub)
                fX = copula_standardize(deepcopy(self.turbo._fX).ravel())  # Use Copula
                X_cand, y_cand, _ = self.turbo._create_candidates(
                    X, fX, length=self.turbo.length, n_training_steps=100, hypers={}
                )
                X_next[-n_adapt:, :] = self.turbo._select_candidates(X_cand, y_cand)[:n_adapt, :]
                X_next[-n_adapt:, :] = from_unit_cube(X_next[-n_adapt:, :], self.lb, self.ub)

        # Unwarp the suggestions
        suggestions = self.space_x.unwarp(X_next)
        return suggestions

    def suggest(self, n_suggestions=1):
        if n_suggestions == 1:
            return self.turbo_suggest(n_suggestions)
        else:
            suggestion = n_suggestions // 2
            turbo_suggest = self.turbo_suggest(suggestion)
            pysot_suggest = self.pysot_suggest(turbo_suggest, n_suggestions - suggestion)
            return turbo_suggest + pysot_suggest

    def _observe(self, x, y):
        # Find the matching proposal and execute its callbacks
        idx = [x == xx for xx in self.history]
        if np.any(idx):
            i = np.argwhere(idx)[0].item()  # Pick the first index if there are ties
            proposal = self.proposals[i]
            proposal.record.complete(y)
            self.proposals.pop(i)
            self.history.pop(i)

    def observe(self, X, y):
        """Send an observation of a suggestion back to the optimizer.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        assert len(X) == len(y)

        for x_, y_ in zip(X, y):
            # Just ignore, any inf observations we got, unclear if right thing
            if np.isfinite(y_):
                self._observe(x_, y_)

        XX, yy = self.space_x.warp(X), np.array(y)[:, None]

        if len(self.turbo._fX) >= self.turbo.n_init:
            self.turbo._adjust_length(yy)

        self.turbo.n_evals += self.turbo_batch_size

        self.turbo._X = np.vstack((self.turbo._X, deepcopy(XX)))
        self.turbo._fX = np.vstack((self.turbo._fX, deepcopy(yy)))
        self.turbo.X = np.vstack((self.turbo.X, deepcopy(XX)))
        self.turbo.fX = np.vstack((self.turbo.fX, deepcopy(yy)))

        # Check for a restart
        if self.turbo.length < self.turbo.length_min:
            self.restart()


if __name__ == "__main__":
    experiment_main(tuSOTOptimizer)
