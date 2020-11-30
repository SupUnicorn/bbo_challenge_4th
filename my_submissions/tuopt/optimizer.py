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

from hyperopt import hp, tpe
from hyperopt.base import JOB_STATE_DONE, JOB_STATE_NEW, STATUS_OK, Domain, Trials, miscs_update_idxs_vals, pyll
from scipy.interpolate import interp1d
from bayesmark.np_util import random as np_random
from bayesmark.np_util import random_seed

# hyperopt tools
DTYPE_MAP = {"real": float, "int": int, "bool": bool, "cat": str, "ordinal": str}


def dummy_f(x):
    assert False, "This is a placeholder, it should never be called."


def only(x):
    y, = x
    return y


# turbo tools
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

        # hyperopt
        self.random = np_random

        space, self.round_to_values = tuSOTOptimizer.get_hyperopt_dimensions(api_config)
        self.domain = Domain(dummy_f, space, pass_expr_memo_ctrl=None)
        self.trials = Trials()

        # Some book keeping like opentuner wrapper
        self.trial_id_lookup = {}

        # Store just for data validation
        self.param_set_chk = frozenset(api_config.keys())

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

    @staticmethod
    def hashable_dict(d):
        hashable_object = frozenset(d.items())
        return hashable_object

    @staticmethod
    def get_hyperopt_dimensions(api_config):
        param_list = sorted(api_config.keys())

        space = {}
        round_to_values = {}
        for param_name in param_list:
            param_config = api_config[param_name]

            param_type = param_config["type"]

            param_space = param_config.get("space", None)
            param_range = param_config.get("range", None)
            param_values = param_config.get("values", None)

            # Some setup for case that whitelist of values is provided:
            values_only_type = param_type in ("cat", "ordinal")
            if (param_values is not None) and (not values_only_type):
                assert param_range is None
                param_values = np.unique(param_values)
                param_range = (param_values[0], param_values[-1])
                round_to_values[param_name] = interp1d(
                    param_values, param_values, kind="nearest", fill_value="extrapolate"
                )

            if param_type == "int":
                low, high = param_range
                if param_space in ("log", "logit"):
                    space[param_name] = hp.qloguniform(param_name, np.log(low), np.log(high), 1)
                else:
                    space[param_name] = hp.quniform(param_name, low, high, 1)
            elif param_type == "bool":
                assert param_range is None
                assert param_values is None
                space[param_name] = hp.choice(param_name, (False, True))
            elif param_type in ("cat", "ordinal"):
                assert param_range is None
                space[param_name] = hp.choice(param_name, param_values)
            elif param_type == "real":
                low, high = param_range
                if param_space in ("log", "logit"):
                    space[param_name] = hp.loguniform(param_name, np.log(low), np.log(high))
                else:
                    space[param_name] = hp.uniform(param_name, low, high)
            else:
                assert False, "type %s not handled in API" % param_type

        return space, round_to_values

    def get_trial(self, trial_id):
        for trial in self.trials._dynamic_trials:
            if trial["tid"] == trial_id:
                assert isinstance(trial, dict)
                # Make sure right kind of dict
                assert "state" in trial and "result" in trial
                assert trial["state"] == JOB_STATE_NEW
                return trial
        assert False, "No matching trial ID"

    def cleanup_guess(self, x_guess):
        assert isinstance(x_guess, dict)
        # Also, check the keys are only the vars we are searching over:
        assert frozenset(x_guess.keys()) == self.param_set_chk

        # Do the rounding
        # Make a copy to be safe, and also unpack singletons
        # We may also need to consider clip_chk at some point like opentuner
        x_guess = {k: only(x_guess[k]) for k in x_guess}
        for param_name, round_f in self.round_to_values.items():
            x_guess[param_name] = round_f(x_guess[param_name])
        # Also ensure this is correct dtype so sklearn is happy
        x_guess = {k: DTYPE_MAP[self.api_config[k]["type"]](x_guess[k]) for k in x_guess}
        return x_guess

    def pysot_suggest(self, n_suggestions=1):
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

        return x_w

    def pysot_get_suggest(self, suggests):
        turbo_suggest_warps = self.space_x.warp(suggests)
        for i, warps in enumerate(turbo_suggest_warps):
            proposal = self.strategy.make_proposal(warps)
            proposal.add_callback(self.strategy.on_initial_proposal)
            record = EvalRecord(proposal.args, status="pending")
            proposal.record = record
            proposal.accept()

            self.history.append(copy(suggests[i]))
            self.proposals.append(proposal)

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

    def _hyperopt_suggest(self):
        new_ids = self.trials.new_trial_ids(1)
        assert len(new_ids) == 1
        self.trials.refresh()

        seed = random_seed(self.random)
        new_trials = tpe.suggest(new_ids, self.domain, self.trials, seed)
        assert len(new_trials) == 1

        self.trials.insert_trial_docs(new_trials)
        self.trials.refresh()

        new_trial, = new_trials  # extract singleton
        return new_trial

    def _hyperopt_transform(self, x):
        new_id = self.trials.new_trial_ids(1)[0]

        domain = self.domain
        rng = np.random.RandomState(1)
        idxs, vals = pyll.rec_eval(
            domain.s_idxs_vals,
            memo={
                domain.s_new_ids: [new_id],
                domain.s_rng: rng,
            })
        rval_miscs = [dict(tid=new_id, cmd=domain.cmd, workdir=domain.workdir)]
        rval_results = domain.new_result()
        for (k, _) in vals.items():
            vals[k][0] = x[k]
        miscs_update_idxs_vals(rval_miscs, idxs, vals)
        rval_docs = self.trials.new_trial_docs([new_id],
                                               [None], rval_results, rval_miscs)

        return rval_docs[0]

    def hyperopt_suggest(self, n_suggestions=1):
        assert n_suggestions >= 1, "invalid value for n_suggestions"

        # Get the new trials, it seems hyperopt either uses random search or
        # guesses one at a time anyway, so we might as welll call serially.
        new_trials = [self._hyperopt_suggest() for _ in range(n_suggestions)]

        X = []
        for trial in new_trials:
            x_guess = self.cleanup_guess(trial["misc"]["vals"])
            X.append(x_guess)

            # Build lookup to get original trial object
            x_guess_ = tuSOTOptimizer.hashable_dict(x_guess)
            assert x_guess_ not in self.trial_id_lookup, "the suggestions should not already be in the trial dict"
            self.trial_id_lookup[x_guess_] = trial["tid"]

        assert len(X) == n_suggestions
        return X

    def hyperopt_get_suggest(self, suggests):
        trials = [self._hyperopt_transform(x) for x in suggests]
        for trial in trials:
            x_guess = self.cleanup_guess(trial["misc"]["vals"])
            x_guess_ = tuSOTOptimizer.hashable_dict(x_guess)
            assert x_guess_ not in self.trial_id_lookup, "the suggestions should not already be in the trial dict"
            self.trial_id_lookup[x_guess_] = trial["tid"]
        self.trials.insert_trial_docs(trials)
        self.trials.refresh()

    def suggest(self, n_suggestions=1):
        if n_suggestions == 1:
            return self.turbo_suggest(n_suggestions)
        else:
            t_suggestion = n_suggestions // 2
            # p_suggestion = int((n_suggestions - t_suggestion) * 3/4)
            h_suggestion = n_suggestions - t_suggestion
            turbo_suggest = self.turbo_suggest(t_suggestion)
            # pysot_suggest = self.pysot_suggest(p_suggestion)
            hyperopt_suggest = self.hyperopt_suggest(h_suggestion)
            self.hyperopt_get_suggest(turbo_suggest)
            # self.pysot_get_suggest(turbo_suggest + hyperopt_suggest)
            return turbo_suggest + hyperopt_suggest

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

        # # pysot observe
        # for x_, y_ in zip(X, y):
        #     # Just ignore, any inf observations we got, unclear if right thing
        #     if np.isfinite(y_):
        #         self._observe(x_, y_)

        # turbo observe
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

        # hyperopt observe
        for x_guess, y_ in zip(X, y):
            x_guess_ = tuSOTOptimizer.hashable_dict(x_guess)
            assert x_guess_ in self.trial_id_lookup, "Appears to be guess that did not originate from suggest"

            assert x_guess_ in self.trial_id_lookup, "trial object not available in trial dict"
            trial_id = self.trial_id_lookup.pop(x_guess_)
            trial = self.get_trial(trial_id)
            assert self.cleanup_guess(trial["misc"]["vals"]) == x_guess, "trial ID not consistent with x values stored"

            # Cast to float to ensure native type
            result = {"loss": float(y_), "status": STATUS_OK}
            trial["state"] = JOB_STATE_DONE
            trial["result"] = result
        self.trials.refresh()


if __name__ == "__main__":
    experiment_main(tuSOTOptimizer)
