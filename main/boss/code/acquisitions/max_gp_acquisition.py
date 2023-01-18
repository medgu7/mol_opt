from typing import Tuple
import scipy.stats
import numpy as np
from emukit.core.interfaces import IModel
from emukit.core.acquisition import Acquisition


class Max_G(Acquisition):

    def __init__(self, model: IModel) -> None:
        """
        This acquisition computes for a given input point the value predicted by the GP surrogate model
        :param model: The underlying model that provides the predictive mean and variance for the given test points
        """
        self.model = model

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the predicted means
        :param x: points where the acquisition is evaluated, shape (number of points, number of dimensions).
        """
        mean, _ = self.model.predict(x)
        return -mean

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple:
        """
        Computes the predicted means and its derivative
        :param x: points where the acquisition is evaluated, shape (number of points, number of dimensions).
        """
        mean, variance = self.model.predict(x)

        dmean_dx, _ = self.model.get_prediction_gradients(x)
        return mean, dmean_dx

    @property
    def has_gradients(self):
        return isinstance(self.model, IDifferentiable)
    
    
    
from __future__ import annotations

from typing import Any, Callable, Optional
import torch
from torch import Tensor

from botorch.models.model import Model
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.sampling.samplers import MCSampler, SobolQMCNormalSampler
from botorch.utils.transforms import (
    concatenate_pending_points,
    t_batch_mode_transform,
)
from botorch.utils.multi_objective.box_decompositions.box_decomposition import (
    BoxDecomposition,
)
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning
)

from botorch.models.utils import fantasize as fantasize_flag, validate_input_scaling
from botorch import settings
from torch.distributions import Normal
from gpytorch.functions import logdet

from math import pi


CLAMP_LB = 1.0e-8
NEG_INF = -1e+10


class JES(AcquisitionFunction):
    r"""The acquisition function for Joint Entropy Search.
    This acquisition function computes the mutual information between the observation
    at a candidate point X and the Pareto optimal input-output pairs.
    The batch case `q > 1` is supported through cyclic optimization and fantasies.
    TODO: Implement a user-defined tolerance for the sampling noise.
    """

    def __init__(
        self,
        model: Model,
        num_pareto_samples: int,
        num_pareto_points: int,
        sample_pareto_sets_and_fronts: Callable[
            [Model, Model, int, int, Tensor], Tensor
        ],
        bounds: Tensor,
        num_fantasies: int,
        partitioning: BoxDecomposition = DominatedPartitioning,
        maximize: bool = True,
        X_pending: Optional[Tensor] = None,
        estimation_type: str = "Noiseless",
        sampling_noise: Optional[bool] = True,
        only_diagonal: Optional[bool] = False,
        sampler: Optional[MCSampler] = None,
        num_samples: Optional[int] = 64,
        num_constraints: Optional[int] = 0,
        **kwargs: Any,
    ) -> None:
        r"""Joint entropy search acquisition function.
        Args:
            model: A fitted batch model with 'M + K' number of outputs. The
                first `M` corresponds to the objectives and the rest corresponds
                to the constraints. An input is feasible if f_k(x) <= 0 for the
                constraint functions f_k.
            num_pareto_samples: The number of samples for the Pareto optimal input
                and outputs.
            num_pareto_points: The number of Pareto points for each sample.
            sample_pareto_sets_and_fronts: A callable that takes the initial model,
                the fantasy model, the number of pareto samples, the input bounds
                and returns the Pareto optimal set of inputs and outputs:
                - pareto_sets: a `num_pareto_samples x num_fantasies x
                    num_pareto_points x d`-dim Tensor
                - pareto_fronts: a `num_pareto_samples x num_fantasies x
                    num_pareto_points x M`-dim Tensor.
            bounds: a `2 x d`-dim Tensor containing the input bounds for
                multi-objective optimization.
            num_fantasies: Number of fantasies to generate. Ignored if `X_pending`
                is `None`.
            partitioning: A `BoxDecomposition` module that is used to obtain the
                hyper-rectangle bounds for integration. In the unconstrained case,
                this gives the partition of the dominated space. In the constrained
                case, this gives the partition of the feasible dominated space union
                the infeasible space.
            maximize: If True, consider the problem a maximization problem.
            X_pending: A `m x d`-dim Tensor of `m` design points that have been
                submitted for function evaluation but have not yet been evaluated.
            estimation_type: A string to determine which entropy estimate is
                computed: "Noiseless", "Noiseless lower bound", "Lower bound"
                or "Monte Carlo".
            sampling_noise: If True we assume that there is noise when sampling the
                Pareto optimal points. We advise always setting `sampling_noise =
                True`. The JES estimate tends to exhibit a large variance when using
                the `Noiseless`, `Noiseless lower bound` or `Lower Bound` entropy
                estimation strategy if this is turned off.
            only_diagonal: If true we only compute the diagonal elements of the
                variance for the `Lower bound` estimation strategy.
            sampler: The sampler used if Monte Carlo is used to estimate the entropy.
                Defaults to 'SobolQMCNormalSampler(num_samples,
                collapse_batch_dims=True)'.
            num_samples: The number of Monte Carlo samples if using the default Monte
                Carlo sampler.
            num_constraints: The number of constraints.
        """
        super().__init__(model=model)

        self._init_model = model
        self.prior_model = model
        self.posterior_model = model

        self.num_pareto_samples = num_pareto_samples
        self.num_pareto_points = num_pareto_points
        self.sample_pareto_sets_and_fronts = sample_pareto_sets_and_fronts

        self.fantasies_sampler = SobolQMCNormalSampler(num_fantasies)
        self.num_fantasies = num_fantasies

        self.partitioning = partitioning

        self.maximize = maximize
        self.weight = 1.0 if maximize else -1.0
        self.bounds = bounds

        self.estimation_type = estimation_type
        if estimation_type not in ["Noiseless", "Noiseless lower bound",
                                   "Lower bound", "Monte Carlo"]:
            raise NotImplementedError(
                "Currently the only supported estimation type are: "
                "['Noiseless', 'Noiseless lower bound', 'Lower bound', 'Monte Carlo'"
                "]."
            )
        self.sampling_noise = sampling_noise
        self.only_diagonal = only_diagonal
        if sampler is None:
            self.sampler = SobolQMCNormalSampler(
                num_samples=num_samples, collapse_batch_dims=True
            )
        else:
            self.sampler = sampler

        self.num_constraints = num_constraints
        self.hypercell_bounds = None
        self.set_X_pending(X_pending)

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        r"""Set pending points.
        Informs the acquisition function about pending design points, fantasizes the
        model on the pending points and draws pareto optimal samples from the
        fantasized model posterior.
        Args:
            X_pending: A `num_pending x d` Tensor with `num_pending` `d`-dim design
                points that have been submitted for evaluation but have not yet been
                evaluated.
        """

        if X_pending is not None:
            # fantasize the model on pending points
            fantasy_model = self._init_model.fantasize(
                X=X_pending,
                sampler=self.fantasies_sampler,
                observation_noise=True
            )
            self.prior_model = fantasy_model

        self._sample_pareto_points()

        # Condition the model on the sampled pareto optimal points
        # Need to call posterior otherwise gpytorch runtime error
        # "Fantasy observations can only be added after making predictions with a
        # model so that all test independent caches exist."
        with fantasize_flag():
            with settings.propagate_grads(False):
                post_ps = self.prior_model.posterior(
                    self.pareto_sets, observation_noise=False
                )
            if self.sampling_noise:
                # condition with observation noise
                self.posterior_model = self.prior_model.condition_on_observations(
                    X=self.prior_model.transform_inputs(self.pareto_sets),
                    Y=self.pareto_fronts
                )
            else:
                # condition without observation noise
                self.posterior_model = self.prior_model.condition_on_observations(
                    X=self.prior_model.transform_inputs(self.pareto_sets),
                    Y=self.pareto_fronts,
                    noise=torch.zeros(self.pareto_fronts.shape)
                )

        # Compute the box decompositions
        with torch.no_grad():
            self.hypercell_bounds = compute_box_decomposition(
                self.pareto_fronts,
                self.partitioning,
                self.maximize,
                self.num_constraints
            )

    def _sample_pareto_points(self) -> None:
        r"""Sample the Pareto optimal input-output pairs for the Monte Carlo
        approximation of the entropy in Joint Entropy Search.
        Note: Sampling exactly `num_pareto_points` of Pareto optimal inputs and
        outputs is achieved by over generating points and then truncating the sample.
        """
        with torch.no_grad():
            # pareto_sets shape:
            # `num_pareto_samples x num_fantasies x num_pareto_points x d`
            # pareto_fronts shape:
            # `num_pareto_samples x num_fantasies x num_pareto_points x M`
            pareto_sets, pareto_fronts = self.sample_pareto_sets_and_fronts(
                model=self._init_model,
                fantasy_model=self.prior_model,
                num_pareto_samples=self.num_pareto_samples,
                num_pareto_points=self.num_pareto_points,
                bounds=self.bounds,
                maximize=self.maximize,
            )

            self.pareto_sets = pareto_sets
            self.pareto_fronts = pareto_fronts

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Compute joint entropy search at the design points `X`.
        Args:
            X: A `batch_shape x 1 x d`-dim Tensor of `batch_shape` t-batches with `1`
            `d`-dim design points each.
        Returns:
            A `batch_shape`-dim Tensor of JES values at the given design points `X`.
        """

        K = self.num_constraints
        M = self._init_model.num_outputs - K

        # compute the prior entropy term depending on `X`
        prior_posterior_plus_noise = self.prior_model.posterior(
            X.unsqueeze(-2), observation_noise=True
        )

        # additional constant term
        add_term = .5 * (M + K) * (1 + torch.log(torch.ones(1) * 2 * pi))
        # the variance initially has shape `batch_shape x num_fantasies x 1 x
        # (M + K)`
        # prior_entropy has shape `batch_shape x num_fantasies`
        prior_entropy = add_term + .5 * torch.log(
            prior_posterior_plus_noise.variance.clamp_min(CLAMP_LB)
        ).sum(-1).squeeze(-1)

        # compute the posterior entropy term
        # Note: we compute the posterior twice here because we need access to
        # the variance with observation noise.
        # [There is probably a better way to do this.]
        post_posterior = self.posterior_model.posterior(
            X.unsqueeze(-2).unsqueeze(-2), observation_noise=False
        )
        post_posterior_plus_noise = self.posterior_model.posterior(
            X.unsqueeze(-2).unsqueeze(-2), observation_noise=True
        )

        post_mean = post_posterior.mean
        post_var = post_posterior.variance.clamp_min(CLAMP_LB)
        post_var_plus_noise = post_posterior_plus_noise.variance.clamp_min(CLAMP_LB)

        # `batch_shape x num_fantasies` dim Tensor of entropy estimates
        if self.estimation_type == "Noiseless":
            post_entropy = _compute_entropy_noiseless(
                hypercell_bounds=self.hypercell_bounds,
                mean=post_mean,
                variance=post_var,
                variance_plus_noise=post_var_plus_noise,
            )
        if self.estimation_type == "Noiseless lower bound":
            prior_posterior = self.prior_model.posterior(
                X.unsqueeze(-2), observation_noise=False
            )
            # the variance initially has shape `batch_shape x num_fantasies x 1 x
            # (M + K)`
            prior_var = prior_posterior.variance.clamp_min(CLAMP_LB)
            prior_var_plus_noise = prior_posterior_plus_noise.variance.clamp_min(
                CLAMP_LB
            )
            # new shape `batch_shape x num_pareto_samples x num_fantasies x 1 x
            # (M + K)`
            new_shape = prior_var.shape[:-3] \
                + torch.Size([self.num_pareto_samples]) + prior_var.shape[-3:]

            prior_var = prior_var.unsqueeze(-4).expand(new_shape)
            prior_var_plus_noise = prior_var_plus_noise.unsqueeze(-4).expand(
                new_shape
            )

            post_entropy = _compute_entropy_noiseless_upper_bound(
                hypercell_bounds=self.hypercell_bounds,
                mean=post_mean,
                variance=post_var,
                initial_variance=prior_var,
                initial_variance_plus_noise=prior_var_plus_noise,
            )
        if self.estimation_type == "Lower bound":
            post_entropy = _compute_entropy_upper_bound(
                hypercell_bounds=self.hypercell_bounds,
                mean=post_mean,
                variance=post_var,
                variance_plus_noise=post_var_plus_noise,
                only_diagonal=self.only_diagonal
            )
        if self.estimation_type == "Monte Carlo":
            # `num_mc_samples x batch_shape x num_pareto_samples x num_fantasies x 1
            # x (M+K)`
            samples = self.sampler(post_posterior_plus_noise)

            # `num_mc_samples x batch_shape x num_pareto_samples x num_fantasies`
            if (M + K) == 1:
                samples_log_prob = post_posterior_plus_noise.mvn.log_prob(
                    samples.squeeze(-1)
                )
            else:
                samples_log_prob = post_posterior_plus_noise.mvn.log_prob(
                    samples
                )

            post_entropy = _compute_entropy_monte_carlo(
                hypercell_bounds=self.hypercell_bounds,
                mean=post_mean,
                variance=post_var,
                variance_plus_noise=post_var_plus_noise,
                samples=samples,
                samples_log_prob=samples_log_prob
            )

        # average over the fantasies
        return (prior_entropy - post_entropy).mean(dim=-1)
