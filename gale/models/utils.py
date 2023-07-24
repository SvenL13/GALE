"""
GALe - Global Adaptive Learning
@author: Sven Lämmle

Models - Utilities
"""
from typing import Callable, List, Literal, Optional, Tuple, Union

import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel, Kernel, Matern

from gale.models.gp import GPRegressor, SurrogateRegressionModel


def cook_model(
    model_name: Literal["GP", "BNN"],
    gp_kernel: Optional[Kernel] = None,
    net_config: dict = None,
    return_class: bool = False,
    n_dim: int = None,
    rnd_state: int = 543532,
    **model_args,
) -> Union[SurrogateRegressionModel, Tuple[SurrogateRegressionModel, Callable]]:
    """
    Create initial surrogate model

    Parameters
    ----------
    model_name: Literal["GP", "SGP", "BNN"]
        model to use
    gp_kernel: instance of sklearn Kernel class, optional (default=None)
        Kernel used in GP model
    net_config: dict, optional(default=None)
        configuration used for BNN, if None -> default bnn is used
    return_class: bool
        indicate if model class should be returned
    n_dim: int, optional(default=None)
        if integer is passed then this information is used for the GP to create
        anisotropic kernels
    rnd_state: int, optional
        seed

    Returns
    -------
    surrogate_model: SurrogateRegressionModel
        created model
    model_class: Callable
        class to create the surrogate model
    """
    surrogate_model: SurrogateRegressionModel

    valid_models: List[str] = ["GP", "BNN", "SGP"]

    if model_name == "GP":

        if gp_kernel is None:  # use default kernel

            if n_dim is None:
                n_dim = 1  # use isotropic kernel
                ls_up_bound = 10
            else:
                ls_up_bound = 5 * np.sqrt(n_dim)  # input should be normalized (0, 1)

            gp_kernel = ConstantKernel(1)

            gp_kernel *= Matern(
                np.array([1] * n_dim),
                length_scale_bounds=(1e-6, ls_up_bound),
                nu=1.5,
            )

        elif not isinstance(gp_kernel, Kernel):
            raise TypeError("Given Kernel is not a valid kernel!")

        # init model
        surrogate_model = GPRegressor(
            kernel=gp_kernel,
            random_state=rnd_state,
            normalize_y=True,
            n_restarts_optimizer=10,
            **model_args,
        )
        model_class = lambda: GPRegressor(
            kernel=gp_kernel,
            random_state=rnd_state,
            normalize_y=True,
            n_restarts_optimizer=10,
            **model_args,
        )
    elif model_name == "SGP":
        from .sparse_gp import SparseGPRegressor

        surrogate_model = SparseGPRegressor()
        model_class = lambda: SparseGPRegressor()

    elif model_name == "BNN":

        if net_config is None:
            # topology used for Stybli4d_norm, Rose4d_norm, Ackley4d_norm (epochs=500, num_batches=5)
            net_top = [
                {"neurons": 64, "activation": "swish", "weight_decay": 0.0001},
                {"neurons": 1, "weight_decay": 0.0005},
            ]

            # topology used for Micha4d_norm (epochs=500, num_batches=5)
            # net_top = [{"neurons": 48, "activation": "tanh", "weight_decay": 0.0001},
            #               {"neurons": 1, "weight_decay": 0.0005},
            #               ]

            net_config: dict = {"net_config": net_top, "lr": 0.03}

        from gale.models.bnn_hot import BnnHot

        surrogate_model = BnnHot(
            config=net_config, random_state=rnd_state, **model_args
        )
        model_class = lambda: BnnHot(
            config=net_config, random_state=rnd_state, **model_args
        )
    else:
        raise ValueError(
            "No valid surrogate model is given. Valid models:" + str(valid_models)
        )

    if return_class:
        return surrogate_model, model_class
    else:
        return surrogate_model
