"""
GALe - Global Adaptive Learning
@author: Sven Lämmle

Models - PNN Ensemble
"""
import os
import pickle
from typing import Optional, Tuple
from warnings import filterwarnings

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam

from gale._typing import ARRAY_LIKE_1D, ARRAY_LIKE_2D
from gale.models.surrogate import SurrogateRegressionModel

from .dmbrl import BNN, FC

filterwarnings("ignore")


__all__ = ["BnnHot"]


class BnnHot(SurrogateRegressionModel, BNN):
    model_name = "Probabilistic Neural Network"
    is_multioutput = False
    """
    Wrapper for the PNN used in [1].
    
    A PNN can be defined as an NN whose output neurons parameterize a probability 
    distribution function (Gaussian in our case). To incorporate epistemic uncertainty, 
    ensembles of bootstrapped models are used.
    
    Parameters
    ----------
    num_networks: int, optional(default=5)
        number of models used for the ensemble
    norm_y: bool, optional(default=True)
        standardize output y to zero mean and unit variance during fitting
    random_state: int, optional(default=None)
        seed
    use_gpu: bool, optional(default=False)
        
    
    References
    ----------
    [1] K. Chua, R. Calandra, R. McAllister and S. Levine - Deep Reinforcement Learning 
    in a Handful of Trials using Probabilistic Dynamics Models, 2018; 
    github: https://github.com/kchua/handful-of-trials    
    """

    def __init__(
        self,
        config: dict,
        num_networks: int = 5,
        norm_y: bool = True,
        random_state: Optional[int] = None,
        use_gpu: bool = False,
    ):

        super(BnnHot, self).__init__(num_nets=num_networks)

        self._init_flag: bool = False

        self.config: dict = config

        self.X_train_: Optional[np.ndarray] = None
        self.y_train_: Optional[np.ndarray] = None
        self._n_dim_input: Optional[int] = None

        self._norm_y: bool = norm_y
        self._scaler_y = StandardScaler()
        self._norm_y_mu: Optional[float] = None
        self._norm_y_std: Optional[float] = None

        if use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        self._rnd_state: Optional[int] = 42 if random_state is None else random_state

        if random_state is not None:
            os.environ["PYTHONHASHSEED"] = "0"
            tf.random.set_seed(self._rnd_state + 3)

    @property
    def params(self):
        return [param.numpy() for param in self.trainable_variables]

    @params.setter
    def params(self, new_params):
        for i, param in enumerate(self.trainable_variables):
            param.assign(new_params[i])

    def _create_net(self, config: dict):

        assert isinstance(config, dict)
        assert isinstance(config["net_config"], list)
        assert len(config["net_config"]) >= 2  # has at least two layers

        net_config: list = config["net_config"]

        # first layer
        first_param = net_config[0]
        self.add(
            FC(
                first_param["neurons"],
                input_dim=self._n_dim_input,
                activation=first_param["activation"],
                weight_decay=first_param["weight_decay"],
            )
        )

        for layer in net_config[1:]:
            self.add(
                FC(
                    layer["neurons"],
                    activation=layer.get("activation", None),
                    weight_decay=layer["weight_decay"],
                )
            )

        self.finalize(Adam, {"learning_rate": config["lr"]})

    def fit(
        self,
        X: ARRAY_LIKE_2D,
        y: ARRAY_LIKE_1D,
        num_batches: int = 5,
        epochs: int = 500,
        verbose: bool = False,
    ):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        X, Y = self.check_input(X, y)

        if len(Y.shape) == 1:
            Y = Y.reshape(-1, 1)

        self.X_train_, self.y_train_ = X, Y

        self._n_dim_input = X.shape[1]

        if not self._init_flag:
            self._create_net(self.config)
            self._init_flag = True

        if X.shape[0] > num_batches:
            batch_size = int(X.shape[0] / num_batches)
        else:
            batch_size = X.shape[0]

        if self._norm_y:
            Y = self._scaler_y.fit_transform(Y)
            self._norm_y_mu = float(self._scaler_y.mean_.item())
            self._norm_y_std = float(np.sqrt(self._scaler_y.var_).item())

        BNN.fit(
            self, X, Y, batch_size=batch_size, epochs=epochs, hide_progress=not verbose
        )
        return self

    def predict(
        self,
        X: ARRAY_LIKE_2D,
        return_std: bool = False,
        return_cov: bool = False,
        return_grad: bool = False,
    ):
        X = self.check_input(X)

        if len(X.shape) == 1:  # single point
            X = X.reshape(1, -1)

        X = tf.convert_to_tensor(X, dtype=tf.float64)

        if return_grad:
            with tf.GradientTape() as tape:
                tape.watch(X)
                y_mu_, y_var_ = BNN.predict(self, X)
                y_grad = tape.gradient(y_mu_, X).numpy()
        else:
            y_mu_, y_var_ = BNN.predict(self, X)
            y_grad = None

        # conv to numpy
        y_mu = y_mu_.numpy().flatten()

        if self._norm_y:
            y_mu = self._norm_y_mu + self._norm_y_std * y_mu

        out = [y_mu]
        if return_std:
            # conv to numpy
            y_var = y_var_.numpy().flatten()
            y_std = np.sqrt(y_var)

            if self._norm_y:
                y_std = y_std * self._norm_y_std
            out.append(y_std)

        if return_grad:

            # undo normalization
            if self._norm_y:
                y_grad = y_grad * self._norm_y_std
            out.append(y_grad)

        if len(out) == 1:
            return out[0]
        return out

    def predict_grad(self, X: ARRAY_LIKE_2D) -> np.ndarray:
        """
        Return gradient at X

        Parameters
        ----------
        X: arr_like, shape=(n_samples, n_features)
            predict gradient of y for given X, dy/dx

        Returns
        -------
        y_mean_grad : shape = (n_samples, n_features)
            gradient of y, dy/dx
        """
        _, y_grad = self.predict(X, return_grad=True)
        return y_grad

    def loo(
        self,
        X: ARRAY_LIKE_2D = None,
        y: ARRAY_LIKE_1D = None,
        return_var: bool = False,
        squared: bool = False,
    ) -> np.ndarray:
        """
        Replacement for the Leave one out Cross validation error (LOO).
        Calculate the absolute error (y - y_pred)**2 instead

        If X is None or y is None, the squared trainings error is returned
        -> (y_train - y_pred)**2

        Parameters
        ----------
        X: array-like, shape = (n_samples, n_features), optional (default=None)
            Query points where the error is evaluated, if None -> squared error is
            returned for the trainings samples
        y: array-like, shape = (n_samples), optional (default=None)
            responses for X, if None -> squared error is returned for the trainings
            samples
        squared
        return_var

        Returns
        -------
        error: np.ndarray, shape = (n_samples)
            squared error
        """
        if X is None or y is None:
            X, y = self.X_train_, self.y_train_
        else:
            X, y = self.check_input(X, y)

        y_pred = self.predict(X)

        if return_var:
            raise NotImplementedError()

        err = (y.flatten() - y_pred) ** 2

        if not squared:
            err = np.sqrt(err)
        return err

    def save(self, path: Optional[str] = None) -> Tuple[dict, str]:
        """
        Save model to given path

        Parameters
        ----------
        path: str, optional (default=None)
            The filepath where to save the model including the name.
        """
        if path is None:
            # temporary path to store trained models
            temp_path = os.path.dirname(os.path.abspath(__file__)) + r"\temp_bnn_hot"

            if not os.path.exists(temp_path):  # create temp folder if not existing
                os.makedirs(temp_path)

            path = os.path.join(temp_path, "bnn_model_%s.dat" % self._rnd_state)

        else:
            path = path + ".dat"

        if self.finalized:

            data: dict = {
                "norm_y_mu": self._norm_y_mu,
                "norm_y_std": self._norm_y_std,
                "X_train_": self.X_train_,
                "y_train_": self.y_train_,
                "config": self.config,
                "params": self.params,
            }

            with open(path, "wb") as f:
                pickle.dump(data, f)

            return data, path
        else:
            raise ValueError("Model should be finalized first.")

    def load(self, path: str):

        assert isinstance(path, str)

        base_path, file_extension = os.path.splitext(path)
        if file_extension != ".dat":
            path = base_path + ".dat"

        if os.path.isfile(path):

            with open(path, "rb") as f:
                data: dict = pickle.load(f)

            # set parameters
            try:
                self.X_train_ = data["X_train_"]
                self.y_train_ = data["y_train_"]
                self._norm_y_mu = data["norm_y_mu"]
                self._norm_y_std = data["norm_y_std"]

                # build model
                self._n_dim_input = self.X_train_.shape[1]
                self.config = data["config"]
                self._create_net(self.config)
                self.scaler.fit(self.X_train_)
                self.params = data["params"]

                return self

            except Exception:
                raise ValueError("Loading model failed.")
        else:
            raise ValueError("File not found. Given path: %s" % path)
