"""
GALe - Global Adaptive Learning
@author: Sven Lämmle

code is copied from
https://github.com/kchua/handful-of-trials/blob/master/dmbrl/modeling/models/BNN.py
https://github.com/kchua/handful-of-trials/blob/master/dmbrl/modeling/utils/TensorStandardScaler.py
https://github.com/kchua/handful-of-trials/blob/master/dmbrl/modeling/layers/FC.py
https://github.com/kchua/handful-of-trials/blob/master/dmbrl/misc/DotmapUtils.py

# published under MIT license
https://github.com/kchua/handful-of-trials/blob/master/LICENSE

BNN adjusted for TensorFlow 2
"""
import numpy as np
import tensorflow as tf
from tqdm import trange

FLOAT_tf = tf.float64


def get_required_argument(dotmap, key, message, default=None):
    val = dotmap.get(key, default)
    if val is default:
        raise ValueError(message)
    return val


class TensorStandardScaler:
    """Helper class for automatically normalizing inputs into the network."""

    def __init__(self, x_dim: int):
        """
        Initializes a scaler.

        Parameters
        ----------
        x_dim: int
            The dimensionality of the inputs into the scaler.
        """
        self.fitted = False
        self.cached_mu, self.cached_sigma = np.zeros([1, x_dim]), np.ones([1, x_dim])
        self.mu = tf.Variable(
            self.cached_mu, name="scaler_mu", shape=[1, x_dim], trainable=False
        )
        self.sigma = tf.Variable(
            self.cached_sigma, name="scaler_std", shape=[1, x_dim], trainable=False
        )

    def fit(self, data):
        """Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.
        Arguments:
        data (np.ndarray): A numpy array containing the input
        Returns: None.
        """
        mu = np.mean(data, axis=0, keepdims=True)
        sigma = np.std(data, axis=0, keepdims=True)
        sigma[sigma < 1e-12] = 1.0

        self.mu.assign(mu)
        self.sigma.assign(sigma)
        self.fitted = True
        self.cache()

    def transform(self, data):
        """Transforms the input matrix data using the parameters of this scaler.
        Arguments:
        data (np.array): A numpy array containing the points to be transformed.
        Returns: (np.array) The transformed dataset.
        """
        return (data - self.mu) / self.sigma

    def inverse_transform(self, data):
        """Undoes the transformation performed by this scaler.
        Arguments:
        data (np.array): A numpy array containing the points to be transformed.
        Returns: (np.array) The transformed dataset.
        """
        return self.sigma * data + self.mu

    def get_vars(self):
        """Returns a list of variables managed by this object.
        Returns: (list<tf.Variable>) The list of variables.
        """
        return [self.mu, self.sigma]

    def cache(self):
        """Caches current values of this scaler.
        Returns: None.
        """
        self.cached_mu = self.mu.numpy()
        self.cached_sigma = self.sigma.numpy()

    def load_cache(self):
        """Loads values from the cache
        Returns: None.
        """
        self.mu.assign(self.cached_mu)
        self.sigma.assign(self.cached_sigma)


class FC:
    """Represents a fully-connected layer in a network."""

    _activations = {
        None: tf.identity,
        "ReLU": tf.nn.relu,
        "tanh": tf.tanh,
        "sigmoid": tf.sigmoid,
        "softmax": tf.nn.softmax,
        "swish": lambda x: x * tf.sigmoid(x),
    }

    def __init__(
        self,
        output_dim,
        input_dim=None,
        activation=None,
        weight_decay=None,
        ensemble_size=1,
    ):
        """Initializes a fully connected layer.
        Arguments:
            output_dim: (int) The dimensionality of the output of this layer.
            input_dim: (int/None) The dimensionality of the input of this layer.
            activation: (str/None) The activation function applied on the outputs.
                                    See FC._activations to see the list of allowed strings.
                                    None applies the identity function.
            weight_decay: (float) The rate of weight decay applied to the weights of this layer.
            ensemble_size: (int) The number of networks in the ensemble within which this layer will be used.
        """
        # Set layer parameters
        self.input_dim, self.output_dim = input_dim, output_dim
        self.activation = activation
        self.weight_decay = weight_decay
        self.ensemble_size = ensemble_size

        # Initialize internal state
        self.variables_constructed = False
        self.weights, self.biases = None, None
        self.decays = None

    def __repr__(self):
        return "FC(output_dim={!r}, input_dim={!r}, activation={!r}, weight_decay={!r}, ensemble_size={!r})".format(
            self.output_dim,
            self.input_dim,
            self.activation,
            self.weight_decay,
            self.ensemble_size,
        )

    #######################
    # Basic Functionality #
    #######################

    def compute_output_tensor(self, input_tensor):
        """Returns the resulting tensor when all operations of this layer are applied to input_tensor.
        If input_tensor is 2D, this method returns a 3D tensor representing the output of each
        layer in the ensemble on the input_tensor. Otherwise, if the input_tensor is 3D, the output
        is also 3D, where output[i] = layer_ensemble[i](input[i]).
        Arguments:
            input_tensor: (tf.Tensor) The input to the layer.
        Returns: The output of the layer, as described above.
        """
        # Get raw layer outputs
        if len(input_tensor.shape) == 2:
            raw_output = (
                tf.einsum("ij,ajk->aik", input_tensor, self.weights) + self.biases
            )
        elif (
            len(input_tensor.shape) == 3 and input_tensor.shape[0] == self.ensemble_size
        ):
            raw_output = tf.matmul(input_tensor, self.weights) + self.biases
        else:
            raise ValueError("Invalid input dimension.")

        # Apply activations if necessary
        return FC._activations[self.activation](raw_output)

    def get_decays(self):
        """Returns the list of losses corresponding to the weight decay imposed on each weight of the
        network.
        Returns: the list of weight decay losses.
        """
        return self.decays

    def copy(self, sess=None):
        """Returns a Layer object with the same parameters as this layer.
        Arguments:
            sess: (tf.Session/None) session containing the current values of the variables to be copied.
                  Must be passed in to copy values.
            copy_vals: (bool) Indicates whether variable values will be copied over.
                       Ignored if the variables of this layer has not yet been constructed.
        Returns: The copied layer.
        """
        new_layer = eval(repr(self))
        return new_layer

    #########################################################
    # Methods for controlling internal Tensorflow variables #
    #########################################################

    def construct_vars(self):
        """Constructs the variables of this fully-connected layer.
        Returns: None
        """
        if (
            self.variables_constructed
        ):  # Ignore calls to this function once variables are constructed.
            return
        if self.input_dim is None or self.output_dim is None:
            raise RuntimeError(
                "Cannot construct variables without fully specifying input and output dimensions."
            )

        # Construct variables
        self.weights = tf.Variable(
            tf.random.truncated_normal(
                [self.ensemble_size, self.input_dim, self.output_dim],
                stddev=1 / (2 * np.sqrt(self.input_dim)),
                dtype=FLOAT_tf,
            ),
            name="FC_weights",
            shape=[self.ensemble_size, self.input_dim, self.output_dim],
        )
        self.biases = tf.Variable(
            tf.zeros([self.ensemble_size, 1, self.output_dim], dtype=FLOAT_tf),
            name="FC_biases",
            shape=[self.ensemble_size, 1, self.output_dim],
        )

        if self.weight_decay is not None:
            self.decays = [
                tf.multiply(
                    self.weight_decay, tf.nn.l2_loss(self.weights), name="weight_decay"
                )
            ]
        self.variables_constructed = True

    def get_vars(self):
        """Returns the variables of this layer."""
        return [self.weights, self.biases]

    ########################################
    # Methods for setting layer parameters #
    ########################################

    def get_input_dim(self):
        """Returns the dimension of the input.
        Returns: The dimension of the input
        """
        return self.input_dim

    def set_input_dim(self, input_dim):
        """Sets the dimension of the input.
        Arguments:
            input_dim: (int) The dimension of the input.
        Returns: None
        """
        if self.variables_constructed:
            raise RuntimeError("Variables already constructed.")
        self.input_dim = input_dim

    def get_output_dim(self):
        """Returns the dimension of the output.
        Returns: The dimension of the output.
        """
        return self.output_dim

    def set_output_dim(self, output_dim):
        """Sets the dimension of the output.
        Arguments:
            output_dim: (int) The dimension of the output.
        Returns: None.
        """
        if self.variables_constructed:
            raise RuntimeError("Variables already constructed.")
        self.output_dim = output_dim

    def get_activation(self, as_func=True):
        """Returns the current activation function for this layer.
        Arguments:
            as_func: (bool) Determines whether the returned value is the string corresponding
                     to the activation function or the activation function itself.
        Returns: The activation function (string/function, see as_func argument for details).
        """
        if as_func:
            return FC._activations[self.activation]
        else:
            return self.activation

    def set_activation(self, activation):
        """Sets the activation function for this layer.
        Arguments:
            activation: (str) The activation function to be used.
        Returns: None.
        """
        if self.variables_constructed:
            raise RuntimeError("Variables already constructed.")
        self.activation = activation

    def unset_activation(self):
        """Removes the currently set activation function for this layer.
        Returns: None
        """
        if self.variables_constructed:
            raise RuntimeError("Variables already constructed.")
        self.set_activation(None)

    def get_weight_decay(self):
        """Returns the current rate of weight decay set for this layer.
        Returns: The weight decay rate.
        """
        return self.weight_decay

    def set_weight_decay(self, weight_decay):
        """Sets the current weight decay rate for this layer.
        Returns: None
        """
        self.weight_decay = weight_decay
        if self.variables_constructed:
            if self.weight_decay is not None:
                self.decays = [
                    tf.multiply(
                        self.weight_decay,
                        tf.nn.l2_loss(self.weights),
                        name="weight_decay",
                    )
                ]

    def unset_weight_decay(self):
        """Removes weight decay from this layer.
        Returns: None
        """
        self.set_weight_decay(None)
        if self.variables_constructed:
            self.decays = []

    def set_ensemble_size(self, ensemble_size):
        if self.variables_constructed:
            raise RuntimeError("Variables already constructed.")
        self.ensemble_size = ensemble_size

    def get_ensemble_size(self):
        return self.ensemble_size


class BNN(tf.Module):
    """Neural network models which model aleatoric uncertainty (and possibly epistemic uncertainty
    with ensembling).
    """

    def __init__(self, num_nets=5):
        """Initializes a class instance.
        Arguments:
            params (DotMap): A dotmap of model parameters.
                .name (str): Model name, used for logging/use in variable scopes.
                    Warning: Models with the same name will overwrite each other.
                .num_networks (int): (optional) The number of networks in the ensemble. Defaults to 1.
                    Ignored if model is being loaded.
                .model_dir (str/None): (optional) Path to directory from which model will be loaded, and
                    saved by default. Defaults to None.
                .load_model (bool): (optional) If True, model will be loaded from the model directory,
                    assuming that the files are generated by a model of the same name. Defaults to False.
                .sess (tf.Session/None): The session that this model will use.
                    If None, creates a session with its own associated graph. Defaults to None.
        """
        super().__init__(name="BNN")
        super(BNN, self).__init__()

        # Instance variables
        self.finalized = False
        self.layers, self.max_logvar, self.min_logvar = [], None, None
        self.decays, self.optvars, self.nonoptvars = [], [], []
        self.end_act, self.end_act_name = None, None
        self.scaler = None

        # Training objects
        self.optimizer = None

        self.num_nets = num_nets
        self.model_loaded = False

        # if self.num_nets == 1:
        #     print("Created a neural network with variance predictions.")
        # else:
        #     print("Created an ensemble of %d neural networks with variance predictions." % (self.num_nets))

    @property
    def is_probabilistic(self):
        return True

    @property
    def is_tf_model(self):
        return True

    ###################################
    # Network Structure Setup Methods #
    ###################################

    def add(self, layer):
        """Adds a new layer to the network.
        Arguments:
            layer: (layer) The new layer to be added to the network.
                   If this is the first layer, the input dimension of the layer must be set.
        Returns: None.
        """
        if self.finalized:
            raise RuntimeError("Cannot modify network structure after finalizing.")
        if len(self.layers) == 0 and layer.get_input_dim() is None:
            raise ValueError("Must set input dimension for the first layer.")
        if self.model_loaded:
            raise RuntimeError("Cannot add layers to a loaded model.")

        layer.set_ensemble_size(self.num_nets)
        if len(self.layers) > 0:
            layer.set_input_dim(self.layers[-1].get_output_dim())
        self.layers.append(layer.copy())

    def pop(self):
        """Removes and returns the most recently added layer to the network.
        Returns: (layer) The removed layer.
        """
        if len(self.layers) == 0:
            raise RuntimeError("Network is empty.")
        if self.finalized:
            raise RuntimeError("Cannot modify network structure after finalizing.")
        if self.model_loaded:
            raise RuntimeError("Cannot remove layers from a loaded model.")

        return self.layers.pop()

    def finalize(self, optimizer, optimizer_args=None, *args, **kwargs):
        """Finalizes the network.
        Arguments:
            optimizer: (tf.train.Optimizer) An optimizer class from those available at tf.train.Optimizer.
            optimizer_args: (dict) A dictionary of arguments for the __init__ method of the chosen optimizer.
        Returns: None
        """
        if len(self.layers) == 0:
            raise RuntimeError("Cannot finalize an empty network.")
        if self.finalized:
            raise RuntimeError("Can only finalize a network once.")

        optimizer_args = {} if optimizer_args is None else optimizer_args
        self.optimizer = optimizer(**optimizer_args)

        # Add variance output.
        self.layers[-1].set_output_dim(2 * self.layers[-1].get_output_dim())

        # Remove last activation to isolate variance from activation function.
        self.end_act = self.layers[-1].get_activation()
        self.end_act_name = self.layers[-1].get_activation(as_func=False)
        self.layers[-1].unset_activation()

        # Construct all variables.
        self.scaler = TensorStandardScaler(self.layers[0].get_input_dim())
        self.max_logvar = tf.Variable(
            np.ones([1, self.layers[-1].get_output_dim() // 2]) / 2.0,
            dtype=FLOAT_tf,
            name="max_log_var",
        )
        self.min_logvar = tf.Variable(
            -np.ones([1, self.layers[-1].get_output_dim() // 2]) * 10.0,
            dtype=FLOAT_tf,
            name="min_log_var",
        )
        for i, layer in enumerate(self.layers):
            layer.construct_vars()
            self.decays.extend(layer.get_decays())
            self.optvars.extend(layer.get_vars())
        self.optvars.extend([self.max_logvar, self.min_logvar])
        self.nonoptvars.extend(self.scaler.get_vars())

        # Set up training
        self.optimizer = optimizer(**optimizer_args)
        self.finalized = True

    #################
    # Model Methods #
    #################

    def loss(self, inputs, targets):
        train_loss = tf.reduce_sum(
            self._compile_losses(inputs, targets, inc_var_loss=True)
        )
        train_loss += tf.add_n(self.decays)
        train_loss += 0.01 * tf.reduce_sum(self.max_logvar) - 0.01 * tf.reduce_sum(
            self.min_logvar
        )
        return train_loss

    def mse_loss(self, inputs, targets):
        return self._compile_losses(inputs, targets, inc_var_loss=False).numpy()

    def fit(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        batch_size: int = 32,
        epochs: int = 100,
        hide_progress: bool = False,
        holdout_ratio: float = 0.0,
        max_logging: int = 5000,
    ):
        """Trains/Continues network training
        Arguments:
            inputs (np.ndarray): Network inputs in the training dataset in rows.
            targets (np.ndarray): Network target outputs in the training dataset in rows corresponding
                to the rows in inputs.
            batch_size (int): The minibatch size to be used for training.
            epochs (int): Number of epochs (full network passes that will be done.
            hide_progress (bool): If True, hides the progress bar shown at the beginning of training.
        Returns: None
        """

        def shuffle_rows(arr):
            idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:, None], idxs]

        # Split into training and holdout sets
        num_holdout = min(int(inputs.shape[0] * holdout_ratio), max_logging)
        permutation = np.random.permutation(inputs.shape[0])
        inputs, holdout_inputs = (
            inputs[permutation[num_holdout:]],
            inputs[permutation[:num_holdout]],
        )
        targets, holdout_targets = (
            targets[permutation[num_holdout:]],
            targets[permutation[:num_holdout]],
        )
        holdout_inputs = np.tile(holdout_inputs[None], [self.num_nets, 1, 1])
        holdout_targets = np.tile(holdout_targets[None], [self.num_nets, 1, 1])

        self.scaler.fit(inputs)

        idxs = np.random.randint(inputs.shape[0], size=[self.num_nets, inputs.shape[0]])
        if hide_progress:
            epoch_range = range(epochs)
        else:
            epoch_range = trange(epochs, unit="epoch(s)", desc="Network training")
        for _ in epoch_range:
            for batch_num in range(int(np.ceil(idxs.shape[-1] / batch_size))):
                batch_idxs = idxs[
                    :, batch_num * batch_size : (batch_num + 1) * batch_size
                ]

                _loss = lambda: self.loss(inputs[batch_idxs], targets[batch_idxs])
                self.optimizer.minimize(_loss, self.optvars)

            idxs = shuffle_rows(idxs)
            # tf.print(inputs[idxs[:, :max_logging]].shape)
            if not hide_progress:
                if holdout_ratio < 1e-12:
                    epoch_range.set_postfix(
                        {
                            "Training loss(es)": self.mse_loss(
                                inputs[idxs[:, :max_logging]],
                                targets[idxs[:, :max_logging]],
                            )
                        }
                    )
                else:
                    epoch_range.set_postfix(
                        {
                            "Training loss(es)": self.mse_loss(
                                inputs[idxs[:, :max_logging]],
                                targets[idxs[:, :max_logging]],
                            ),
                            "Holdout loss(es)": self.mse_loss(
                                holdout_inputs, holdout_targets
                            ),
                        }
                    )

    def predict(self, inputs, factored=False, *args, **kwargs):
        """Returns the distribution predicted by the model for each input vector in inputs.
        Behavior is affected by the dimensionality of inputs and factored as follows:
        inputs is 2D, factored=True: Each row is treated as an input vector.
            Returns a mean of shape [ensemble_size, batch_size, output_dim] and variance of shape
            [ensemble_size, batch_size, output_dim], where N(mean[i, j, :], diag([i, j, :])) is the
            predicted output distribution by the ith model in the ensemble on input vector j.
        inputs is 2D, factored=False: Each row is treated as an input vector.
            Returns a mean of shape [batch_size, output_dim] and variance of shape
            [batch_size, output_dim], where aggregation is performed as described in the paper.
        inputs is 3D, factored=True/False: Each row in the last dimension is treated as an input vector.
            Returns a mean of shape [ensemble_size, batch_size, output_dim] and variance of sha
            [ensemble_size, batch_size, output_dim], where N(mean[i, j, :], diag([i, j, :])) is the
            predicted output distribution by the ith model in the ensemble on input vector [i, j].
        Arguments:
            inputs (np.ndarray): An array of input vectors in rows. See above for behavior.
            factored (bool): See above for behavior.
        """
        factored_mean, factored_variance = self.create_prediction_tensors(
            inputs, factored=True
        )
        mean = tf.reduce_mean(factored_mean, axis=0)
        var = tf.reduce_mean(factored_variance, axis=0) + tf.reduce_mean(
            tf.square(factored_mean - mean), axis=0
        )
        var = tf.linalg.diag(var)
        return mean, var

    def create_prediction_tensors(self, inputs, factored=False, *args, **kwargs):
        """See predict() above for documentation."""
        factored_mean, factored_variance = self._compile_outputs(inputs)
        if len(inputs.shape) == 2 and not factored:
            mean = tf.reduce_mean(factored_mean, axis=0)
            variance = tf.reduce_mean(
                tf.square(factored_mean - mean), axis=0
            ) + tf.reduce_mean(factored_variance, axis=0)
            return mean, variance
        return factored_mean, factored_variance

    def save(self, savedir=None):
        """Saves all information required to recreate this model in two files in savedir
        (or self.model_dir if savedir is None), one containing the model structure and the other
        containing all variables in the network.
        savedir (str): (Optional) Path to which files will be saved. If not provided, self.model_dir
            (the directory provided at initialization) will be used.
        """
        if not self.finalized:
            raise RuntimeError()

        params = [v.numpy() for v in self.trainable_variables]
        return params

    #######################
    # Compilation methods #
    #######################

    def _compile_outputs(self, inputs, ret_log_var=False):
        """Compiles the output of the network at the given inputs.
        If inputs is 2D, returns a 3D tensor where output[i] is the output of the ith network in the ensemble.
        If inputs is 3D, returns a 3D tensor where output[i] is the output of the ith network on the ith input matrix.
        Arguments:
            inputs: (tf.Tensor) A tensor representing the inputs to the network
            ret_log_var: (bool) If True, returns the log variance instead of the variance.
        Returns: (tf.Tensors) The mean and variance/log variance predictions at inputs for each network
            in the ensemble.
        """
        dim_output = self.layers[-1].get_output_dim()
        cur_out = self.scaler.transform(inputs)
        for layer in self.layers:
            cur_out = layer.compute_output_tensor(cur_out)

        mean = cur_out[:, :, : dim_output // 2]
        if self.end_act is not None:
            mean = self.end_act(mean)

        logvar = self.max_logvar - tf.nn.softplus(
            self.max_logvar - cur_out[:, :, dim_output // 2 :]
        )
        logvar = self.min_logvar + tf.nn.softplus(logvar - self.min_logvar)

        if ret_log_var:
            return mean, logvar
        else:
            return mean, tf.exp(logvar)

    def _compile_losses(self, inputs, targets, inc_var_loss=True):
        """Helper method for compiling the loss function.
        The loss function is obtained from the log likelihood, assuming that the output
        distribution is Gaussian, with both mean and (diagonal) covariance matrix being determined
        by network outputs.
        Arguments:
            inputs: (tf.Tensor) A tensor representing the input batch
            targets: (tf.Tensor) The desired targets for each input vector in inputs.
            inc_var_loss: (bool) If True, includes log variance loss.
        Returns: (tf.Tensor) A tensor representing the loss on the input arguments.
        """
        mean, log_var = self._compile_outputs(inputs, ret_log_var=True)
        inv_var = tf.exp(-log_var)

        if inc_var_loss:
            mse_losses = tf.reduce_mean(
                tf.reduce_mean(tf.square(mean - targets) * inv_var, axis=-1), axis=-1
            )
            var_losses = tf.reduce_mean(tf.reduce_mean(log_var, axis=-1), axis=-1)
            total_losses = mse_losses + var_losses
        else:
            total_losses = tf.reduce_mean(
                tf.reduce_mean(tf.square(mean - targets), axis=-1), axis=-1
            )

        return total_losses
