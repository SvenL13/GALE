"""
GALe - Global Adaptive Learning
@author: Sven Lämmle
"""
import numpy as np
import matplotlib.pyplot as plt

from gale.doe import SeqED, adaptive_sampling
from gale.experiments import bench_functions as bf

seed = 31

# benchmark function
bench = bf.benchmarks["GramLee_norm"]

# maximum number of samples, if not stated otherwise we use 10*n_dim initial samples,
# i.e. we sample from the GramLee 25 point with the selected acquisition function
n_max = 35

# sampling
method = adaptive_sampling["GUESS"](bounds=bench.bounds, rnd_state=seed, model="GP")
doe = SeqED(fun=bench.run, adaptive_method=method, n_calls=n_max)

res = doe.run()

n_init = doe.n_init
model = res["model"]
x_train = res["obs_cond"]
y_train = res["obs_y"]

# visualize results from sampling
fig, ax = plt.subplots(figsize=(15, 3))
x = np.linspace(0, 1.0, 200)
x_init = x_train[:n_init]
x_adapt = x_train[n_init:]
y_pred, y_pred_std = model.predict(x[:, None], return_std=True)
ax.plot()
y_test = bench.run(x).flatten()
y_pred = y_pred.flatten()
y_pred_std = y_pred_std.flatten()

ax.fill_between(
    x,
    y_pred - y_pred_std,
    y_pred + y_pred_std,
    alpha=0.2,
    label=r"Model Uncertainty $\sigma_Y$",
)
ax.plot(x, y_test, label=r"Target $f$", ls="--", color="k", lw=2)
ax.plot(x, y_pred, label=r"Model $\hat{f}$", lw=2)

ax.plot(
    x_train[:n_init, 0],
    bench.run(x_train[:n_init, 0]).flatten(),
    "o",
    color="k",
    markersize=8,
    label="Initial Sample",
)

ax.plot(
    x_train[n_init:, 0],
    bench.run(x_train[n_init:, 0]).flatten(),
    "o",
    markersize=8,
    label="Adaptive Sample",
)

ax.set_xlim(0.0, 1.0)
ax.set_ylabel(r"$\mathrm{Y}$")
plt.legend()
plt.tight_layout()
plt.show()
