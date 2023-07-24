"""
GALe - Global Adaptive Learning
@author: Sven Lämmle

Experiments - Configuration
"""
# default configuration GP & BNN
EX_GP_CONFIG = {
    "bench_config": "all",
    "ex": {
        "repetitions": 10,
        "save_pred": False,
        "save_pred_sep": False,
    },
    "eval": {
    },
}

EX_SGP_CONFIG = {
    "bench_config": "High",
    "ex": {
        "repetitions": 10,
        "save_pred": False,
        "save_pred_sep": False,
        "fit_model_every": 2,
    },
    "eval": {
    },
}

net_top1 = [
    {"neurons": 64, "activation": "swish", "weight_decay": 0.0001},
    {"neurons": 1, "weight_decay": 0.0005},
]
net_config1 = {"net_config": net_top1, "lr": 0.03}

net_top2 = [
    {"neurons": 48, "activation": "tanh", "weight_decay": 0.0001},
    {"neurons": 1, "weight_decay": 0.0005},
]
net_config2 = {"net_config": net_top2, "lr": 0.03}

EX_BNN_CONFIG = {
    "bench_config": "4D",
    "ex": {
        "repetitions": 10,
        "save_pred": False,
        "save_pred_sep": False,
        "model_param": {
            "Stybli4d": {"net_config": net_config1},
            "Rose4d": {"net_config": net_config1},
            "Ackley4d": {"net_config": net_config1},
            "Micha4d": {"net_config": net_config2},
        },
    },
    "eval": {
    },
}
