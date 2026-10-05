import torch
import numpy as np


_MODELS = {}


def register_model(cls=None, *, name=None):

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _MODELS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _MODELS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_model(name):
    return _MODELS[name]


def get_sigmas(config):

    sigmas = np.exp(
        np.linspace(
            np.log(config.model.sigma_max),
            np.log(config.model.sigma_min),
            config.model.num_scales,
        )
    )

    return sigmas


def create_model(config):
    model_name = config.model.name
    score_model = get_model(model_name)(config)
    score_model = score_model.to(config.device)

    num_params = 0
    for p in score_model.parameters():
        num_params += p.numel()
    print("Number of Parameters in the Score Model:", num_params)

    score_model = torch.nn.DataParallel(score_model)
    return score_model


def get_model_fn(model, train=False):

    def model_fn(x, labels):
        if not train:
            model.eval()
            return model(x, labels)
        else:
            model.train()
            return model(x, labels)

    return model_fn


def to_flattened_numpy(x):
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    return torch.from_numpy(x.reshape(shape))
