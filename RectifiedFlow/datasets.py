def get_data_scaler(config):
    if config.data.centered:
        return lambda x: x * 2.0 - 1.0
    else:
        return lambda x: x


def get_data_inverse_scaler(config):
    if config.data.centered:
        return lambda x: (x + 1.0) / 2.0
    else:
        return lambda x: x
