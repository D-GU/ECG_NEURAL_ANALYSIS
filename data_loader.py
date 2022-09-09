from numpy import load


def get_data(path: str):
    return load(path, allow_pickle=True)

