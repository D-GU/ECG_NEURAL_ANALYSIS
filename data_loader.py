from numpy import load


def get_data(_path: str):
    return load(_path, allow_pickle=True)

