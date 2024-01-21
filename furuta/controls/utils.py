import json


def read_parameters_file(path: str):
    with open(path) as f:
        parameters = json.load(f)
    return parameters
