import json


def read_logfile(filename):
    with open(filename) as file:
        return [json.loads(line) for line in file]
