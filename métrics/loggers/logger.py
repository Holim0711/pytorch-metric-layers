import os
import json
import logging


class SimpleLogger():

    def __init__(self, *path):
        self.filename = os.path.join(*[str(x) for x in path]) + '.log'

        if os.path.exists(self.filename):
            raise Exception(f"{self.filename} already exists.")

        try:
            os.makedirs(os.path.dirname(self.filename))
        except OSError:
            pass

        self.logger = logging.getLogger(self.filename)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.FileHandler(self.filename))

    def write(self, obj):
        self.logger.info(json.dumps(obj))
