import logging

class Logger(object):
    def __init__(self, name, level=logging.DEBUG, path="logfile.log"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(level)

        self.file_handler = logging.FileHandler(path)
        self.file_handler.setLevel(level)

        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.file_handler.setFormatter(self.formatter)
        self.console_handler.setFormatter(self.formatter)

        self.logger.addHandler(self.console_handler)
        self.logger.addHandler(self.file_handler)