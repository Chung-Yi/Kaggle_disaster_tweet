import configparser

class ConfigParser:

    def __init__(self, config_path):

        self.config = configparser.ConfigParser()
        self.config.read(config_path)

        print(self.config['data_loader']['MAX_LEN'])
