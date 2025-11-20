import configparser

def readConfig(fileName):
    config = configparser.ConfigParser()
    config.read(fileName)
    return config
