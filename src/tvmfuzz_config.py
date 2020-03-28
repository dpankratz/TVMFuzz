from configparser import ConfigParser
from ast import literal_eval

config_parser = ConfigParser()
config_parser.read("../settings/tvmfuzz_settings.ini")

class TVMFuzzConfig(object):
	def __init__(self,parameters):
		for name,value in parameters.items():
			setattr(TVMFuzzConfig,name,literal_eval(value))

parameters = {}
for section in config_parser.sections():
	for option in config_parser.options(section):
		parameters[option] = config_parser.get(section,option)

TVMFuzzConfig(parameters)
