import json


class JsonParser:
    """This class is used by training and testing entrypoints to provide proper
    configurations from according .json files."""
    def __init__(self):
        pass

    @staticmethod
    def read_config(json_file) -> dict:
        with open(json_file, "r") as config_file:
            json_dict = json.load(config_file)
        return json_dict
