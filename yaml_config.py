import yaml


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries = entries

    def print(self):
        print(self.entries)


def load_config_from_yaml(file_path):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return Config(**config)


def write_config_to_yaml(config, file_path):
    with open(file_path, "w") as file:
        yaml.safe_dump(config, file)
