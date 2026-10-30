import yaml

def load_config(path):
    """Load yaml file"""
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

