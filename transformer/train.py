import json


def train(args):
    with open(args.config, "r") as f:
        config = json.load(f)
