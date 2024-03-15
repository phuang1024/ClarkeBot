import argparse

from train import train


def main():
    parser = argparse.ArgumentParser()
    subp = parser.add_subparsers(dest="action")

    train = subp.add_parser("train")
    train.add_argument("--outputs", default="outputs", help="Output directory")
    train.add_argument("--config", required=True, help="Path to config file")

    args = parser.parse_args()

    if args.action == "train":
        train(args)


if __name__ == "__main__":
    main()
