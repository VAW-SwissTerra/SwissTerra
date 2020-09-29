import argparse


def main():
    args = parse_args()
    args.func(args)


def parse_args():
    parser = argparse.ArgumentParser(description="Processing of SwissTerra archival photographs")

    commands = parser.add_subparsers(dest="commands")
    commands.required = True

    overview_parser = commands.add_parser("overview")
    overview_parser.set_defaults(func=overview)

    return parser.parse_args()


def overview(args):
    print("Overview!")
