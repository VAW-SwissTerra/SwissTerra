import argparse

from . import files


def main():
    args = parse_args()
    args.func(args)


def parse_args():
    parser = argparse.ArgumentParser(description="Processing of SwissTerra archival photographs")

    commands = parser.add_subparsers(dest="commands")
    commands.required = True

    cache_parser = commands.add_parser("cache")
    cache_parser.add_argument("action", choices=["clear", "list"])
    cache_parser.set_defaults(func=cache_commands)

    files_parser = commands.add_parser("files")
    files_parser.add_argument("action", choices=["check"])
    files_parser.set_defaults(func=files_commands)

    overview_parser = commands.add_parser("overview")
    overview_parser.set_defaults(func=overview_commands)

    return parser.parse_args()


def overview_commands(args):
    print("Overview!")


def cache_commands(args):

    if args.action == "clear":
        files.clear_cache()
    elif args.action == "list":
        files.list_cache()


def files_commands(args):

    if args.action == "check":
        files.check_data()
