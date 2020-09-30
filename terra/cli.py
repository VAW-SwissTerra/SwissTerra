import argparse

from terra import files, metadata


def main():
    """CLI entry point."""
    args = parse_args()
    args.func(args)


def parse_args():
    """Parse the arguments given through the CLI."""
    parser = argparse.ArgumentParser(description="Processing of SwissTerra archival photographs")

    # Subcommands delimit different modules or themes.
    commands = parser.add_subparsers(dest="commands")
    commands.required = True

    # Subcommands for cache (temp folder) handling
    cache_parser = commands.add_parser("cache")
    cache_parser.add_argument("action", choices=["clear", "list"])
    cache_parser.set_defaults(func=cache_commands)

    # Subcommands for input file handling
    files_parser = commands.add_parser("files")
    files_parser.add_argument("action", choices=["check"])
    files_parser.set_defaults(func=files_commands)

    # Subcommands for the overview module
    overview_parser = commands.add_parser("overview")
    overview_parser.add_argument("action", choices=["capture-dates"])
    overview_parser.set_defaults(func=overview_commands)

    # Subcommands for metadata handling
    metadata_parser = commands.add_parser("metadata")
    metadata_parser.add_argument("action", choices=["collect-files", "show"])
    metadata_parser.set_defaults(func=metadata_commands)

    return parser.parse_args()


def overview_commands(args):
    """Run the overview module."""
    if args.action == "capture-dates":
        metadata.overview.get_capture_date_distribution()

    print("Overview!")


def cache_commands(args):
    """Run the cache subcommands."""
    if args.action == "clear":
        files.clear_cache()
    elif args.action == "list":
        files.list_cache()


def files_commands(args):
    """Run the file handling subcommands."""

    if args.action == "check":
        files.check_data()


def metadata_commands(args):
    """Run the metadata handling subcommands."""
    if args.action == "collect-files":
        metadata.image_meta.collect_metadata(use_cached=False)
    elif args.action == "show":
        meta_file = metadata.image_meta.collect_metadata(use_cached=True)
        print(meta_file)
        print(meta_file.describe())
        print(meta_file.dtypes)
