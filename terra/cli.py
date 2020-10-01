import argparse
import subprocess

from terra import fiducials, files, metadata


def main():
    """CLI entry point."""
    args = parse_args()
    args.func(args)


def parse_args():
    """Parse the arguments given through the CLI."""
    parser = argparse.ArgumentParser(prog="terra", description="Processing of SwissTerra archival photographs",
                                     formatter_class=argparse.RawTextHelpFormatter)

    # Subcommands delimit different modules or themes.
    commands = parser.add_subparsers(dest="commands", metavar="module")
    commands.required = True

    # Subcommands for cache (temp folder) handling
    choices = {
        "clear": "Remove everything in the cache folder.",
        "list": "List files in the cache"
    }
    cache_parser = commands.add_parser(
        "cache", formatter_class=argparse.RawTextHelpFormatter, help="Handle the cache (temporary files).")
    cache_parser.add_argument("action",
                              help=generate_help_text(choices), metavar="action", choices=choices.keys())
    cache_parser.set_defaults(func=cache_commands)

    # Subcommands for input file handling
    choices = {
        "check": "Check that all files can be found and have the correct file type."
    }
    files_parser = commands.add_parser(
        "files", formatter_class=argparse.RawTextHelpFormatter, help="Handle input files.")
    files_parser.add_argument("action",
                              help=generate_help_text(choices), metavar="action", choices=choices.keys())
    files_parser.set_defaults(func=files_commands)

    # Subcommands for the overview module
    choices = {
        "capture-dates": "Plot the dates for when photographs were captured"
    }
    overview_parser = commands.add_parser(
        "overview", formatter_class=argparse.RawTextHelpFormatter, help="Dataset overview statistics.")
    overview_parser.add_argument("action",
                                 help=generate_help_text(choices), metavar="action", choices=choices.keys())
    overview_parser.set_defaults(func=overview_commands)

    # Subcommands for metadata handling
    choices = {
        "collect-files": "Collect the files into an easily accessible format.",
        "show": "Display properties about the images' metadata."
    }
    metadata_parser = commands.add_parser(
        "metadata", formatter_class=argparse.RawTextHelpFormatter, help="Handle image metadata.")
    metadata_parser.add_argument("action",
                                 help=generate_help_text(choices), metavar="action", choices=choices.keys())
    metadata_parser.set_defaults(func=metadata_commands)

    # Preprocessing data
    preprocessing_parser = commands.add_parser("preprocess", help="WIP")
    preprocessing_parser.add_argument("action")
    preprocessing_parser.set_defaults(func=preprocessing_commands)

    # Fiducial marker handling
    choices = {
        "train": "Train the fiducial matcher with manually picked references",
        "animate": "Animate the automated fiducial matches."
    }
    fiducials_parser = commands.add_parser(
        "fiducials", formatter_class=argparse.RawTextHelpFormatter, help="Fiducial identification.")
    fiducials_parser.add_argument("action",
                                  help=generate_help_text(choices), metavar="action", choices=choices.keys())
    fiducials_parser.add_argument("--redo", action="store_true", help="Clear the cache and start from the beginning")
    fiducials_parser.set_defaults(func=fiducials_commands)

    # hello_parser = argparse.ArgumentParser(parents=[parser])

    return parser.parse_args()


def generate_help_text(choice_explanations):
    help_text = """"""
    for choice, explanation in choice_explanations.items():
        help_text += f"{choice}\t{explanation}\n"

    return help_text


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


def preprocessing_commands(args):
    print("preprocessing?")


def fiducials_commands(args):
    """Run the fiducial handling subcommands."""

    if args.action == "train":
        matcher = fiducials.fiducials.FrameMatcher(cache=True)
        # matcher.filenames = matcher.filenames[:2]
        if args.redo:
            matcher.clear_cache()
        matcher.train()
    elif args.action == "animate":
        matcher = fiducials.fiducials.FrameMatcher(cache=True)
        # Rerun the training if not cached, otherwise use the cache
        matcher.train()

        matcher.transform_images()
        output_filename = fiducials.fiducials.generate_fiducial_animation()
        # TODO: Add support for more operating systems than linux
        subprocess.run(["xdg-open", output_filename], check=True)
