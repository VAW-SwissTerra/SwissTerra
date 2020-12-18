#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import subprocess
import time
from typing import List

from terra import files  # TODO: Change to import submodules instead
from terra import preprocessing, processing
from terra.preprocessing import fiducials


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

    # Subcommands for input file handling
    choices = {
        "check-inputs": "Check that all files can be found and have the correct file type.",
        "clear-cache": "Remove everything in the cache folder.",
        "list-cache": "List files in the cache",
        "remove-locks": "Remove Metashape lock files if they exist",
    }
    files_parser = commands.add_parser(
        "files", formatter_class=argparse.RawTextHelpFormatter, help="Handle input, output and cache files.")
    files_parser.add_argument("action",
                              help=generate_help_text(choices), metavar="action", choices=choices.keys())
    files_parser.set_defaults(func=files_commands)

    # Subcommands for the overview module
    choices = {
        "show-metadata": "Display properties about the images' metadata.",
        "capture-dates": "Plot the dates for when photographs were captured",
        "process-log": "Show the processing progress log"
    }
    overview_parser = commands.add_parser(
        "overview", formatter_class=argparse.RawTextHelpFormatter, help="Dataset overview statistics.")
    overview_parser.add_argument("action",
                                 help=generate_help_text(choices), metavar="action", choices=choices.keys())
    overview_parser.set_defaults(func=overview_commands)

    # Preprocessing data
    choices = {
        "collect-metadata": "Collect the files into an easily accessible format.",
        "generate-masks": "Generate frame masks for each image with an estimated frame transform.",
        "show-reference-mask": "Show the generated reference mask.",
        "train-fiducials": "Train the frame matcher with manually picked reference fiducials",
        "estimate-fiducials": "Estimate frame transforms for all images.",
        "animate-fiducials": "Animate the automated fiducial matches.",
        "transform-images": "Apply the estimated fiducial transforms to the images and save them."
    }
    preprocessing_parser = commands.add_parser(
        "preprocessing", help="Run preprocessing tasks.", formatter_class=argparse.RawTextHelpFormatter)
    preprocessing_parser.add_argument("action", help=generate_help_text(choices),
                                      metavar="action", choices=choices.keys())
    preprocessing_parser.add_argument("--dataset", type=str, default="full", help="limit the estimation to one dataset")
    preprocessing_parser.add_argument("--redo", action="store_true",
                                      help="clear the cache and start from the beginning")
    preprocessing_parser.set_defaults(func=preprocessing_commands)

    # Processing
    datasets = processing.inputs.get_dataset_names() + ["full"]
    choices = {dataset: f"Process the {dataset} dataset" for dataset in datasets}
    processing_parser = commands.add_parser("processing", formatter_class=argparse.RawTextHelpFormatter,
                                            help="Main data processing.", description="Main data processing")
    processing_parser.add_argument("dataset", help="The dataset to process (type anything to see valid options)",
                                   metavar="dataset", choices=datasets)

    choices = {
        "run": "Run the main processing pipeline for the dataset",
        "rerun": "Run the main processing pipeline from the start",
        "check-inputs": "Check that all required dataset inputs exist.",
        "generate-inputs": "Generate all required dataset inputs."
    }
    processing_parser.add_argument("action", help=generate_help_text(choices), metavar="action", choices=choices.keys())
    processing_parser.set_defaults(func=processing_commands)

    return parser.parse_args()


def generate_help_text(choice_explanations: dict[str, str]):
    """Generate help text for argparse subcommands."""
    help_text = """\n"""
    for choice, explanation in choice_explanations.items():
        help_text += f"{choice}\t{explanation}\n"

    return help_text


def files_commands(args):
    """Run the file handling subcommands."""
    if args.action == "check-inputs":
        files.check_data()
    elif args.action == "clear-cache":
        files.clear_cache()
    elif args.action == "list-cache":
        files.list_cache()
    elif args.action == "remove-locks":
        files.remove_locks()


def overview_commands(args):
    """Run the overview module."""
    if args.action == "show-metadata":
        meta_file = preprocessing.image_meta.collect_metadata(use_cached=True)
        print(meta_file)
        print(meta_file.describe())
        print(meta_file.dtypes)
    elif args.action == "capture-dates":
        preprocessing.overview.get_capture_date_distribution()
    elif args.action == "process-log":
        processing.processing_tools.show_processing_log()


def preprocessing_commands(args):
    """Run any of the preprocessing subcommands."""
    if args.action == "collect-metadata":
        preprocessing.image_meta.collect_metadata(use_cached=False)
    elif args.action == "generate-masks":
        preprocessing.masks.generate_masks()
    elif args.action == "show-reference-mask":
        preprocessing.masks.show_reference_mask()

    def fetch_dataset_filenames(dataset: str) -> List[str]:
        try:
            filenames = processing.inputs.get_dataset_filenames(args.dataset)
        except KeyError:
            raise ValueError(f"Dataset '{dataset}' not configured")

        return filenames

    if args.action == "train-fiducials":
        matcher = preprocessing.fiducials.FrameMatcher(cache=True)

        if args.dataset != "full":
            matcher.filenames = fetch_dataset_filenames(args.dataset)

        # matcher.filenames = matcher.filenames[:2]
        if args.redo:
            matcher.clear_cache()
        matcher.train()

    elif args.action == "estimate-fiducials":
        matcher = preprocessing.fiducials.FrameMatcher(cache=True)
        if args.dataset != "full":
            matcher.filenames = fetch_dataset_filenames(args.dataset)

        matcher.estimate()

    elif args.action == "animate-fiducials":
        matcher = preprocessing.fiducials.FrameMatcher(cache=True)
        # Rerun the training if not cached, otherwise use the cache
        matcher.train()

        if not os.path.isdir(preprocessing.fiducials.CACHE_FILES["transformed_image_dir"])\
                or len(os.listdir(preprocessing.fiducials.CACHE_FILES["transformed_image_dir"])) == 0:
            matcher.transform_images()
        output_filename = preprocessing.fiducials.generate_fiducial_animation()
        # TODO: Add support for more operating systems than linux
        subprocess.run(["xdg-open", output_filename], check=True, close_fds=True)

    elif args.action == "transform-images":
        matcher = preprocessing.fiducials.FrameMatcher(cache=True)

        matcher.transform_images()

        print(f"Saved images to {preprocessing.fiducials.CACHE_FILES['transformed_image_dir']}")


def processing_commands(args):
    """Run the main processing subcommands."""
    if args.action in ["run", "rerun"]:
        if args.dataset == "full":
            for dataset in processing.inputs.get_dataset_names():
                print(f"Processing {dataset}")
                processing.main.process_dataset(dataset, redo=args.action == "rerun")
                time.sleep(1)  # Allow for logging to be slightly nicer.
        processing.main.process_dataset(args.dataset, redo=args.action == "rerun")
    elif args.action == "check-inputs":
        processing.inputs.check_inputs(args.dataset)
    elif args.action == "generate-inputs":
        processing.inputs.generate_inputs(args.dataset)
