import os
from collections import namedtuple

from terra import fiducials, files, preprocessing


def process_dataset(dataset: str):
    if dataset == "rhone":
        process_rhone()


def process_rhone():
    print("I FINSHED IT WAS THIS SIMPLE ALL ALONG")


def check_inputs(dataset: str):
    missing_inputs = []
    MissingInput = namedtuple("MissingInput", ["filepath", "filetype"])
    image_filenames = open(files.INPUT_FILES["rhone_image_filenames"]).read().splitlines()
    image_filepaths = [os.path.join(files.INPUT_DIRECTORIES["image_dir"], filename) for filename in image_filenames]

    frame_matcher = fiducials.fiducials.FrameMatcher(verbose=False)
    transforms = frame_matcher.load_transforms("merged_transforms.pkl")

    for filepath in image_filepaths:
        if not os.path.isfile(filepath):
            missing_inputs.append(MissingInput(filepath, "image"))

        meta_name = filepath.replace(files.INPUT_DIRECTORIES["image_dir"],
                                     files.INPUT_DIRECTORIES["image_meta_dir"]).replace(".tif", ".txt")

        if not os.path.isfile(meta_name):
            missing_inputs.append(MissingInput(meta_name, "image_meta"))

        if not os.path.basename(filepath) in transforms.keys():
            missing_inputs.append(MissingInput(os.path.basename(filepath), "transform"))

        mask_name = os.path.join(preprocessing.masks.CACHE_FILES["mask_dir"], os.path.basename(filepath))

        if not os.path.isfile(mask_name):
            missing_inputs.append(MissingInput(mask_name, "frame_mask"))

    if len(missing_inputs) == 0:
        print("Every file located. Processing pipeline is ready.")
        return

    print("\n============================")
    print(f"Missing {len(missing_inputs)} input file{'s' if len(missing_inputs) > 1 else ''}.")
    print("=============================\n")

    pretty_names = {
        "image": ("image", "Check the image input folder.", "multiple"),
        "image_meta": ("image metadata file", "Check the image metadata folder.", "multiple"),
        "transform": ("image frame transform", "Run the fiducial estimation.", "multiple"),
        "frame_mask": ("image frame mask", f"Run 'terra preprocessing generate-masks'", "multiple"),
    }

    for key, (pretty_name, help_text, count) in pretty_names.items():
        missing_keys = [missing_input for missing_input in missing_inputs if missing_input.filetype == key]

        if len(missing_keys) == 0:
            continue

        if count == "one":
            print(f"Missing {pretty_name}.")

        elif count == "multiple":
            print(f"Missing {len(missing_keys)} {pretty_name}{'s' if len(missing_keys) > 1 else ''}")
            if len(missing_keys) < 10:
                print(*(f"\tMissing {missing.filepath}" for missing in missing_keys))

        print(f"Suggested fix: {help_text}\n")
