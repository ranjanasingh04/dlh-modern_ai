#!/usr/bin/env python3
"""Prepare a Pascal VOC 2012 subset in YOLOv8 detection format."""

import argparse
import shutil
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


VOC_URL = (
    "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/"
    "VOCtrainval_11-May-2012.tar"
)

CLASS_NAMES = ["person", "car", "bicycle"]
CLASS_TO_ID = {
    class_name: class_id
    for class_id, class_name in enumerate(CLASS_NAMES)
}


def parse_arguments():
    """Parse and return command-line arguments."""
    project_dir = Path(__file__).resolve().parent
    default_cache = Path.home() / "datasets" / "pascal-voc-2012"

    parser = argparse.ArgumentParser(
        description=(
            "Convert selected Pascal VOC 2012 images and annotations "
            "to YOLOv8 format."
        )
    )
    parser.add_argument(
        "--train-list",
        type=Path,
        default=project_dir / "train_samples.txt",
        help="Path to the training sample-name file.",
    )
    parser.add_argument(
        "--val-list",
        type=Path,
        default=project_dir / "val_samples.txt",
        help="Path to the validation sample-name file.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=default_cache,
        help="Directory used for the full Pascal VOC download.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "datasets" / "detection",
        help="Destination directory for the YOLO dataset.",
    )
    parser.add_argument(
        "--voc-root",
        type=Path,
        default=None,
        help=(
            "Optional path to an existing VOC2012 directory. "
            "It must contain JPEGImages and Annotations."
        ),
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not download Pascal VOC if it is missing.",
    )
    return parser.parse_args()


def download_archive(url, destination):
    """Download a file from URL to destination with progress output."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        print("Using existing archive: {}".format(destination))
        return

    print("Downloading Pascal VOC 2012.")
    print("Source: {}".format(url))
    print("Destination: {}".format(destination))
    print("The archive is large, so this may take some time.")

    def report_progress(block_count, block_size, total_size):
        """Display approximate download progress."""
        downloaded = block_count * block_size

        if total_size <= 0:
            return

        percentage = min(100, downloaded * 100 / total_size)
        print(
            "\rDownloading: {:.1f}%".format(percentage),
            end="",
            flush=True,
        )

    try:
        urllib.request.urlretrieve(
            url,
            destination,
            reporthook=report_progress,
        )
    except Exception:
        if destination.exists():
            destination.unlink()
        raise

    print("\nDownload completed.")


def validate_archive_member(destination, member_name):
    """Return whether an archive member stays inside destination."""
    destination = destination.resolve()
    member_path = (destination / member_name).resolve()

    try:
        member_path.relative_to(destination)
    except ValueError:
        return False

    return True


def extract_archive(archive_path, destination):
    """Safely extract a tar archive into destination."""
    expected_root = destination / "VOCdevkit" / "VOC2012"

    if expected_root.exists():
        print("Using extracted dataset: {}".format(expected_root))
        return expected_root

    print("Extracting Pascal VOC archive.")

    with tarfile.open(archive_path, "r") as archive:
        for member in archive.getmembers():
            if not validate_archive_member(
                destination,
                member.name,
            ):
                raise ValueError(
                    "Unsafe archive member: {}".format(member.name)
                )

        archive.extractall(destination)

    if not expected_root.exists():
        raise FileNotFoundError(
            "VOC2012 was not found after extracting the archive."
        )

    print("Extraction completed.")
    return expected_root


def validate_voc_root(voc_root):
    """Validate and return a Pascal VOC 2012 root directory."""
    voc_root = voc_root.expanduser().resolve()
    image_dir = voc_root / "JPEGImages"
    annotation_dir = voc_root / "Annotations"

    if not image_dir.is_dir():
        raise FileNotFoundError(
            "Missing JPEGImages directory: {}".format(image_dir)
        )

    if not annotation_dir.is_dir():
        raise FileNotFoundError(
            "Missing Annotations directory: {}".format(annotation_dir)
        )

    return voc_root


def obtain_voc_root(arguments):
    """Return an existing or newly downloaded VOC2012 directory."""
    if arguments.voc_root is not None:
        return validate_voc_root(arguments.voc_root)

    cache_dir = arguments.cache_dir.expanduser().resolve()
    archive_path = cache_dir / "VOCtrainval_11-May-2012.tar"
    extracted_root = cache_dir / "VOCdevkit" / "VOC2012"

    if extracted_root.exists():
        return validate_voc_root(extracted_root)

    if arguments.no_download:
        raise FileNotFoundError(
            "Pascal VOC was not found and --no-download was specified."
        )

    download_archive(VOC_URL, archive_path)
    voc_root = extract_archive(archive_path, cache_dir)
    return validate_voc_root(voc_root)


def normalize_sample_name(sample_name):
    """Return a Pascal VOC image identifier without an extension."""
    return Path(sample_name.strip()).stem


def read_sample_names(file_path):
    """Read unique image identifiers from a sample-list file."""
    file_path = file_path.expanduser().resolve()

    if not file_path.is_file():
        raise FileNotFoundError(
            "Sample-list file not found: {}".format(file_path)
        )

    sample_names = []
    observed_names = set()

    with file_path.open("r", encoding="utf-8") as sample_file:
        for line in sample_file:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            sample_name = normalize_sample_name(line)

            if sample_name not in observed_names:
                sample_names.append(sample_name)
                observed_names.add(sample_name)

    if not sample_names:
        raise ValueError(
            "No sample names were found in {}".format(file_path)
        )

    return sample_names


def clamp(value, minimum, maximum):
    """Return value restricted to the inclusive given range."""
    return max(minimum, min(value, maximum))


def convert_box(box, image_width, image_height):
    """Convert a Pascal VOC bounding box to normalized YOLO format."""
    xmin, ymin, xmax, ymax = box

    xmin = clamp(xmin - 1.0, 0.0, image_width)
    ymin = clamp(ymin - 1.0, 0.0, image_height)
    xmax = clamp(xmax - 1.0, 0.0, image_width)
    ymax = clamp(ymax - 1.0, 0.0, image_height)

    box_width = xmax - xmin
    box_height = ymax - ymin
    center_x = xmin + box_width / 2.0
    center_y = ymin + box_height / 2.0

    return (
        center_x / image_width,
        center_y / image_height,
        box_width / image_width,
        box_height / image_height,
    )


def parse_annotation(annotation_path):
    """Read a VOC XML annotation and return YOLO label lines."""
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    size_element = root.find("size")

    if size_element is None:
        raise ValueError(
            "Missing image size in {}".format(annotation_path)
        )

    width_element = size_element.find("width")
    height_element = size_element.find("height")

    if width_element is None or height_element is None:
        raise ValueError(
            "Incomplete image size in {}".format(annotation_path)
        )

    image_width = float(width_element.text)
    image_height = float(height_element.text)

    if image_width <= 0 or image_height <= 0:
        raise ValueError(
            "Invalid image size in {}".format(annotation_path)
        )

    labels = []

    for object_element in root.findall("object"):
        name_element = object_element.find("name")

        if name_element is None:
            continue

        class_name = name_element.text.strip()

        if class_name not in CLASS_TO_ID:
            continue

        box_element = object_element.find("bndbox")

        if box_element is None:
            continue

        coordinate_names = ("xmin", "ymin", "xmax", "ymax")
        coordinate_values = []

        for coordinate_name in coordinate_names:
            coordinate_element = box_element.find(coordinate_name)

            if coordinate_element is None:
                coordinate_values = []
                break

            coordinate_values.append(float(coordinate_element.text))

        if len(coordinate_values) != 4:
            continue

        xmin, ymin, xmax, ymax = coordinate_values

        if xmax <= xmin or ymax <= ymin:
            continue

        center_x, center_y, box_width, box_height = convert_box(
            coordinate_values,
            image_width,
            image_height,
        )

        if box_width <= 0 or box_height <= 0:
            continue

        class_id = CLASS_TO_ID[class_name]
        label = (
            "{} {:.6f} {:.6f} {:.6f} {:.6f}".format(
                class_id,
                center_x,
                center_y,
                box_width,
                box_height,
            )
        )
        labels.append(label)

    return labels


def reset_directory(directory):
    """Create a directory and remove its existing contents."""
    directory.mkdir(parents=True, exist_ok=True)

    for item in directory.iterdir():
        if item.is_file() or item.is_symlink():
            try:
                item.unlink()
            except PermissionError:
                item.chmod(0o666)
                item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def find_image(image_directory, sample_name):
    """Find a Pascal VOC image using common image extensions."""
    extensions = (".jpg", ".jpeg", ".JPG", ".JPEG")

    for extension in extensions:
        image_path = image_directory / (sample_name + extension)

        if image_path.is_file():
            return image_path

    raise FileNotFoundError(
        "Image not found for sample: {}".format(sample_name)
    )


def prepare_split(
    split_name,
    sample_names,
    voc_root,
    output_root,
):
    """Prepare one train or validation split in YOLO format."""
    source_images = voc_root / "JPEGImages"
    source_annotations = voc_root / "Annotations"
    output_images = output_root / "images" / split_name
    output_labels = output_root / "labels" / split_name

    reset_directory(output_images)
    reset_directory(output_labels)

    copied_count = 0
    label_count = 0

    for sample_name in sample_names:
        source_image = find_image(source_images, sample_name)
        annotation_path = (
            source_annotations / (sample_name + ".xml")
        )

        if not annotation_path.is_file():
            raise FileNotFoundError(
                "Annotation not found: {}".format(annotation_path)
            )

        destination_image = output_images / source_image.name
        destination_label = output_labels / (sample_name + ".txt")

        shutil.copy2(source_image, destination_image)
        labels = parse_annotation(annotation_path)

        label_content = "\n".join(labels)

        if label_content:
            label_content += "\n"

        destination_label.write_text(
            label_content,
            encoding="utf-8",
            newline="\n",
        )

        copied_count += 1
        label_count += len(labels)

    print(
        "{}: copied {} images and wrote {} objects.".format(
            split_name,
            copied_count,
            label_count,
        )
    )


def create_data_yaml(output_root):
    """Create the YOLOv8 data.yaml configuration file."""
    yaml_content = (
        "path: datasets/detection/\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "nc: 3\n"
        'names: ["person", "car", "bicycle"]\n'
    )

    yaml_path = output_root / "data.yaml"
    yaml_path.write_text(
        yaml_content,
        encoding="utf-8",
        newline="\n",
    )

    print("Created {}".format(yaml_path))


def main():
    """Download and convert the selected Pascal VOC 2012 samples."""
    arguments = parse_arguments()
    output_root = arguments.output_dir.expanduser().resolve()

    train_samples = read_sample_names(arguments.train_list)
    val_samples = read_sample_names(arguments.val_list)
    voc_root = obtain_voc_root(arguments)

    print("VOC source: {}".format(voc_root))
    print("YOLO output: {}".format(output_root))
    print("Training samples: {}".format(len(train_samples)))
    print("Validation samples: {}".format(len(val_samples)))

    output_root.mkdir(parents=True, exist_ok=True)

    prepare_split(
        "train",
        train_samples,
        voc_root,
        output_root,
    )
    prepare_split(
        "val",
        val_samples,
        voc_root,
        output_root,
    )
    create_data_yaml(output_root)

    print("Pascal VOC preparation completed successfully.")


if __name__ == "__main__":
    main()