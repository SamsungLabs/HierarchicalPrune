#!/usr/bin/env python3
import os
import tarfile
from argparse import ArgumentParser
from glob import glob
from io import BytesIO
from json import dump, dumps, load
from os.path import basename, isdir, join, normpath, splitext
from uuid import uuid4
from zipfile import ZipFile

import pyarrow as pa
import pyarrow.parquet as pq


def process_chunk_dir(chunk_dir, output_dir, caption="cogvlm_caption"):
    os.makedirs(output_dir, exist_ok=True)
    json_filename = glob(join(chunk_dir, "*.json"))[0]
    zip_filename = glob(join(chunk_dir, "*.zip"))[0]
    chunk_num = chunk_dirname_to_num(chunk_dir)
    new_tar_file = join(output_dir, f"{chunk_num:03d}.tar")

    with open(json_filename) as json_file:
        json_data = load(json_file)

    with ZipFile(zip_filename, "r") as zip, tarfile.open(new_tar_file, "w") as new_tar:
        jpg_members = [member for member in zip.namelist() if member.endswith(".jpg")]
        jpg_members.sort()
        metadata = {
            "uid": [],
            "key": [],
            "text": [],
            "original_image_filename": [],
            "url": [],
        }
        # TODO: could save all metadata listed in https://github.com/rom1504/img2dataset/blob/main/README.md

        for i, member in enumerate(jpg_members):
            image_key = splitext(basename(normpath(member)))[0].lstrip("0")
            json_obj = json_data[image_key]
            new_basename = f"{chunk_num:03d}{i:06d}"
            new_jpg_name = new_basename + ".jpg"
            new_txt_name = new_basename + ".txt"

            # Extract and rename .jpg and .txt members
            jpg_data = zip.read(member)
            new_tarinfo = tarfile.TarInfo(new_jpg_name)
            new_tarinfo.size = len(jpg_data)
            new_tar.addfile(new_tarinfo, BytesIO(jpg_data))

            txt_data = json_obj[caption]
            txt_bytes = txt_data.encode("utf-8")
            new_tarinfo = tarfile.TarInfo(new_txt_name)
            new_tarinfo.size = len(txt_bytes)
            new_tar.addfile(new_tarinfo, BytesIO(txt_bytes))

            metadata_member = {
                "uid": uuid4().hex,
                "key": new_basename,
                "text": txt_data,
                "original_image_filename": json_obj["filename"],
                "url": json_obj["url"],
            }

            json_tarinfo = tarfile.TarInfo(new_basename + ".json")
            json_bytes = dumps(metadata_member).encode("utf-8")
            json_tarinfo.size = len(json_bytes)
            new_tar.addfile(json_tarinfo, BytesIO(json_bytes))

            # Add metadata
            metadata["uid"].append(metadata_member["uid"])
            metadata["key"].append(metadata_member["key"])
            metadata["text"].append(metadata_member["text"])
            metadata["original_image_filename"].append(metadata_member["original_image_filename"])
            metadata["url"].append(metadata_member["url"])
    return metadata


def save_metadata_to_parquet(parquet_filename, metadata):
    table = pa.Table.from_pydict(metadata)
    pq.write_table(table, parquet_filename)


def save_stats_json(stats_json_filename, count):
    stats_json = {"count": count, "successes": count}
    with open(stats_json_filename, "w") as stats_json_file:
        dump(stats_json, stats_json_file)


def chunk_dirname_to_num(chunk_dirname):
    return int(normpath(chunk_dirname).split("_")[-1]) - 1


def main():
    parser = ArgumentParser()
    parser.add_argument("--input_directory", help="Path to ye-pop directory")
    parser.add_argument("--output_directory", help="Path to the output directory")
    parser.add_argument(
        "--caption",
        choices=["cogvlm_caption", "llava_caption", "alt_txt"],
        default="llava_caption",
        help="Which caption to use in the converted dataset, others will be ignored",
    )

    args = parser.parse_args()

    input_directory = args.input_directory
    output_directory = args.output_directory

    # Process all .tar files in the directory
    for chunk_dirname in sorted(glob(join(input_directory, "images/*"))):
        if not isdir(chunk_dirname):
            continue
        print(f"chunk_dirname {chunk_dirname}")
        metadata = process_chunk_dir(chunk_dirname, output_directory, caption=args.caption)
        chunk_num = chunk_dirname_to_num(chunk_dirname)
        metadata_filename = join(output_directory, f"{chunk_num:03d}.parquet")
        stats_filename = join(output_directory, f"{chunk_num:03d}_stats.json")
        save_metadata_to_parquet(metadata_filename, metadata)
        save_stats_json(stats_filename, len(next(iter(metadata.values()))))


if __name__ == "__main__":
    main()
