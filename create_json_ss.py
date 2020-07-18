#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import json
import os
import sys
import base64


def bytes_to_utf8(byte_arr):
    b64_enc = base64.b64encode(byte_arr)
    utf8_str = b64_enc.decode('utf-8')
    return utf8_str


def parse_args(args_to_parse):
    description = "Compresses the output folder for the OpenEDS 2.0 Challenge\n"
    parser = argparse.ArgumentParser(description=description)

    # Processing arguments
    process_args = parser.add_argument_group('Processing arguments')
    process_args.add_argument('--root-folder', type=str, required=True,
                        help="path to output folder")
    process_args.add_argument('--submission-json', type=str, required=True,
                        help='The output compressed file path')

    args = parser.parse_args(args_to_parse)
    return args


def compress_image(image_path):
    with open(image_path, 'rb') as f:
        data = f.read()
    compressed = bytearray()
    # Check if the file is not empty
    if len(data) == 0:
        return compressed

    # RLE
    current_byte = 0
    current_count = 0
    for byte in data:
        if current_byte == byte:
            current_count += 1
            if current_count == 255:
                compressed.append(current_count)
                compressed.append(current_byte)
                current_count = 0
        else:
            if current_count > 0:
                compressed.append(current_count)
                compressed.append(current_byte)
            current_byte = byte
            current_count = 1
    if current_count > 0:
        compressed.append(current_count)
        compressed.append(current_byte)
    return bytes_to_utf8(compressed)


def compress_folder(folder_path):
    result = {}
    with open(os.path.join(folder_path, 'output.txt'), "rt") as f:
        lines = f.readlines()
        lines = [l.rstrip() for l in lines]

    for idx, image_name in enumerate(lines):
        if idx % 100 == 0:
            print("Saving {}...".format(idx))

        image_path = os.path.join(folder_path, image_name)
        if not os.path.isfile(image_path):
            raise("File does not exist: {}".format(image_path))
        compressed_image = compress_image(image_path)
        result[image_name] = compressed_image
    return result


def main(args):
    root_folder = os.path.expanduser(args.root_folder)
    submission_json = os.path.expanduser(args.submission_json)
    compressed_result = compress_folder(root_folder)
    with open(submission_json, "w") as f:
        json.dump(compressed_result, f)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)
