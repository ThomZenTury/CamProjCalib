# coding: UTF-8

import os
import os.path
import argparse
from pathlib import Path

import cv2
import numpy as np

TARGETDIR = './graycode_pattern'
CAPTUREDDIR = './capture_*'


def generate_graycode_pattern_images(output_dir, proj_height, proj_width, graycode_step=1):
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    height = int(proj_height)
    width = int(proj_width)
    step = max(1, int(graycode_step))
    if height <= 0 or width <= 0:
        raise ValueError('Projector height and width must both be positive.')

    gc_height = int((height - 1) / step) + 1
    gc_width = int((width - 1) / step) + 1

    structured_light = getattr(cv2, 'structured_light_GrayCodePattern', None)
    if structured_light is not None and hasattr(structured_light, 'create'):
        graycode = structured_light.create(gc_width, gc_height)
        patterns = graycode.generate()[1]
    else:
        bit_count_x = max(1, int(np.ceil(np.log2(gc_width)))
                          ) if gc_width > 1 else 1
        bit_count_y = max(1, int(np.ceil(np.log2(gc_height)))
                          ) if gc_height > 1 else 1
        patterns = []

        for bit in range(bit_count_x):
            pattern = np.zeros((gc_height, gc_width), dtype=np.uint8)
            for y in range(gc_height):
                for x in range(gc_width):
                    code = x ^ (x >> 1)
                    bit_value = (code >> (bit_count_x - 1 - bit)) & 1
                    pattern[y, x] = 255 if bit_value else 0
            patterns.append(pattern)

        for bit in range(bit_count_y):
            pattern = np.zeros((gc_height, gc_width), dtype=np.uint8)
            for y in range(gc_height):
                for x in range(gc_width):
                    code = y ^ (y >> 1)
                    bit_value = (code >> (bit_count_y - 1 - bit)) & 1
                    pattern[y, x] = 255 if bit_value else 0
            patterns.append(pattern)

    expanded_patterns = []
    for pattern in patterns:
        expanded = np.zeros((height, width), dtype=np.uint8)
        for y in range(height):
            source_y = min(gc_height - 1, int(y / step))
            for x in range(width):
                source_x = min(gc_width - 1, int(x / step))
                expanded[y, x] = pattern[source_y, source_x]
        expanded_patterns.append(expanded)

    expanded_patterns.append(np.ones((height, width), dtype=np.uint8) * 255)
    expanded_patterns.append(np.zeros((height, width), dtype=np.uint8))

    files = []
    for index, pat in enumerate(expanded_patterns):
        path = output_dir / f'pattern_{index:02d}.png'
        cv2.imwrite(str(path), pat)
        files.append(str(path))
    return files


def main():
    parser = argparse.ArgumentParser(
        description='Generate graycode pattern images')
    parser.add_argument('proj_height', type=int, help='projector pixel height')
    parser.add_argument('proj_width', type=int, help='projector pixel width')
    parser.add_argument('-graycode_step', type=int,
                        default=1, help='step size of graycode [default:1](increase if moire appears)')

    args = parser.parse_args()

    files = generate_graycode_pattern_images(
        TARGETDIR, args.proj_height, args.proj_width, args.graycode_step)

    print('=== Result ===')
    print(
        f"'{TARGETDIR}/pattern_00.png ~ pattern_{len(files)-1:02d}.png' were generated")
    print()
    print('=== Next step ===')
    print(
        f"Project patterns and save captured images as '{CAPTUREDDIR}/graycode_*.png'")
    print()
    print(
        '    ./ --- capture_1/ --- graycode_00.png\n'
        '        |              |- graycode_01.png\n'
        '        |              |        .\n'
        '        |              |        .\n'
        '        |              |- graycode_' +
        str(len(files)-1).zfill(2) + '.png\n'
        '        |- capture_2/ --- graycode_00.png\n'
        '        |              |- graycode_01.png\n'
        '        |      .       |        .\n'
        '        |      .       |        .\n')
    print()
    print('It is recommended to capture more than 5 times')
    print()


if __name__ == '__main__':
    main()
