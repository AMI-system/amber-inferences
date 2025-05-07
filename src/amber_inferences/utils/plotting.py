#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Plotting utils for Amber Inferences"""

import os
from PIL import ImageDraw, Image, ImageFont


def image_annotation(
    image_path, img=None, boxes={}, scale=False, default_colour="grey"
):
    if img is None:
        img = Image.open(image_path)

    draw = ImageDraw.Draw(img)

    for box in boxes:
        x0 = float(box["x_min"])
        y0 = float(box["y_min"])
        x1 = float(box["x_max"])
        y1 = float(box["y_max"])
        if scale:
            og_width, og_height = img.size
            x0 = x0 / 300 * og_width
            y0 = y0 / 300 * og_height
            x1 = x1 / 300 * og_width
            y1 = y1 / 300 * og_height
        if "ann_col" not in box.keys():
            box["ann_col"] = default_colour
        if "label" not in box.keys():
            box["label"] = ""

        draw.rectangle([x0, y0, x1, y1], outline=box["ann_col"], width=3)
        draw.text(
            (x0, y0),
            box["label"],
            fill=box["ann_col"],
            font=ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=50
            ),
        )

    return img


def gif_creater(input_dir, output_path):
    # Open images and convert to a sequence
    image_paths = os.listdir(input_dir)
    image_paths = [os.path.join(input_dir, x) for x in image_paths]
    images = [Image.open(img) for img in image_paths]

    # Save as GIF
    images[0].save(
        output_path, save_all=True, append_images=images[1:], duration=500, loop=0
    )

    del images
