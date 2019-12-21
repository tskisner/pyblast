#!/usr/bin/env python3

# Copyright (c) 2019-2020 BLAST Collaboration.

"""Load and plot a focalplane layout
"""
import os
import sys
import argparse

from pyblast.focalplane import (Focalplane, plot_focalplane)


def main():

    parser = argparse.ArgumentParser(
        description="Read a bolotable file and plot the detector layout."
    )

    parser.add_argument(
        "--bolotable", required=True, default=None, help="The bolotable file"
    )

    parser.add_argument(
        "--output",
        required=False,
        default="focalplane.pdf",
        help="The output PDF file.",
    )

    args = parser.parse_args()

    # Create the focalplane

    fp = Focalplane(args.bolotable)

    # Make the plot

    plot_focalplane(fp, args.output)


if __name__ == "__main__":
    main()
