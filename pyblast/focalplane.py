# -*- coding: utf-8 -*-
# Copyright (C) 2019-2020:
#    Scott Coughlin
#    Ted Kisner
#
# This file is part of pyblast.
#
# pyblast is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pyblast is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pyblast.  If not, see <http://www.gnu.org/licenses/>.

"""Focalplane model for Blast Data.
"""

import os
import sys
import re

import numpy as np

import hashlib

import toast.qarray as qa
from toast.tod import Noise


def detector_UID(name):
    """Return a unique integer for a specified detector name.

    Args:
        name (str):  The detector name.

    Returns:
        (int):  A unique integer based on a hash of the name.

    """
    bdet = name.encode("utf-8")
    dhash = hashlib.md5()
    dhash.update(bdet)
    bdet = dhash.digest()
    uid = None
    try:
        ind = int.from_bytes(bdet, byteorder="little")
        uid = int(ind & 0xFFFFFFFF)
    except:
        raise RuntimeError(
            "Cannot convert detector name {} to a unique integer-\
            maybe it is too long?".format(
                name
            )
        )
    return uid


class Focalplane(object):
    """Class representing the bolo properties.

    Args:
        path (str):  Path to a bolo table file.

    """
    def __init__(self, path):
        self._path = path

        # Read the file

        self._bolonames = list()
        self._bolokeys = None
        self._boloprops = dict()
        with open(self._path, "r") as f:
            name_pat = re.compile(r"^#(Name.*)")
            begin_pat = re.compile(r"^Begin:.*")
            end_pat = re.compile(r"^End:.*")
            in_data = False
            # "Scroll forward" until we get to the row containing the
            # column names.  There is arbitrary stuff before that.
            for line in f:
                name_mat = name_pat.match(line)
                if name_mat is not None:
                    # get the column names
                    self._bolokeys = name_mat.group(1).split()
                    continue
                begin_mat = begin_pat.match(line)
                if begin_mat is not None:
                    in_data = True
                    continue
                end_mat = end_pat.match(line)
                if end_mat is not None:
                    in_data = False
                    continue
                if in_data:
                    # We are in the data region
                    vals = line.split()
                    if len(vals) != len(self._bolokeys):
                        raise RuntimeError(
                            "Failed to parse bolo table line '{}'".format(line)
                        )
                    thisname = None
                    for k, v in zip(self._bolokeys, vals):
                        if k == "Name":
                            if v in self._boloprops:
                                raise RuntimeError(
                                    "Found duplicate detector '{}'".format(v)
                                )
                            self._boloprops[v] = dict()
                            thisname = v
                            self._bolonames.append(v)
                        else:
                            self._boloprops[thisname][k] = v

        # Compute a UID for every detector- useful for streamed random
        # number generation in simulations.
        self._detindx = {x: detector_UID(x) for x in self._bolonames}

        # # Use inverse white noise weights
        # self._detweights = {
        #     x: 1.0 / float(self._boloprops[x]["WhiteNoise"])
        #     for x in self._bolonames
        # }

        # Use uniform noise weights for now
        self._detweights = {x: 1.0 for x in self._bolonames}

        # Add the nominal FWHM to the detector properties.
        fwhm = {
            "250": 22.0,
            "350": 30.0,
            "500": 42.0
        }
        for d in self._bolonames:
            type = self._boloprops[d]["Type"]
            if re.match(r".*250.*", type) is not None:
                self._boloprops[d]["FWHM"] = fwhm["250"]
            elif re.match(r".*350.*", type) is not None:
                self._boloprops[d]["FWHM"] = fwhm["350"]
            elif re.match(r".*500.*", type) is not None:
                self._boloprops[d]["FWHM"] = fwhm["500"]
            else:
                self._boloprops[d]["FWHM"] = 0.0

        # Compute the offset of each detector from the boresight as
        # a quaternion.
        self._detquats = {
            x: self.elxel2quat(
                float(self._boloprops[x]["EL"]),
                float(self._boloprops[x]["XEL"]),
                float(self._boloprops[x]["Angle"])
            ) for x in self._bolonames
        }


    def elxel2quat(self, el, xel, alpha):
        """Compute detector quaternion rotation from boresight.

        These notes come from the original code written by Steve Benton
        in TOAST version 1:

        Create a rotation with conventions suitable for transforming TOAST
        dir-orient vectors using el (pitch), xel (yaw), alpha (roll) offsets
        defined according to BLAST conventions.  For BLAST, "direction" is
        along y-axis, and "orientation" along z-axis, and (direction "cross"
        orientation) along x-axis.  Yaw-pitch-roll rotations occur along
        z-, x-, and y-axes.  With right-to-left ordering of quaternion product,
        this means:

            Q = q_roll * q_pitch * q_yaw
            q_yaw   = [ cos(y/2),        0,        0, sin(y/2) ]
            q_pitch = [ cos(p/2), sin(p/2),        0,        0 ]
            q_roll  = [ cos(r/2),        0, sin(r/2),        0 ]

        NB: this ordering assumed INTRINSIC rotations. Since these are
        specified in EXTRINSIC manner, order should be reversed

        yaw rotates direction (Y) away from cross (X)
        pitch rotates direction (Y) towards orientation (Z)
        roll rotates orientation (Z) towards cross (X)

        For TOAST, the reference axes are direction: Z, orientation: X, and
        cross: Y

        yaw rotates direction (Z) away from cross (Y)
            --> positive rotation about x-axis

        pitch rotates direction (Z) towards orientation (X)
            --> positive rotation about y-axis

        roll rotates orientation (X) towards cross (Y)
            --> positive rotation about z-axis

        BLAST's polarization angle increases in same direction as parallactic
        angle: CCW along line of sight.  Increasing alpha --> CCW rotation
        along z-axis --> negative rotation about z-axis.

        Args:
            el (float):  The EL offset in arcsec from the bolotable.
            xel (float):  The XEL offset in arcsec from the bolotable.
            alpha (float):  The Angle parameter in deg from the bolotable.

        Returns:
            (array): the quaternion that rotates focalplane boresight coordinate
                frame to detector location and primary polarization orientation.

        """
        xaxis = np.array([1.0, 0.0, 0.0])
        yaxis = np.array([0.0, 1.0, 0.0])
        zaxis = np.array([0.0, 0.0, 1.0])
        el = np.radians(el / 3600.0)
        xel = np.radians(xel / 3600.0)
        alpha = np.radians(alpha)

        qel = qa.rotation(xaxis, xel)
        qxel = qa.rotation(yaxis, el)
        qpol = qa.rotation(zaxis, alpha)
        qrot = qa.mult(qxel, qel)
        quat = qa.mult(qrot, qpol)
        return qa.norm(quat)


    @property
    def detweights(self):
        """Return the inverse noise variance weights
        """
        return self._detweights


    @property
    def data(self):
        """Return the full dictionary of bolo properties.
        """
        return self._boloprops


    @property
    def detquats(self):
        return self._detquats

    # @property
    # def noise(self):
    #     if self._noise is None:
    #         detectors = sorted(self.detector_data.keys())
    #         fmin = {}
    #         fknee = {}
    #         alpha = {}
    #         NET = {}
    #         rates = {}
    #         for detname in detectors:
    #             detdata = self.detector_data[detname]
    #             if "fsample" in detdata:
    #                 rates[detname] = detdata["fsample"]
    #             else:
    #                 rates[detname] = self.sample_rate
    #             fmin[detname] = detdata["fmin"]
    #             fknee[detname] = detdata["fknee"]
    #             alpha[detname] = detdata["alpha"]
    #             NET[detname] = detdata["NET"]
    #         self._noise = AnalyticNoise(
    #             rate=rates,
    #             fmin=fmin,
    #             detectors=detectors,
    #             fknee=fknee,
    #             alpha=alpha,
    #             NET=NET,
    #         )
    #     return self._noise
    #
    # def __repr__(self):
    #     value = (
    #         "(Focalplane : {} detectors, sample_rate = {} Hz, radius = {} deg, "
    #         "detectors = ("
    #         "".format(len(self.detector_data), self.sample_rate, self.radius)
    #     )
    #     for detector_name, detector_data in self.detector_data.items():
    #         value += "{}, ".format(detector_name)
    #     value += "))"
    #     return value


def plot_focalplane(fp, outpdf, width=None, height=None):
    """Make a simple plot of the focalplane layout.

    Args:
        fp (Focalplane): The focalplane.
        outpdf (str): The output path of the generated PDF.

    Returns:
        None

    """
    # Although not PEP8 compliant, we import matplotlib here so that it is
    # only imported when plotting, rather than when thousands of processes load
    # this file simultaneously.
    import matplotlib
    matplotlib.use("pdf")
    import matplotlib.pyplot as plt

    band_colors = {
        "250": (0.4, 0.4, 1.0, 0.3),
        "B250": (0.4, 0.4, 1.0, 0.1),
        "D250": (0.4, 0.4, 1.0, 0.1),
        "350": (0.4, 1.0, 0.4, 0.3),
        "B350": (0.4, 1.0, 0.4, 0.1),
        "D350": (0.4, 1.0, 0.4, 0.1),
        "500": (1.0, 0.4, 0.4, 0.3),
        "B500": (1.0, 0.4, 0.4, 0.1),
        "D500": (1.0, 0.4, 0.4, 0.1),
    }
    xaxis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    zaxis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    wmin = 1.0
    wmax = -1.0
    hmin = 1.0
    hmax = -1.0
    if (width is None) or (height is None):
        # We are autoscaling.  Compute the angular extent of all detectors
        # and add some buffer.
        for d, quat in fp.detquats.items():
            type = fp.data[d]["Type"]
            if type[0] in ["R", "T", "N"]:
                # Skip over sensors with no position data.
                continue
            dir = qa.rotate(quat, zaxis).flatten()
            if (dir[0] > wmax):
                wmax = dir[0]
            if (dir[0] < wmin):
                wmin = dir[0]
            if (dir[1] > hmax):
                hmax = dir[1]
            if (dir[1] < hmin):
                hmin = dir[1]
        wmin = np.arcsin(wmin) * 180.0 / np.pi
        wmax = np.arcsin(wmax) * 180.0 / np.pi
        hmin = np.arcsin(hmin) * 180.0 / np.pi
        hmax = np.arcsin(hmax) * 180.0 / np.pi
        wbuf = 0.1 * (wmax - wmin)
        hbuf = 0.1 * (hmax - hmin)
        wmin -= wbuf
        wmax += wbuf
        hmin -= hbuf
        hmax += hbuf
        width = wmax - wmin
        height = hmax - hmin
    else:
        half_width = 0.5 * width
        half_height = 0.5 * height
        wmin = -half_width
        wmax = half_width
        hmin = -half_height
        hmax = half_height

    xfigsize = 10.0
    yfigsize = xfigsize * (height / width)
    figdpi = 75
    yfigpix = int(figdpi * yfigsize)
    ypixperdeg = yfigpix / height

    fig = plt.figure(figsize=(xfigsize, yfigsize), dpi=figdpi)
    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlabel("Degrees", fontsize="large")
    ax.set_ylabel("Degrees", fontsize="large")
    ax.set_xlim([wmin, wmax])
    ax.set_ylim([hmin, hmax])

    for d, quat in fp.detquats.items():
        type = fp.data[d]["Type"]
        if type[0] in ["R", "T", "N"]:
            # Skip over sensors with no position data.
            continue
        fwhm = fp.data[d]["FWHM"]

        # el = float(fp.data[d]["EL"])
        # if el < -180.0:
        #     el += 360.0
        # xel = float(fp.data[d]["XEL"])
        # if xel < -180.0:
        #     xel += 360.0
        # circ = plt.Circle(
        #     (el, xel),
        #     radius=(fwhm/100),
        #     fc="none",
        #     ec="black"
        # )
        # ax.add_artist(circ)


        # radius in degrees
        detradius = 0.5 * fwhm / 3600.0

        # rotation from boresight
        rdir = qa.rotate(quat, zaxis).flatten()
        ang = np.arctan2(rdir[1], rdir[0])

        orient = qa.rotate(quat, xaxis).flatten()
        polang = np.arctan2(orient[1], orient[0])

        mag = np.arccos(rdir[2]) * 180.0 / np.pi
        xpos = mag * np.cos(ang)
        ypos = mag * np.sin(ang)

        detface = band_colors[type]

        circ = plt.Circle((xpos, ypos), radius=detradius, fc=detface,
                          ec="black", linewidth=0.05*detradius)
        ax.add_artist(circ)

        ascale = 1.5

        xtail = xpos - ascale * detradius * np.cos(polang)
        ytail = ypos - ascale * detradius * np.sin(polang)
        dx = ascale * 2.0 * detradius * np.cos(polang)
        dy = ascale * 2.0 * detradius * np.sin(polang)

        detcolor = "black"
        if fp.data[d]["Angle"] == "90":
            detcolor = (1.0, 0.0, 0.0, 1.0)
        else:
            detcolor = (0.0, 0.0, 1.0, 1.0)

        ax.arrow(xtail, ytail, dx, dy, width=0.1*detradius,
                 head_width=0.3*detradius, head_length=0.3*detradius,
                 fc=detcolor, ec="none", length_includes_head=True)

        # if labels:
        #     # Compute the font size to use for detector labels
        #     fontpix = 0.1 * detradius * ypixperdeg
        #     ax.text((xpos), (ypos), pixel,
        #             color='k', fontsize=fontpix, horizontalalignment='center',
        #             verticalalignment='center',
        #             bbox=dict(fc='white', ec='none', pad=0.2, alpha=1.0))
        #     xsgn = 1.0
        #     if dx < 0.0:
        #         xsgn = -1.0
        #     labeloff = 1.0 * xsgn * fontpix * len(pol) / ypixperdeg
        #     ax.text((xtail+1.0*dx+labeloff), (ytail+1.0*dy), pol,
        #             color='k', fontsize=fontpix, horizontalalignment='center',
        #             verticalalignment='center',
        #             bbox=dict(fc='none', ec='none', pad=0, alpha=1.0))

    plt.savefig(outpdf)
    plt.close()
