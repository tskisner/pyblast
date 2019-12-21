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

"""Unit tests for focalplane tools
"""
import os
import unittest

from pkg_resources import resource_filename

from pyblast.focalplane import Focalplane, plot_focalplane


class TestFocalplane(unittest.TestCase):

    def setUp(self):
        self._bolopath = resource_filename(
            "pyblast.tests", os.path.join(
                "data",
                "fake_bolotable.tsv"
            )
        )

    def tearDown(self):
        pass

    def test_load(self):
        fp = Focalplane(self._bolopath)

    def test_vis(self):
        fp = Focalplane(self._bolopath)
        plot_focalplane(fp, "focalplane.pdf")
