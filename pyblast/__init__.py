#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) Katie Breivik (2017)
#
# This file is part of pyblast
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
# along with pyblast.  If not, see <http://www.gnu.org/licenses/>

"""Python BLAST Tools.
"""

from ._version import get_versions
__version__ = get_versions()['version']
__author__ = 'Scott Coughlin <scottcoughlin2014@u.northwestern.edu>'
__credits__ = ['Scott Coughlin <scottcoughlin2014@u.northwestern.edu>']
del get_versions
