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

"""Tools for loading BLAST observations into TOAST.
"""
# Importing this first, in case we are using MPI and the system
# initialization happens to be slow.
from toast.mpi import MPI

import os
import sys
import re

import traceback

import numpy as np

import pygetdata as gd

from toast.dist import Data, distribute_discrete

import toast.qarray as qa

from toast.tod import TOD

from toast.utils import Logger, Environment, memreport


class BlastTOD(TOD):
    """This class represents the timestream data for one observation.

    Args:
        path (str):  The path to the observation dirfile directory.
        focalplane (Focalplane):  The focalplane to use for this observation.
        mpicomm (mpi4py.MPI.Comm): the MPI communicator over which this
            observation data is distributed.
        detranks (int):  The dimension of the process grid in the detector
            direction.  The MPI communicator size must be evenly divisible
            by this number.

    """
    def __init__(self, path, focalplane, mpicomm, detranks=1):
        self._path = path
        self._focalplane = focalplane

        # Open and read metadata from the dirfile on one process.

        # Broadcast metadata to other processes.


        # call base class constructor to distribute data
        super().__init__(
            mpicomm, list(sorted(detquats.keys())), nsamp,
            detindx=self._detindx, detranks=detranks,
            sampsizes=sampsizes, meta=dict())

        # Every process opens the dirfile and caches a handle.




    def detoffset(self):
        return dict(self._detquats)

    def _get_boresight(self, start, n):
        ref = self.cache.reference("boresight_radec")[start:start+n, :]
        return ref

    def _put_boresight(self, start, data):
        ref = self.cache.reference("boresight_radec")
        ref[start:(start+data.shape[0]), :] = data
        del ref
        return

    def _get_boresight_azel(self, start, n):
        ref = self.cache.reference("boresight_azel")[start:start+n, :]
        return ref

    def _put_boresight_azel(self, start, data):
        ref = self.cache.reference("boresight_azel")
        ref[start:(start+data.shape[0]), :] = data
        del ref
        return

    def _get(self, detector, start, n):
        name = "{}_{}".format("signal", detector)
        ref = self.cache.reference(name)[start:start+n]
        return ref

    def _put(self, detector, start, data):
        name = "{}_{}".format("signal", detector)
        ref = self.cache.reference(name)
        ref[start:(start+data.shape[0])] = data
        del ref
        return

    def _get_flags(self, detector, start, n):
        name = "{}_{}".format("flags", detector)
        ref = self.cache.reference(name)[start:start+n]
        return ref

    def _put_flags(self, detector, start, flags):
        name = "{}_{}".format("flags", detector)
        ref = self.cache.reference(name)
        ref[start:(start+flags.shape[0])] = flags
        del ref
        return

    def _get_common_flags(self, start, n):
        ref = self.cache.reference("flags_common")[start:start+n]
        return ref

    def _put_common_flags(self, start, flags):
        ref = self.cache.reference("flags_common")
        ref[start:(start+flags.shape[0])] = flags
        del ref
        return

    def _get_hwp_angle(self, start, n):
        if self.cache.exists(self.HWP_ANGLE_NAME):
            hwpang = self.cache.reference(self.HWP_ANGLE_NAME)[start:start+n]
        else:
            hwpang = None
        return hwpang

    def _put_hwp_angle(self, start, hwpang):
        ref = self.cache.reference(self.HWP_ANGLE_NAME)
        ref[start:(start + hwpang.shape[0])] = hwpang
        del ref
        return

    def _get_times(self, start, n):
        ref = self.cache.reference("timestamps")[start:start+n]
        tm = 1.0e-9 * ref.astype(np.float64)
        del ref
        return tm

    def _put_times(self, start, stamps):
        ref = self.cache.reference("timestamps")
        ref[start:(start+stamps.shape[0])] = np.array(1.0e9 * stamps,
                                                      dtype=np.int64)
        del ref
        return

    def _get_pntg(self, detector, start, n):
        # Get boresight pointing (from disk or cache)
        bore = self._get_boresight(start, n)
        # Apply detector quaternion and return
        return qa.mult(bore, self._detquats[detector])

    def _put_pntg(self, detector, start, data):
        raise RuntimeError("SOTOD computes detector pointing on the fly."
                           " Use the write_boresight() method instead.")
        return

    def _get_position(self, start, n):
        ref = self.cache.reference("site_position")[start:start+n, :]
        return ref

    def _put_position(self, start, pos):
        ref = self.cache.reference("site_position")
        ref[start:(start+pos.shape[0]), :] = pos
        del ref
        return

    def _get_velocity(self, start, n):
        ref = self.cache.reference("site_velocity")[start:start+n, :]
        return ref

    def _put_velocity(self, start, vel):
        ref = self.cache.reference("site_velocity")
        ref[start:(start+vel.shape[0]), :] = vel
        del ref
        return

    def read_boresight_az(self, local_start=0, n=0):
        if n == 0:
            n = self.local_samples[1] - local_start
        if self.local_samples[1] <= 0:
            raise RuntimeError(
                "cannot read boresight azimuth - process "
                "has no assigned local samples"
            )
        if (local_start < 0) or (local_start + n > self.local_samples[1]):
            raise ValueError(
                "local sample range {} - {} is invalid".format(
                    local_start, local_start + n - 1
                )
            )
        return self._az[local_start : local_start + n]




def load_observation(path, dets=None, mpicomm=None, prefix=None, **kwargs):
    """Loads an observation into memory.

    Given an observation directory, load frame files into memory.  Observation
    and Calibration frames are stored in corresponding toast objects and Scan
    frames are loaded and distributed.  Further selection of a subset of
    detectors is done based on an explicit list or a Hardware object.

    Additional keyword arguments are passed to the SOTOD constructor.

    Args:
        path (str):  The path to the observation directory.
        dets (list):  Either a list of detectors, a Hardware object, or None.
        mpicomm (mpi4py.MPI.Comm):  The communicator.
        prefix (str):  Only consider frame files with this prefix.

    Returns:
        (dict):  The observation dictionary.

    """
    log = Logger.get()
    rank = 0
    if mpicomm is not None:
        rank = mpicomm.rank
    frame_sizes = {}
    frame_sizes_by_offset = {}
    frame_sample_offs = {}
    file_names = list()
    file_sample_offs = {}
    nframes = {}

    obs = dict()

    latest_obs = None
    latest_cal_frames = []
    first_offset = None

    if rank == 0:
        pat = None
        if prefix is None:
            pat = re.compile(r".*_(\d{8}).g3")
        else:
            pat = re.compile(r"{}_(\d{{8}}).g3".format(prefix))
        for root, dirs, files in os.walk(path, topdown=True):
            for f in sorted(files):
                fmat = pat.match(f)
                if fmat is not None:
                    ffile = os.path.join(path, f)
                    fsampoff = int(fmat.group(1))
                    if first_offset is None:
                        first_offset = fsampoff
                    file_names.append(ffile)
                    allframes = 0
                    file_sample_offs[ffile] = fsampoff
                    frame_sizes[ffile] = []
                    frame_sample_offs[ffile] = []
                    for frame in core3g.G3File(ffile):
                        allframes += 1
                        if frame.type == core3g.G3FrameType.Scan:
                            # This is a scan frame, process it.
                            fsz = len(frame["boresight"]["az"])
                            if fsampoff not in frame_sizes_by_offset:
                                frame_sizes_by_offset[fsampoff] = fsz
                            else:
                                if frame_sizes_by_offset[fsampoff] != fsz:
                                    raise RuntimeError(
                                        "Frame size at {} changes. {} != {}"
                                        "".format(
                                            fsampoff,
                                            frame_sizes_by_offset[fsampoff],
                                            fsz))
                            frame_sample_offs[ffile].append(fsampoff)
                            frame_sizes[ffile].append(fsz)
                            fsampoff += fsz
                        else:
                            frame_sample_offs[ffile].append(0)
                            frame_sizes[ffile].append(0)
                            if frame.type == core3g.G3FrameType.Observation:
                                latest_obs = frame
                            elif frame.type == core3g.G3FrameType.Calibration:
                                if fsampoff == first_offset:
                                    latest_cal_frames.append(frame)
                            else:
                                # Unknown frame type- skip it.
                                pass
                    frame_sample_offs[ffile] = np.array(frame_sample_offs[ffile], dtype=np.int64)
                    nframes[ffile] = allframes
                    log.debug("{} starts at {} and has {} frames".format(
                        ffile, file_sample_offs[ffile], nframes[ffile]))
            break
        if len(file_names) == 0:
            raise RuntimeError(
                "No frames found at '{}' with prefix '{}'"
                .format(path, prefix))

    if mpicomm is not None:
        latest_obs = mpicomm.bcast(latest_obs, root=0)
        latest_cal_frames = mpicomm.bcast(latest_cal_frames, root=0)
        nframes = mpicomm.bcast(nframes, root=0)
        file_names = mpicomm.bcast(file_names, root=0)
        file_sample_offs = mpicomm.bcast(file_sample_offs, root=0)
        frame_sizes = mpicomm.bcast(frame_sizes, root=0)
        frame_sizes_by_offset = mpicomm.bcast(frame_sizes_by_offset, root=0)
        frame_sample_offs = mpicomm.bcast(frame_sample_offs, root=0)

    if latest_obs is None:
        raise RuntimeError("No observation frame was found!")
    for k, v in latest_obs.iteritems():
        obs[k] = s3utils.from_g3_type(v)

    if len(latest_cal_frames) == 0:
        raise RuntimeError("No calibration frame with detector offsets!")
    detoffset, noise = parse_cal_frames(latest_cal_frames, dets)

    obs["noise"] = noise

    obs["tod"] = SOTOD(path, file_names, nframes, file_sample_offs,
                       frame_sizes, frame_sizes_by_offset,
                       frame_sample_offs, detquats=detoffset,
                       mpicomm=mpicomm, **kwargs)
    return obs


def obsweight(path, prefix=None):
    """Compute frame file sizes.

    This uses the sizes of the frame files in an observation as a proxy for
    the amount of data in that observation.  This allows us to approximately
    load balance the observations across process groups.

    Args:
        path (str):  The directory of frame files.

    Returns:
        (float):  Approximate total size in MB.

    """
    pat = None
    if prefix is None:
        pat = re.compile(r".*_\d{8}.g3")
    else:
        pat = re.compile(r"{}_\d{{8}}.g3".format(prefix))
    total = 0
    for root, dirs, files in os.walk(path, topdown=True):
        for f in files:
            mat = pat.match(f)
            if mat is not None:
                statinfo = os.stat(os.path.join(root, f))
                total += statinfo.st_size
        break
    return float(total) / 1.0e6


def load_data(dir, obs=None, comm=None, prefix=None, **kwargs):
    """Loads data into memory.

    Given a directory tree of observations, load one or more observations.
    The observations are distributed among groups in the toast communicator.
    Additional keyword arguments are passed to the load_observation()
    function.

    Args:
        dir (str):  The top-level directory that contains subdirectories (one
            per observation).
        obs (list):  The list of observations to load.
        comm (toast.Comm): the toast Comm class for distributing the data.
        prefix (str):  Only consider frame files with this prefix.

    Returns:
        (toast.Data):  The distributed data object.

    """
    log = Logger.get()
    # the global communicator
    cworld = comm.comm_world
    # the communicator within the group
    cgroup = comm.comm_group

    # One process gets the list of observation directories
    obslist = list()
    weight = dict()

    worldrank = 0
    if cworld is not None:
        worldrank = cworld.rank

    if worldrank == 0:
        for root, dirs, files in os.walk(dir, topdown=True):
            for d in dirs:
                # FIXME:  Add some check here to make sure that this is a
                # directory of frame files.
                obslist.append(d)
                weight[d] = obsweight(os.path.join(root, dir), prefix=prefix)
            break
        obslist = sorted(obslist)
        # Filter by the requested obs
        fobs = list()
        if obs is not None:
            for ob in obslist:
                if ob in obs:
                    fobs.append(ob)
            obslist = fobs

    if cworld is not None:
        obslist = cworld.bcast(obslist, root=0)
        weight = cworld.bcast(weight, root=0)

    # Distribute observations based on approximate size
    dweight = [weight[x] for x in obslist]
    distobs = distribute_discrete(dweight, comm.ngroups)

    # Distributed data
    data = Data(comm)

    # Now every group adds its observations to the list

    firstobs = distobs[comm.group][0]
    nobs = distobs[comm.group][1]
    for ob in range(firstobs, firstobs+nobs):
        opath = os.path.join(dir, obslist[ob])
        log.debug("Loading {}".format(opath))
        # In case something goes wrong on one process, make sure the job
        # is killed.
        try:
            data.obs.append(
                load_observation(opath, mpicomm=cgroup, prefix=prefix, **kwargs)
            )
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            lines = traceback.format_exception(exc_type, exc_value,
                                               exc_traceback)
            lines = ["Proc {}: {}".format(worldrank, x)
                     for x in lines]
            print("".join(lines), flush=True)
            if cworld is not None:
                cworld.Abort()

    return data
