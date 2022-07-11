# This script contains a list of routines to set mappraiser parameters, and
# apply the OpMappraiser operator during a TOD2MAP TOAST pipeline

# @author: Hamza El Bouhargani
# @date: January 2020

import argparse
import copy
import os
import re

import numpy as np
from toast.timing import function_timer, Timer
from toast.utils import Logger, Environment

from TOAST_interface import OpMappraiser


def add_mappraiser_args(parser):
    """ Add libmappraiser arguments
    """

    parser.add_argument(
        "--outpath", required=False, default="./", help="Output path"
    )
    parser.add_argument(
        "--ref", required=False, default="run0", help="Output maps references"
    )
    parser.add_argument("--uniform_w", required=False, default=0, type=np.int, help="Activate for uniform white noise model: 0->off, 1->on"
                        )
    parser.add_argument(
        "--Lambda", required=False, default=16384, type=np.int,
        help="Half bandwidth (lambda) of noise covariance"
    )
    parser.add_argument(
        "--pair-diff",
        dest="pair_diff",
        required=False,
        action="store_true",
        help="Process differenced TOD [default]",
    )
    parser.add_argument("--precond", required=False, default=0, type=np.int, help="Choose map-making preconditioner: 0->BD, 1->2lvl a priori, 2->2lvl a posteriori"
                        )
    parser.add_argument("--Z_2lvl", required=False, default=0, type=np.int, help="2lvl deflation size"
                        )
    parser.add_argument(
        "--solver", required=False, default=0, type=np.int,
        help="Choose map-making solver: 0->PCG, 1->ECG"
    )
    parser.add_argument(
        "--ptcomm_flag", required=False, default=6, type=np.int,
        help="Choose collective communication scheme"
    )
    parser.add_argument(
        "--tol", required=False, default=1e-6, type=np.double,
        help="Tolerance parameter for convergence"
    )
    parser.add_argument("--maxiter", required=False, default=500, type=np.int, help="Maximum number of iterations in Mappraiser"
                        )
    parser.add_argument("--enlFac", required=False, default=1, type=np.int, help="Enlargement factor for ECG"
                        )
    parser.add_argument("--ortho_alg", required=False, default=1, type=np.int, help="Orthogonalization scheme for ECG. O:odir, 1:omin"
                        )
    parser.add_argument("--bs_red", required=False, default=0, type=np.int, help="Use dynamic search reduction"
                        )
    parser.add_argument(
        "--conserve-memory",
        dest="conserve_memory",
        required=False,
        action="store_true",
        help="Conserve memory when staging libMappraiser buffers [default]",
    )
    parser.add_argument(
        "--no-conserve-memory",
        dest="conserve_memory",
        required=False,
        action="store_false",
        help="Do not conserve memory when staging libMappraiser buffers",
    )
    parser.set_defaults(conserve_memory=True)

    # `nside` may already be added
    try:
        parser.add_argument(
            "--nside", required=False, default=512, type=np.int, help="Healpix NSIDE"
        )
    except argparse.ArgumentError:
        pass
    # Common flag mask may already be added
    try:
        parser.add_argument(
            "--common-flag-mask",
            required=False,
            default=1,
            type=np.uint8,
            help="Common flag mask",
        )
    except argparse.ArgumentError:
        pass
    # `sample-rate` may be already added
    try:
        parser.add_argument(
            "--sample-rate",
            required=False,
            default=100.0,
            type=np.float,
            help="Detector sample rate (Hz)",
        )
    except argparse.ArgumentError:
        pass

    parser.add_argument(
        "--epsilon-frac-mean",
        dest="epsilon_frac_mean",
        required=False,
        default=None,
        type=np.double,
        help="Mean fractional noise difference in detector pairs"
    )

    parser.add_argument(
        "--epsilon-frac-sd",
        dest="epsilon_frac_sd",
        required=False,
        default=None,
        type=np.double,
        help="Std deviation of fractional noise difference in detector pairs (if 0, all pairs will take the value given by --epsilon-frac-mean"
    )

    parser.add_argument(
        "--epsilon-seed",
        dest="epsilon_seed",
        required=False,
        default=None,
        type=np.int,
        help="Specify seed for distribution of epsilon values in detector pairs"
    )

    parser.add_argument(
        "--white-noise",
        dest="white_noise",
        required=False,
        action="store_true",
        help="Generate simple white noise (gaussian) for each detector independently."
    )
    parser.add_argument(
        "--no-white-noise",
        dest="white_noise",
        required=False,
        action="store_false",
        help="Do not generate white noise [default]"
    )
    parser.set_defaults(white_noise=False)

    parser.add_argument(
        "--white-noise-NET",
        dest="white_noise_NET",
        required=False,
        # 400 µK.√s typical for SAT detector NET (high PSD freqs)
        default=400e-6,
        type=np.double,
        help="NET (K.√s) of the white noise signal."
    )

    parser.add_argument(
        "--my-common-mode",
        dest="my_common_mode",
        required=False,
        default=None,
        help="String defining analytical parameters of a common mode added to all detectors: 'fmin[Hz],fknee[Hz],alpha,NET[K.√s]'. Remove argument if no common mode is to be generated."
    )

    parser.add_argument(
        "--custom-noise-only",
        dest="custom_noise_only",
        required=False,
        action="store_true",
        help="Only generate custom noise (white and/or common mode)."
    )
    parser.add_argument(
        "--no-custom-noise-only",
        dest="custom_noise_only",
        required=False,
        action="store_false",
        help="Include TOAST generated noise [default]"
    )
    parser.set_defaults(custom_noise_only=False)

    parser.add_argument(
        "--rng-seed",
        dest="rng_seed",
        required=False,
        default=None,
        type=np.int,
        help="Specify seed for custom noise generation (white noise and common mode). MUST be specified when using custom common mode noise."
    )

    parser.add_argument(
        "--ignore-dets",
        dest="ignore_dets",
        required=False,
        default=None,
        type=np.uint8,
        help="Ignore detectors to make half maps. 0->take all dets. 1->ignore odd dets. 2->ignore even dets"
    )

    parser.add_argument(
        "--save-noise-psd",
        dest="save_noise_psd",
        required=False,
        action="store_true",
        help="Save the PSD of the simulated noise timestream for detector 0."
    )
    parser.set_defaults(save_noise_psd=False)

    return


@function_timer
def setup_mappraiser(args):
    """ Create a Mappraiser parameter dictionary.

    Initialize the Mappraiser parameters from the command line arguments.

    """
    params = {}

    params["nside"] = args.nside
    params["Lambda"] = args.Lambda
    params["uniform_w"] = args.uniform_w
    params["samplerate"] = args.sample_rate
    params["output"] = args.outpath
    params["ref"] = args.ref
    params["solver"] = args.solver
    params["precond"] = args.precond
    params["Z_2lvl"] = args.Z_2lvl
    params["pointing_commflag"] = args.ptcomm_flag
    params["tol"] = args.tol
    params["maxiter"] = args.maxiter
    params["enlFac"] = args.enlFac
    params["ortho_alg"] = args.ortho_alg
    params["bs_red"] = args.bs_red

    # custom noise generation
    params["white_noise"] = args.white_noise
    params["white_noise_NET"] = args.white_noise_NET
    params["my_common_mode"] = args.my_common_mode
    params["rng_seed"] = args.rng_seed
    params["custom_noise_only"] = args.custom_noise_only

    # noise level differences in detector pairs
    params["epsilon_frac_mean"] = args.epsilon_frac_mean
    params["epsilon_frac_sd"] = args.epsilon_frac_sd
    params["epsilon_seed"] = args.epsilon_seed
    params["ignore_dets"] = args.ignore_dets

    params["save_noise_psd"] = args.save_noise_psd

    return params


@function_timer
def apply_mappraiser(
    args,
    comm,
    data,
    params,
    signalname,
    noisename,
    time_comms=None,
    telescope_data=None,
    verbose=True,
):
    """ Use libmappraiser to run the ML map-making

    Args:
        time_comms (iterable) :  Series of disjoint communicators that
            map, e.g., seasons and days.  Each entry is a tuple of
            the form (`name`, `communicator`)
        telescope_data (iterable) : series of disjoint TOAST data
            objects.  Each entry is tuple of the form (`name`, `data`).
    """
    if comm.comm_world is None:
        raise RuntimeError("Mappraiser requires MPI")

    log = Logger.get()
    total_timer = Timer()
    total_timer.start()
    if comm.world_rank == 0 and verbose:
        log.info("Making maps")

    mappraiser = OpMappraiser(
        params=params,
        purge=True,
        name=signalname,
        noise_name=noisename,
        conserve_memory=args.conserve_memory,
        pair_diff=args.pair_diff,
    )

    if time_comms is None:
        time_comms = [("all", comm.comm_world)]

    if telescope_data is None:
        telescope_data = [("all", data)]

    timer = Timer()
    for time_name, time_comm in time_comms:
        for tele_name, tele_data in telescope_data:
            if len(time_name.split("-")) == 3:
                # Special rules for daily maps
                if args.do_daymaps:
                    continue
                if len(telescope_data) > 1 and tele_name == "all":
                    # Skip daily maps over multiple telescopes
                    continue

            timer.start()
            # N.B: code below is for Madam but may be useful to copy in Mappraiser
            # once we start doing multiple maps in one run
            # madam.params["file_root"] = "{}_telescope_{}_time_{}".format(
            #     file_root, tele_name, time_name
            # )
            # if time_comm == comm.comm_world:
            #     madam.params["info"] = info
            # else:
            #     # Cannot have verbose output from concurrent mapmaking
            #     madam.params["info"] = 0
            # if (time_comm is None or time_comm.rank == 0) and verbose:
            #     log.info("Mapping {}".format(madam.params["file_root"]))
            mappraiser.exec(tele_data, time_comm)

            if time_comm is not None:
                time_comm.barrier()
            if comm.world_rank == 0 and verbose:
                timer.report_clear("Mapping {}_telescope_{}_time_{}".format(
                    args.outpath,
                    tele_name,
                    time_name,
                ))

    if comm.comm_world is not None:
        comm.comm_world.barrier()
    total_timer.stop()
    if comm.world_rank == 0 and verbose:
        total_timer.report("Mappraiser total")

    return
