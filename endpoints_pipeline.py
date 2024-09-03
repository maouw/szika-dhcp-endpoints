#!/usr/bin/env python3

# Built-in modules:
import logging
import os
import posixpath
import inspect
import shutil
import sys
import time
from pathlib import Path
import re

# Third-party modules:
import nibabel as nib
import numpy as np
import pandas as pd
import fsspec  # for S3 file system operations
import sh  # for shell commands
import tqdm  # for progress bars
from tqdm.contrib.concurrent import thread_map  # for parallel processing
from fsspec.callbacks import TqdmCallback  # for progress bars
from loguru import logger  # for logging


def timecmd(*args, quiet=False, **kwargs):
    # def on_done(cmd, success, exit_code):
    #     if success:
    #         logger.info(open(tf.name).readlines()[0])
    kwargs.update(dict(_tty_out=False, _err_to_out=True))
    logger.debug(f"Running {args=} with {kwargs=}")
    if quiet:
        logger.debug("sh.{cmd}: {line}", cmd=args[0])
        kwargs["_no_out"] = True
    else:
        kwargs["_iter"] = True
    _log_level = kwargs.pop("_log_level", "INFO" if not quiet else "DEBUG")
    c = sh.Command("time").bake(**kwargs)
    if not args:
        raise ValueError("No command provided")
    if not isinstance(args[0], str):
        if isinstance(args[0][0], str):
            args = args[0]
        else:
            raise ValueError(f"Expected string, got {type(args[0])} for args[0] ({args=})")
    actual_cmd = args[0]
    if not isinstance(actual_cmd, str):
        raise ValueError(f"Expected string, got {type(actual_cmd)} for actual_cmd ({args=})")
    c = c.bake(actual_cmd)
    c = c.bake(*args[1:])
    if quiet:
        return c()
    else:
        for line in c():  # type: ignore
            line = line.rstrip()  # remove trailing newline # type: ignore
            if not line.strip():
                continue
            if actual_cmd == "mri_vol2surf" and not any(
                x in line for x in ("exit_status=", "srcvol")
            ):  # Skip some extra output
                continue
            logger.log(_log_level, "sh.{cmd}: {line}", cmd=actual_cmd, line=line)


# Wrapper for timecmd that suppresses output in a way usable by thread_map
def timecmd_quiet(*args, **kwargs):
    return timecmd(*args, quiet=True, **kwargs)


# Bundle names

# Required bundles:
requiredLeftBundles = sorted(set(['ARCL', 'ATRL', 'CGCL', 'CSTL', 'FA', 'FP', 'IFOL', 'ILFL', 'MdLFL', 'ORL', 'SLFL', 'UNCL', 'VOFL', 'pARCL']), key=lambda x: x.lower())  # fmt: skip
requiredRightBundles = sorted(set(['ARCR', 'ATRR', 'CGCR', 'CSTR', 'FA', 'FP', 'IFOR', 'ILFR', 'MdLFR', 'ORR', 'SLFR', 'UNCR', 'VOFR', 'pARCR']), key=lambda x: x.lower())  # fmt: skip
required_bundles = sorted({*requiredRightBundles, *requiredLeftBundles})
bundles_in_both_hemispheres = sorted(set(requiredLeftBundles).intersection(requiredRightBundles))

# Projection values to generate
proj_values = tuple(np.arange(-3, 3.5, 0.5))

# TQDM settings
default_tqdm_kwargs = {
    "ascii": True,
}


# Progress bar functions
def progress(*args, **kwargs):
    """progress bar"""
    kwargs.update(default_tqdm_kwargs)
    return tqdm.tqdm(
        *args,
        **kwargs,
    )


def progress_fs(*args, **kwargs):
    """progress bar for fsspec, tailored for file system operations"""
    kwargs.update(default_tqdm_kwargs)
    kwargs.setdefault("unit", "file")
    return TqdmCallback(tqdm_kwargs=kwargs)


# Intercept standard logging and redirect it to Loguru
class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists.
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        # Find caller from where originated the logged message.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


# Check if session output directory exists:
def check_session_output_dir_exists(subject, session, s3_output_root, **kwargs):
    return fsspec.filesystem("s3", use_listings_cache=False).exists(
        f"s3://{s3_output_root}/sub-{subject}/ses-{session}"
    )


# Find session output bundles:
def find_session_output_bundles(subject, session, s3_output_root, **kwargs):
    fs = kwargs.get("fs", fsspec.filesystem("s3"))
    csvs = fs.glob(f"{s3_output_root}/output/sub-{subject}/ses-{session}/t1t2values/GreyMatterT1T2_*_[RL]H.csv")
    found = {}
    for csv in csvs:
        bundle = (
            posixpath.basename(csv)
            .removeprefix("GreyMatterT1T2_")
            .removesuffix(".csv")
            .removesuffix("_LH")
            .removesuffix("_RH")
        )
        if bundle not in required_bundles:
            continue
        found[bundle] = csv
    return found


# Get bundle name from trk filename:
def trk_filename_to_bundle_name(trk_filename):
    f = str(trk_filename)
    try:
        return f[f.index("desc-") :].split("_")[0].removeprefix("desc-")
    except (ValueError, IndexError):
        logger.warning(f"Could not parse bundle name from {f}")
        return None


# Find bundles for a session:
def bundles_for_session(subject, session, s3_input_root, **kwargs) -> dict[str, str]:
    fs = kwargs.get("fs", fsspec.filesystem("s3"))
    return {
        bundle_name: path
        for path in fs.glob(
            f"{s3_input_root}/sub-{subject}/ses-{session}/bundles/sub-{subject}_ses-{session}_coordsys-RASMM_trkmethod-probCSD_recogmethod-AFQ_desc-*_tractography.trk"
        )
        if (bundle_name := trk_filename_to_bundle_name(path)) and bundle_name in required_bundles
    }


# List all sessions:
def list_all_sessions(s3_input_root, **kwargs):
    fs = kwargs.get("fs", fsspec.filesystem("s3"))
    return [
        (
            posixpath.basename(posixpath.dirname(posixpath.dirname(paths[0]))).removeprefix("sub-"),
            posixpath.basename(posixpath.dirname(paths[0])).removeprefix("ses-"),
        )
        for x in progress(fs.glob(f"{s3_input_root}/sub-*/"))
        if progress(paths := fs.glob(f"{x}/ses-*/bundles/"), desc="Finding all sessions", unit="session")
    ]


# Find sessions that do not have all required bundles:
def find_invalid_sessions(s3_input_root, as_df=False, **kwargs):
    fs = kwargs.pop("fs", fsspec.filesystem("s3"))
    logger.info(f"Looking for valid sessions in {s3_input_root}... This may take a while.")
    sessions = kwargs.pop("sessions", list_all_sessions(s3_input_root=s3_input_root, fs=fs))
    invalid_sessions = {
        k: b
        for k in progress(sessions, desc="Scanning for valid sessions", unit="session")
        if (b := set(required_bundles).difference(bundles_for_session(*k, fs=fs).keys()))
    }
    if as_df:
        invalid_sessions = pd.DataFrame(
            [(*k, ",".join(sorted(v))) for k, v in invalid_sessions.items()],
            index=None,
            columns=("subject", "session", "missing"),
        )
    return invalid_sessions


# Find sessions that have all required bundles:
def find_valid_sessions(s3_input_root, as_df=False, **kwargs):
    fs = kwargs.pop("fs", fsspec.filesystem("s3"))
    logger.info(f"Looking for valid sessions in {s3_input_root}... This may take a while.")
    sessions = kwargs.pop("sessions", list_all_sessions(s3_input_root=s3_input_root, fs=fs))
    valid_sessions = [
        s
        for s in progress(sessions, desc="Scanning for valid sessions", unit="session")
        if (b := bundles_for_session(*s, fs=fs)) and set(b.keys()).issuperset(required_bundles)
    ]
    if as_df:
        valid_sessions = pd.DataFrame(valid_sessions, index=None, columns=("subject", "session"))
        valid_sessions["bundles"] = ",".join(required_bundles)
    return valid_sessions


# Pipeline function:
def dhcp_pyafq_pipeline(subject, session, **kwargs):
    """
    For each `subject`, `session` pair run tractography pipeline.
    systemctl --user start docker-desktop
    0. Perpare local environment
    1. Generate isotropic DWI files
    2. Realign DWI to anatomy using rigid transformation
    3. Estimate CSD response function
    4. Reassign ribbon values
    5. Generate five-tissue-type (5TT) segmented tissue image
    6. Tractography
    7. Create BIDS derivatives
    8. Upload to s3

    Parameters
    ----------
    subject : string
    session : string
    aws_access_key : string
    aws_secret_key : string
    streamline_count : int
    local_env : boolean
    """
    from loguru import logger

    started_at = time.localtime()
    timestamp_start = time.strftime("%Y%m%dT%H%M%S%Z", started_at)

    # Global settings:
    os.environ.setdefault("MRTRIX_QUIET", "1")
    os.environ.setdefault(
        "TIME", "elapsed=%E system_time=%S cpu_percent=%P max_rss=%M major_faults=%F exit_status=%x user=%U command=%C"
    )

    # Disable PAGER for sh commands:
    os.environ.pop("PAGER", None)

    # Initialize environment variables:
    pipeline_ncores = int(os.environ.setdefault("DHCP_PIPELINE_NCORES", value=str(len(os.sched_getaffinity(0)))))
    s3_input_root = os.environ.setdefault("DHCP_PIPELINE_S3_INPUT_ROOT", value="dhcp-afq/afq_rel3/output")
    s3_output_root = os.environ.setdefault("DHCP_PIPELINE_S3_OUTPUT_ROOT", value="dhcp-afq/rel3PipelineSteffi/output")

    os.environ.setdefault("DHCP_PIPELINE_IGNORE_EXISTING", "0")  # Set to 1 to ignore existing output directories
    os.environ.setdefault("OMP_NUM_THREADS", str(pipeline_ncores))
    os.environ.setdefault("SUBJECTS_DIR", "output")
    log_dir = os.environ.setdefault("DHCP_PIPELINE_LOG_DIR", str(Path(Path.cwd(), "logs").expanduser().resolve()))
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Set up loguru logger:
    logger.remove()
    logger_fmt = " ".join(
        ["{time:YYYY-MM-DDTHH:mm:ss}", f"{subject}_{session}", "{level}", "{module}", "<level>{message}</level>"]
    )
    logger_fmt_nocolor = " ".join(
        ["{time:YYYY-MM-DDTHH:mm:ss}", f"{subject}_{session}", "{level}", "{module}", "{message}"]
    )
    logger.add(sys.stderr, level=os.environ.get("DHCP_LOG_LEVEL", "INFO"), format=logger_fmt)
    log_path = Path(log_dir, f"steffi_pipeline_endpoints_sub-{subject}_ses-{session}-{timestamp_start}.log").resolve()
    logger.add(log_path, level=os.environ.get("DHCP_LOG_LEVEL", "INFO"), format=logger_fmt_nocolor, mode="a")

    # Intercept standard logging and redirect it to loguru:
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    logging.getLogger("sh").setLevel(logging.WARNING)

    # Create workdir and change to it:
    Path("workdir").mkdir(exist_ok=True)
    os.chdir("workdir")

    # Log the start of the pipeline:
    logger.info(f"Running pipeline for {subject=} {session=} with logging to {log_path}")
    logger.info(f"{s3_input_root=} {s3_output_root=}")

    # Set up S3 file system:
    fs = kwargs.get("fs", fsspec.filesystem("s3"))

    ###############################################################################
    # Step 0: Prepare local environment and download data
    ###############################################################################
    logger.info("--- Step 0: Prepare local environment")
    logger.info(f"Using {pipeline_ncores} cores")

    found_bundles = bundles_for_session(subject, session, s3_input_root=s3_input_root, fs=fs)
    if not found_bundles:
        logger.error(f"Missing bundle files for {subject=} {session=}")
        return ()

    missing_bundles = set(required_bundles).difference(found_bundles.keys())
    if missing_bundles:
        logger.warning(f"Missing bundle files for {subject=} {session=}: ({missing_bundles})")

    leftBundles = set(requiredLeftBundles).intersection(found_bundles.keys())
    rightBundles = set(requiredRightBundles).intersection(found_bundles.keys())

    if os.environ.get("DHCP_PIPELINE_IGNORE_EXISTING", "0") != "1" and fs.exists(
        found_output_dir := posixpath.join(s3_output_root, f"sub-{subject}/ses-{session}")
    ):
        logger.warning(f"Output directory for {subject=}, {session=} already exists at {found_output_dir}")
        return ()

    # Create output directories:
    Path("input", f"sub-{subject}", f"ses-{session}").mkdir(parents=True, exist_ok=True)
    logger.info("Deleting old labels and creating new directories")
    sh.Command("rm")("-rfv", f"output/sub-{subject}/ses-{session}/label")
    for x in (
        ("surf",),
        ("tracts",),
        ("volume",),
        ("label",),
        ("t1t2values",),
        ("label", "annotation"),
        ("label", "ROIs"),
        ("label", "FinalLabels"),
    ):
        Path("output", f"sub-{subject}", f"ses-{session}", *x).mkdir(parents=True, exist_ok=True)

    # Identify tracts that need to be downloaded:
    tracts_to_download = {}
    for k, v in found_bundles.items():
        if not Path(v.replace(s3_input_root, "output", 1).replace("/bundles/", "/tracts/", 1)).exists():
            tracts_to_download[k] = v

    if tracts_to_download:
        logger.info(
            f"Downloading tract files for {len(tracts_to_download)} of {len(required_bundles)} bundles not already present"
        )

        fs.get(
            list(tracts_to_download.values()),
            f"output/sub-{subject}/ses-{session}/tracts/",
            callback=progress_fs(desc="Downloading tractography files"),
        )
    else:
        logger.info("All required tract files already present")

    # Download anatomy files:
    anat_files = [
        f"sub-{subject}_ses-{session}_T1w.nii.gz",
        f"sub-{subject}_ses-{session}_hemi-left_wm.surf.gii",
        f"sub-{subject}_ses-{session}_hemi-right_wm.surf.gii",
        f"sub-{subject}_ses-{session}_hemi-left_myelinmap.shape.gii",
        f"sub-{subject}_ses-{session}_hemi-right_myelinmap.shape.gii",
    ]
    anat_files_to_download = [
        f"dhcp-afq/rel3_dhcp_anat_pipeline/sub-{subject}/ses-{session}/anat/{anat_file}"
        for anat_file in anat_files
        if not Path("input", f"sub-{subject}", f"ses-{session}", anat_file).exists()
    ]
    if anat_files_to_download:
        logger.info("Downloading anatomy files")
        fs.get(
            anat_files_to_download,
            f"input/sub-{subject}/ses-{session}/",
            callback=progress_fs(desc="Downloading anatomy files"),
        )

    # Download annotation files:
    annotation_files = [
        f"sub-{subject}_ses-{session}_hemi-left_desc-drawem_dseg.label.gii",
        f"sub-{subject}_ses-{session}_hemi-right_desc-drawem_dseg.label.gii",
    ]
    annotation_files_to_download = [
        f"dhcp-afq/rel3_dhcp_anat_pipeline/sub-{subject}/ses-{session}/anat/{annotation_file}"
        for annotation_file in annotation_files
        if not Path("output", f"sub-{subject}", f"ses-{session}", annotation_file).exists()
    ]
    if annotation_files_to_download:
        logger.info("Downloading annotation files")
        fs.get(
            annotation_files_to_download,
            f"output/sub-{subject}/ses-{session}/label/",
            callback=progress_fs(desc="Downloading annotation files"),
        )

    ###############################################################################
    # Step 1: Convert tract files from .trk to .tck
    ###############################################################################
    logger.info(f"--- Step 1: Convert tract files from .trk to .tck for bundles {*found_bundles.keys(),}")
    bundles_to_convert = {}
    for bundle in found_bundles:
        path = Path(found_bundles[bundle].replace(s3_input_root, "output", 1).replace("/bundles/", "/tracts/", 1))
        if path.with_suffix(".tck").exists():
            logger.info(f"Skipping {bundle=} as output already exists")
        else:
            bundles_to_convert[bundle] = path

    # Set up function to convert trk to tck:
    def run_nib_trk2tck(path):
        timecmd_quiet("nib-trk2tck", path)

    # Run trk2tck conversion in parallel:
    thread_map(
        run_nib_trk2tck,
        list(bundles_to_convert.values()),
        max_workers=pipeline_ncores,
        desc="uni32->uni64",
        unit="file",
        **default_tqdm_kwargs,
    )

    ###############################################################################
    # Step 2: Use tckmap to find endpoints in volume space
    ###############################################################################
    logger.info("--- Step 2: Use tckmap to find endpoints in volume space")
    for bundle in progress(found_bundles, desc="tckmap", unit="bundle"):
        if Path(
            f"output/sub-{subject}/ses-{session}/volume/sub-{subject}_ses-{session}_tckmap_bundle-{bundle}.nii.gz"
        ).exists():
            logger.info(f"Skipping {bundle=} as output already exists")
            continue
        timecmd(
            "tckmap",
            "-nthreads",
            pipeline_ncores,
            "-template",
            f"input/sub-{subject}/ses-{session}/sub-{subject}_ses-{session}_T1w.nii.gz",
            "-ends_only",
            "-quiet",
            "-contrast",
            "tdi",
            "-force",
            f"output/sub-{subject}/ses-{session}/tracts/sub-{subject}_ses-{session}_coordsys-RASMM_trkmethod-probCSD_recogmethod-AFQ_desc-{bundle}_tractography.tck",
            f"output/sub-{subject}/ses-{session}/volume/sub-{subject}_ses-{session}_tckmap_bundle-{bundle}.nii.gz",
        )

    ###############################################################################
    # Step 3: Double endpoint-niftii files from uni32 to uni64
    ###############################################################################
    logger.info("--- Step 3: Double endpoint-niftii files from uni32 to uni64")

    def run_nii_uni32_to_uni64(*args):
        bundle_input, bundle_output = args[0][:2]
        nii = nib.load(str(bundle_input))  # type: ignore
        niidata = nii.get_fdata()  # type: ignore
        niidouble = np.double(niidata)
        niidouble_img = nib.Nifti1Image(niidouble, nii.affine)  # type: ignore
        nib.save(  # type: ignore
            niidouble_img,
            str(bundle_output),
        )

    to_convert = [
        [
            str(bundle_output).replace("_tckmapUNI64_bundle-", "_tckmap_bundle-"),
            str(bundle_output),
        ]
        for bundle in found_bundles
        if not (
            bundle_output := Path(
                f"output/sub-{subject}/ses-{session}/volume/sub-{subject}_ses-{session}_tckmapUNI64_bundle-{bundle}.nii.gz"
            )
        ).exists()
    ]

    thread_map(
        run_nii_uni32_to_uni64,
        to_convert,
        max_workers=pipeline_ncores,
        desc="uni32->uni64",
        unit="file",
        **default_tqdm_kwargs,
    )

    ###############################################################################
    # Step 4: Function to use mrivol2surf without freesurfer preprocessing (3rd release)
    ###############################################################################
    logger.info(
        "Step 4: Function to use mrivol2surf without freesurfer preprocessing",
    )

    def process_trg_surfaces(ref_volume, trg_surface, work_dir):
        ### create working volume directory
        mri_dir = Path(work_dir, "mri")
        mri_dir.mkdir(parents=True, exist_ok=True)

        ### convert reference volume into freesurfer mgz
        nib.save(nib.load(ref_volume), Path(mri_dir, "ref.mgz"))  # type: ignore
        ref_volume = nib.load(Path(mri_dir, "ref.mgz"))  # type: ignore
        xyz = ref_volume.header.get("Pxyz_c")  # type: ignore # volume center

        ### create working surface directory
        surf_dir = Path(work_dir, "surf")
        surf_dir.mkdir(parents=True, exist_ok=True)

        for trg_fname in progress(trg_surface, desc="Process coregistered surface file"):
            ### save coregistered surface file
            hemi_str = "lh" if "hemi-left" in trg_fname else "rh"
            suffix = re.sub("\..+", "", trg_fname.split("_")[-1])
            write_geometry_to = Path(surf_dir, f"{hemi_str}.{suffix}")

            if os.environ.get("DHCP_PIPELINE_IGNORE_EXISTING", "0") != "1" and write_geometry_to.exists():
                logger.info(f"Skipping {trg_fname=} as output already exists")
                continue
            ### load coordinates and faces of surface file
            coords, faces = nib.load(trg_fname).agg_data()  # type: ignore

            ### swap y and z axes (results in RAS coordinates)
            tmp = coords.copy()
            coords[:, 1] = tmp[:, 2]
            coords[:, 2] = tmp[:, 1]

            ### center surface vertices
            coords = np.apply_along_axis(lambda x: coords - x, -1, xyz)

            ### rotate surface vertices (flip left and right axes)
            theta = np.deg2rad(180)
            R = np.array(
                [
                    [np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)],
                ]
            )
            coords = np.matmul(R, coords.T).T
            nib.freesurfer.io.write_geometry(str(write_geometry_to), coords, faces)

    ###############################################################################
    # Step 5: Setting up trg_surfaces for mri_vol2surf
    ###############################################################################
    logger.info("--- Step 5: Setting up mri_vol2surf")
    ref_volume = f"input/sub-{subject}/ses-{session}/sub-{subject}_ses-{session}_T1w.nii.gz"
    trg_surfaces = [
        f"input/sub-{subject}/ses-{session}/sub-{subject}_ses-{session}_hemi-left_wm.surf.gii",
        f"input/sub-{subject}/ses-{session}/sub-{subject}_ses-{session}_hemi-right_wm.surf.gii",
    ]
    work_dir = f"output/sub-{subject}/ses-{session}/"
    process_trg_surfaces(ref_volume, trg_surfaces, work_dir)

    ###############################################################################
    # Step 6: Running mrivol2surf on both hemispheres
    ###############################################################################
    logger.info("--- Step 6: Running mrivol2surf on both hemispheres")

    # Generate mri_vol2surf commands to run in parallel:
    def generate_mri_vol2surf_cmds(subject, session, hemisphere, bundles):
        hemisphere = hemisphere[0].upper() + "H"
        return {
            (bundle, p): [
                "mri_vol2surf",
                "--sd",
                "output/",
                "--o",
                str(outpath),
                "--regheader",
                f"sub-{subject}/ses-{session}",
                "--hemi",
                hemisphere.lower(),
                "--surf",
                "wm",
                "--mov",
                f"output/sub-{subject}/ses-{session}/volume/sub-{subject}_ses-{session}_tckmapUNI64_bundle-{bundle}.nii.gz",
                "--ref",
                "ref.mgz",
                "--projdist",
                f"{p}",
                "--fwhm",
                "1",
            ]
            for p in proj_values
            for bundle in bundles
            if not (
                outpath := Path(
                    f"output/sub-{subject}/ses-{session}/surf/sub-{subject}_ses-{session}_mrivol2surf_bundle-{bundle}_{hemisphere.lower()}_surface_{p}.mgh"
                )
            ).exists()
        }

    left_cmds = generate_mri_vol2surf_cmds(subject, session, "LH", leftBundles)
    right_cmds = generate_mri_vol2surf_cmds(subject, session, "RH", rightBundles)
    bundle_cmds = [*left_cmds.values(), *right_cmds.values()]
    bundles_to_process = list(left_cmds.keys()) + list(right_cmds.keys())

    logger.info(
        f"Running mri_vol2surf on both hemispheres for {len(bundle_cmds)} bundles ({bundles_to_process[:20]}...)"
    )

    # Run mri_vol2surf in parallel:
    thread_map(
        timecmd_quiet,
        bundle_cmds,
        max_workers=pipeline_ncores,
        desc="mri_vol2surf",
        unit="bundle",
        **default_tqdm_kwargs,
    )

    ###############################################################################
    # Step 7: Running mriconcat on left hemisphere to find maximum endpoints
    ###############################################################################
    logger.info(
        "Step 7: Running mriconcat on left hemisphere to find maximum endpoints",
    )

    def do_mri_concat(subject, session, hemisphere, bundles, skip_existing=False):
        hemisphere = hemisphere[0].upper() + "H"
        for bundle in bundles:
            proj_max_mgh = Path(
                f"output/sub-{subject}/ses-{session}/surf/sub-{subject}_ses-{session}_bundle-{bundle}_{hemisphere.lower()}_proj_max.mgh"
            )
            if skip_existing and proj_max_mgh.exists():
                logger.info(f"Skipping {bundle=} as output already exists at {proj_max_mgh}")
                continue
            Path(proj_max_mgh).unlink(missing_ok=True)
            in_files = [
                proj_max_mgh.parent
                / f"sub-{subject}_ses-{session}_mrivol2surf_bundle-{bundle}_{hemisphere.lower()}_surface_{proj_value}.mgh"
                for proj_value in proj_values
            ]
            for f, p in zip(in_files, proj_values):
                if not Path(f).exists():
                    logger.error(f"Missing {f=} for {bundle=} {hemisphere=} {p=}")
                    raise FileNotFoundError(f)
            timecmd_quiet(
                "mri_concat",
                "--i",
                *in_files,
                "--o",
                f"output/sub-{subject}/ses-{session}/surf/sub-{subject}_ses-{session}_bundle-{bundle}_{hemisphere.lower()}_proj_max.mgh",
                "--max",
            )

    do_mri_concat(subject, session, "LH", leftBundles)

    ###############################################################################
    # Step 8: Running mriconcat on right hemisphere to find maximum endpoints
    ###############################################################################
    logger.info(
        "Step 8: Running mriconcat on right hemisphere to find maximum endpoints",
    )
    do_mri_concat(subject, session, "RH", rightBundles)

    ###############################################################################
    # Step 9: converting dHCP-surface files to freesurfer surface files left hemisphere
    ###############################################################################
    logger.info(
        "Step 9: converting dHCP-surface files to freesurfer surface files left hemisphere",
    )
    timecmd(
        "mris_convert",
        f"input/sub-{subject}/ses-{session}/sub-{subject}_ses-{session}_hemi-left_wm.surf.gii",
        f"output/sub-{subject}/ses-{session}/surf/lh.white",
    )

    ###############################################################################
    # Step 10: converting dHCP-surface files to freesurfer surface files right hemisphere
    ###############################################################################
    logger.info(
        "Step 10: converting dHCP-surface files to freesurfer surface files right hemisphere",
    )
    timecmd(
        "mris_convert",
        f"input/sub-{subject}/ses-{session}/sub-{subject}_ses-{session}_hemi-right_wm.surf.gii",
        f"output/sub-{subject}/ses-{session}/surf/rh.white",
    )

    ###############################################################################
    # Step 11: Load in annotation file and split it into different labels for left hemi
    ###############################################################################
    logger.info(
        "Step 11: Load in annotation file and split it into different labels for left hemi",
    )
    timecmd_quiet(
        "mri_annotation2label",
        "--subject",
        f"sub-{subject}/ses-{session}",
        "--sd",
        "output",
        "--hemi",
        "lh",
        "--annotation",
        f"output/sub-{subject}/ses-{session}/label/sub-{subject}_ses-{session}_hemi-left_desc-drawem_dseg.label.gii",
        "--outdir",
        f"output/sub-{subject}/ses-{session}/label/annotation",
    )

    ###############################################################################
    # Step 12: Load in annotation file and split it into different labels for right hemi
    ###############################################################################
    logger.info(
        "Step 12: Load in annotation file and split it into different labels for right hemi",
    )
    timecmd_quiet(
        "mri_annotation2label",
        "--subject",
        f"sub-{subject}/ses-{session}",
        "--sd",
        "output",
        "--hemi",
        "rh",
        "--annotation",
        f"output/sub-{subject}/ses-{session}/label/sub-{subject}_ses-{session}_hemi-right_desc-drawem_dseg.label.gii",
        "--outdir",
        f"output/sub-{subject}/ses-{session}/label/annotation",
    )

    ###############################################################################
    # Step 13: Function to insert text within .label file and to generate label tables
    ###############################################################################
    logger.info("--- Step 13: : Function to insert text within .label file and to generate label tables")

    def prepend_multiple_lines(file_name, list_of_lines):
        """Insert given list of strings as a new lines at the beginning of a file"""
        file_name = str(file_name)
        # define name of temporary dummy file
        dummy_file = file_name + ".bak"
        # open given original file in read mode and dummy file in write mode
        with open(file_name, "r") as read_obj, open(dummy_file, "w") as write_obj:
            # Iterate over the given list of strings and write them to dummy file as lines
            for line in list_of_lines:
                write_obj.write(line + "\n")
            # Read lines from original file one by one and append them to the dummy file
            for line in read_obj:
                write_obj.write(line)

        # remove original file
        Path(file_name).unlink(missing_ok=True)
        # Rename dummy file as the original file
        Path(dummy_file).rename(file_name)

    def process_label_table(file_path, skiprows=2, sep="  ", header=None, engine="python"):
        label_table = pd.read_table(file_path, skiprows=skiprows, sep=sep, header=header, engine=engine)
        label_table[4] = [x.split(" ")[0] for x in label_table[3]]
        label_table[5] = [x.split(" ")[-1] for x in label_table[3]]
        return label_table.drop(columns=3)

    def process_hemisphere_labels(subject: str, session: str, hemisphere: str, outfolder: str | os.PathLike):
        hemisphere = hemisphere[0].upper() + "H"
        other_hemisphere = "LH" if hemisphere == "RH" else "RH"

        Path(outfolder).mkdir(parents=True, exist_ok=True)
        logger.info(f"Processing labels for hemisphere {hemisphere} with {outfolder=}")
        bundle_to_labels = {
            "ARC": (
                ("Frontal_lobe",),
                (
                    "Anterior_temporal_lobe_lateral",
                    "Superior_temporal_gyrus_middle",
                    "Medial_and_inferior_temporal_gyri_anterior",
                    "Lateral_occipitotemporal_gyrus_gyrus_fusiformis_anterior",
                    "Lateral_occipitotemporal_gyrus_gyrus_fusiformis_posterior",
                    "Medial_and_inferior_temporal_gyri_posterior",
                ),
            ),
            "ATR": (("Frontal_lobe",), ()),
            "CGC": (
                ("Cingulate_gyrus_anterior", "Frontal_lobe"),
                (
                    "Gyri_parahippocampalis_et_ambiens_posterior",
                    "Cingulate_gyrus_posterior",
                    "Parietal_lobe",
                ),
            ),
            "CST": (("Superior_temporal_gyrus_posterior", "Frontal_lobe", "Parietal_lobe"), ()),
            "FA": (("Frontal_lobe",), ()),
            "FP": (("Occipital_lobe", "Parietal_lobe"), ()),
            "IFO": (("Frontal_lobe",), ("Occipital_lobe",)),
            "ILF": (
                (
                    "Anterior_temporal_lobe_medial",
                    "Anterior_temporal_lobe_lateral",
                    "Superior_temporal_gyrus_middle",
                    "Medial_and_inferior_temporal_gyri_anterior",
                ),
                (
                    "Occipital_lobe",
                    "Lateral_occipitotemporal_gyrus_gyrus_fusiformis_posterior",
                    "Medial_and_inferior_temporal_gyri_posterior",
                    "Parietal_lobe",
                ),
            ),
            "MdLF": (
                (
                    "Anterior_temporal_lobe_medial",
                    "Anterior_temporal_lobe_lateral",
                    "Gyri_parahippocampalis_et_ambiens_anterior",
                    "Superior_temporal_gyrus_middle",
                    "Medial_and_inferior_temporal_gyri_anterior",
                ),
                ("Occipital_lobe", "Parietal_lobe"),
            ),
            "OR": (("Occipital_lobe",), ()),
            "pARC": (
                (
                    "Superior_temporal_gyrus_posterior",
                    "Medial_and_inferior_temporal_gyri_posterior",
                    "Medial_and_inferior_temporal_gyri_anterior",
                    "Lateral_occipitotemporal_gyrus_gyrus_fusiformis_posterior",
                    "Lateral_occipitotemporal_gyrus_gyrus_fusiformis_anterior",
                    "Occipital_lobe",
                ),
                ("Parietal_lobe",),
            ),
            "SLF": (
                ("Insula", "Frontal_lobe"),
                ("Superior_temporal_gyrus_posterior", "Parietal_lobe"),
            ),
            "UNC": (
                ("Insula", "Frontal_lobe"),
                ("Anterior_temporal_lobe_medial", "Anterior_temporal_lobe_lateral"),
            ),
            "VOF": (
                (
                    "Medial_and_inferior_temporal_gyri_posterior",
                    "Lateral_occipitotemporal_gyrus_gyrus_fusiformis_posterior",
                    "Occipital_lobe",
                ),
                ("Parietal_lobe",),
            ),
        }

        ann_root = Path(f"output/sub-{subject}/ses-{session}/label/annotation")
        if not Path(ann_path := ann_root / f"{hemisphere.lower()}.???.label").exists():
            logger.error(f"MISSING: Skipping bundle_name=??? in {hemisphere=} because the file does not exist.")
        else:
            try:
                process_label_table(ann_path)
            except Exception as e:
                logger.error(
                    f"MISSING: Skipping bundle_name=??? in {hemisphere=} because the file could not be loaded due to {e}"
                )

        for bundle_name in bundle_to_labels:
            real_bundle_name = (
                bundle_name if bundle_name in bundles_in_both_hemispheres else bundle_name + hemisphere[0].upper()
            )
            if real_bundle_name not in found_bundles:
                logger.error(
                    f"MISSING: Skipping bundle_name=??? in {hemisphere=} because it is not among the found bundles."
                )
                continue
            for i, g in enumerate(bundle_to_labels[bundle_name]):
                lts = []
                for name in g:
                    ann_path = ann_root / f"{hemisphere.lower()}.{hemisphere[0].upper()}_{name}.label"
                    alternative_ann_path = ann_root / f"{hemisphere.lower()}.{other_hemisphere[0].upper()}_{name}.label"
                    if not ann_path.exists():
                        ann_found = alternative_ann_path.name if alternative_ann_path.exists() else ""
                        with (Path(outfolder) / "missing_label_files.txt").open("a") as f:
                            f.write(f"{hemisphere=} ann_needed={ann_path.name} {ann_found=}\n")
                        if ann_found:
                            logger.warning(
                                f"{ann_path.name} is missing for bundle {real_bundle_name} in {hemisphere=} but alternative found at {ann_found}. Will use this instead. You should rename the file so that it matches the hemisphere."
                            )
                            ann_path.symlink_to(alternative_ann_path.relative_to(ann_path.parent))
                        else:
                            logger.error(
                                f"MISSING: Skipping {ann_path.name} for bundle {real_bundle_name} in {hemisphere=} because the file does not exist and no alternative was found."
                            )
                            continue
                    try:
                        lts.append(process_label_table(ann_path))
                    except Exception as e:
                        logger.error(
                            f"Skipping {ann_path.name} for bundle {real_bundle_name} in {hemisphere=} because the file could not be loaded due to {e}"
                        )
                        continue
                if lts:
                    lt_output_path = Path(
                        outfolder, f"sub-{subject}_ses-{session}_{hemisphere}_{real_bundle_name}_{i+1:02d}.label"
                    )
                    try:
                        pd.concat(lts).to_csv(lt_output_path, header=None, index=None, sep=" ")
                    except Exception as e:
                        logger.error(f"Failed to save {lt_output_path} due to {e}. Continuing.")
                        with (Path(outfolder) / "failed_label_tables.txt").open("a") as f:
                            f.write(f"{lt_output_path}\n")

    ###############################################################################
    # Step 14: Load in labels for left hemisphere and save them in different ROIs
    ###############################################################################
    outfolder = f"output/sub-{subject}/ses-{session}/label/ROIs"
    logger.info(f"Step 14: Load in labels for left hemisphere and save them in different ROIs in {outfolder}")
    process_hemisphere_labels(subject, session, "LH", outfolder)

    ###############################################################################
    # Step 15: Load in labels for right hemisphere and save them in different ROIs
    ###############################################################################
    logger.info(f"Step 15: Load in labels for right hemisphere and save them in different ROIs in {outfolder=}")
    process_hemisphere_labels(subject, session, "RH", outfolder)

    ###############################################################################
    # Step 16: Merge surface endpoint file and label together by vertices to get endpoint-vertices
    ###############################################################################
    logger.info(
        "Step 16: Merge surface endpoint file and label together by vertices to get endpoint-vertices",
    )
    outfolder = f"output/sub-{subject}/ses-{session}/label/FinalLabels"

    def process_bundles_step16(subject, session, hemisphere, bundles, outfolder):
        hemisphere = hemisphere.upper()[0] + "H"
        for bundle in bundles:
            try:
                if not (
                    mgh_path := Path(
                        f"output/sub-{subject}/ses-{session}/surf",
                        f"sub-{subject}_ses-{session}_bundle-{bundle}_{hemisphere.lower()}_proj_max.mgh",
                    )
                ).exists():
                    logger.warning(f"MISSING: File {mgh_path} does not exist. Skipping {bundle=}")
                    continue
                endpoints = nib.freesurfer.mghformat.load(mgh_path)
                endpointValues = endpoints.get_fdata()
                dfEndpointMap = pd.DataFrame(endpointValues[:, 0])
                dfEndpointMap.columns = ["endpointValues"]
                dfEndpointMap["verticeNr"] = dfEndpointMap.index
                dfEndpointMap["verticeNr"] = dfEndpointMap["verticeNr"].astype(str)

                labelTableBundle01 = pd.read_table(
                    f"output/sub-{subject}/ses-{session}/label/ROIs/sub-{subject}_ses-{session}_{hemisphere}_{bundle}_01.label",
                    sep=" ",
                    header=None,
                )
                labelTableBundle01.columns = ["verticeNr", "x", "y", "z", "value"]
                labelTableBundle01["verticeNr"] = labelTableBundle01["verticeNr"].astype(str)
                labelTableMatch1 = pd.merge(labelTableBundle01, dfEndpointMap, on="verticeNr")

                labelTableMatch1 = labelTableMatch1[labelTableMatch1["endpointValues"] > 0.0001]
                labelTableMatch1 = labelTableMatch1.drop(["value"], axis=1)
                labelTableMatch1 = labelTableMatch1.drop_duplicates("verticeNr")
                labelTableMatch1.to_csv(
                    f"{outfolder}/sub-{subject}_ses-{session}_{hemisphere}_{bundle}_01_matched.label",
                    header=None,
                    index=None,
                    sep=" ",
                    mode="a",
                )

                length1 = len(labelTableMatch1.index)
                list_of_lines1 = [
                    f"#!ascii label  , from subject sub-{subject}_ses-{session} vox2ras=TkReg",
                    f"{length1}",
                ]
                prepend_multiple_lines(
                    f"output/sub-{subject}/ses-{session}/label/FinalLabels/sub-{subject}_ses-{session}_{hemisphere}_{bundle}_01_matched.label",
                    list_of_lines1,
                )

                if (
                    labelTableBundle02Path := Path(
                        f"output/sub-{subject}/ses-{session}/label/ROIs/sub-{subject}_ses-{session}_{hemisphere}_{bundle}_02.label"
                    )
                ).exists():
                    try:
                        labelTableBundle02 = pd.read_table(
                            labelTableBundle02Path,
                            sep=" ",
                            header=None,
                        )
                        labelTableBundle02.columns = ["verticeNr", "x", "y", "z", "value"]
                        labelTableBundle02["verticeNr"] = labelTableBundle02["verticeNr"].astype(str)

                        labelTableMatch2 = pd.merge(labelTableBundle02, dfEndpointMap, on="verticeNr")
                        labelTableMatch2 = labelTableMatch2[labelTableMatch2["endpointValues"] > 0.0001]
                        labelTableMatch2 = labelTableMatch2.drop(["value"], axis=1)
                        labelTableMatch2 = labelTableMatch2.drop_duplicates("verticeNr")
                        labelTableMatch2.to_csv(
                            f"{outfolder}/sub-{subject}_ses-{session}_{hemisphere}_{bundle}_02_matched.label",
                            header=None,
                            index=None,
                            sep=" ",
                            mode="a",
                        )

                        length2 = len(labelTableMatch2.index)

                        list_of_lines2 = [
                            f"#!ascii label  , from subject sub-{subject}_ses-{session} vox2ras=TkReg",
                            f"{length2}",
                        ]
                        prepend_multiple_lines(
                            f"output/sub-{subject}/ses-{session}/label/FinalLabels/sub-{subject}_ses-{session}_{hemisphere}_{bundle}_02_matched.label",
                            list_of_lines2,
                        )
                    except Exception as e:
                        logger.error(f"Skipping second label table for {bundle=} due to {e}")
                else:
                    pass

            except Exception as e:
                logger.error(f"Error processing {bundle=}: {e}")
                with Path(f"output/sub-{subject}/ses-{session}/label/FinalLabels/failed_bundles.txt").open("a") as f:
                    f.write(f"{bundle=}\n")
                continue

    process_bundles_step16(subject, session, "LH", leftBundles, outfolder)
    process_bundles_step16(subject, session, "RH", rightBundles, outfolder)

    ###############################################################################
    # Step 17. Convert surface myelinmap.gii files to .mgh files
    ###############################################################################
    logger.info("--- Step 17. Convert surface myelinmap.gii files to .mgh files")
    timecmd(
        "mri_convert",
        f"input/sub-{subject}/ses-{session}/sub-{subject}_ses-{session}_hemi-left_myelinmap.shape.gii",
        f"output/sub-{subject}/ses-{session}/surf/sub-{subject}_ses-{session}_hemi-L_space-T2w_myelinmap.shape.mgh",
    )

    timecmd(
        "mri_convert",
        f"input/sub-{subject}/ses-{session}/sub-{subject}_ses-{session}_hemi-right_myelinmap.shape.gii",
        f"output/sub-{subject}/ses-{session}/surf/sub-{subject}_ses-{session}_hemi-R_space-T2w_myelinmap.shape.mgh",
    )

    ###############################################################################
    # Step 18: Find myelinvalue in vertices of endpoints (left hemisphere)
    ###############################################################################
    logger.info(
        "Step 18: Find myelinvalue in vertices of endpoints (left hemisphere)",
    )

    def find_myelinvalues_in_endpoints(hemisphere, bundles):
        hemisphere = hemisphere.upper()[0] + "H"
        for bundle in bundles:
            logger.info(f"Processing bundle {bundle}")
            if not (
                mgh_path := Path(
                    f"output/sub-{subject}/ses-{session}/surf/sub-{subject}_ses-{session}_hemi-{hemisphere[0]}_space-T2w_myelinmap.shape.mgh"
                )
            ).exists():
                logger.warning(f"File {mgh_path} does not exist. Skipping {bundle=}")
                continue
            try:
                t1wt2wMap = nib.freesurfer.mghformat.load(mgh_path)
                t1wt2wValues = t1wt2wMap.get_fdata()
                t1wt2wValuesMap = pd.DataFrame(t1wt2wValues[:, 0])
                t1wt2wValuesMap.columns = ["t1wt2wValues"]
                t1wt2wValuesMap["verticeNr"] = t1wt2wValuesMap.index
                t1wt2wValuesMap["verticeNr"] = t1wt2wValuesMap["verticeNr"].astype(str)

                rois1 = pd.read_table(
                    f"output/sub-{subject}/ses-{session}/label/FinalLabels/sub-{subject}_ses-{session}_{hemisphere}_{bundle}_01_matched.label",
                    skiprows=[0, 1],
                    sep=" ",
                    header=None,
                )
                rois1.columns = ["verticeNr", "x", "y", "z", "endpointdensity"]
                rois1["verticeNr"] = rois1["verticeNr"].astype(str)
                t1wt2w_data_rois1 = pd.merge(t1wt2wValuesMap, rois1, on="verticeNr")
                t1wt2w_data_rois1["subjectID"] = f"{subject}"
                t1wt2w_data_rois1["sessionID"] = f"{session}"
                t1wt2w_data_rois1["tractID"] = f"{bundle}"
                t1wt2w_data_rois1["ROI"] = "1"
                t1wt2w_data_rois1["hemisphere"] = hemisphere[0].upper()
                try:
                    rois2 = pd.read_table(
                        f"output/sub-{subject}/ses-{session}/label/FinalLabels/sub-{subject}_ses-{session}_{hemisphere}_{bundle}_02_matched.label",
                        skiprows=[0, 1],
                        sep=" ",
                        header=None,
                    )
                    rois2.columns = ["verticeNr", "x", "y", "z", "endpointdensity"]
                    rois2["verticeNr"] = rois2["verticeNr"].astype(str)
                    t1wt2w_data_rois2 = pd.merge(t1wt2wValuesMap, rois2, on="verticeNr")
                    t1wt2w_data_rois2["subjectID"] = f"{subject}"
                    t1wt2w_data_rois2["sessionID"] = f"{session}"
                    t1wt2w_data_rois2["tractID"] = f"{bundle}"
                    t1wt2w_data_rois2["ROI"] = "2"
                    t1wt2w_data_rois2["hemisphere"] = hemisphere[0].upper()
                    t1wt2wdata = pd.concat([t1wt2w_data_rois1, t1wt2w_data_rois2], ignore_index=True)
                    t1wt2wdata.to_csv(
                        f"output/sub-{subject}/ses-{session}/t1t2values/GreyMatterT1T2_{bundle}_{hemisphere}.csv",
                        sep=" ",
                        mode="a",
                    )
                except Exception as e1:
                    try:
                        t1wt2w_data_rois1.to_csv(
                            f"output/sub-{subject}/ses-{session}/t1t2values/GreyMatterT1T2_{bundle}_{hemisphere}.csv",
                            sep=" ",
                            mode="a",
                        )
                    except Exception as e2:
                        logger.error(f"Error processing {bundle=} due to {e1} and {e2}")
                        with Path(f"output/sub-{subject}/ses-{session}/t1t2values/failed_bundles.txt").open("a") as f:
                            f.write(f"{bundle=}\n")
                        continue
            except Exception as e:
                logger.error(f"Error processing {bundle=}: {e}")
                with Path(f"output/sub-{subject}/ses-{session}/t1t2values/failed_bundles.txt").open("a") as f:
                    f.write(f"{bundle=}\n")
                continue

    find_myelinvalues_in_endpoints("LH", leftBundles)

    ###############################################################################
    # Step 19: Find myelinvalue in vertices of endpoints (right hemisphere)
    ###############################################################################
    logger.info(
        "Step 19: Find myelinvalue in vertices of endpoints (right hemisphere)",
    )
    find_myelinvalues_in_endpoints("RH", rightBundles)

    # Write pipeline finish time:
    completed_file = Path(f"output/sub-{subject}/ses-{session}") / (
        ".pipeline_completed" if len(set(found_bundles)) == len(set(required_bundles)) else ""
    )
    sh.date("-Is", _out=str(completed_file))

    ###############################################################################
    # Step 20. Upload to s3
    ###############################################################################
    # Remove intermediate files
    if os.environ.get("DHCP_PIPELINE_KEEP_PROJ_FILES", "0") == "0":
        logger.debug("Removing intermediate projection files (set DHCP_PIPELINE_KEEP_PROJ_FILES=1 to keep)")
        for p in proj_values:
            Path(
                f"output/sub-{subject}/ses-{session}/surf/sub-{subject}_ses-{session}_mrivol2surf_bundle-{bundle}_lh_surface_{p}.mgh"
            ).unlink(missing_ok=True)
            Path(
                f"output/sub-{subject}/ses-{session}/surf/sub-{subject}_ses-{session}_mrivol2surf_bundle-{bundle}_rh_surface_{p}.mgh"
            ).unlink(missing_ok=True)

    logger.info("--- Step 20. Upload to s3")
    if os.environ.get("DHCP_PIPELINE_NO_UPLOAD", "0") != "0":
        logger.warning("Skipping upload to s3 as DHCP_PIPELINE_NO_UPLOAD is set")
    else:
        destination = posixpath.join(s3_output_root, f"sub-{subject}/ses-{session}")
        if os.environ.get("DHCP_PIPELINE_IGNORE_EXISTING_S3", "0") != "0" and fs.exists(destination):
            logger.error(f"Output directory for {subject=}, {session=} already exists at {destination}")
            raise FileExistsError(f"Output directory already exists for {subject=} {session=} at {destination}")
        fs.put(f"output/sub-{subject}/ses-{session}", destination, recursive=True)
        logger.info(f"Uploaded output for {len(found_bundles)} bundles to {destination}")
        if os.environ.get("DHCP_PIPELINE_CLEAN_ALL", "0") != "0":
            logger.info("Removing all input and output files (set DHCP_PIPELINE_CLEAN_ALL=0 to keep)")
            shutil.rmtree(f"input/sub-{subject}/ses-{session}")
            shutil.rmtree(f"output/sub-{subject}/ses-{session}")
    elapsed = time.time() - time.mktime(started_at)
    logger.success(f"Pipeline complete in {int(elapsed)} seconds")
    return tuple(found_bundles)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Invalid arguments. Usage: python -m dhcp_pyafq_pipeline <subject> <session>", file=sys.stderr)
        sys.exit(1)
    subject, session = sys.argv[1:3]
    res = dhcp_pyafq_pipeline(subject.removeprefix("sub-"), session.removeprefix("ses-"))
    if not res:
        logger.error("Didn't process any bundles")
        sys.exit(2)
    else:
        sys.exit(0)
