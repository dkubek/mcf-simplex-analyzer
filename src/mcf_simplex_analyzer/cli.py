""" CLI interface for mcf-simplex """
import logging
import sys
from pathlib import Path

import click

from .load_instance import SUPPORTED_INSTANCES, load_instance
from .formulate import formulate_concurrent_flow_problem


def initialize_logging(log_level=1):
    """ Initialize logger """

    logging.basicConfig()
    logging.root.setLevel(logging.DEBUG)
    logger = logging.getLogger("mcf_simplex")

    if log_level == 1:
        logger.setLevel(logging.INFO)
    elif log_level > 1:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

    return logger


@click.group()
@click.option("-v", "--verbose", count=True)
def debug(verbose):
    initialize_logging(verbose)


@debug.command()
@click.argument("instance_format", type=click.Choice(SUPPORTED_INSTANCES))
@click.option(
    "--basename",
    type=str,
    default=None,
    help="Basename of the file."
    "Will try to find nod, sup/od, arc, mut files in the directory."
    "Other options will be ignored.",
)
@click.option(
    "-n",
    "--nod-file",
    type=click.Path(exists=True, file_okay=True),
    help="Path to nod file.",
)
@click.option(
    "-a",
    "--arc-file",
    type=click.Path(exists=True, file_okay=True),
    help="Path to arc file.",
)
@click.option(
    "-s",
    "--sup-file",
    type=click.Path(exists=True, file_okay=True),
    help="Path to sup file.",
)
@click.option(
    "-m",
    "--mut-file",
    type=click.Path(exists=True, file_okay=True),
    help="Path to mut file.",
)
def load(instance_format, basename, nod_file, arc_file, sup_file, mut_file):
    logger = logging.getLogger(__name__)
    logger.debug(
        "Instance_format=%s, basename=%s, nod_file=%s, "
        "sup_file=%s, arc_file=%s, mut_file=%s",
        instance_format,
        basename,
        nod_file,
        sup_file,
        arc_file,
        mut_file,
    )

    if basename is None and nod_file is None:
        click.echo(
            click.style("No instance has been provided! Exiting.", fg="red")
        )
        sys.exit(1)

    if basename:
        nod_file = Path(basename + ".nod")
        arc_file = Path(basename + ".arc")
        sup_file = Path(basename + ".sup")
        mut_file = Path(basename + ".mut")

        assert nod_file.exists()
        assert arc_file.exists()
        assert sup_file.exists()
        assert mut_file.exists()

    instance = load_instance(
        instance_format, nod_file, arc_file, sup_file, mut_file
    )
    flow = formulate_concurrent_flow_problem(instance)

    logger.debug("Loaded instance=%s", instance)
    logger.debug("Summed flow=%s", flow)
