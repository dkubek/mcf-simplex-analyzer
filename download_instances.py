#!/usr/bin/env python
"""Tool for automatically downloading and extracting Linear MCF instances. """

import pathlib
import tarfile
import urllib.request

DEFAULT_DATA_DIR = "data/instances"
""" Path to the defeault instances directory """

INSTANCE_LINKS = {
    "mnetgen": "http://groups.di.unipi.it/optimize/Data/MMCF/Mnetgen.tgz",
    "pds": "http://groups.di.unipi.it/optimize/Data/MMCF/Pds100.tar.gz",
    "jpf": "http://groups.di.unipi.it/optimize/Data/MMCF/JLF.tgz",
    "hydrothermal": (
        "http://groups.di.unipi.it/optimize/Data/MMCF/hydroter.tar.gz"
    ),
    "vance": "http://groups.di.unipi.it/optimize/Data/MMCF/vance.tar.gz",
    "aertranspo": (
        "http://groups.di.unipi.it/optimize/Data/MMCF/AerTrans.tar.gz"
    ),
    "planar": "http://groups.di.unipi.it/optimize/Data/MMCF/planarmnet.tgz",
    "grid": "http://groups.di.unipi.it/optimize/Data/MMCF/gridmnet.tgz",
}


def main():
    # Ensure path exists
    data_path = pathlib.Path(DEFAULT_DATA_DIR).absolute()
    data_path.mkdir(parents=True, exist_ok=True)

    print("Downloading ...")
    for instance in INSTANCE_LINKS:
        print(f"\tDownloading {instance} instance ...")

        with urllib.request.urlopen(INSTANCE_LINKS[instance]) as fin:
            with tarfile.open(fileobj=fin, mode="r:gz") as tar:
                print(f"\tExtracting {instance} ... ")
                tar.extractall(data_path)
                print(f"\t{instance} instance extracted.")

        print(f"\t{instance} instance downloaded\n")

    print("Downloaded ... ")


if __name__ == "__main__":
    main()
