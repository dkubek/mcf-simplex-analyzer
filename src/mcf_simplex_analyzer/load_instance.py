"""
Load a MCF instance

TODO:
    - Add option to specify the precision for input floating point values
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, MutableMapping, Sequence, Dict

import numpy as np
from mcf_simplex_analyzer.fractionarray import FractionArray


SUPPORTED_INSTANCES = ("mnetgen", "pds", "planar", "grid", "jlf")
""" Supported instance formats """

MUT_FIELD_TYPES = {
    "mutual_ptr": np.int32,
    "capacity": Fraction,
}

ARC_FIELD_TYPES = {
    "fromnode": np.int32,
    "tonode": np.int32,
    "commodity": np.int32,
    "cost": Fraction,
    "individual_capacity": Fraction,
    "mutual_ptr": np.uint32,
}

SUP_FIELD_TYPES = {
    "node": np.int32,
    "commodity": np.int32,
    "origin": np.int32,
    "destination": np.int32,
    "flow": Fraction,
}


# TODO: report real number of edges, the links_no currently denotes the number
#       edges before being merged together
@dataclass
class InstanceInfo:
    """ Information about the instance. """

    products_no: int
    nodes_no: int
    links_no: int
    bundled_links_no: int


@dataclass
class SupplyInfo:
    """ Inforormation about supply """

    origin: np.ndarray
    destination: np.ndarray
    commodity: np.ndarray
    flow: FractionArray

    def __len__(self):
        return len(self.origin)

    def __getitem__(self, key):
        return (
            self.origin[key],
            self.destination[key],
            self.commodity[key],
            self.flow[key],
        )

    def __iter__(self):
        return iter(
            zip(
                self.origin,
                self.destination,
                self.commodity,
                self.flow,
            )
        )

    def __reversed__(self):
        return reversed(self.__iter__())


@dataclass
class ArcInfo:
    """ Information about arcs """

    fromnode: np.ndarray
    tonode: np.ndarray
    commodity: np.ndarray
    cost: FractionArray
    individual_capacity: FractionArray
    mutual_ptr: np.ndarray

    def __len__(self):
        return len(self.fromnode)

    def __getitem__(self, key):
        return (
            self.fromnode[key],
            self.tonode[key],
            self.commodity[key],
            self.cost[key],
            self.individual_capacity[key],
            self.mutual_ptr[key],
        )

    def __iter__(self):
        return iter(
            zip(
                self.fromnode,
                self.tonode,
                self.commodity,
                self.cost,
                self.individual_capacity,
                self.mutual_ptr,
            )
        )

    def __reversed__(self):
        return reversed(self.__iter__())


@dataclass
class MutualInfo:
    """ Information about mutual cappacities """

    mapping: Dict[int, Fraction]


@dataclass
class Instance:
    """ Instance """

    info: InstanceInfo
    arcs: ArcInfo
    mutual: MutualInfo
    supply: SupplyInfo


def read_nod(nod_file: Path) -> InstanceInfo:
    """" Read a nod_file """

    logger = logging.getLogger(__name__)
    logger.debug("Reading nod file: %s.", nod_file)

    content = None
    with open(nod_file, "r") as fin:
        content = fin.read()

    products_no, nodes_no, links_no, bundled_links_no, *_ = map(
        int, content.split()
    )

    logger.debug(
        "products_no=%d, nodes_no=%d, links_no=%d, bundled_links_no=%d",
        products_no,
        nodes_no,
        links_no,
        bundled_links_no,
    )

    return InstanceInfo(
        products_no=products_no,
        nodes_no=nodes_no,
        links_no=links_no,
        bundled_links_no=bundled_links_no,
    )


def _to_array_types(
    data: MutableMapping[str, Any], types: Sequence[str]
) -> MutableMapping[str, Any]:
    ans = {}

    for field in data:
        dtype = types[field]

        arr = None
        if dtype is Fraction:
            numerators = np.fromiter(
                map(lambda frac: frac.numerator, data[field]), dtype=np.int64
            )
            denominators = np.fromiter(
                map(lambda frac: frac.denominator, data[field]), dtype=np.int64
            )

            arr = FractionArray(
                numerator=numerators, denominator=denominators
            )
        else:
            arr = np.array(data[field], dtype=dtype)

        ans[field] = arr

    return ans


def _read_file(
    file: Path,
    types: MutableMapping[str, Any],
    fields: Sequence[str],
) -> MutableMapping[str, Any]:
    """ Read a file with a template. '_' means ignore """

    data = defaultdict(list)
    with open(file, "r") as fin:
        for line in fin:
            for field, value in zip(fields, line.split()):
                if field == "_":
                    continue

                data[field].append(types[field](value))

    return _to_array_types(data, types)


def read_arc(arc_file: Path, instance_format: str) -> ArcInfo:
    """ Read arc file """

    logger = logging.getLogger(__name__)
    logger.debug(
        "Reading arc file %s, instance_format=%s", arc_file, instance_format
    )

    fields = None
    if instance_format in ["mnetgen", "pds", "planar", "grid"]:
        fields = (
            "_",
            "fromnode",
            "tonode",
            "commodity",
            "cost",
            "individual_capacity",
            "mutual_ptr",
        )
    elif instance_format == "jlf":
        fields = (
            "fromnode",
            "tonode",
            "commodity",
            "cost",
            "individual_capacity",
            "_",
            "_",
            "mutual_ptr",
        )
    else:
        raise NotImplementedError("read_arc: Not implemented!")

    data = _read_file(arc_file, ARC_FIELD_TYPES, fields)

    return ArcInfo(**data)


def read_sup(sup_file: Path, instance_format: str):
    """ Read sup file """

    logger = logging.getLogger(__name__)
    logger.debug(
        "Reading sup file %s, instance_format=%s", sup_file, instance_format
    )

    fields = None
    if instance_format in ["mnetgen", "pds", "planar", "grid"]:
        fields = ("node", "commodity", "flow")
    elif instance_format == "jlf":
        fields = ("origin", "destination", "commodity", "flow")
    else:
        raise NotImplementedError("read_sup: Not implemented!")

    data = _read_file(sup_file, SUP_FIELD_TYPES, fields)

    # Split the node field into origin/destination by inspecting the flow
    if instance_format in ["mnetgen", "pds", "planar", "grid"]:
        nums = data["flow"].numerator
        sign = np.sign(nums)
        origin = -np.ones_like(data["node"])
        destination = -np.ones_like(data["node"])

        mask = sign < 0
        destination[mask] = data["node"][mask]

        mask = sign > 0
        origin[mask] = data["node"][mask]

        data["origin"] = origin
        data["destination"] = destination

        nums *= sign

        del data["node"]

    return SupplyInfo(**data)


def read_mut(mut_file: Path, instance_format: str):
    """ Read mut file """

    logger = logging.getLogger(__name__)
    logger.debug(
        "Reading mut file %s, instance_format=%s", mut_file, instance_format
    )

    fields = ("mutual_ptr", "capacity")
    if instance_format not in ["mnetgen", "pds", "jlf", "planar", "grid"]:
        raise NotImplementedError("read_arc: Not implemented!")

    data = _read_file(mut_file, MUT_FIELD_TYPES, fields)
    mutual_mapping = dict(zip(data["mutual_ptr"], data["capacity"]))

    return MutualInfo(mapping=mutual_mapping)


def load_instance(
    instance_format: str,
    nod_file: Path,
    arc_file: Path,
    sup_file: Path,
    mut_file: Path,
) -> Instance:
    """ Load given instance """

    instanceinfo = read_nod(nod_file)
    arcinfo = read_arc(arc_file, instance_format)
    supinfo = read_sup(sup_file, instance_format)
    mutinfo = read_mut(mut_file, instance_format)

    return Instance(
        info=instanceinfo, arcs=arcinfo, supply=supinfo, mutual=mutinfo
    )
