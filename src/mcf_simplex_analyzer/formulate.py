"""
Formulate a linear programming instance

TODO:
    - Use numpy types? Do they offer speed benefits?
    - Use Fractions in networkx? Does it impact performance?
"""

import logging

from fractions import Fraction

import attr
import numpy as np
import networkx as nx

from mcf_simplex_analyzer.load_instance import Instance


SOURCE_IDENTIFIER = "_#SOURCE_"
DESTINATION_IDENTIFIER = "_#DESTINATION_"
SOURCE_FMT = SOURCE_IDENTIFIER + "{}"
DESTINATION_FMT = DESTINATION_IDENTIFIER + "{}"


@attr.s
class NetworkInfo:
    """ Contains info about the input network """

    sources = attr.ib()
    destinations = attr.ib()
    capacities = attr.ib()
    lcm = attr.ib()


def formulate_concurrent_flow_problem(instance: Instance):
    """ Formulate a LP problem for the given instance. """

    logger = logging.getLogger(__name__)
    logger.info("Formulating the LP problem ...")

    flow = find_max_flows(instance)

    logger.debug("%s", flow)
    logger.info("Done formulating the problem")

    return flow


def find_max_flows(instance: Instance):
    """
    Find the maximal flow for each commodity satisfying the `y` multiple of
    demands.
    """

    logger = logging.getLogger(__name__)
    logger.info("Finding maximal flows for commodities.")

    network_info = collect_network_info(instance)
    logger.debug("Network info=%s", network_info)

    graph = consturct_network(network_info)

    flow_sum = {}
    for commodity in range(1, instance.info.products_no + 1):
        flow_dict = find_max_flow(commodity, graph)
        sum_flows(flow_dict, flow_sum)

    for from_node in flow_sum:
        for to_node in flow_sum[from_node]:
            flow_sum[from_node][to_node] = Fraction(
                flow_sum[from_node][to_node], network_info.lcm
            )

    return flow_sum


def sum_flows(flow_dict, flow_sum):
    """ Sum two flows. """

    for from_node in flow_dict:
        if isinstance(from_node, str) and SOURCE_IDENTIFIER in from_node:
            continue

        for to_node in flow_dict[from_node]:
            if isinstance(to_node, str) and DESTINATION_IDENTIFIER in to_node:
                continue

            to_dict = flow_sum.setdefault(from_node, {})
            to_dict[to_node] = (
                to_dict.get(to_node, 0) + flow_dict[from_node][to_node]
            )


def find_max_flow(commodity, graph):
    """ Find maximum flow for the given commodity. """

    logger = logging.getLogger(__name__)
    logger.info("Computing maximum flow for commodity=%s", commodity)

    _, flow_dict = nx.maximum_flow(
        graph,
        SOURCE_FMT.format(commodity),
        DESTINATION_FMT.format(commodity),
    )

    return flow_dict


def collect_network_info(instance: Instance):
    """ Collect information about the network """

    logger = logging.getLogger(__name__)
    logger.debug("Collecting network information ...")

    capacities = collect_capacities(instance)
    sources, destinations = collect_sources_destinations(instance)

    sources, destinations, capacities, lcm = _normalize_to_whole_numbers(
        capacities, destinations, sources
    )

    network_info = NetworkInfo(sources, destinations, capacities, lcm)

    logger.debug("Done collecting network information.")

    return network_info


def _normalize_to_whole_numbers(capacities, destinations, sources):
    """ Normalize the fractions to whole numbers. """

    logger = logging.getLogger(__name__)
    logger.debug("Normalizing fractions to whole numbers ...")

    # Normalize to whole numbers
    denominators = (
        [
            capacity.denominator
            for capacity in capacities.values()
            if capacity is not None
        ]
        + [sources[s][k].denominator for s in sources for k in sources[s]]
        + [
            destinations[t][k].denominator
            for t in destinations
            for k in destinations[t]
        ]
    )
    lcm = np.lcm.reduce(denominators)

    logger.debug("lcm=%d", lcm)

    # Normalize capacities
    for key in capacities:
        capacity = capacities[key]
        if capacity is not None:
            capacities[key] = capacity.numerator * (
                lcm // capacity.denominator
            )

    # Normalize sources
    for source in sources:
        for commodity in sources[source]:
            supply = sources[source][commodity]
            sources[source][commodity] = supply.numerator * (
                lcm // supply.denominator
            )

    # Normalize destinations
    for destination in destinations:
        for commodity in destinations[destination]:
            demand = destinations[destination][commodity]
            destinations[destination][commodity] = demand.numerator * (
                lcm // demand.denominator
            )

    return sources, destinations, capacities, lcm


def consturct_network(network_info):
    """ Construct network graph given network info """

    graph = nx.DiGraph()

    for (from_node, to_node) in network_info.capacities:
        capacity = network_info.capacities[(from_node, to_node)]
        if capacity is not None:
            graph.add_edge(from_node, to_node, capacity=capacity)
        else:
            graph.add_edge(from_node, to_node)

    for source in network_info.sources:
        for commodity in network_info.sources[source]:
            supply = network_info.sources[source][commodity]
            graph.add_edge(
                SOURCE_FMT.format(commodity), source, capacity=supply
            )

    for destination in network_info.destinations:
        for commodity in network_info.destinations[destination]:
            demand = network_info.destinations[destination][commodity]
            graph.add_edge(
                destination, DESTINATION_FMT.format(commodity), capacity=demand
            )

    return graph


def collect_capacities(instance: Instance):
    """
    Collect edge capacities and commodities from the instance.
    """

    logger = logging.getLogger(__name__)
    logger.debug("Collecting edge capacities and commodities ...")

    capacities = _collect_capacities(instance)
    _merge_capacity_with_mutual(capacities)

    return capacities


def _collect_capacities(instance: Instance):
    """ Collect capacities from instance """

    logger = logging.getLogger(__name__)

    capacities = {}
    for arc in instance.arcs:
        logger.debug("Processing arc: %s", arc)

        (
            from_node,
            to_node,
            _,
            _,
            individual_capacity,
            mutual_ptr,
        ) = arc

        if from_node < 0 or to_node < 0:
            logger.warning("Invalid source or destination node:\n%s", arc)

        mutual, capacity = capacities.get((from_node, to_node), (None, None))

        if mutual_ptr > 0:
            mutual = instance.mutual.mapping[mutual_ptr]
        else:
            logger.warning("Invalid mutual ptr:\n%s", arc)

        if individual_capacity >= 0:
            capacity = (
                capacity + individual_capacity
                if capacity is not None
                else individual_capacity
            )

        capacities[(from_node, to_node)] = (mutual, capacity)

    return capacities


def _merge_capacity_with_mutual(capacities):
    """ Merge capacities with mutual capacity """

    logger = logging.getLogger(__name__)
    logger.debug("Merging capacities with mutual capacities ...")

    for key in capacities:
        mutual, total = capacities[key]
        final_capacity = (
            max(mutual, total)
            if mutual is not None and total is not None
            else total
        )
        logger.debug("%s: %s", key, final_capacity)

        capacities[key] = final_capacity


def collect_sources_destinations(instance: Instance):
    """
    Collect source and destination nodes, with their respective supplies and
    demands.
    """

    logger = logging.getLogger(__name__)
    logger.debug("Collecting source and destination nodes...")

    sources = {}
    destinations = {}

    for supply in instance.supply:
        source, destination, commodity, flow = supply

        if source != -1 and destination != -1:
            logger.warning("Invalid supply entry: %s", supply)

        if source == -1:
            logger.debug(
                "Adding destination node: %s with demand %s", destination, flow
            )
            destinations.setdefault(destination, dict())[commodity] = flow

        if destination == -1:
            logger.debug("Adding source node: %s with supply %s", source, flow)
            sources.setdefault(source, dict())[commodity] = flow

    return sources, destinations
