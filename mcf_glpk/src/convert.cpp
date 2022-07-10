#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "Graph.h"

const Graph::CNumber C_INF = Graph::Inf<Graph::CNumber>();
const Graph::FNumber F_INF = Graph::Inf<Graph::FNumber>();

std::ostream&
print_network_info(std::ostream& os, Graph& graph)
{
    os << graph.NrNodes() << '\n';
    os << graph.NrArcs() << '\n';
    os << graph.NrComm() << '\n';

    // Output all arcs and their capacities
    auto start_nodes = graph.StartN();
    auto end_nodes = graph.EndN();
    auto mutual_capacities = graph.TotCapacities();
    for (Graph::Index i = 0; i < graph.NrArcs(); i++) {

        // Sum capacities over all commodities
        Graph::CNumber sum_capacity{ 0 };
        for (Graph::Index k = 0; k < graph.NrComm(); k++) {
            auto capacity = graph.CapacityKJ(k, i);
            sum_capacity = capacity;

            if (sum_capacity >= F_INF)
                break;
        }

        os << start_nodes[i] << '\t' << end_nodes[i];

        auto arc_capacity = std::min(sum_capacity, mutual_capacities[i]);
        os << '\t';
        if (arc_capacity == F_INF) {
            os << -1;
        } else {
            os << arc_capacity;
        }

        os << '\n';
    }

    // Print information about the deficits of nodes
    for (Graph::Index i = 0; i < graph.NrNodes(); i++) {
        os << (i + 1) << ":\n";
        for (Graph::Index k = 0; k < graph.NrComm(); k++) {
            auto deficit = graph.DeficitKJ(k, i);
            os << '\t' << k << " -> " << deficit << '\n';
        }
    }

    return os;
}

int
main(int argc, char** argv)
{

    std::vector<std::string> args{ argv, argv + argc };

    char type;
    bool fxdc = false;
    bool fxdf = false;
    char optn = 'n';

    switch (argc) {
        case (5):
            optn = *argv[4];

        case (4):
            type = *argv[3];
            break;

        default:
            std::cerr
              << "Usage: convert in_file out_file format [option]"
              << '\n'
              << " format = s, c, p, o, d, u, m (lower or uppercase)" << '\n'
              << " option = n, p, s, b (none, pre-process, single-source, both)"
              << std::endl;

            return EXIT_FAILURE;
    }

    // Read the problem
    Graph Gh(argv[1], type);

    // Make the Graph single-sourced
    if ((optn == 's') || (optn == 'b'))
        Gh.MakeSingleSourced();

    // Pre-process the problem
    if ((optn == 'p') || (optn == 'b'))
        Gh.PreProcess();

    // Output the problem in MPS format
    print_network_info(std::cout, Gh);

    return EXIT_SUCCESS;
}
