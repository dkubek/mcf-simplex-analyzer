#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <ratio>
#include <string>
#include <vector>

#include <Graph.h>
#define HAVE_GMP
#include <glpk.h>

constexpr char USAGE[] = "Usage: mcfglpk mpsfile";
constexpr glp_smcp DEFAULT_GLP_SMCP = {
        .msg_lev = GLP_MSG_ALL,
        .meth = GLP_PRIMAL,
        .pricing = GLP_PT_PSE,
        .r_test = GLP_RT_STD,
        .tol_bnd = 1e-7,
        .tol_dj = 1e-7,
        .tol_piv = 1e-9,
        .obj_ll = -std::numeric_limits<double>::max(),
        .obj_ul = std::numeric_limits<double>::max(),
        .it_lim = std::numeric_limits<int>::max(),
        .tm_lim = std::numeric_limits<int>::max(),
        .out_frq = 500,
        .out_dly = 0,
        .presolve = GLP_OFF};

std::ostream &
get_basis_value(std::ostream &os, const std::string &filename) {
    glp_prob *P;
    P = glp_create_prob();

    glp_smcp params = DEFAULT_GLP_SMCP;
    //params.it_lim = 1;
    params.msg_lev = GLP_MSG_ERR;

    // Read the problem from a file
    glp_read_mps(P, GLP_MPS_FILE, NULL, filename.c_str());

    // Construct initial basis
    glp_adv_basis(P, 0);
    // glp_std_basis(P);

    auto ncols = glp_get_num_cols(P);
    auto nrows = glp_get_num_rows(P);

    auto info = glp_create_dbginfo();

    //int count = 0;
    //size_t it = 0;
    //while (glp_get_status(P) != GLP_OPT) {
    //    // Solve the problem step by step, day by day
    //    glp_exact_debug(P, &params, info);

    //    std::cerr << count++ << ":\t";
    //    if (!info->updated)
    //        std::cerr << "NOT UPDATED" << '\n';
    //    else {
    //        std::cerr << "UPDATED" << '\n';
    //        os << "Objective val: " << info->objective_values[it] << '\n';
    //        for (int i = 0; i < info->no_basic; ++i) {
    //            size_t offset = it * info->no_basic;
    //            os << info->basic_values[offset + i] << ' ';
    //        }
    //        os << '\n';
    //        it++;
    //        if (it > 10)
    //            break;

    //    }
    //}

    glp_exact_debug(P, &params, info);

    for (size_t i = 0; i < info->no_iterations; ++i)
        os << info->objective_values[i] << '\n';

    glp_dbginfo_free(info);
    glp_delete_prob(P);

    return os;
}

std::ostream &
get_basis_factorizations(std::ostream &os, const std::string &filename) {
    glp_prob *P;
    P = glp_create_prob();

    glp_smcp params = DEFAULT_GLP_SMCP;
    params.it_lim = 1;
    params.msg_lev = GLP_MSG_ERR;

    // Read the problem from a file
    glp_read_mps(P, GLP_MPS_FILE, NULL, filename.c_str());

    // Construct initial basis
    glp_adv_basis(P, 0);
    // glp_std_basis(P);

    auto ncols = glp_get_num_cols(P);
    auto nrows = glp_get_num_rows(P);

    // Solve the problem step by step, day by day
    while (glp_get_status(P) != GLP_OPT) {
        glp_exact(P, &params);

        if (!glp_bf_exists(P))
            glp_factorize(P);

        for (int i = 1; i <= nrows; i++) {
            os << glp_get_bhead(P, i) << ' ';
        }
        os << '\n';
    }

    glp_delete_prob(P);
    return os;
}

void compare_execution_time(const std::string &filename) {
    glp_prob *P;
    P = glp_create_prob();

    glp_smcp params = DEFAULT_GLP_SMCP;
    params.it_lim = 1;
    params.msg_lev = GLP_MSG_ON;

    // Read the problem from a file
    glp_read_mps(P, GLP_MPS_FILE, NULL, filename.c_str());

    // Construct initial basis
    glp_adv_basis(P, 0);
    // glp_std_basis(P);

    // Print solution
    // glp_print_sol(P, "out.txt");

    auto ncols = glp_get_num_cols(P);
    auto nrows = glp_get_num_rows(P);

    // if (!glp_bf_exists(P))
    //     glp_factorize(P);

    // for (int i = 1; i <= nrows; i++) {
    //     std::cout << glp_get_bhead(P, i) << ' ';
    // }
    // std::cout << '\n';

    // Solve the problem step by step, day by day
    std::cout << "Computing solution iteration by iteration ...\n";
    auto start = std::chrono::steady_clock::now();
    while (glp_get_status(P) != GLP_OPT) {
        glp_exact(P, &params);
    }
    auto end = std::chrono::steady_clock::now();
    auto diff_partial = end - start;
    std::cout << "Finished.\n";

    // Solve the problem all at once
    std::cout << "Computing all at once ...\n";

    glp_read_mps(P, GLP_MPS_FILE, NULL, filename.c_str());
    glp_adv_basis(P, 0);

    start = std::chrono::steady_clock::now();
    params.it_lim = std::numeric_limits<int>::max();
    glp_exact(P, &params);
    end = std::chrono::steady_clock::now();
    auto diff_complete = end - start;

    std::cout << "Finished.\n";

    // Report results
    std::cout << "STEP BY STEP:\t"
              << std::chrono::duration<double, std::milli>(diff_partial).count()
              << " ms" << std::endl;
    std::cout
            << "COMPLETE:\t"
            << std::chrono::duration<double, std::milli>(diff_complete).count()
            << " ms" << std::endl;

    // Cleanup
    glp_delete_prob(P);
}

int main(int argc, char *argv[]) {
    std::vector<std::string> args{argv, argv + argc};

    if (args.size() != 2) {
        std::cerr << USAGE << '\n';
        return EXIT_FAILURE;
    }

    //compare_execution_time(args[1]);
    //{
    //    std::ofstream fout{"basis_factorizations.txt"};
    //    get_basis_factorizations(fout, args[1]);
    //}
    {
        std::ofstream fout{"basis_values.txt"};
        get_basis_value(std::cout, args[1]);
    }

    return EXIT_SUCCESS;
}
