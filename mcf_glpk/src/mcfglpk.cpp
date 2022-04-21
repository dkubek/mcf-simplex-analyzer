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


constexpr glp_stmcp GLP_DEFAULT_STMCP = {
        .objective_trace = GLP_OBJECTIVE_TRACE_ON,
        .basis_trace = GLP_BASIS_TRACE_ON,
        .nonbasis_trace = GLP_NONBASIS_TRACE_ON,
        .complexity_trace = GLP_COMPLEXITY_TRACE_ON,
        .pivot_rule = GLP_TRACE_PIVOT_DANTZIG,
};

std::ostream &get_trace(std::ostream &os, const std::string &filename) {
    glp_prob *P;
    P = glp_create_prob();

    glp_smcp params = DEFAULT_GLP_SMCP;
    // params.it_lim = 1;
    params.msg_lev = GLP_MSG_ERR;

    // Read the problem from a file
    glp_read_lp(P, NULL, filename.c_str());

    // Construct initial basis
    glp_adv_basis(P, 0);
    // glp_std_basis(P);

    auto ncols = glp_get_num_cols(P);
    auto nrows = glp_get_num_rows(P);

    std::vector<std::string> names;
    names.push_back("");
    for (size_t i = 1; i <= nrows; i++) {
        std::string name(glp_get_row_name(P, i));
        names.push_back(name);
    }

    for (size_t j = 1; j <= ncols; j++) {
        std::string name(glp_get_col_name(P, j));
        names.push_back(name);
    }

    auto trace = glp_create_ssxtrace(&GLP_DEFAULT_STMCP);

    glp_exact_trace(P, &params, trace);
    // glp_exact(P, &params);

    // Print solution
    mpz_t num, den;
    mpq_t value;
    mpq_init(value);
    mpz_init(num);
    mpz_init(den);

    for (size_t i = 0; i < trace->no_iterations; ++i) {
        mpq_set(value, trace->objective_values[i]);
        mpq_get_den(den, value);
        mpq_get_num(num, value);
        auto bits = mpz_sizeinbase(den, 2) + mpz_sizeinbase(num, 2);

        os << value << '\t' << bits << '\n';
    }

    size_t start = (trace->no_iterations - 1) * trace->no_basic;
    size_t k = trace->no_basic + trace->no_nonbasic;
    os << "Number of names: " << names.size() - 1 << '\n';
    for (size_t i = 0; i < trace->no_basic; ++i) {
        auto variable = trace->bases[start + i];
        auto s = trace->status[(trace->no_iterations - 1) * (k + 1) + variable];

        mpq_set(value, trace->basic_values[start + i]);
        mpq_get_den(den, value);
        mpq_get_num(num, value);
        auto bits = mpz_sizeinbase(den, 2) + mpz_sizeinbase(num, 2);

        os << variable << '\t' << s << '\t' << names[variable] << '\t' << value
           << '\t' << bits << '\n';
    }

    for (size_t i = 1; i <= k; i++) {
        auto s = trace->status[(trace->no_iterations - 1) * (k + 1) + i];
        if (s == GLP_BS) continue;

        os << i << '\t' << s << '\t' << names[i] << '\t';

        switch (s) {
            case GLP_NL:
                mpq_set(value, trace->lb[i]);
                break;
            case GLP_NU:
                mpq_set(value, trace->ub[i]);
                break;
            case GLP_NS:
                mpq_set(value, trace->lb[i]);
                break;
            default:
                throw std::invalid_argument("UNKNOWN STATUS");
        }

        mpq_get_den(den, value);
        mpq_get_num(num, value);
        auto bits = mpz_sizeinbase(den, 2) + mpz_sizeinbase(num, 2);
        os << value << '\t' << bits << '\n';
    }

    glp_print_sol(P, "out.txt");

    mpq_clear(value);
    mpz_clear(den);
    mpz_clear(num);
    glp_ssxtrace_free(trace);
    glp_delete_prob(P);

    return os;
}

std::ostream &get_basis_factorizations(std::ostream &os,
                                       const std::string &filename) {
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

        if (!glp_bf_exists(P)) glp_factorize(P);

        for (int i = 1; i <= nrows; i++) { os << glp_get_bhead(P, i) << ' '; }
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
    while (glp_get_status(P) != GLP_OPT) { glp_exact(P, &params); }
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

    // compare_execution_time(args[1]);
    //{
    //     std::ofstream fout{"basis_factorizations.txt"};
    //     get_basis_factorizations(fout, args[1]);
    // }
    {
        std::ofstream fout{"basis_values.txt"};
        get_trace(std::cout, args[1]);
    }

    return EXIT_SUCCESS;
}
