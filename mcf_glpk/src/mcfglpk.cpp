#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <Graph.h>

#define HAVE_GMP

#include <glpk.h>

#include <string.h>
#include <cxxopts.hpp>

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
        .store_mem = GLP_STORE_TRACE_MEM_OFF,

        .objective_trace = GLP_OBJECTIVE_TRACE_ON,
        .basis_trace = GLP_BASIS_TRACE_ON,
        .nonbasis_trace = GLP_NONBASIS_TRACE_ON,
        .complexity_trace = GLP_COMPLEXITY_TRACE_ON,
        //.pivot_rule = GLP_TRACE_PIVOT_BEST,
        .pivot_rule = GLP_TRACE_PIVOT_DANTZIG,

        .info_file_basename = {'\0'},
        .objective_values_file_basename = {'\0'},
        .status_file_basename = {'\0'},
        .variable_values_file_basename = {'\0'},
};

std::vector<std::string> get_names(glp_prob *P);

void print_info(std::ostream &os, glp_prob *P);

std::ostream& manual_memory_trace_example(std::ostream& os, const std::string& lp_filename) {
    // TODO: Pretty output
    glp_prob *P;
    P = glp_create_prob();

    glp_smcp params = DEFAULT_GLP_SMCP;
    // params.it_lim = 1;
    params.msg_lev = GLP_MSG_ERR;

    // Read the problem from a file
    glp_read_lp(P, NULL, lp_filename.c_str());

    // Construct initial basis
    glp_adv_basis(P, 0);

    auto trace_params = GLP_DEFAULT_STMCP;
    auto trace = glp_create_ssxtrace(&trace_params);
    glp_exact_trace(P, &params, trace);

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

    std::vector<std::string> names = get_names(P);
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


std::ostream &get_trace(std::ostream &os, const std::string &lp_filename, const std::string &info_filename,
                        const std::string &objective_filename, const std::string &status_filename,
                        const std::string &variable_filename, int pivot_rule) {
    glp_prob *P;
    P = glp_create_prob();

    glp_smcp params = DEFAULT_GLP_SMCP;
    // params.it_lim = 1;
    params.msg_lev = GLP_MSG_ERR;

    // Read the problem from a file
    glp_read_lp(P, NULL, lp_filename.c_str());

    if (!info_filename.empty()) {
        std::ofstream fout{info_filename};
        if (fout) {
            print_info(fout, P);
        }
    }

    // Construct initial basis
    glp_adv_basis(P, 0);

    auto trace_params = GLP_DEFAULT_STMCP;
    strcpy(trace_params.status_file_basename, status_filename.c_str());
    strcpy(trace_params.objective_values_file_basename, objective_filename.c_str());
    strcpy(trace_params.variable_values_file_basename, variable_filename.c_str());
    trace_params.pivot_rule = pivot_rule;

    auto trace = glp_create_ssxtrace(&trace_params);

    glp_exact_trace(P, &params, trace);

    glp_ssxtrace_free(trace);
    glp_delete_prob(P);

    return os;
}

void print_info(std::ostream &os, glp_prob *P) {
    auto ncols = glp_get_num_cols(P);
    auto nrows = glp_get_num_rows(P);

    os << "--- START INFO ---" << '\n';
    os << "Rows: " << nrows << '\n';
    os << "Cols: " << ncols << '\n';
    os << "--- END INFO ---" << '\n';

    os << "--- START NAMES ---" << '\n';
    std::vector<std::string> names = get_names(P);
    for (size_t i = 1; i < names.size(); i++)
        os << names[i] << '\n';
    os << "--- END NAMES ---" << '\n';
}

std::vector<std::string> get_names(glp_prob *P) {
    auto ncols = glp_get_num_cols(P);
    auto nrows = glp_get_num_rows(P);

    std::vector<std::string> names;
    names.resize(1);
    for (size_t i = 1; i <= nrows; i++) {
        std::string name(glp_get_row_name(P, i));
        names.push_back(name);
    }

    for (size_t j = 1; j <= ncols; j++) {
        std::string name(glp_get_col_name(P, j));
        names.push_back(name);
    }

    return names;
}


int main(int argc, char *argv[]) {
    cxxopts::Options options("mcfglpk", "Compute trace statistics from exact simplex.");

    options.add_options()
            ("lp-file", "Linear programming problem in LP format", cxxopts::value<std::string>())
            ("pivot", "Pivoting rule to use. Available : dantzig, best", cxxopts::value<std::string>())
            ("info-file", "Info file name", cxxopts::value<std::string>())
            ("obj-file", "Objective values file name", cxxopts::value<std::string>())
            ("status-file", "Status file name", cxxopts::value<std::string>())
            ("var-file", "Variables file name", cxxopts::value<std::string>())
            ("h,help", "Print usage");
    options.parse_positional({"lp-file"});
    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    std::string lp_file;
    std::string info_filename;
    std::string objective_filename;
    std::string status_filename;
    std::string variable_filename;
    if (result.count("lp-file"))
        lp_file = result["lp-file"].as<std::string>();
    if (result.count("info-file"))
        info_filename = result["info-file"].as<std::string>();
    if (result.count("obj-file"))
        objective_filename = result["obj-file"].as<std::string>();
    if (result.count("status-file"))
        status_filename = result["status-file"].as<std::string>();
    if (result.count("var-file"))
        variable_filename = result["var-file"].as<std::string>();

    int pivot_rule;
    if (!result.count("pivot")) {
        std::cerr << "Need to specify pivoting rule!" << '\n';
        return EXIT_FAILURE;
    }

    auto pivot_rule_name = result["pivot"].as<std::string>();
    if (pivot_rule_name == "dantzig") {
        pivot_rule = GLP_TRACE_PIVOT_DANTZIG;
    }
    else if ( pivot_rule_name == "best" ) {
        pivot_rule = GLP_TRACE_PIVOT_BEST;
    }
    else  {
        std::cerr << "Unknown pivoting rule: " << pivot_rule_name << " !" << '\n';
        return EXIT_FAILURE;
    }

    get_trace(std::cout,
              lp_file,
              info_filename,
              objective_filename,
              status_filename,
              variable_filename,
              pivot_rule);

    return EXIT_SUCCESS;
}
