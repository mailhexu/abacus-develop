#include "projector_netcdf.h"

#include "atomic_projectors.h"
#include "build_info.h"
#include "native_state.h"
#include "source_hamilt/module_xc/xc_functional.h"
#include "source_io/module_parameter/input_parameter.h"
#include "source_main/version.h"

#ifdef COMMIT_INFO
#include "commit.h"
#endif

#include <netcdf.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace ModuleIO
{
namespace Tb2j
{
// Serial root writer for the existing tb2j.projector_green/1.0 common schema
// (canonical TB2J ProjectorGreenData::save_netcdf/load_netcdf contract).
// Raw primal coefficients, the full covariant Gram and the complete BZ-weighted
// onsite XC vertex are published verbatim; no dualization, channel drop or
// schema fork happens here.
namespace
{
static_assert(sizeof(std::complex<double>) == 2 * sizeof(double),
              "TB2J NetCDF export relies on the standard complex-double layout");

constexpr double kHermitianTolerance = 1e-10; // module convention (metric_condition)

void check(const int status, const std::string& what)
{
    if (status != NC_NOERR)
    {
        throw std::runtime_error("TB2J projector NetCDF " + what + ": " + nc_strerror(status));
    }
}

bool finite(const std::complex<double> value)
{
    return std::isfinite(value.real()) && std::isfinite(value.imag());
}

void require_finite(const std::complex<double>* values, const std::size_t count,
                    const std::string& what)
{
    for (std::size_t i = 0; i < count; ++i)
    {
        if (!finite(values[i]))
        {
            throw std::runtime_error("Nonfinite TB2J " + what);
        }
    }
}

/// Storage-Hermiticity integrity check, never a repair: reject improper data.
void require_hermitian(const std::complex<double>* matrix, const int n, const std::string& what)
{
    double scale = 0.0;
    const std::size_t size = static_cast<std::size_t>(n) * n;
    for (std::size_t i = 0; i < size; ++i)
    {
        scale = std::max(scale, std::abs(matrix[i]));
    }
    for (int p = 0; p < n; ++p)
    {
        for (int q = p; q < n; ++q)
        {
            if (std::abs(matrix[static_cast<std::size_t>(p) * n + q]
                         - std::conj(matrix[static_cast<std::size_t>(q) * n + p]))
                > kHermitianTolerance * scale)
            {
                throw std::runtime_error("Non-Hermitian TB2J " + what);
            }
        }
    }
}

int define_dim(const int ncid, const char* name, const std::size_t length)
{
    int dim = -1;
    check(nc_def_dim(ncid, name, length, &dim), std::string("defining dimension ") + name);
    return dim;
}

/// Fixed-shape variable; an explicit one-slab chunk layout keeps the streamed
/// one-k/one-site writes contiguous. Oversized chunks fall back to contiguous.
int define_var(const int ncid, const char* name, const nc_type type, const std::vector<int>& dims,
               const std::vector<std::size_t>* chunks = nullptr)
{
    int var = -1;
    check(nc_def_var(ncid, name, type, static_cast<int>(dims.size()), dims.data(), &var),
          std::string("defining variable ") + name);
    if (chunks != nullptr)
    {
        double bytes = 8.0; // all chunked variables of this writer are NC_DOUBLE
        for (const std::size_t c : *chunks)
        {
            bytes *= static_cast<double>(c);
        }
        if (bytes <= 2147483647.0)
        {
            check(nc_def_var_chunking(ncid, var, NC_CHUNKED, chunks->data()),
                  std::string("chunking variable ") + name);
        }
        else
        {
            check(nc_def_var_chunking(ncid, var, NC_CONTIGUOUS, nullptr),
                  std::string("storing variable ") + name);
        }
    }
    return var;
}

void put_text(const int ncid, const int varid, const char* name, const std::string& value)
{
    check(nc_put_att_text(ncid, varid, name, value.size(), value.c_str()),
          std::string("writing attribute ") + name);
}

void put_scalar(const int ncid, const char* name, const double value)
{
    check(nc_put_att_double(ncid, NC_GLOBAL, name, NC_DOUBLE, 1, &value),
          std::string("writing attribute ") + name);
}

/// Unique destination-sibling partial reserved through mkstemp.
std::string make_partial(const std::string& destination)
{
    const auto slash = destination.find_last_of('/');
    const std::string dir = slash == std::string::npos ? std::string(".") : destination.substr(0, slash);
    const std::string base = slash == std::string::npos ? destination : destination.substr(slash + 1);
    if (base.empty() || base == "." || base == "..")
    {
        throw std::runtime_error("Invalid TB2J projector NetCDF destination: " + destination);
    }
    std::string pattern = dir + "/" + base + ".partial.XXXXXX";
    std::vector<char> buffer(pattern.begin(), pattern.end());
    buffer.push_back('\0');
    const int fd = ::mkstemp(buffer.data());
    if (fd < 0)
    {
        throw std::runtime_error("Cannot reserve a TB2J projector partial sibling of " + destination);
    }
    ::close(fd);
    return std::string(buffer.data());
}

const char* func_type_meaning(const int func_type)
{
    switch (func_type)
    {
    case 0:
        return "none";
    case 1:
        return "lda";
    case 2:
        return "gga";
    case 3:
        return "mgga";
    case 4:
        return "hybrid lda/gga";
    case 5:
        return "hybrid mgga";
    default:
        return "unknown";
    }
}
} // namespace

ProjectorNetcdf::ProjectorNetcdf(const std::string& destination, const NativeState& state,
                                 const AtomicBasis& basis, const std::vector<SiteBlock>& sites,
                                 const Input_para& input)
    : state_(state), basis_(basis), sites_(sites), destination_(destination)
{
    try
    {
        const int nkpt = state_.nkpt;
        const int nband = state_.nband;
        const int nproj = basis_.nproj();
        if (input.nspin != 2)
        {
            throw std::runtime_error("TB2J common export requires native two-spin collinear input");
        }
        if (nkpt <= 0 || nband <= 0 || nproj <= 0 || state_.kpoints.size() != static_cast<std::size_t>(nkpt)
            || state_.weights.size() != static_cast<std::size_t>(nkpt)
            || state_.eigenvalues.size() != static_cast<std::size_t>(2) * nkpt * nband
            || state_.occupations.size() != state_.eigenvalues.size())
        {
            throw std::runtime_error("Invalid certified TB2J projector dataset dimensions");
        }
        for (const double weight : state_.weights)
        {
            if (!std::isfinite(weight) || weight <= 0.0)
            {
                throw std::runtime_error("Certified TB2J unique-k weights must be finite and positive");
            }
        }
        if (std::abs(std::accumulate(state_.weights.begin(), state_.weights.end(), 0.0) - 1.0) > 1e-12)
        {
            throw std::runtime_error("Certified TB2J unique-k weights do not sum to one");
        }
        for (const double value : state_.efermi_spin)
        {
            if (!std::isfinite(value))
            {
                throw std::runtime_error("Certified TB2J Fermi data is nonfinite");
            }
        }
        for (const double value : state_.cell)
        {
            if (!std::isfinite(value))
            {
                throw std::runtime_error("Certified TB2J lattice is nonfinite");
            }
        }
        const auto expected = atomic_sites(basis_);
        const bool same_blocks = sites_.size() == expected.size()
            && std::equal(expected.begin(), expected.end(), sites_.begin(),
                          [](const SiteBlock& a, const SiteBlock& b)
                          { return a.first == b.first && a.count == b.count && a.offset == b.offset; });
        if (!same_blocks || sites_.empty())
        {
            throw std::runtime_error("TB2J site blocks disagree with the immutable atomic basis");
        }
        const std::size_t nat = sites_.size();
        if (state_.positions.size() != nat || state_.atomic_numbers.size() != nat)
        {
            throw std::runtime_error("TB2J site count and captured physical geometry disagree");
        }
        std::size_t species_atoms = 0;
        for (int it = 0; it < basis_.ntype(); ++it)
        {
            if (basis_.species(it).natoms < 0)
            {
                throw std::runtime_error("Invalid TB2J species atom count");
            }
            species_atoms += static_cast<std::size_t>(basis_.species(it).natoms);
        }
        if (species_atoms != nat)
        {
            throw std::runtime_error("TB2J species atom counts and site blocks disagree");
        }
        nmax_ = 0;
        for (const SiteBlock& block : sites_)
        {
            if (block.count <= 0 || block.first < 0 || block.first + block.count > nproj)
            {
                throw std::runtime_error("Invalid TB2J site projector block extent");
            }
            nmax_ = std::max(nmax_, block.count);
        }
        seen_.assign(static_cast<std::size_t>(nkpt), 0);

        // Harmonic-row maps over ALL channels in deterministic source order:
        // species it -> atom ia -> source channel ic -> m = 0..2l (lm = l*l + m).
        std::vector<int> p_site(nproj), p_atom(nproj), p_l(nproj), p_m(nproj), p_radial(nproj);
        const std::vector<ProjectorSite>& channels = basis_.sites();
        std::size_t block = 0;
        int filled = 0;
        for (std::size_t c = 0; c < channels.size(); ++c)
        {
            const ProjectorSite& channel = channels[c];
            const bool same_atom = c > 0 && channel.it == channels[c - 1].it
                && channel.ia == channels[c - 1].ia;
            if (!same_atom)
            {
                if (filled != 0 || block >= sites_.size() || sites_[block].first != channel.start)
                {
                    throw std::runtime_error("TB2J channel order and site blocks disagree");
                }
                filled = 0;
            }
            if (channel.count != 2 * channel.l + 1 || channel.count <= 0 || channel.start < 0
                || channel.start + channel.count > nproj
                || filled + channel.count > sites_[block].count)
            {
                throw std::runtime_error("Invalid TB2J source-channel extent in the harmonic map");
            }
            for (int m = 0; m < channel.count; ++m)
            {
                const int row = channel.start + m;
                p_site[row] = static_cast<int>(block);
                p_atom[row] = static_cast<int>(block); // dense zero-based producer atom id
                p_l[row] = channel.l;
                p_m[row] = m;
                p_radial[row] = channel.channel;
            }
            filled += channel.count;
            if (filled == sites_[block].count)
            {
                ++block;
                filled = 0;
            }
        }
        if (block != sites_.size() || filled != 0)
        {
            throw std::runtime_error("TB2J harmonic map does not cover every projector row exactly once");
        }

        // ---- provenance: existing nlohmann backend, ordered for determinism ----
        nlohmann::ordered_json meta;
        meta["source_code"] = "abacus";
        meta["producer"] = "ABACUS plane-wave UPF atomic-wavefunction projector exporter";
#ifdef VERSION
        meta["source_version"] = std::string(VERSION);
#else
        meta["source_version"] = "unknown";
#endif
#ifdef COMMIT_INFO
        meta["source_commit"] = std::string(COMMIT);
        meta["source_commit_scope"] = "configured build revision; no full worktree dirty digest exists";
#else
        meta["source_commit"] = "unknown";
#endif
        meta["build"] = {{"platform", std::string(ABACUS_PLATFORM_NAME)},
                         {"build_type", std::string(ABACUS_BUILD_TYPE)},
                         {"cxx_compiler", std::string(ABACUS_CXX_COMPILER_ID) + " "
                                              + std::string(ABACUS_CXX_COMPILER_VERSION)},
                         {"cxx_flags", std::string(ABACUS_CXX_FLAGS)},
                         {"mpi_implementation", std::string(ABACUS_MPI_IMPLEMENTATION)}};
        meta["operator_basis"] = "norm_conserving_pao";
        meta["source_basis"] = "explicit UPF atomic wavefunctions (no PAW, no beta projectors)";
        meta["units"] = {{"cell", "Angstrom"},
                         {"positions", "Angstrom"},
                         {"eigenvalues", "eV"},
                         {"efermi", "eV"},
                         {"delta_total", "eV"}};
        meta["kpoint_convention"] = "fractional_reciprocal";
        meta["phase_convention"] = "exp(-2*pi*i*k.R)";
        meta["complex_component"] = {"real", "imag"};
        meta["storage_level"] = "spectral";
        meta["nspin"] = 2;
        meta["spin_channels"] = {"up", "down"};
        meta["basis_type"] = input.basis_type;
        meta["ks_solver"] = input.ks_solver;
        meta["cutoff"] = {{"ecutwfc_ry", input.ecutwfc}, {"ecutrho_ry", input.ecutrho}};
        meta["smearing"] = {{"method", input.smearing_method}, {"sigma_ry", input.smearing_sigma}};
        meta["scf_controls"] = {{"nmax", input.scf_nmax},
                                {"thr", input.scf_thr},
                                {"thr_type", input.scf_thr_type},
                                {"ene_thr_ev", input.scf_ene_thr},
                                {"mixing_mode", input.mixing_mode},
                                {"mixing_beta", input.mixing_beta}};
        meta["occupations"] = {
            {"custom", state_.custom_occupations},
            {"definition",
             "native weighted occupations divided by the corresponding native unique-k weight "
             "(dimensionless)"},
            {"native_electrons", {state_.native_electrons[0], state_.native_electrons[1]}},
            {"common_electrons", {state_.electrons[0], state_.electrons[1]}}};
        if (std::isfinite(state_.native_weight_sum) && state_.native_weight_sum > 0.0)
        {
            meta["native_weight_sum"] = state_.native_weight_sum;
        }
        meta["grid_recipe"] = {{"generated", state_.recipe.generated},
                               {"convention", state_.recipe.convention},
                               {"symmetry", state_.recipe.symmetry},
                               {"input_count", state_.recipe.input_count},
                               {"original_count", state_.recipe.original_count},
                               {"dimensions", {state_.recipe.dimensions[0],
                                               state_.recipe.dimensions[1],
                                               state_.recipe.dimensions[2]}},
                               {"offsets", {state_.recipe.offsets[0], state_.recipe.offsets[1],
                                            state_.recipe.offsets[2]}},
                               {"k_weight_sum", std::accumulate(state_.weights.begin(),
                                                                state_.weights.end(), 0.0)}};
        meta["functional"] = {
            {"requested", input.dft_functional},
            {"actual_func_type", XC_Functional::get_func_type()},
            {"actual_func_type_meaning", func_type_meaning(XC_Functional::get_func_type())},
            {"libxc_func_ids", XC_Functional::get_func_id()}};
        meta["coefficient_definition"] =
            "raw primal projection C[spin,k,band,p] = <phi_p|psi_band> against the native UPF "
            "atomic-wavefunction PW fill (real harmonics lm = l*l + m with m = 0..2l, radial phase "
            "pow(-i,l)); never dualized";
        meta["overlap_metric_policy"] =
            "raw covariant full projector overlap stored per k; the consumer applies the inverse "
            "metric with the inherited condition limit 1e12 (producer-side guard 1e12)";
        meta["operator_scope"] =
            "complete onsite R=0 BZ-weighted XC-only spin-difference vertex for every atom "
            "(v_xc_up - v_xc_dn from native reevaluation of the converged charge); no "
            "Hubbard/U/SOC contribution";
        nlohmann::ordered_json species_meta = nlohmann::ordered_json::array();
        for (int it = 0; it < basis_.ntype(); ++it)
        {
            const SpeciesBasis& sp = basis_.species(it);
            nlohmann::ordered_json entry;
            entry["label"] = sp.label;
            entry["pseudo_file"] = sp.pseudo_file;
            entry["sha256"] = sp.sha256;
            entry["natoms"] = sp.natoms;
            entry["mesh"] = sp.mesh;
            nlohmann::ordered_json channel_meta = nlohmann::ordered_json::array();
            for (const AtomicChannel& channel : sp.channels)
            {
                nlohmann::ordered_json item;
                item["label"] = channel.label;
                item["l"] = channel.l;
                item["occupation"] = channel.occupation;
                channel_meta.push_back(item);
            }
            entry["channels"] = channel_meta;
            species_meta.push_back(entry);
        }
        meta["species"] = species_meta;

        // ---- partial file and NetCDF-4 dataset definition ----
        partial_ = make_partial(destination_);
        check(nc_create(partial_.c_str(), NC_NETCDF4, &ncid_), "creating partial dataset");
        const int dim_spin = define_dim(ncid_, "nspin", 2);
        const int dim_k = define_dim(ncid_, "nkpt", nkpt);
        const int dim_band = define_dim(ncid_, "nband", nband);
        const int dim_proj = define_dim(ncid_, "nproj", nproj);
        const int dim_three = define_dim(ncid_, "three", 3);
        const int dim_complex = define_dim(ncid_, "complex", 2);
        const int dim_site = define_dim(ncid_, "nsite", nat);
        const int dim_smax = define_dim(ncid_, "nproj_site_max", nmax_);

        int structure = -1, kpoints_group = -1, bands = -1, operators = -1;
        check(nc_def_grp(ncid_, "structure", &structure), "creating structure group");
        check(nc_def_grp(ncid_, "kpoints", &kpoints_group), "creating kpoints group");
        check(nc_def_grp(ncid_, "bands", &bands), "creating bands group");
        check(nc_def_grp(ncid_, "projectors", &projectors_), "creating projectors group");
        check(nc_def_grp(ncid_, "operators", &operators), "creating operators group");
        check(nc_def_grp(operators, "operator_components", &components_),
              "creating operator_components group");

        // Canonical writer also keeps group-local site dimensions in projectors.
        const int proj_site_dim = define_dim(projectors_, "nsite", nat);
        const int proj_smax_dim = define_dim(projectors_, "nproj_site_max", nmax_);
        const int natom_dim = define_dim(structure, "natom", nat);

        const int cell_var = define_var(structure, "cell", NC_DOUBLE, {dim_three, dim_three});
        const int pos_var = define_var(structure, "positions", NC_DOUBLE, {natom_dim, dim_three});
        const int z_var = define_var(structure, "atomic_numbers", NC_INT, {natom_dim});
        const int kpt_var = define_var(kpoints_group, "kpoints", NC_DOUBLE, {dim_k, dim_three});
        const int weight_var = define_var(kpoints_group, "weights", NC_DOUBLE, {dim_k});
        const int efermi_spin_var = define_var(bands, "efermi_spin", NC_DOUBLE, {dim_spin});
        const int eigen_var = define_var(bands, "eigenvalues", NC_DOUBLE, {dim_spin, dim_k, dim_band});
        const int occ_var = define_var(bands, "occupations", NC_DOUBLE, {dim_spin, dim_k, dim_band});

        const std::vector<std::size_t> coefficient_chunks = {1, 1, static_cast<std::size_t>(nband),
                                                             static_cast<std::size_t>(nproj), 2};
        const std::vector<std::size_t> overlap_chunks = {1, static_cast<std::size_t>(nproj),
                                                         static_cast<std::size_t>(nproj), 2};
        const std::vector<std::size_t> delta_chunks = {1, static_cast<std::size_t>(nmax_),
                                                       static_cast<std::size_t>(nmax_), 2};
        coefficients_ = define_var(projectors_, "coefficients", NC_DOUBLE,
                                   {dim_spin, dim_k, dim_band, dim_proj, dim_complex},
                                   &coefficient_chunks);
        const int site_var = define_var(projectors_, "projector_site", NC_INT, {dim_proj});
        const int atom_var = define_var(projectors_, "projector_atom", NC_INT, {dim_proj});
        const int l_var = define_var(projectors_, "projector_l", NC_INT, {dim_proj});
        const int m_var = define_var(projectors_, "projector_m", NC_INT, {dim_proj});
        const int radial_var = define_var(projectors_, "projector_radial", NC_INT, {dim_proj});
        overlap_ = define_var(projectors_, "overlap_k", NC_DOUBLE, {dim_k, dim_proj, dim_proj, dim_complex},
                              &overlap_chunks);
        const int site_nproj_var = define_var(projectors_, "site_nproj", NC_INT, {proj_site_dim});
        const int site_indices_var
            = define_var(projectors_, "site_projector_indices", NC_INT, {proj_site_dim, proj_smax_dim});
        delta_ = define_var(components_, "delta_total", NC_DOUBLE,
                            {dim_site, dim_smax, dim_smax, dim_complex}, &delta_chunks);

        put_text(ncid_, NC_GLOBAL, "schema_name", "tb2j.projector_green");
        put_text(ncid_, NC_GLOBAL, "schema_version", "1.0");
        put_text(ncid_, NC_GLOBAL, "metadata_json", meta.dump());
        put_text(ncid_, NC_GLOBAL, "complex_component", nlohmann::ordered_json({"real", "imag"}).dump());
        put_scalar(bands, "efermi", 0.5 * (state_.efermi_spin[0] + state_.efermi_spin[1]));
        put_text(projectors_, NC_GLOBAL, "coefficient_source", "nc_pao");
        put_text(projectors_, NC_GLOBAL, "coefficient_projector", "upf_atomic_wavefunction");
        put_text(projectors_, NC_GLOBAL, "channel_interpretation",
                 "native real harmonics: lm = l*l + m with m = 0..2l (ABACUS Ylm_Real convention)");
        put_text(projectors_, NC_GLOBAL, "population_metric", "projector");
        put_text(projectors_, overlap_, "definition",
                 "k-dependent full projector overlap <phi_p|phi_q>, covariant, raw");
        put_text(components_, delta_, "units", "eV");
        put_text(components_, delta_, "definition",
                 "covariant R=0 BZ-weighted onsite XC-only spin-difference vertex: "
                 "sum_k w_k <phi_p| (v_xc_up - v_xc_dn) |phi_q> in eV; no metric applied");
        put_text(components_, delta_, "source",
                 "ABACUS native semilocal XC potential difference reevaluated from the converged "
                 "charge (XC_Functional)");
        put_text(components_, delta_, "operator_basis", "norm_conserving_pao");
        put_text(components_, delta_, "completeness", "complete");
        put_text(components_, delta_, "exchange_ready", "true");

        check(nc_enddef(ncid_), "ending define mode");

        // ---- structure / kpoints / bands ----
        check(nc_put_var_double(structure, cell_var, state_.cell.data()), "writing structure/cell");
        std::vector<double> flat;
        flat.reserve(3 * nat);
        for (const auto& position : state_.positions)
        {
            flat.push_back(position.x);
            flat.push_back(position.y);
            flat.push_back(position.z);
        }
        check(nc_put_var_double(structure, pos_var, flat.data()), "writing structure/positions");
        check(nc_put_var_int(structure, z_var, state_.atomic_numbers.data()),
              "writing structure/atomic_numbers");
        flat.clear();
        flat.reserve(3 * static_cast<std::size_t>(nkpt));
        for (const auto& kpoint : state_.kpoints)
        {
            flat.push_back(kpoint.x);
            flat.push_back(kpoint.y);
            flat.push_back(kpoint.z);
        }
        check(nc_put_var_double(kpoints_group, kpt_var, flat.data()), "writing kpoints/kpoints");
        check(nc_put_var_double(kpoints_group, weight_var, state_.weights.data()),
              "writing kpoints/weights");
        const double efermi_spin[2] = {state_.efermi_spin[0], state_.efermi_spin[1]};
        check(nc_put_var_double(bands, efermi_spin_var, efermi_spin), "writing bands/efermi_spin");
        const bool finite_spectrum = std::all_of(state_.eigenvalues.begin(), state_.eigenvalues.end(),
                                                 [](const double value) { return std::isfinite(value); });
        if (!finite_spectrum)
        {
            throw std::runtime_error("Nonfinite certified TB2J eigenvalues");
        }
        check(nc_put_var_double(bands, eigen_var, state_.eigenvalues.data()),
              "writing bands/eigenvalues");
        const bool finite_occupations = std::all_of(state_.occupations.begin(),
                                                    state_.occupations.end(),
                                                    [](const double value) { return std::isfinite(value); });
        if (!finite_occupations)
        {
            throw std::runtime_error("Nonfinite certified TB2J occupations");
        }
        check(nc_put_var_double(bands, occ_var, state_.occupations.data()),
              "writing bands/occupations");

        // ---- projector maps (all channels, source order) ----
        check(nc_put_var_int(projectors_, site_var, p_site.data()), "writing projector_site");
        check(nc_put_var_int(projectors_, atom_var, p_atom.data()), "writing projector_atom");
        check(nc_put_var_int(projectors_, l_var, p_l.data()), "writing projector_l");
        check(nc_put_var_int(projectors_, m_var, p_m.data()), "writing projector_m");
        check(nc_put_var_int(projectors_, radial_var, p_radial.data()), "writing projector_radial");
        std::vector<int> site_nproj(sites_.size());
        for (std::size_t s = 0; s < sites_.size(); ++s)
        {
            site_nproj[s] = sites_[s].count;
        }
        check(nc_put_var_int(projectors_, site_nproj_var, site_nproj.data()),
              "writing site_nproj");
        std::vector<int> indices(sites_.size() * static_cast<std::size_t>(nmax_), -1);
        for (std::size_t s = 0; s < sites_.size(); ++s)
        {
            for (int j = 0; j < sites_[s].count; ++j)
            {
                indices[s * static_cast<std::size_t>(nmax_) + static_cast<std::size_t>(j)]
                    = sites_[s].first + j;
            }
        }
        check(nc_put_var_int(projectors_, site_indices_var, indices.data()),
              "writing site_projector_indices");
    }
    catch (...)
    {
        cleanup();
        throw;
    }
}

ProjectorNetcdf::~ProjectorNetcdf()
{
    cleanup();
}

void ProjectorNetcdf::cleanup() noexcept
{
    if (ncid_ >= 0)
    {
        nc_abort(ncid_); // discard the owned partial; never throw from a destructor
        ncid_ = -1;
    }
    if (!published_ && !partial_.empty())
    {
        ::unlink(partial_.c_str());
    }
    partial_.clear();
}

void ProjectorNetcdf::write_k(const int unique_k, const std::complex<double>* coefficients,
                              const std::complex<double>* gram)
{
    if (published_ || ncid_ < 0)
    {
        throw std::runtime_error("TB2J projector dataset is not open for k streaming");
    }
    if (unique_k < 0 || unique_k >= state_.nkpt)
    {
        throw std::runtime_error("TB2J projector k index out of range: " + std::to_string(unique_k));
    }
    if (seen_[unique_k])
    {
        throw std::runtime_error("Duplicate TB2J projector k block: " + std::to_string(unique_k));
    }
    const int nband = state_.nband;
    const int nproj = basis_.nproj();
    if (coefficients == nullptr || gram == nullptr)
    {
        throw std::runtime_error("Null TB2J projector k block at k " + std::to_string(unique_k));
    }
    const std::string at_k = " at k " + std::to_string(unique_k);
    require_finite(coefficients, static_cast<std::size_t>(2) * nband * nproj, "coefficients" + at_k);
    require_finite(gram, static_cast<std::size_t>(nproj) * nproj, "projector overlap" + at_k);
    require_hermitian(gram, nproj, "projector overlap" + at_k);

    const std::size_t slab = static_cast<std::size_t>(nband) * nproj;
    for (int spin = 0; spin < 2; ++spin)
    {
        const std::size_t start[5] = {static_cast<std::size_t>(spin),
                                      static_cast<std::size_t>(unique_k), 0, 0, 0};
        const std::size_t count[5] = {1, 1, static_cast<std::size_t>(nband),
                                      static_cast<std::size_t>(nproj), 2};
        check(nc_put_vara_double(projectors_, coefficients_, start, count,
                                 reinterpret_cast<const double*>(coefficients
                                                                 + static_cast<std::size_t>(spin) * slab)),
              "writing coefficients" + at_k);
    }
    const std::size_t start[4] = {static_cast<std::size_t>(unique_k), 0, 0, 0};
    const std::size_t count[4] = {1, static_cast<std::size_t>(nproj), static_cast<std::size_t>(nproj), 2};
    check(nc_put_vara_double(projectors_, overlap_, start, count, reinterpret_cast<const double*>(gram)),
          "writing overlap_k" + at_k);
    seen_[unique_k] = 1;
    ++written_;
}

void ProjectorNetcdf::finish(const std::complex<double>* site_delta_ev)
{
    if (published_ || ncid_ < 0)
    {
        throw std::runtime_error("TB2J projector dataset is not open for completion");
    }
    if (written_ != state_.nkpt
        || std::find(seen_.begin(), seen_.end(), static_cast<unsigned char>(0)) != seen_.end())
    {
        throw std::runtime_error("TB2J projector coverage incomplete before completion: "
                                 + std::to_string(written_) + "/" + std::to_string(state_.nkpt)
                                 + " unique k points written");
    }
    if (site_delta_ev == nullptr)
    {
        throw std::runtime_error("Null TB2J packed onsite vertex at completion");
    }
    std::size_t packed = 0;
    for (const SiteBlock& block : sites_)
    {
        packed += static_cast<std::size_t>(block.count) * block.count;
    }
    // One reusable padded site buffer; never a full all-site padded array.
    std::vector<std::complex<double>> padded(static_cast<std::size_t>(nmax_) * nmax_,
                                             std::complex<double>{});
    for (std::size_t s = 0; s < sites_.size(); ++s)
    {
        const SiteBlock& block = sites_[s];
        const std::size_t n = static_cast<std::size_t>(block.count);
        if (block.count <= 0 || block.offset + n * n > packed)
        {
            throw std::runtime_error("TB2J onsite vertex packing extent is invalid for site "
                                     + std::to_string(s));
        }
        const std::complex<double>* values = site_delta_ev + block.offset;
        require_finite(values, n * n, "onsite vertex at site " + std::to_string(s));
        require_hermitian(values, block.count, "onsite vertex at site " + std::to_string(s));
        std::fill(padded.begin(), padded.end(), std::complex<double>{});
        for (int p = 0; p < block.count; ++p)
        {
            for (int q = 0; q < block.count; ++q)
            {
                padded[static_cast<std::size_t>(p) * nmax_ + q] = values[static_cast<std::size_t>(p) * n + q];
            }
        }
        const std::size_t start[4] = {s, 0, 0, 0};
        const std::size_t count[4] = {1, static_cast<std::size_t>(nmax_),
                                      static_cast<std::size_t>(nmax_), 2};
        check(nc_put_vara_double(components_, delta_, start, count,
                                 reinterpret_cast<const double*>(padded.data())),
              "writing operators/operator_components/delta_total at site " + std::to_string(s));
    }
    check(nc_close(ncid_), "closing TB2J projector dataset");
    ncid_ = -1;
    if (std::rename(partial_.c_str(), destination_.c_str()) != 0)
    {
        throw std::runtime_error("TB2J projector publication rename to " + destination_
                                 + " failed: " + std::strerror(errno));
    }
    published_ = true;
}

} // namespace Tb2j
} // namespace ModuleIO
