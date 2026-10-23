#include "native_state.h"

#include "source_base/constants.h"
#include "source_base/element_name.h"
#include "source_base/global_variable.h"
#include "source_base/module_parallel/para_band_output.h"
#include "source_base/parallel_common.h"
#include "source_base/tool_quit.h"
#include "source_cell/klist.h"
#include "source_cell/unitcell.h"
#include "source_estate/elecstate.h"
#include "source_io/module_parameter/input_parameter.h"
#include "source_io/module_parameter/parameter.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace ModuleIO
{
namespace Tb2j
{
namespace
{
bool finite_vector(const ModuleBase::Vector3<double>& value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

void collective_error(std::string& error)
{
#ifdef __MPI
    Parallel_Common::bcast_string(error);
#endif
    if (!error.empty())
    {
        ModuleBase::WARNING_QUIT("TB2J native state", error);
    }
}
}

std::unique_ptr<NativeState> NativeState::create(const GridRecipe& source, const int bands, std::string& error)
{
    error.clear();
    if (!source.generated || source.input_count != 0 || source.symmetry != -1)
    {
        error = "A generated full-BZ Gamma/MP recipe with symmetry -1 is required";
        return nullptr;
    }
    int kind = -1;
    if (source.convention == "Gamma") kind = 0;
    if (source.convention == "Monkhorst-Pack" || source.convention == "MP" || source.convention == "mp") kind = 1;
    if (kind < 0 || bands <= 0)
    {
        error = "Invalid native generated-mesh convention or band count";
        return nullptr;
    }
    int count = 1;
    for (int axis = 0; axis < 3; ++axis)
    {
        const int size = source.dimensions[axis];
        if (size <= 0 || count > std::numeric_limits<int>::max() / 2 / size
            || !std::isfinite(source.offsets[axis]))
        {
            error = "Invalid or unrepresentable generated-mesh dimensions/offsets";
            return nullptr;
        }
        count *= size;
    }
    if (source.original_count != count || count > std::numeric_limits<int>::max() / 2 / bands)
    {
        error = "Generated full-BZ point count or spectral array extent is inconsistent";
        return nullptr;
    }
    K_Vectors expected;
    expected.Monkhorst_Pack(source.dimensions.data(), source.offsets.data(), kind);
    if (expected.kvec_d.size() != static_cast<std::size_t>(count))
    {
        error = "Native generator returned an inconsistent expected point count";
        return nullptr;
    }
    auto state = std::unique_ptr<NativeState>(new NativeState);
    state->recipe = source;
    state->nkpt = count;
    state->nband = bands;
    state->kpoints = std::move(expected.kvec_d);
    state->weights.resize(count);
    state->eigenvalues.resize(static_cast<std::size_t>(2) * count * bands);
    state->occupations.resize(state->eigenvalues.size());
    state->seen_.assign(2 * count, 0);
    state->native_weights_.resize(2 * count);
    return state;
}

bool NativeState::append(const SpectralBlock& block, std::string& error)
{
    if (finished_ || block.nrow < 0 || !block.energies_ry || !block.weighted_occupations
        || (block.nrow > 0 && (!block.global_k || !block.spin || !block.kpoints || !block.weights))
        || block.energies_ry->nr != block.nrow || block.weighted_occupations->nr != block.nrow
        || (block.nrow > 0 && (block.energies_ry->nc != nband || block.weighted_occupations->nc != nband)))
    {
        error = "Inconsistent native spectral block shape or finalized state";
        return false;
    }
    for (int row = 0; row < block.nrow; ++row)
    {
        const int global = block.global_k[row];
        const int spin = block.spin[row];
        // Native KListIO::build_ik2iktot assigns the second spin an nkpt offset.
        // Local row order is immaterial: both actual global map and isk are checked.
        if (global < 0 || global >= 2 * nkpt || spin < 0 || spin > 1 || global / nkpt != spin)
        {
            error = "Native global k map and spin identity disagree";
            return false;
        }
        const int physical = global - spin * nkpt;
        if (seen_[global])
        {
            error = "Native spin/k coverage contains a duplicate global identity";
            return false;
        }
        if (!finite_vector(block.kpoints[row])
            || !ModuleCell::kpoints_equivalent(block.kpoints[row], kpoints[physical], 1e-10))
        {
            error = "Native point does not match its generated full-BZ mesh identity";
            return false;
        }
        const double weight = block.weights[row];
        if (!std::isfinite(weight) || weight <= 0.0)
        {
            error = "Native k weight must be finite and strictly positive";
            return false;
        }
        for (int band = 0; band < nband; ++band)
        {
            const double energy = (*block.energies_ry)(row, band);
            const double weighted = (*block.weighted_occupations)(row, band);
            if (!std::isfinite(energy) || !std::isfinite(weighted))
            {
                error = "Native spectrum or weighted occupation is nonfinite";
                return false;
            }
            const std::size_t index = static_cast<std::size_t>(global) * nband + band;
            eigenvalues[index] = energy * ModuleBase::Ry_to_eV;
            occupations[index] = weighted / weight;
            if (!std::isfinite(eigenvalues[index]) || !std::isfinite(occupations[index]))
            {
                error = "Native energy conversion or occupation unweighting is nonfinite";
                return false;
            }
            native_electrons[spin] += weighted;
        }
        native_weights_[global] = weight;
        seen_[global] = 1;
    }
    return true;
}

bool NativeState::finish(const double native_nelec, const std::array<double, 2>& fermi_ry,
                         const bool custom, std::string& error)
{
    if (finished_ || std::find(seen_.begin(), seen_.end(), 0) != seen_.end())
    {
        error = "Native spin/k coverage is incomplete or already finalized";
        return false;
    }
    const double sum = std::accumulate(native_weights_.begin(), native_weights_.begin() + nkpt, 0.0);
    if (!std::isfinite(sum) || sum <= 0.0)
    {
        error = "Native k weight sum is invalid";
        return false;
    }
    for (int ik = 0; ik < nkpt; ++ik)
    {
        const double up = native_weights_[ik];
        const double down = native_weights_[nkpt + ik];
        weights[ik] = up / sum;
        if (std::abs(up - down) > 1e-12 || std::abs(weights[ik] - 1.0 / nkpt) > 1e-12)
        {
            error = "Native spin-pair weights disagree or generated-grid weights are nonuniform";
            return false;
        }
        for (int spin = 0; spin < 2; ++spin)
            for (int band = 0; band < nband; ++band)
                electrons[spin] += weights[ik] * occupations[(static_cast<std::size_t>(spin) * nkpt + ik) * nband + band];
    }
    if (std::abs(std::accumulate(weights.begin(), weights.end(), 0.0) - 1.0) > 1e-12)
    {
        error = "Normalized unique k weights do not sum to one";
        return false;
    }
    if (!std::isfinite(native_nelec) || native_nelec < 0.0
        || std::abs(native_electrons[0] + native_electrons[1] - native_nelec)
            > 1e-10 * std::max(1.0, std::abs(native_nelec)))
    {
        error = "Native weighted occupations do not conserve the total electron count";
        return false;
    }
    for (int spin = 0; spin < 2; ++spin)
    {
        if (!std::isfinite(electrons[spin])
            || std::abs(electrons[spin] - native_electrons[spin])
                > 1e-10 * std::max(1.0, std::abs(native_electrons[spin]))
            || !std::isfinite(fermi_ry[spin]))
        {
            error = "Common occupations do not conserve native spin electron counts or Fermi data is nonfinite";
            return false;
        }
        efermi_spin[spin] = fermi_ry[spin] * ModuleBase::Ry_to_eV;
        if (!std::isfinite(efermi_spin[spin]))
        {
            error = "Native Fermi energy conversion is nonfinite";
            return false;
        }
    }
    custom_occupations = custom;
    native_weight_sum = sum;
    finished_ = true;
    return true;
}

std::unique_ptr<NativeState> NativeState::capture(const UnitCell& ucell, const K_Vectors& kv,
                                                const elecstate::ElecState& estate, const Input_para& input,
                                                const Parallel::ParaBandOutput& bands)
try
{
    std::string error;
    std::unique_ptr<NativeState> result;
    if (GlobalV::MY_RANK == 0)
    {
        GridRecipe source;
        source.generated = kv.get_is_mp();
        source.input_count = kv.get_k_nkstot();
        source.original_count = kv.get_nkstot_nospin();
        source.symmetry = input.symmetry == "-1" ? -1 : 0;
        source.convention = kv.get_k_kword();
        for (int axis = 0; axis < 3; ++axis)
        {
            source.dimensions[axis] = kv.nmp[axis];
            source.offsets[axis] = kv.get_koffset(axis);
        }
        result = create(source, input.nbands, error);
        if (result && (input.nspin != 2 || kv.get_spin_mult() != 2 || kv.get_nkstot() != 2 * result->nkpt))
            error = "Observed native spin/full-BZ point count does not match the generated recipe";
        if (result && error.empty())
        {
            if (!std::isfinite(ucell.lat0) || !std::isfinite(ucell.omega) || ucell.lat0 <= 0.0 || ucell.omega <= 0.0)
                error = "Native physical cell is not finite with positive volume";
            const double scale = ucell.lat0 * ModuleBase::BOHR_TO_A;
            result->cell = {ucell.latvec.e11 * scale, ucell.latvec.e12 * scale, ucell.latvec.e13 * scale,
                            ucell.latvec.e21 * scale, ucell.latvec.e22 * scale, ucell.latvec.e23 * scale,
                            ucell.latvec.e31 * scale, ucell.latvec.e32 * scale, ucell.latvec.e33 * scale};
            for (const double component : result->cell)
                if (!std::isfinite(component)) error = "Native lattice vector is nonfinite";
            result->positions.reserve(ucell.nat);
            result->atomic_numbers.reserve(ucell.nat);
            for (int it = 0; it < ucell.ntype; ++it)
            {
                const auto& atom = ucell.atoms[it];
                const auto element = std::find(ModuleBase::element_name.begin(), ModuleBase::element_name.end(), atom.ncpp.psd);
                if (element == ModuleBase::element_name.end())
                {
                    error = "Native UPF element cannot be mapped to an atomic number: " + atom.ncpp.psd;
                    break;
                }
                const int z = static_cast<int>(element - ModuleBase::element_name.begin()) + 1;
                for (int ia = 0; ia < atom.na; ++ia)
                {
                    const auto position = atom.tau[ia] * scale;
                    if (!finite_vector(position)) error = "Native atomic position is nonfinite";
                    result->positions.push_back(position);
                    result->atomic_numbers.push_back(z);
                }
            }
            if (result->positions.size() != static_cast<std::size_t>(ucell.nat))
                error = "Native atom count and physical geometry disagree";
        }
    }
    collective_error(error);

    const int nlocal = kv.get_nks();
    const int local_bands = PARAM.globalv.nbands_l;
    int bad_shape = local_bands < 0 || estate.ekb.nr != nlocal || estate.wg.nr != nlocal
                    || estate.ekb.nc != local_bands || estate.wg.nc != local_bands
                    || kv.ik2iktot.size() < static_cast<std::size_t>(nlocal)
                    || kv.isk.size() < static_cast<std::size_t>(nlocal)
                    || kv.kvec_d.size() < static_cast<std::size_t>(nlocal)
                    || kv.wk.size() < static_cast<std::size_t>(nlocal);
#ifdef __MPI
    MPI_Allreduce(MPI_IN_PLACE, &bad_shape, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
    if (bad_shape) error = "Native local spectral shape or global k/spin map is inconsistent";
    collective_error(error);
    ModuleBase::matrix gathered_energies, gathered_weighted;
    const ModuleBase::matrix* energies = &estate.ekb;
    const ModuleBase::matrix* weighted = &estate.wg;
    if (GlobalV::RANK_IN_POOL == 0 && !bands.bands_are_replicated())
    {
        gathered_energies = bands.gather_matrix(*energies);
        gathered_weighted = bands.gather_matrix(*weighted);
        energies = &gathered_energies;
        weighted = &gathered_weighted;
    }

    // Only PW roots gather band columns. One representative per physical pool
    // streams its native identities and spectrum; other PW ranks hold no copies.
    std::vector<int> global, spin;
    std::vector<ModuleBase::Vector3<double>> coordinates;
    std::vector<double> weights, packed;
    ModuleBase::matrix received_energy, received_weighted;
    for (int pool = 0; pool < GlobalV::KPAR; ++pool)
    {
        const int owner = kv.para_k.get_startpro_pool(pool);
        if (GlobalV::MY_RANK == owner)
        {
            if (owner == 0)
            {
                const SpectralBlock block{nlocal, kv.ik2iktot.data(), kv.isk.data(), kv.kvec_d.data(),
                                          kv.wk.data(), energies, weighted};
                result->append(block, error);
            }
#ifdef __MPI
            else
            {
                packed.resize(static_cast<std::size_t>(nlocal) * 4);
                for (int row = 0; row < nlocal; ++row)
                {
                    packed[4 * row] = kv.kvec_d[row].x;
                    packed[4 * row + 1] = kv.kvec_d[row].y;
                    packed[4 * row + 2] = kv.kvec_d[row].z;
                    packed[4 * row + 3] = kv.wk[row];
                }
                MPI_Send(&nlocal, 1, MPI_INT, 0, 8200, MPI_COMM_WORLD);
                MPI_Send(kv.ik2iktot.data(), nlocal, MPI_INT, 0, 8201, MPI_COMM_WORLD);
                MPI_Send(kv.isk.data(), nlocal, MPI_INT, 0, 8202, MPI_COMM_WORLD);
                MPI_Send(packed.data(), nlocal * 4, MPI_DOUBLE, 0, 8203, MPI_COMM_WORLD);
                MPI_Send(energies->c, nlocal * input.nbands, MPI_DOUBLE, 0, 8204, MPI_COMM_WORLD);
                MPI_Send(weighted->c, nlocal * input.nbands, MPI_DOUBLE, 0, 8205, MPI_COMM_WORLD);
            }
#endif
        }
#ifdef __MPI
        else if (GlobalV::MY_RANK == 0)
        {
            int rows = 0;
            MPI_Recv(&rows, 1, MPI_INT, owner, 8200, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (rows < 0 || rows > kv.get_nkstot() || rows > std::numeric_limits<int>::max() / 4
                || rows > std::numeric_limits<int>::max() / input.nbands)
                throw std::runtime_error("Unrepresentable native pool metadata block");
            global.resize(rows);
            spin.resize(rows);
            packed.resize(static_cast<std::size_t>(rows) * 4);
            coordinates.resize(rows);
            weights.resize(rows);
            received_energy.create(rows, input.nbands, false);
            received_weighted.create(rows, input.nbands, false);
            MPI_Recv(global.data(), rows, MPI_INT, owner, 8201, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(spin.data(), rows, MPI_INT, owner, 8202, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(packed.data(), rows * 4, MPI_DOUBLE, owner, 8203, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(received_energy.c, rows * input.nbands, MPI_DOUBLE, owner, 8204, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(received_weighted.c, rows * input.nbands, MPI_DOUBLE, owner, 8205, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int row = 0; row < rows; ++row)
            {
                coordinates[row] = {packed[4 * row], packed[4 * row + 1], packed[4 * row + 2]};
                weights[row] = packed[4 * row + 3];
            }
            if (error.empty())
            {
                const SpectralBlock block{rows, global.data(), spin.data(), coordinates.data(), weights.data(),
                                          &received_energy, &received_weighted};
                result->append(block, error);
            }
        }
#endif
    }
    if (GlobalV::MY_RANK == 0 && error.empty())
    {
        // Match calculate_weights' authoritative mode. The PW constructor does
        // not initialize Efermi::two_efermi, so its generic getter can return
        // the unused scalar ef even when ef_up/ef_dw were actually calculated.
        const std::array<double, 2> fermi = PARAM.globalv.two_fermi
            ? std::array<double, 2>{estate.eferm.ef_up, estate.eferm.ef_dw}
            : std::array<double, 2>{estate.eferm.ef, estate.eferm.ef};
        result->finish(input.nelec, fermi, input.ocp, error);
        if (error.empty())
        {
            GlobalV::ofs_running << "\n TB2J certified generated full BZ: " << result->recipe.convention
                                 << " expected/observed " << result->nkpt << '/' << result->nkpt
                                 << " per spin; weights sum 1; native electrons "
                                 << result->native_electrons[0] << ' ' << result->native_electrons[1]
                                 << "; reconstructed " << result->electrons[0] << ' ' << result->electrons[1]
                                 << "; custom occupations " << result->custom_occupations << '\n';
            GlobalV::ofs_running << " TB2J grid dimensions "
                                 << result->recipe.dimensions[0] << ' ' << result->recipe.dimensions[1] << ' '
                                 << result->recipe.dimensions[2] << "; offsets "
                                 << result->recipe.offsets[0] << ' ' << result->recipe.offsets[1] << ' '
                                 << result->recipe.offsets[2] << "; spin Fermi energies (eV) "
                                 << result->efermi_spin[0] << ' ' << result->efermi_spin[1] << '\n';
        }
    }
    collective_error(error);
    return result;
}
catch (const std::exception& exception)
{
    std::cerr << "TB2J native metadata rank " << GlobalV::MY_RANK << ": " << exception.what() << std::endl;
#ifdef __MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
#endif
    throw;
}
} // namespace Tb2j
} // namespace ModuleIO
