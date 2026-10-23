#ifndef ABACUS_TB2J_NATIVE_STATE_H
#define ABACUS_TB2J_NATIVE_STATE_H

#include "source_base/matrix.h"
#include "source_base/vector3.h"
#include <array>
#include <memory>
#include <string>
#include <vector>

class K_Vectors;
class UnitCell;
struct Input_para;
namespace elecstate { class ElecState; }
namespace Parallel { class ParaBandOutput; }

namespace ModuleIO
{
namespace Tb2j
{
struct GridRecipe
{
    bool generated = false;
    int input_count = -1;
    int original_count = 0;
    int symmetry = 0;
    std::array<int, 3> dimensions{};
    std::array<double, 3> offsets{};
    std::string convention;
};

/// Non-owning native rows; band columns have already been assembled by ParaBandOutput.
struct SpectralBlock
{
    int nrow;
    const int* global_k;
    const int* spin;
    const ModuleBase::Vector3<double>* kpoints;
    const double* weights;
    const ModuleBase::matrix* energies_ry;
    const ModuleBase::matrix* weighted_occupations;
};

/// Private root-owned metadata; no KS coefficients or projected matrices live here.
class NativeState
{
  public:
    static std::unique_ptr<NativeState> create(const GridRecipe& recipe, int nband, std::string& error);
    bool append(const SpectralBlock& block, std::string& error);
    bool finish(double native_nelec, const std::array<double, 2>& fermi_ry,
                bool custom, std::string& error);

    /// Collective, after converged SCF and before estate teardown. Non-root returns nullptr.
    static std::unique_ptr<NativeState> capture(const UnitCell& cell, const K_Vectors& kv,
                                               const elecstate::ElecState& estate, const Input_para& input,
                                               const Parallel::ParaBandOutput& bands);

    GridRecipe recipe;
    int nkpt = 0;
    int nband = 0;
    std::vector<ModuleBase::Vector3<double>> kpoints;
    std::vector<double> weights;
    double native_weight_sum = 0.0; ///< unique native up-spin sum, for identical per-pool normalization
    std::vector<double> eigenvalues; ///< spin,k,global-band; eV
    std::vector<double> occupations; ///< wg / corresponding native wk
    std::array<double, 2> efermi_spin{}; ///< eV, native two-Fermi state retained
    std::array<double, 2> native_electrons{};
    std::array<double, 2> electrons{};
    bool custom_occupations = false;
    std::array<double, 9> cell{}; ///< row-major lattice vectors, Angstrom
    std::vector<ModuleBase::Vector3<double>> positions; ///< native species/atom order, Angstrom
    std::vector<int> atomic_numbers;

  private:
    std::vector<unsigned char> seen_;
    std::vector<double> native_weights_;
    bool finished_ = false;
};
} // namespace Tb2j
} // namespace ModuleIO
#endif
