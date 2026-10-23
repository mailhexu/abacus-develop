#ifndef ABACUS_TB2J_ATOMIC_PROJECTORS_H
#define ABACUS_TB2J_ATOMIC_PROJECTORS_H

#include "source_base/matrix.h"
#include "source_base/vector3.h"
#include "source_cell/atom_pseudo.h"

#include <cmath>
#include <complex>
#include <memory>
#include <string>
#include <vector>

class UnitCell;
class Structure_Factor;
namespace ModulePW
{
class PW_Basis_K;
}

namespace ModuleIO
{
namespace Tb2j
{

/// @brief Pure channel guard for the TB2J atomic-projector scope: scalar NC
/// pseudopotentials only (no USPP/PAW/SOC/fully-relativistic), with complete
/// chi/r/rab/lchi arrays and a finite, nonzero per-channel square sum.
/// Returns nullptr when the basis is usable, otherwise a static error text.
inline const char* atomic_channel_error(const Atom_pseudo& pp, const bool fully_relativistic = false)
{
    if (pp.tvanp || pp.pp_type != "NC" || pp.has_so || fully_relativistic)
    {
        return "Only scalar norm-conserving atomic projectors are supported";
    }
    if (pp.nchi <= 0 || pp.mesh <= 1 || pp.chi.nr != pp.nchi || pp.chi.nc < pp.mesh
        || pp.r.size() != static_cast<std::size_t>(pp.mesh)
        || pp.rab.size() != static_cast<std::size_t>(pp.mesh)
        || pp.lchi.size() != static_cast<std::size_t>(pp.nchi))
    {
        return "Missing or invalid atomic-wavefunction channels";
    }
    for (int channel = 0; channel < pp.nchi; ++channel)
    {
        double square_sum = 0.0;
        for (int ir = 0; ir < pp.mesh; ++ir)
        {
            const double value = pp.chi(channel, ir);
            square_sum += value * value;
        }
        if (pp.lchi[channel] < 0 || !std::isfinite(square_sum) || square_sum == 0.0)
        {
            return "Atomic-wavefunction channel has an invalid radial norm";
        }
    }
    return nullptr;
}

/// One immutable radial channel copied from the source UPF.
struct AtomicChannel
{
    int l = 0;                ///< angular momentum of the channel
    double occupation = 0.0;  ///< source occupancy (may be zero or negative)
    std::string label;        ///< source els label
    std::vector<double> chi;  ///< radial values on the FULL source UPF mesh,
                              ///< normalized in the basis (immutable copy)
    std::vector<double> table; ///< cell-independent q-space transform, [nqx]
};

/// One species' immutable full-mesh radial basis plus raw-file provenance.
struct SpeciesBasis
{
    std::string label;       ///< species label
    std::string pseudo_file; ///< source UPF filename (as referenced)
    std::string sha256;      ///< SHA256 of the raw source UPF bytes
    int natoms = 0;          ///< number of atoms of this species in the cell
    int mesh = 0;            ///< full UPF mesh (even meshes preserved)
    std::vector<double> r;   ///< full UPF radial grid [mesh]
    std::vector<double> rab; ///< original UPF PP_RAB integration factors [mesh]
    std::vector<AtomicChannel> channels; ///< ALL source channels, source order
};

/// Native index map of one (atom, source channel) entry in the deterministic
/// fill order: species it -> atom ia -> source channel ic -> m = 0..2l with
/// lm = l*l + m.
struct ProjectorSite
{
    int it = 0;      ///< species index
    int ia = 0;      ///< atom index within the species
    int channel = 0; ///< source radial channel index (distinct for repeated l)
    int l = 0;       ///< angular momentum
    int start = 0;   ///< first harmonic row of this site in fill order
    int count = 0;   ///< 2l + 1 harmonic rows
};

/**
 * @brief Immutable all-channel atomic basis captured from the original UPF
 * files before the native initial-guess mutates the pseudopotential data
 * (opt-in, out_tb2j only).
 *
 * MathDoc: tb2j-dev/qe-atomic-pao-trace
 *
 * The capture keeps the complete UPF r/rab/mesh (preserve-mesh parse), ALL
 * source channels in source order (including repeated-l and zero/negative
 * occupations), normalizes each channel on the full mesh, hashes the raw
 * source bytes (SHA256 provenance) and tabulates cell-independent q-space
 * transforms on the native nqx/dq grid. fill_pw reproduces the native
 * psi_init_atomic collinear conventions for every channel, applying
 * 4pi/sqrt(omega) from the CURRENT cell at fill time.
 */
class AtomicBasis
{
  public:
    /// Caller-owned reusable scratch for fill_pw; no per-k output allocation.
    struct FillWorkspace
    {
        std::vector<ModuleBase::Vector3<double>> gk;
        std::vector<double> q; ///< |G+k| * tpiba, computed once per PW row
        ModuleBase::matrix ylm;     ///< (lmax+1)^2 x npw real harmonics
        std::vector<double> radial; ///< channel-major npw factors, reused across atoms
    };

    /**
     * @brief Capture every species' original UPF basis (root-only IO).
     *
     * Root parses each ucell.pseudo_fn in preserve-mesh mode, validates and
     * hashes it; errors are broadcast and every rank quits uniformly with the
     * native pseudopotential IO policy. The immutable snapshots are
     * broadcast so every rank tabulates its own local q tables. Allocates
     * only when called (call under the opt-in flag).
     *
     * @param ucell cell with the final species/atom layout
     * @param pseudo_dir pseudopotential directory
     * @param nqx native q-table length (PARAM.globalv.nqx)
     * @param dq native q-table spacing (PARAM.globalv.dq)
     */
    static std::unique_ptr<AtomicBasis> capture(const UnitCell& ucell,
                                                const std::string& pseudo_dir,
                                                const int nqx,
                                                const double dq);

    /**
     * @brief Root-only capture of one original UPF into an immutable
     * snapshot: preserve-mesh parse, channel validation, full-mesh
     * normalization, raw-byte SHA256. No broadcast inside; pure given the
     * file.
     * @param filename UPF file to read
     * @param label species label recorded in the snapshot
     * @param error set to a non-empty message when the input is rejected
     * @param pseudo_format native STRU format declaration; empty means auto
     * @return the snapshot, or nullptr with error set on any invalid input
     */
    static std::unique_ptr<SpeciesBasis> capture_species(const std::string& filename,
                                                         const std::string& label,
                                                         const std::string& pseudo_format,
                                                         std::string& error);

    /**
     * @brief Build from an already captured snapshot set (all ranks) and
     * tabulate the cell-independent q tables on the native nqx/dq grid.
     * Takes ownership of deep copies; later mutation of the caller's
     * snapshots cannot affect the basis.
     */
    AtomicBasis(std::vector<SpeciesBasis> species, const int nqx, const double dq);

    int ntype() const
    {
        return static_cast<int>(species_.size());
    }
    const SpeciesBasis& species(int it) const
    {
        return species_[it];
    }
    /// total harmonic rows: sum over sites of 2l+1
    int nproj() const
    {
        return nproj_;
    }
    /// deterministic native-order site map (species -> atom -> channel)
    const std::vector<ProjectorSite>& sites() const
    {
        return sites_;
    }

    /**
     * @brief Fill the native atomic PW coefficients of ALL source channels
     * for one k-point into a caller-owned row-major buffer.
     *
     * Native psi_init_atomic collinear conventions: getgpluskcar, Ylm_Real on
     * (lmax_ppwf+1)^2 real harmonics with lm = l*l + m, get_sk,
     * pow(NEG_IMAG_UNIT, l) phase, radial interpolation at |G+k| * tpiba.
     * Rows [0, nproj()) are written for columns [0, npwk[ik]); the whole
     * row-major region [0, nproj()) x stride is zeroed first, matching the
     * native zero-then-fill behavior for reused buffers. The cell-dependent
     * 4pi/sqrt(omega) is applied here from the current cell.
     *
     * @param ik k-point index
     * @param ucell current cell (omega/tpiba/lmax_ppwf/atom layout)
     * @param sf structure factor provider
     * @param pw_wfc plane-wave basis of the k-point
     * @param work caller-owned reusable workspace
     * @param out row-major output [nproj() rows, stride columns]
     * @param stride row stride of out (>= pw_wfc->npwk[ik])
     * @return nullptr on success, otherwise a static diagnostic. On failure
     * output is untouched; the caller must propagate the error collectively.
     */
    [[nodiscard]] const char* fill_pw(const int ik,
                 const UnitCell& ucell,
                 const Structure_Factor& sf,
                 const ModulePW::PW_Basis_K* pw_wfc,
                 FillWorkspace& work,
                 std::complex<double>* out,
                 const int stride) const;

  private:
    void tabulate();

    std::vector<SpeciesBasis> species_;
    std::vector<ProjectorSite> sites_;
    int nproj_ = 0;
    int nqx_ = 0;
    double dq_ = 0.0;
};

} // namespace Tb2j
} // namespace ModuleIO

#endif // ABACUS_TB2J_ATOMIC_PROJECTORS_H
