#ifndef ABACUS_TB2J_PW_PROJECTOR_KERNELS_H
#define ABACUS_TB2J_PW_PROJECTOR_KERNELS_H

#include <complex>
#include <cstddef>
#include <vector>

class Charge;
class UnitCell;
namespace ModulePW
{
class PW_Basis;
class PW_Basis_K;
}

namespace ModuleIO
{
namespace Tb2j
{
class AtomicBasis;

/// One complete atom's contiguous all-channel block, not one radial channel.
struct SiteBlock
{
    int first = 0;
    int count = 0;
    std::size_t offset = 0; ///< packed site-matrix offset
};
std::vector<SiteBlock> atomic_sites(const AtomicBasis& basis);

/// Reused one-k/largest-site scratch. No whole-wavefunction host mirror.
struct ProjectionWorkspace
{
    std::vector<std::complex<double>> real, weighted, column, lapack, work;
    std::vector<double> eigenvalues, rwork;
};

/// Local PW contributions. Row-major C[band,projector] = <phi|psi>;
/// S[projector,projector] = <phi_p|phi_q>, including intersite blocks.
/// Padding is not read. Zero local bands/PWs are valid distributed layouts.
void primal_overlaps(int nband, int nproj, int npw,
                     const std::complex<double>* psi, int psi_stride,
                     const std::complex<double>* phi, int phi_stride,
                     std::complex<double>* coefficients);
void full_gram(int nproj, int npw, const std::complex<double>* phi,
               int phi_stride, std::complex<double>* gram);

/// On a pool-reduced full Gram: finite Hermitian positive definite, cond<=1e12.
/// Returns the actual eigenvalue condition; never inverts or drops channels.
double metric_condition(int nproj, const std::complex<double>* gram,
                        ProjectionWorkspace& work);

/// Reevaluate only native semilocal XC on converged charge, retaining native Ry.
/// Dense-to-smooth interpolation follows Potential::interpolate_vrs without
/// changing any native potential buffer. The returned slab matches pw_wfc.
std::vector<double> xc_difference(const Charge& charge, const UnitCell& cell,
                                  ModulePW::PW_Basis& dense,
                                  ModulePW::PW_Basis& smooth,
                                  const ModulePW::PW_Basis_K& pw_wfc);

/// Local slab contribution to each complete atom's covariant XC vertex in Ry.
/// Native IFFT coefficients require 1/nxyz, NOT omega/nxyz. Pool reduction and
/// unique-k ownership belong to the caller. Uses one FFT per projector.
void project_site_xc(int ik, const ModulePW::PW_Basis_K& pw,
                     const std::vector<SiteBlock>& sites,
                     const std::complex<double>* phi, int phi_stride,
                     const double* delta_v_ry, ProjectionWorkspace& work,
                     std::complex<double>* site_vertex_ry);

/// Add one certified unique-k contribution, converting Ry to common-file eV.
/// The caller supplies each unique normalized BZ weight exactly once.
void accumulate_site_vertex(double weight, std::size_t count,
                            const std::complex<double>* vertex_ry,
                            std::complex<double>* accumulated_ev);

} // namespace Tb2j
} // namespace ModuleIO
#endif
