#ifndef ABACUS_TB2J_PROJECTOR_NETCDF_H
#define ABACUS_TB2J_PROJECTOR_NETCDF_H

#include "projection.h"
#include <complex>
#include <string>
#include <vector>

struct Input_para;
namespace ModuleIO
{
namespace Tb2j
{
class AtomicBasis;
class NativeState;

/// One serial root writer for the existing tb2j.projector_green/1.0 contract.
/// A unique destination-sibling partial becomes final only after full coverage,
/// the complete onsite vertex and a successful NetCDF close. Throws on error;
/// the distributed caller must abort MPI_COMM_WORLD, never retry the stream.
class ProjectorNetcdf
{
  public:
    ProjectorNetcdf(const std::string& destination, const NativeState& state,
                    const AtomicBasis& basis, const std::vector<SiteBlock>& sites,
                    const Input_para& input);
    ~ProjectorNetcdf();
    ProjectorNetcdf(const ProjectorNetcdf&) = delete;
    ProjectorNetcdf& operator=(const ProjectorNetcdf&) = delete;

    /// Complete indexed one-k block: C[2,global-band,projector], S[projector,projector].
    void write_k(int unique_k, const std::complex<double>* coefficients,
                 const std::complex<double>* gram);
    /// Packed complete-site R=0/BZ-averaged delta_total in eV, per SiteBlock.offset.
    void finish(const std::complex<double>* site_delta_ev);

  private:
    void cleanup() noexcept;
    const NativeState& state_;
    const AtomicBasis& basis_;
    const std::vector<SiteBlock>& sites_;
    std::string destination_, partial_;
    int ncid_ = -1, projectors_ = -1, components_ = -1;
    int coefficients_ = -1, overlap_ = -1, delta_ = -1;
    int nmax_ = 0, written_ = 0;
    bool published_ = false;
    std::vector<unsigned char> seen_;
};
} // namespace Tb2j
} // namespace ModuleIO
#endif
