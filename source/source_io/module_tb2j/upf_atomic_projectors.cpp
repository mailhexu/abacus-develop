#include "upf_atomic_projectors.h"

#include "source_base/constants.h"
#include "source_base/global_variable.h"
#include "source_base/math_integral.h"
#include "source_base/math_sphbes.h"
#include "source_base/math_polyint.h"
#include "source_base/math_ylmreal.h"
#include "source_base/parallel_common.h"
#include "source_base/tool_quit.h"
#include "source_basis/module_pw/pw_basis_k.h"
#include "source_cell/read_pp.h"
#include "source_cell/unitcell.h"
#include "source_pw/module_pwdft/stru_fac.h"

#include <cassert>
#include <cmath>

namespace ModuleIO
{
namespace Tb2j
{

namespace
{
// MathDoc: tb2j-dev/qe-atomic-pao-trace
// Odd meshes use the native UPF measure directly; even meshes apply its
// pointwise factors before the existing composite Simpson rule.
double integrate_measure(const int mesh, double* integrand, const double* rab)
{
    if (mesh % 2 == 1)
    {
        double sum = 0.0;
        ModuleBase::Integral::Simpson_Integral(mesh, integrand, rab, sum);
        return sum;
    }
    for (int ir = 0; ir < mesh; ++ir)
    {
        integrand[ir] *= rab[ir];
    }
    return ModuleBase::Integral::simpson(mesh, integrand, 1.0);
}

// MathDoc: tb2j-dev/qe-atomic-pao-trace
// Transform one complete radial channel with the native spherical-Bessel and
// integration primitives on the immutable source mesh.
void transform_channel(const int mesh,
                       const double* r,
                       const double* rab,
                       const double* chi,
                       const int l,
                       const double* q,
                       const int nq,
                       double* bessel,
                       double* integrand,
                       double* table)
{
    assert(mesh > 0);
    assert(l >= 0);
    assert(nq >= 0);

    for (int iq = 0; iq < nq; ++iq)
    {
        ModuleBase::Sphbes::Spherical_Bessel(mesh, r, q[iq], l, bessel);
        for (int ir = 0; ir < mesh; ++ir)
        {
            integrand[ir] = chi[ir] * bessel[ir] * r[ir];
        }
        table[iq] = integrate_measure(mesh, integrand, rab);
    }
}
} // namespace

std::unique_ptr<SpeciesBasis> AtomicBasis::capture_species(const std::string& filename,
                                                           const std::string& label,
                                                           const std::string& pseudo_format,
                                                           std::string& error)
{
    error.clear();
    // Opt-in full-mesh parse of the ORIGINAL UPF: the native ncpp on the unit
    // cell keeps its own truncated/averaged data. Single parse for guard and
    // snapshot; no second parse for validation.
    Atom_pseudo original;
    Pseudopot_upf reader(true); // preserve-mesh mode: keep the full even mesh
    std::string format = pseudo_format.empty() ? "auto" : pseudo_format;
    if (reader.init_pseudo_reader(filename, format, original) != 0
        || (format != "upf" && format != "upf201"))
    {
        error = "Atomic projectors require a readable UPF pseudopotential";
        return nullptr;
    }
    if (const char* channel_error = atomic_channel_error(original, reader.relativistic == "full"))
    {
        error = channel_error;
        return nullptr;
    }
    // the exporter quadrature needs a finite grid with a positive measure
    for (int ir = 0; ir < original.mesh; ++ir)
    {
        if (!std::isfinite(original.r[ir]) || original.r[ir] < 0.0 || !std::isfinite(original.rab[ir])
            || original.rab[ir] <= 0.0)
        {
            error = "Atomic projectors require a finite radial grid with a positive measure";
            return nullptr;
        }
    }
    auto snapshot = std::make_unique<SpeciesBasis>();

    snapshot->label = label;
    snapshot->pseudo_file = filename;
    snapshot->mesh = original.mesh;
    snapshot->r = original.r;   // immutable full-mesh copies
    snapshot->rab = original.rab;
    snapshot->channels.resize(original.nchi);

    std::vector<double> work(original.mesh);
    for (int ic = 0; ic < original.nchi; ++ic)
    {
        AtomicChannel& channel = snapshot->channels[ic];
        channel.l = original.lchi[ic];
        channel.occupation = original.oc[ic];
        channel.label = original.els[ic];
        channel.chi.resize(original.mesh);
        for (int ir = 0; ir < original.mesh; ++ir)
        {
            channel.chi[ir] = original.chi(ic, ir);
            work[ir] = channel.chi[ir] * channel.chi[ir];
        }
        // Normalize the exporter channel on the full UPF mesh; the source
        // Atom_pseudo data stay untouched.
        // The work buffer holds chi squared; the even-mesh measure weighting
        // can reuse it in place without allocating another mesh-sized array.
        const double norm2 = integrate_measure(original.mesh, work.data(), original.rab.data());
        if (!std::isfinite(norm2) || norm2 <= 0.0)
        {
            error = "Atomic-wavefunction channel has an invalid radial norm";
            return nullptr;
        }
        const double inverse_norm = 1.0 / std::sqrt(norm2);
        for (int ir = 0; ir < original.mesh; ++ir)
        {
            channel.chi[ir] *= inverse_norm;
        }
    }
    return snapshot;
}

std::unique_ptr<AtomicBasis> AtomicBasis::capture(const UnitCell& ucell,
                                                  const std::string& pseudo_dir,
                                                  const int nqx,
                                                  const double dq)
{
    // Native rank-zero-only pseudopotential IO policy: only the root rank
    // reads and validates each original UPF once per species; errors are
    // broadcast and every rank quits uniformly.
    std::string error_message;
    std::vector<SpeciesBasis> species;
    if (GlobalV::MY_RANK == 0)
    {
        species.resize(ucell.ntype);
        for (int it = 0; it < ucell.ntype; ++it)
        {
            std::string error;
            const std::string filename = pseudo_dir + ucell.pseudo_fn[it];
            auto snapshot = capture_species(filename, ucell.atoms[it].label, ucell.pseudo_type[it], error);
            if (!snapshot)
            {
                error_message = error;
                break;
            }
            snapshot->natoms = ucell.atoms[it].na;
            species[it] = std::move(*snapshot);
        }
    }
    Parallel_Common::bcast_string(error_message);
    if (!error_message.empty())
    {
        ModuleBase::WARNING_QUIT("out_tb2j", error_message);
    }

#ifdef __MPI
    // Broadcast the immutable snapshots; each rank then tabulates locally.
    species.resize(ucell.ntype);
    for (auto& sp : species)
    {
        Parallel_Common::bcast_string(sp.label);
        Parallel_Common::bcast_string(sp.pseudo_file);
        Parallel_Common::bcast_int(sp.natoms);
        Parallel_Common::bcast_int(sp.mesh);
        sp.r.resize(sp.mesh);
        Parallel_Common::bcast_double(sp.r.data(), sp.mesh);
        sp.rab.resize(sp.mesh);
        Parallel_Common::bcast_double(sp.rab.data(), sp.mesh);
        int nchannels = static_cast<int>(sp.channels.size());
        Parallel_Common::bcast_int(nchannels);
        sp.channels.resize(nchannels);
        for (auto& channel : sp.channels)
        {
            Parallel_Common::bcast_int(channel.l);
            Parallel_Common::bcast_double(channel.occupation);
            Parallel_Common::bcast_string(channel.label);
            channel.chi.resize(sp.mesh);
            Parallel_Common::bcast_double(channel.chi.data(), sp.mesh);
        }
    }
#endif

    return std::unique_ptr<AtomicBasis>(new AtomicBasis(std::move(species), nqx, dq));
}

AtomicBasis::AtomicBasis(std::vector<SpeciesBasis> species, const int nqx, const double dq)
    : species_(std::move(species)), nqx_(nqx), dq_(dq)
{
    if (nqx_ < 4 || !std::isfinite(dq_) || dq_ <= 0.0)
    {
        ModuleBase::WARNING_QUIT("out_tb2j",
                                 "Atomic-basis q table needs at least four samples and positive finite dq.");
    }
    // Deterministic native order: species -> atom -> source channel ->
    // m = 0..2l with lm = l*l + m. Repeated-l channels stay distinct entries
    // and unoccupied/negative-occupation channels are never dropped.
    int next_row = 0;
    for (int it = 0; it < this->ntype(); ++it)
    {
        const SpeciesBasis& sp = species_[it];
        for (int ia = 0; ia < sp.natoms; ++ia)
        {
            for (std::size_t ic = 0; ic < sp.channels.size(); ++ic)
            {
                ProjectorSite site;
                site.it = it;
                site.ia = ia;
                site.channel = static_cast<int>(ic);
                site.l = sp.channels[ic].l;
                site.count = 2 * site.l + 1;
                site.start = next_row;
                next_row += site.count;
                sites_.push_back(site);
            }
        }
    }
    nproj_ = next_row;
    this->tabulate();
}

void AtomicBasis::tabulate()
{
    // Cell-independent q-space tables on the native grid; the cell-dependent
    // 4pi/sqrt(omega) is applied at fill time from the current cell.
    std::vector<double> q(nqx_);
    for (int iq = 0; iq < nqx_; ++iq)
    {
        q[iq] = dq_ * iq;
    }
    for (auto& sp : species_)
    {
        std::vector<double> bessel(sp.mesh);
        std::vector<double> integrand(sp.mesh);
        for (auto& channel : sp.channels)
        {
            channel.table.resize(nqx_);
            transform_channel(sp.mesh,
                              sp.r.data(),
                              sp.rab.data(),
                              channel.chi.data(),
                              channel.l,
                              q.data(),
                              nqx_,
                              bessel.data(),
                              integrand.data(),
                              channel.table.data());
        }
    }
}

const char* AtomicBasis::fill_pw(const int ik,
                          const UnitCell& ucell,
                          const Structure_Factor& sf,
                          const ModulePW::PW_Basis_K* pw_wfc,
                          FillWorkspace& work,
                          std::complex<double>* out,
                          const int stride) const
{
    // MathDoc: tb2j-dev/qe-atomic-pao-trace
    // Native collinear atomic-PW construction: real harmonics lm = l*l + m,
    // phase pow(NEG_IMAG_UNIT, l), radial interpolation at |G+k| * tpiba and
    // the cell-dependent 4pi/sqrt(omega) applied here from the current cell.
    const int npw = pw_wfc->npwk[ik];
    const int lmax = ucell.lmax_ppwf;
    const int total_lm = (lmax + 1) * (lmax + 1);

    // caller-owned reusable workspace, resized on demand only
    work.gk.resize(npw);
    work.q.resize(npw);
    for (int ig = 0; ig < npw; ++ig)
    {
        work.gk[ig] = pw_wfc->getgpluskcar(ik, ig);
        work.q[ig] = work.gk[ig].norm() * ucell.tpiba;
        const double position = work.q[ig] / dq_;
        // The native four-point stencil requires floor(position) <= nqx-4.
        // Reject before writing output, rather than taking PolyInt's zero
        // fallback. No rank-local quit before the caller's collective checks.
        if (!std::isfinite(position) || position < 0.0 || position >= nqx_ - 3)
        {
            return "Atomic-projector radial table does not cover the requested G+k";
        }
    }
    work.ylm.create(total_lm, npw, false);
    ModuleBase::YlmReal::Ylm_Real(total_lm, npw, work.gk.data(), work.ylm);

    // Every active PW is assigned below; clear only padding in reused rows.
    for (int row = 0; row < nproj_; ++row)
    {
        for (int col = npw; col < stride; ++col)
        {
            out[row * stride + col] = std::complex<double>(0.0, 0.0);
        }
    }

    // cell-dependent normalization from the CURRENT cell; tables stay
    // cell-independent across cell relaxation
    const double pref = ModuleBase::FOUR_PI / std::sqrt(ucell.omega);

    int row = 0;
    for (int it = 0; it < this->ntype(); ++it)
    {
        const SpeciesBasis& sp = species_[it];
        const int nchannels = static_cast<int>(sp.channels.size());
        work.radial.resize(static_cast<std::size_t>(nchannels) * npw);
        for (int ic = 0; ic < nchannels; ++ic)
        {
            for (int ig = 0; ig < npw; ++ig)
            {
                work.radial[static_cast<std::size_t>(ic) * npw + ig]
                    = ModuleBase::PolyInt::Polynomial_Interpolation(
                        sp.channels[ic].table.data(), nqx_, dq_, work.q[ig]);
            }
        }
        for (int ia = 0; ia < sp.natoms; ++ia)
        {
            std::complex<double>* sk = sf.get_sk(ik, it, ia, pw_wfc);
            for (int ic = 0; ic < nchannels; ++ic)
            {
                const AtomicChannel& channel = sp.channels[ic];
                const int l = channel.l;
                // native phase convention, native real-harmonic indexing
                const std::complex<double> lphase = pow(ModuleBase::NEG_IMAG_UNIT, l);
                for (int m = 0; m < 2 * l + 1; ++m)
                {
                    const int lm = l * l + m;
                    for (int ig = 0; ig < npw; ++ig)
                    {
                        out[row * stride + ig]
                            = lphase * pref * sk[ig] * work.ylm(lm, ig)
                              * work.radial[static_cast<std::size_t>(ic) * npw + ig];
                    }
                    ++row;
                }
            }
            delete[] sk;
        }
    }
    return nullptr;
}

} // namespace Tb2j
} // namespace ModuleIO
