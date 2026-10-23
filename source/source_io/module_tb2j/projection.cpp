#include "projection.h"
#include "atomic_projectors.h"
#include "source_base/constants.h"
#include "source_base/module_external/blas_connector.h"
#include "source_base/module_external/lapack_connector.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_basis/module_pw/pw_basis_k.h"
#include "source_cell/unitcell.h"
#include "source_estate/module_charge/charge.h"
#include "source_hamilt/module_xc/xc_functional.h"
#include "source_io/module_parameter/parameter.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ModuleIO
{
namespace Tb2j
{
// MathDoc: tb2j-dev/qe-atomic-pao-trace
// Primal coefficients, full covariant metric, and a BZ-averaged onsite vertex
// retain the existing common-consumer pairing; no dualization happens here.
namespace
{
bool finite(const std::complex<double> value)
{
    return std::isfinite(value.real()) && std::isfinite(value.imag());
}
void dimensions(const int nband, const int nproj, const int npw,
                const int psi_stride, const int phi_stride)
{
    if (nband < 0 || nproj <= 0 || npw < 0 || psi_stride < npw || phi_stride < npw)
        throw std::runtime_error("Invalid TB2J local projection dimensions");
}
} // namespace

std::vector<SiteBlock> atomic_sites(const AtomicBasis& basis)
{
    std::vector<SiteBlock> result;
    int previous_it = -1, previous_ia = -1;
    for (const auto& channel : basis.sites())
    {
        if (channel.it != previous_it || channel.ia != previous_ia)
        {
            const std::size_t offset = result.empty() ? 0 : result.back().offset
                + static_cast<std::size_t>(result.back().count)*result.back().count;
            result.push_back({channel.start, channel.count, offset});
            previous_it = channel.it;
            previous_ia = channel.ia;
        }
        else
        {
            result.back().count += channel.count;
        }
    }
    return result;
}

void primal_overlaps(const int nband, const int nproj, const int npw,
                     const std::complex<double>* psi, const int psi_stride,
                     const std::complex<double>* phi, const int phi_stride,
                     std::complex<double>* coefficients)
{
    dimensions(nband,nproj,npw,psi_stride,phi_stride);
    if (nband == 0) return;
    if (npw == 0)
    {
        std::fill_n(coefficients,static_cast<std::size_t>(nband)*nproj,std::complex<double>{});
        return;
    }
    // BlasConnector's pointer overload is ROW-major. psi * phi^H gives
    // C[band,projector] = sum_G psi[band,G] conj(phi[projector,G]).
    BlasConnector::gemm('N','C',nband,nproj,npw,std::complex<double>(1.),
                        psi,psi_stride,phi,phi_stride,std::complex<double>(0.),coefficients,nproj);
}

void full_gram(const int nproj, const int npw, const std::complex<double>* phi,
               const int phi_stride, std::complex<double>* gram)
{
    dimensions(0,nproj,npw,npw,phi_stride);
    if (npw == 0)
    {
        std::fill_n(gram,static_cast<std::size_t>(nproj)*nproj,std::complex<double>{});
        return;
    }
    const char upper='U', conjugate='C';
    const double one=1., zero=0.;
    // Row-major phi[P,G] is column-major phi^T[G,P]. HERK computes
    // <phi_p|phi_q> in its column-major upper triangle. Repack in place,
    // computing only one triangle rather than doing two full GEMMs.
    zherk_(&upper,&conjugate,&nproj,&npw,&one,phi,&phi_stride,&zero,gram,&nproj);
    for (int p=0; p<nproj; ++p)
        for (int q=p+1; q<nproj; ++q)
        {
            const auto value=gram[static_cast<std::size_t>(q)*nproj+p];
            gram[static_cast<std::size_t>(p)*nproj+q]=value;
            gram[static_cast<std::size_t>(q)*nproj+p]=std::conj(value);
        }
}

double metric_condition(const int nproj, const std::complex<double>* gram,
                        ProjectionWorkspace& work)
{
    if (nproj <= 0 || nproj > (std::numeric_limits<int>::max()-1)/3)
        throw std::runtime_error("Invalid TB2J full metric dimension");
    const auto size=static_cast<std::size_t>(nproj)*nproj;
    double scale=0.;
    for (std::size_t i=0; i<size; ++i)
    {
        if (!finite(gram[i]) || !std::isfinite(std::abs(gram[i])))
            throw std::runtime_error("Nonfinite TB2J full atomic metric");
        scale=std::max(scale,std::abs(gram[i]));
    }
    for (int p=0; p<nproj; ++p)
        for (int q=p; q<nproj; ++q)
            if (std::abs(gram[static_cast<std::size_t>(p)*nproj+q]
                         -std::conj(gram[static_cast<std::size_t>(q)*nproj+p])) > 1e-10*scale)
                throw std::runtime_error("Non-Hermitian TB2J full atomic metric");
    work.lapack.assign(gram,gram+size); // LAPACK destroys its input; preserve exported metric.
    work.eigenvalues.resize(nproj);
    const int lwork=std::max(1,2*nproj-1);
    work.work.resize(lwork);
    work.rwork.resize(std::max(1,3*nproj-2));
    const char values='N', upper='U';
    int info=0;
    zheev_(&values,&upper,&nproj,work.lapack.data(),&nproj,work.eigenvalues.data(),
            work.work.data(),&lwork,work.rwork.data(),&info);
    if (info != 0)
        throw std::runtime_error("TB2J full metric Hermitian eigensolver failed, info="+std::to_string(info));
    const double smallest=work.eigenvalues.front(), largest=work.eigenvalues.back();
    const double condition=largest/smallest;
    if (!(smallest > 0.) || !std::isfinite(largest) || !std::isfinite(condition) || condition > 1e12)
        throw std::runtime_error("Singular/ill-conditioned TB2J full atomic metric (inverse limit 1e12)");
    return condition;
}

std::vector<double> xc_difference(const Charge& charge, const UnitCell& cell,
                                  ModulePW::PW_Basis& dense,
                                  ModulePW::PW_Basis& smooth,
                                  const ModulePW::PW_Basis_K& pw_wfc)
{
    if (charge.nspin != 2 || charge.nrxx != dense.nrxx
        || XC_Functional::get_ked_flag()
        || (XC_Functional::get_func_type() != 1 && XC_Functional::get_func_type() != 2))
        throw std::runtime_error("TB2J XC-only vertex requires native two-spin semilocal LDA/GGA charge (no meta/hybrid)");
    if (smooth.nx != pw_wfc.nx || smooth.ny != pw_wfc.ny || smooth.nz != pw_wfc.nz
        || smooth.nrxx != pw_wfc.nrxx || smooth.nplane != pw_wfc.nplane
        || smooth.startz_current != pw_wfc.startz_current)
        throw std::runtime_error("TB2J XC and wavefunction FFT slab layouts differ");
    const double hybrid_alpha=XC_Functional::get_hybrid_alpha();
#ifdef __EXX
    const double hse_omega=XC_Functional::get_hse_omega();
#else
    const double hse_omega=0.;
#endif
    auto evaluated=XC_Functional::v_xc(charge.nrxx,&charge,&cell,2,
                                      PARAM.globalv.domag,PARAM.globalv.domag_z,
                                      PARAM.inp.gga_grad,hybrid_alpha,hse_omega);
    const auto& v=std::get<2>(evaluated);
    if (v.nr != 2 || v.nc != charge.nrxx)
        throw std::runtime_error("Invalid native XC-only potential shape");
    std::vector<double> difference(charge.nrxx);
    for (int ir=0; ir<charge.nrxx; ++ir)
    {
        difference[ir]=v(0,ir)-v(1,ir);
        if (!std::isfinite(difference[ir])) throw std::runtime_error("Nonfinite native XC spin difference");
    }
    if (&dense == &smooth) return difference;
    // Exactly the native Potential::interpolate_vrs path, on owned buffers.
    std::vector<std::complex<double>> reciprocal(dense.npw);
    std::vector<double> interpolated(smooth.nrxx);
    dense.real2recip(difference.data(),reciprocal.data());
    smooth.recip2real(reciprocal.data(),interpolated.data());
    for (const auto value:interpolated)
        if (!std::isfinite(value)) throw std::runtime_error("Nonfinite interpolated XC spin difference");
    return interpolated;
}

void project_site_xc(const int ik, const ModulePW::PW_Basis_K& pw,
                     const std::vector<SiteBlock>& sites,
                     const std::complex<double>* phi, const int phi_stride,
                     const double* delta_v_ry, ProjectionWorkspace& work,
                     std::complex<double>* site_vertex_ry)
{
    if (ik < 0 || ik >= pw.nks || phi_stride < pw.npwk[ik] || pw.nxyz <= 0)
        throw std::runtime_error("Invalid TB2J XC projection grid");
    const char conjugate='C';
    const int one_index=1;
    const std::complex<double> alpha(1./pw.nxyz), zero(0.);
    for (const auto& site:sites)
    {
        if (site.first < 0 || site.count <= 0)
            throw std::runtime_error("Invalid TB2J atomic site projection block");
        work.real.resize(std::max<std::size_t>(1,static_cast<std::size_t>(site.count)*pw.nrxx));
        work.weighted.resize(pw.nrxx);
        work.column.resize(site.count);
        for (int p=0; p<site.count; ++p)
        {
            const auto* row=pw.npwk[ik] > 0
                ? phi+static_cast<std::size_t>(site.first+p)*phi_stride : work.column.data();
            pw.recip2real(row,work.real.data()+static_cast<std::size_t>(p)*pw.nrxx,ik);
        }
        for (int q=0; q<site.count; ++q)
        {
            for (int ir=0; ir<pw.nrxx; ++ir)
                work.weighted[ir]=delta_v_ry[ir]*work.real[static_cast<std::size_t>(q)*pw.nrxx+ir];
            const int columns=q+1;
            if (pw.nrxx > 0)
                zgemv_(&conjugate,&pw.nrxx,&columns,&alpha,work.real.data(),&pw.nrxx,
                         work.weighted.data(),&one_index,&zero,work.column.data(),&one_index);
            else
                std::fill_n(work.column.data(),columns,std::complex<double>{});
            for (int p=0; p<=q; ++p)
            {
                const auto value=work.column[p];
                if (!finite(value)) throw std::runtime_error("Nonfinite TB2J XC projected vertex");
                site_vertex_ry[site.offset+static_cast<std::size_t>(p)*site.count+q]=value;
                if (p != q)
                    site_vertex_ry[site.offset+static_cast<std::size_t>(q)*site.count+p]=std::conj(value);
            }
        }
    }
}

void accumulate_site_vertex(const double weight, const std::size_t count,
                            const std::complex<double>* vertex_ry,
                            std::complex<double>* accumulated_ev)
{
    if (!std::isfinite(weight) || weight <= 0.)
        throw std::runtime_error("Invalid certified TB2J BZ weight for onsite vertex");
    const double factor=weight*ModuleBase::Ry_to_eV;
    for (std::size_t i=0; i<count; ++i)
    {
        const auto value=accumulated_ev[i]+factor*vertex_ry[i];
        if (!finite(value)) throw std::runtime_error("Nonfinite TB2J BZ-averaged onsite vertex");
        accumulated_ev[i]=value;
    }
}
} // namespace Tb2j
} // namespace ModuleIO
