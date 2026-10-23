#include "atomic_radial_transform.h"

#include "source_base/math_integral.h"
#include "source_base/math_sphbes.h"

#include <cassert>

// MathDoc: tb2j-dev/qe-atomic-pao-trace
double atomic_radial::integrate_measure(const int mesh, double* integrand, const double* rab)
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
// Shared radial mechanics extracted from psi_init_atomic::tabulate. The
// initializer passes its native odd msh; the TB2J atomic-basis capture passes
// the full preserved UPF mesh. Callers keep all physical policy.
void atomic_radial::transform_channel(const int mesh,
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
