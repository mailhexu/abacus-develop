#include "gtest/gtest.h"

/************************************************
 *  unit test of atomic_radial_transform
 ***********************************************/

/**
 * - Tested Functions:
 *   - transform_channel
 *     - the shared non-mutating radial transform mechanics extracted from
 *       psi_init_atomic::tabulate: table[iq] = int chi(r) j_l(q_iq r) r dr
 *   - OddMeshMatchesNativeInitializerArithmetic
 *     - on an odd mesh the helper is bit-identical to the pre-change native
 *       tabulate body (Sphbes::Spherical_Bessel + Integral::Simpson_Integral)
 *   - EvenMeshMatchesAnalyticReference
 *     - the full-even-mesh path (existing composite 1/3 + 3/8 simpson on the
 *       rab-weighted integrand) reproduces the analytic reference
 *       int e^{-r} j_0(q r) r dr = 1 / (1 + q^2)
 *   - EvenMeshMatchesFineTrapezoidReference
 *     - l = 1 full-even-mesh result against an independent fine-grid
 *       trapezoidal quadrature computed here
 *   - EvenAndOddMeshAgreeWithinDiscretization
 *     - even and odd-mesh evaluations of the same channel agree to
 *       discretization error
 *   - InputsAreNotMutated
 *     - the helper writes only its work and output buffers
 */

#include "source_base/math_integral.h"
#include "source_base/math_sphbes.h"
#include "source_psi/atomic_radial_transform.h"

#include <cmath>
#include <vector>

namespace
{
/// Uniform grid r_i = i * h, i = 0..mesh-1, with constant measure rab = h.
struct UniformGrid
{
    std::vector<double> r;
    std::vector<double> rab;
    std::vector<double> chi;
};

UniformGrid make_grid(const int mesh, const double h)
{
    UniformGrid g;
    g.r.resize(mesh);
    g.rab.assign(mesh, h);
    g.chi.resize(mesh);
    for (int i = 0; i < mesh; ++i)
    {
        g.r[i] = i * h;
        g.chi[i] = std::exp(-g.r[i]);
    }
    return g;
}

/// Native replica of the pre-change psi_init_atomic::tabulate inner body.
double native_tabulate_point(const int mesh,
                             const double* r,
                             const double* rab,
                             const double* chi,
                             const int l,
                             const double q)
{
    std::vector<double> aux(mesh);
    std::vector<double> vchi(mesh);
    ModuleBase::Sphbes::Spherical_Bessel(mesh, r, q, l, aux.data());
    for (int ir = 0; ir < mesh; ++ir)
    {
        vchi[ir] = chi[ir] * aux[ir] * r[ir];
    }
    double vqint = 0.0;
    ModuleBase::Integral::Simpson_Integral(mesh, vchi.data(), rab, vqint);
    return vqint;
}
} // namespace

TEST(AtomicRadialTransform, OddMeshMatchesNativeInitializerArithmetic)
{
    const int mesh = 11; // odd, as complete_default forces for the initializer
    UniformGrid g = make_grid(mesh, 0.1);
    const int nq = 5;
    const double dq = 0.01;
    std::vector<double> q(nq);
    for (int iq = 0; iq < nq; ++iq)
    {
        q[iq] = dq * iq;
    }

    std::vector<double> bessel(mesh);
    std::vector<double> integrand(mesh);
    std::vector<double> table(nq);

    for (int l = 0; l <= 3; ++l)
    {
        atomic_radial::transform_channel(mesh,
                                         g.r.data(),
                                         g.rab.data(),
                                         g.chi.data(),
                                         l,
                                         q.data(),
                                         nq,
                                         bessel.data(),
                                         integrand.data(),
                                         table.data());
        for (int iq = 0; iq < nq; ++iq)
        {
            const double ref = native_tabulate_point(mesh, g.r.data(), g.rab.data(), g.chi.data(), l, q[iq]);
            EXPECT_EQ(ref, table[iq]) << "l=" << l << " iq=" << iq;
        }
    }
}

TEST(AtomicRadialTransform, EvenMeshMatchesAnalyticReference)
{
    // chi(r) = e^{-r}, l = 0 on a full even mesh: the integral
    // int_0^rmax e^{-r} j_0(q r) r dr = (1/(1+q^2)) (1 - e^{-rmax}(cos(q rmax)
    // + sin(q rmax)/q)) reduces to 1/(1+q^2) for negligible rmax tail.
    const int mesh = 1000; // even: exercises the full-mesh quadrature path
    UniformGrid g = make_grid(mesh, 0.1);
    const double rmax = g.r[mesh - 1];
    const std::vector<double> qs = {0.01, 0.1, 1.0};

    std::vector<double> bessel(mesh);
    std::vector<double> integrand(mesh);
    std::vector<double> table(qs.size());

    for (int iq = 0; iq < static_cast<int>(qs.size()); ++iq)
    {
        atomic_radial::transform_channel(mesh,
                                         g.r.data(),
                                         g.rab.data(),
                                         g.chi.data(),
                                         0,
                                         &qs[iq],
                                         1,
                                         bessel.data(),
                                         integrand.data(),
                                         &table[iq]);
        const double q = qs[iq];
        const double ref = (1.0 / (1.0 + q * q))
                           * (1.0 - std::exp(-rmax) * (std::cos(q * rmax) + std::sin(q * rmax) / q));
        EXPECT_NEAR(ref, table[iq], 1e-5) << "q=" << q;
    }
}

TEST(AtomicRadialTransform, EvenMeshMatchesFineTrapezoidReference)
{
    // l = 1: independent reference quadrature (fine composite trapezoid on a
    // 10x finer grid, a different rule from both native primitives).
    const int mesh = 1000;
    const double h = 0.1;
    UniformGrid g = make_grid(mesh, h);

    const double rmax_ref = 60.0;
    const int nref = static_cast<int>(rmax_ref / 0.01) + 1;
    std::vector<double> rref(nref), href(nref, 0.01);
    for (int i = 0; i < nref; ++i)
    {
        rref[i] = i * 0.01;
    }

    const std::vector<double> qs = {0.5, 1.0};
    std::vector<double> bessel(mesh);
    std::vector<double> integrand(mesh);
    std::vector<double> table(qs.size());

    for (int iq = 0; iq < static_cast<int>(qs.size()); ++iq)
    {
        atomic_radial::transform_channel(mesh,
                                         g.r.data(),
                                         g.rab.data(),
                                         g.chi.data(),
                                         1,
                                         &qs[iq],
                                         1,
                                         bessel.data(),
                                         integrand.data(),
                                         &table[iq]);

        // reference: int e^{-r} j_1(q r) r dr by fine trapezoid
        std::vector<double> j1ref(nref);
        ModuleBase::Sphbes::Spherical_Bessel(nref, rref.data(), qs[iq], 1, j1ref.data());
        std::vector<double> fref(nref);
        for (int i = 0; i < nref; ++i)
        {
            fref[i] = std::exp(-rref[i]) * j1ref[i] * rref[i];
        }
        double ref = 0.0;
        for (int i = 0; i + 1 < nref; ++i)
        {
            ref += 0.5 * (fref[i] + fref[i + 1]) * href[i];
        }
        EXPECT_NEAR(ref, table[iq], 2e-5) << "q=" << qs[iq];
    }
}

TEST(AtomicRadialTransform, EvenAndOddMeshAgreeWithinDiscretization)
{
    UniformGrid even = make_grid(1000, 0.1);
    UniformGrid odd = make_grid(999, 0.1); // same spacing, odd point count
    const double q = 0.7;
    std::vector<double> bessel(1000), integrand(1000), table_even(1), table_odd(1);

    atomic_radial::transform_channel(1000, even.r.data(), even.rab.data(), even.chi.data(), 2, &q, 1,
                                     bessel.data(), integrand.data(), table_even.data());
    std::vector<double> bessel_odd(999), integrand_odd(999);
    atomic_radial::transform_channel(999, odd.r.data(), odd.rab.data(), odd.chi.data(), 2, &q, 1,
                                     bessel_odd.data(), integrand_odd.data(), table_odd.data());
    EXPECT_NEAR(table_odd[0], table_even[0], 1e-4);
}

TEST(AtomicRadialTransform, InputsAreNotMutated)
{
    const int mesh = 11;
    UniformGrid g = make_grid(mesh, 0.1);
    const std::vector<double> r0 = g.r;
    const std::vector<double> rab0 = g.rab;
    const std::vector<double> chi0 = g.chi;

    const int nq = 3;
    std::vector<double> q = {0.01, 0.02, 0.03};
    const std::vector<double> q0 = q;
    std::vector<double> bessel(mesh);
    std::vector<double> integrand(mesh);
    std::vector<double> table(nq);

    atomic_radial::transform_channel(mesh, g.r.data(), g.rab.data(), g.chi.data(), 0, q.data(), nq,
                                     bessel.data(), integrand.data(), table.data());

    EXPECT_EQ(r0, g.r);
    EXPECT_EQ(rab0, g.rab);
    EXPECT_EQ(chi0, g.chi);
    EXPECT_EQ(q0, q);
    for (int iq = 0; iq < nq; ++iq)
    {
        EXPECT_TRUE(std::isfinite(table[iq]));
    }
}
