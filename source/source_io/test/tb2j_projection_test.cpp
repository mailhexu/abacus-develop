#include "../module_tb2j/projection.h"
#include "source_base/constants.h"
#include "source_base/global_variable.h"
#include "source_base/parallel_global.h"
#include "source_basis/module_pw/pw_basis_k.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

using namespace ModuleIO::Tb2j;
using Complex = std::complex<double>;

TEST(Tb2jProjection, PrimalOrientationPaddingAndIntersiteGram)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const Complex phi[] = {1., {0.,1.}, 2., nan, nan, {0.,1.}, 2., -1., nan, nan};
    const Complex psi[] = {1., {0.,2.}, -1., nan, 2., {0.,-1.}, 1., nan};
    Complex c[4], gram[4];
    primal_overlaps(2, 2, 3, psi, 4, phi, 5, c);
    full_gram(2, 3, phi, 5, gram);
    const Complex expected_c[] = {1., {1.,3.}, 3., {-1.,-4.}};
    const Complex expected_s[] = {6., {-2.,-1.}, {-2.,1.}, 6.};
    for (int i = 0; i < 4; ++i)
    {
        EXPECT_NEAR(std::abs(c[i] - expected_c[i]), 0., 1e-14);
        EXPECT_NEAR(std::abs(gram[i] - expected_s[i]), 0., 1e-14);
    }
    ProjectionWorkspace work;
    EXPECT_NEAR(metric_condition(2, gram, work), (6.+std::sqrt(5.))/(6.-std::sqrt(5.)), 1e-13);
    primal_overlaps(0, 2, 3, nullptr, 4, phi, 5, nullptr);
    std::fill(c, c+4, Complex(9.));
    primal_overlaps(2, 2, 0, nullptr, 1, nullptr, 1, c);
    full_gram(2, 0, nullptr, 1, gram);
    for (int i=0; i<4; ++i) { EXPECT_EQ(c[i], Complex(0.)); EXPECT_EQ(gram[i], Complex(0.)); }
}

TEST(Tb2jProjection, RefusesUnusableFullMetricWithoutDroppingChannels)
{
    ProjectionWorkspace work;
    Complex singular[] = {1., 1., 1., 1.};
    Complex ill[] = {1., 0., 0., 1e-13};
    Complex indefinite[] = {1., 2., 2., 1.};
    Complex nonhermitian[] = {1., {0.,1.}, {0.,1.}, 1.};
    Complex nonfinite[] = {1., 0., 0., std::numeric_limits<double>::infinity()};
    for (const auto* gram : {singular, ill, indefinite, nonhermitian, nonfinite})
        EXPECT_THROW(metric_condition(2, gram, work), std::runtime_error);
    const Complex acceptable[] = {1., 0., 0., 2e-12};
    EXPECT_NEAR(metric_condition(2, acceptable, work), 5e11, 1.);
}

TEST(Tb2jProjection, WeightedOnsiteVertexUsesAllKAndNativeEnergyUnits)
{
    const Complex a[] = {2., {1.,-2.}, {1.,2.}, 4.};
    const Complex b[] = {6., {-3.,1.}, {-3.,-1.}, 8.};
    Complex result[4] = {};
    accumulate_site_vertex(.25, 4, a, result);
    accumulate_site_vertex(.75, 4, b, result);
    for (int i=0; i<4; ++i)
        EXPECT_NEAR(std::abs(result[i] / ModuleBase::Ry_to_eV - (.25*a[i]+.75*b[i])), 0., 1e-14);
    EXPECT_GT(std::abs(result[0]/ModuleBase::Ry_to_eV-a[0]), 1.);
    EXPECT_THROW(accumulate_site_vertex(0., 4, a, result), std::runtime_error);
}

TEST(Tb2jProjection, NativeFftMatchesIndependentComplexQuadratureAtTwoK)
{
    ModulePW::PW_Basis_K pw("cpu", "double");
#ifdef __MPI
    pw.initmpi(1, 0, MPI_COMM_SELF);
#endif
    const ModuleBase::Matrix3 lattice(1.,0.,0.,0.,1.,0.,0.,0.,1.);
    ModuleBase::Vector3<double> k[] = {{.17,.11,.07}, {-.13,.09,.21}};
    pw.initgrids(4., lattice, 8, 8, 8);
    pw.initparameters(false, 10., 2, k, 1, false);
    pw.setuptransform();
    pw.collect_local_pw();
    const std::vector<SiteBlock> sites = {{0,2,0}, {2,2,4}};
    const ModuleBase::Vector3<double> tau[] = {{0.,0.,0.}, {.21,.16,.12}};
    ProjectionWorkspace work;
    for (int ik=0; ik<2; ++ik)
    {
        const int npw = pw.npwk[ik];
        ASSERT_GT(npw, 4);
        std::vector<Complex> phi(4*npw), gram(16), vertex(8), exact(8), constant(8);
        std::vector<double> dv(pw.nrxx), uniform(pw.nrxx, 2.5);
        for (int p=0; p<4; ++p)
            for (int g=0; g<npw; ++g)
            {
                const auto q = pw.getgpluskcar(ik,g);
                const double phase = -ModuleBase::TWO_PI * (q*tau[p/2]);
                phi[p*npw+g] = std::polar(1.+.17*p+.03*q.norm2(),phase)
                               * Complex(1., .11*(p+1)*q.x) / 10.;
            }
        // Independent explicit inverse DFT, including complex intersite phases.
        for (int ix=0; ix<pw.nx; ++ix)
            for (int iy=0; iy<pw.ny; ++iy)
                for (int iz=0; iz<pw.nz; ++iz)
                {
                    const int ir=(ix*pw.ny+iy)*pw.nz+iz;
                    dv[ir]=1.+.2*std::cos(ModuleBase::TWO_PI*ix/pw.nx)
                               +.3*std::sin(ModuleBase::TWO_PI*iz/pw.nz);
                    Complex real[4] = {};
                    for (int p=0; p<4; ++p)
                        for (int g=0; g<npw; ++g)
                        {
                            const auto gv=pw.getgpluskcar(ik,g)-k[ik];
                            const double phase=ModuleBase::TWO_PI*(gv.x*ix/pw.nx+gv.y*iy/pw.ny+gv.z*iz/pw.nz);
                            real[p]+=phi[p*npw+g]*std::polar(1.,phase);
                        }
                    for (const auto& site:sites)
                        for (int p=0; p<site.count; ++p)
                            for (int q=0; q<site.count; ++q)
                                exact[site.offset+p*site.count+q] += std::conj(real[site.first+p])*dv[ir]*real[site.first+q]/double(pw.nxyz);
                }
        project_site_xc(ik,pw,sites,phi.data(),npw,dv.data(),work,vertex.data());
        project_site_xc(ik,pw,sites,phi.data(),npw,uniform.data(),work,constant.data());
        full_gram(4,npw,phi.data(),npw,gram.data());
        for (const auto& site:sites)
            for (int p=0; p<site.count; ++p)
                for (int q=0; q<site.count; ++q)
                {
                    const auto i=site.offset+p*site.count+q;
                    EXPECT_NEAR(std::abs(vertex[i]-exact[i]),0.,2e-12);
                    EXPECT_NEAR(std::abs(constant[i]-2.5*gram[(site.first+p)*4+site.first+q]),0.,2e-12);
                    EXPECT_NEAR(std::abs(vertex[i]-std::conj(vertex[site.offset+q*site.count+p])),0.,1e-14);
                }
    }
}

int main(int argc, char** argv)
{
#ifdef __MPI
    MPI_Init(&argc,&argv);
#endif
    ::testing::InitGoogleTest(&argc,argv);
    const int result=RUN_ALL_TESTS();
#ifdef __MPI
    MPI_Finalize();
#endif
    return result;
}
