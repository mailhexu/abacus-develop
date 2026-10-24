#include "gtest/gtest.h"

/************************************************
 *  unit test of the TB2J immutable atomic basis
 ***********************************************/

/**
 * - Tested Functions:
 *   - atomic_channel_error
 *     - rejects unsupported or malformed channel records
 *   - AtomicBasis (mapper)
 *     - deterministic native order species -> atom -> source channel ->
 *       m = 0..2l; nproj and per-site l/channel/start/count; deep-copy
 *       ownership immune to later mutation of the source snapshots
 *   - AtomicBasis q-space tables
 *     - even/full and odd mesh quadrature against analytic and independent
 *       numerical references; source mesh and channels remain unchanged
 *   - AtomicBasis::fill_pw
 *     - native conventions: getgpluskcar, Ylm_Real on (lmax+1)^2 real
 *       harmonics with lm = l*l + m, get_sk, pow(NEG_IMAG_UNIT, l) phase,
 *       radial interpolation at |G+k| * tpiba, 4pi/sqrt(omega) from the
 *       current cell, all channels filled including unoccupied/negative
 *       occupation, rows zeroed up to caller stride, reusable workspace
 *     - output is independent of subsequent native initializer data
 */

#include "source_io/module_tb2j/upf_atomic_projectors.h"

#include "source_base/constants.h"
#include "source_base/math_polyint.h"
#include "source_base/math_sphbes.h"
#include "source_base/math_ylmreal.h"
#include "source_cell/atom_spec.h"
#include "source_pw/module_pwdft/stru_fac.h"

#include <cmath>
#include <complex>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Empty definitions for Structure_Factor members used by the code under
// test; stru_fac.cpp is not linked into this target (mirrors the
// source_psi/test convention of stubbing the not-linked class members).
Structure_Factor::Structure_Factor() {}
Structure_Factor::~Structure_Factor() {}
std::complex<double>* Structure_Factor::get_sk(const int ik,
                                               const int it,
                                               const int ia,
                                               const ModulePW::PW_Basis_K* wfc_basis) const
{
    const int npw = wfc_basis->npwk[ik];
    std::complex<double>* sk = new std::complex<double>[npw];
    for (int ipw = 0; ipw < npw; ++ipw)
    {
        sk[ipw] = std::complex<double>(1.0, 0.0);
    }
    return sk;
}

// Empty definitions for UnitCell data-member classes whose implementations
// are not linked into this target (mirrors source_psi/test/psi_init_unit_test.cpp).
// Atom_pseudo/pseudo are linked as real sources here; Sep_Cell/SepPot come
// with mock_unitcell.cpp; only the Magnetism member needs a stub.
Magnetism::Magnetism() {}
Magnetism::~Magnetism() {}

namespace
{
using ModuleIO::Tb2j::AtomicBasis;
using ModuleIO::Tb2j::AtomicChannel;
using ModuleIO::Tb2j::SpeciesBasis;

SpeciesBasis make_test_species()
{
    SpeciesBasis species;
    species.label = "Fe";
    species.pseudo_file = "synthetic";
    species.natoms = 2;
    species.mesh = 10;
    species.r.resize(species.mesh);
    species.rab.assign(species.mesh, 0.1);
    for (int ir = 0; ir < species.mesh; ++ir)
    {
        species.r[ir] = 0.1 * ir;
    }

    const int angular_momenta[] = {0, 0, 1};
    const double occupations[] = {2.0, 0.0, -1.0};
    const char* labels[] = {"3S", "4S", "2P"};
    species.channels.resize(3);
    for (int ic = 0; ic < 3; ++ic)
    {
        AtomicChannel& channel = species.channels[ic];
        channel.l = angular_momenta[ic];
        channel.occupation = occupations[ic];
        channel.label = labels[ic];
        channel.chi.resize(species.mesh);
        for (int ir = 0; ir < species.mesh; ++ir)
        {
            const double r = species.r[ir];
            if (ic == 0)
            {
                channel.chi[ir] = std::exp(-r);
            }
            else if (ic == 1)
            {
                channel.chi[ir] = 0.5 * std::exp(-0.5 * r);
            }
            else
            {
                channel.chi[ir] = r * std::exp(-r);
            }
        }
    }
    return species;
}

SpeciesBasis make_exponential_species(const int mesh, const double spacing, const int l = 0)
{
    SpeciesBasis species;
    species.label = "H";
    species.natoms = 1;
    species.mesh = mesh;
    species.r.resize(mesh);
    species.rab.assign(mesh, spacing);
    AtomicChannel channel;
    channel.l = l;
    channel.occupation = 1.0;
    channel.label = "1S";
    channel.chi.resize(mesh);
    for (int ir = 0; ir < mesh; ++ir)
    {
        species.r[ir] = spacing * ir;
        channel.chi[ir] = std::exp(-species.r[ir]);
    }
    species.channels.push_back(std::move(channel));
    return species;
}

double fine_radial_transform(const double q, const int l, const double rmax)
{
    constexpr double step = 0.01;
    const int nref = static_cast<int>(rmax / step) + 1;
    std::vector<double> r(nref), bessel(nref), integrand(nref);
    for (int ir = 0; ir < nref; ++ir)
    {
        r[ir] = ir * step;
    }
    ModuleBase::Sphbes::Spherical_Bessel(nref, r.data(), q, l, bessel.data());
    for (int ir = 0; ir < nref; ++ir)
    {
        integrand[ir] = std::exp(-r[ir]) * bessel[ir] * r[ir];
    }
    double result = 0.0;
    for (int ir = 0; ir + 1 < nref; ++ir)
    {
        result += 0.5 * (integrand[ir] + integrand[ir + 1]) * step;
    }
    return result;
}
} // namespace

TEST(Tb2jAtomicBasis, AtomicChannelGuardRejectsUnsupportedOrMalformedChannels)
{
    Atom_pseudo pp;
    pp.pp_type = "NC";
    pp.nchi = 1;
    pp.mesh = 3;
    pp.r = {0.0, 0.5, 1.0};
    pp.rab = {0.5, 0.5, 0.5};
    pp.lchi = {0};
    pp.chi.create(1, 3);
    pp.chi(0, 0) = 0.5;
    pp.chi(0, 1) = 1.0;
    pp.chi(0, 2) = 0.5;
    EXPECT_EQ(nullptr, ModuleIO::Tb2j::atomic_channel_error(pp, false));

    for (const auto& type : {"PAW", "US"})
    {
        pp.pp_type = type;
        EXPECT_NE(nullptr, ModuleIO::Tb2j::atomic_channel_error(pp, false));
    }
    pp.pp_type = "NC";
    pp.tvanp = true;
    EXPECT_NE(nullptr, ModuleIO::Tb2j::atomic_channel_error(pp, false));
    pp.tvanp = false;
    pp.has_so = true;
    EXPECT_NE(nullptr, ModuleIO::Tb2j::atomic_channel_error(pp, false));
    pp.has_so = false;
    EXPECT_NE(nullptr, ModuleIO::Tb2j::atomic_channel_error(pp, true));
    pp.nchi = 0;
    EXPECT_NE(nullptr, ModuleIO::Tb2j::atomic_channel_error(pp, false));
    pp.nchi = 1;
    pp.lchi[0] = -1;
    EXPECT_NE(nullptr, ModuleIO::Tb2j::atomic_channel_error(pp, false));
    pp.lchi[0] = 0;
    pp.chi(0, 0) = 0.0;
    pp.chi(0, 1) = 0.0;
    pp.chi(0, 2) = 0.0;
    EXPECT_NE(nullptr, ModuleIO::Tb2j::atomic_channel_error(pp, false));
    pp.chi(0, 1) = std::numeric_limits<double>::infinity();
    EXPECT_NE(nullptr, ModuleIO::Tb2j::atomic_channel_error(pp, false));
    pp.chi(0, 1) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_NE(nullptr, ModuleIO::Tb2j::atomic_channel_error(pp, false));
}


TEST(Tb2jAtomicBasis, MapperIsDeterministicNativeOrderAndOwnsCopies)
{
    SpeciesBasis snap = make_test_species();
    std::vector<SpeciesBasis> species;
    species.push_back(snap);

    const int nqx = 1000;
    const double dq = 0.01;
    ModuleIO::Tb2j::AtomicBasis basis(species, nqx, dq);

    ASSERT_EQ(1, basis.ntype());
    // (1 + 1 + 3) harmonics per atom, two atoms
    EXPECT_EQ(10, basis.nproj());

    const auto& sites = basis.sites();
    ASSERT_EQ(6u, sites.size());
    EXPECT_EQ(0, sites[0].it);
    EXPECT_EQ(0, sites[0].ia);
    EXPECT_EQ(0, sites[0].channel);
    EXPECT_EQ(0, sites[0].l);
    EXPECT_EQ(0, sites[0].start);
    EXPECT_EQ(1, sites[0].count);
    EXPECT_EQ(0, sites[1].it);
    EXPECT_EQ(0, sites[1].ia);
    EXPECT_EQ(1, sites[1].channel); // repeated-l second channel, distinct row
    EXPECT_EQ(0, sites[1].l);
    EXPECT_EQ(1, sites[1].start);
    EXPECT_EQ(1, sites[1].count);
    EXPECT_EQ(0, sites[2].it);
    EXPECT_EQ(0, sites[2].ia);
    EXPECT_EQ(2, sites[2].channel);
    EXPECT_EQ(1, sites[2].l);
    EXPECT_EQ(2, sites[2].start);
    EXPECT_EQ(3, sites[2].count);
    EXPECT_EQ(0, sites[3].it);
    EXPECT_EQ(1, sites[3].ia);
    EXPECT_EQ(0, sites[3].channel);
    EXPECT_EQ(0, sites[3].l);
    EXPECT_EQ(5, sites[3].start);
    EXPECT_EQ(1, sites[3].count);
    EXPECT_EQ(0, sites[4].it);
    EXPECT_EQ(1, sites[4].ia);
    EXPECT_EQ(1, sites[4].channel);
    EXPECT_EQ(0, sites[4].l);
    EXPECT_EQ(6, sites[4].start);
    EXPECT_EQ(1, sites[4].count);
    EXPECT_EQ(0, sites[5].it);
    EXPECT_EQ(1, sites[5].ia);
    EXPECT_EQ(2, sites[5].channel);
    EXPECT_EQ(1, sites[5].l);
    EXPECT_EQ(7, sites[5].start);
    EXPECT_EQ(3, sites[5].count);

    // deep-copy ownership: later mutation of the source snapshot or of the
    // constructor input must not change the basis (initializer immunity)
    const double chi3 = basis.species(0).channels[0].chi[3];
    snap.channels[0].chi[3] *= 7.0;
    species[0].channels[0].chi[3] *= 7.0;
    EXPECT_EQ(chi3, basis.species(0).channels[0].chi[3]);
}

TEST(Tb2jAtomicBasis, TablesAreCellIndependentQSpaceTransforms)
{
    // Odd-mesh s-wave table; chi = e^{-r} gives the analytic transform
    // int e^{-r} j_0(qr) r dr = 1/(1+q^2) for a negligible tail.
    const int nqx = 1000;
    const double dq = 0.01;
    std::vector<SpeciesBasis> species{make_exponential_species(801, 0.1)};
    AtomicBasis basis(species, nqx, dq);

    const auto& table = basis.species(0).channels[0].table;
    ASSERT_EQ(static_cast<std::size_t>(nqx), table.size());
    // The tolerances cover the Simpson discretization error of this 801-point
    // mesh for the exponential s-wave channel.
    // q = 0: int e^{-r} r dr = 1
    EXPECT_NEAR(1.0, table[0], 1e-5);
    // q = 0.01: 1/(1+q^2)
    EXPECT_NEAR(1.0 / (1.0 + 0.0001), table[1], 1e-5);
    // q = 0.9: 1/(1+q^2)
    EXPECT_NEAR(1.0 / (1.0 + 0.81), table[90], 1e-5);
}

TEST(Tb2jAtomicBasis, EvenMeshRadialTransformMatchesAnalyticReference)
{
    std::vector<SpeciesBasis> species{make_exponential_species(1000, 0.1)};
    AtomicBasis basis(std::move(species), 101, 0.01);
    const auto& table = basis.species(0).channels[0].table;
    const double rmax = 99.9;
    const int indexes[] = {1, 10, 100};
    for (const int iq : indexes)
    {
        const double q = 0.01 * iq;
        const double reference = (1.0 / (1.0 + q * q))
                                 * (1.0 - std::exp(-rmax)
                                    * (std::cos(q * rmax) + std::sin(q * rmax) / q));
        EXPECT_NEAR(reference, table[iq], 1e-5) << "q=" << q;
    }
}

TEST(Tb2jAtomicBasis, EvenMeshPWaveMatchesFineIndependentQuadrature)
{
    std::vector<SpeciesBasis> species{make_exponential_species(1000, 0.1, 1)};
    AtomicBasis basis(std::move(species), 101, 0.01);
    const auto& table = basis.species(0).channels[0].table;
    for (const int iq : {50, 100})
    {
        const double q = 0.01 * iq;
        EXPECT_NEAR(fine_radial_transform(q, 1, 60.0), table[iq], 2e-5) << "q=" << q;
    }
}

TEST(Tb2jAtomicBasis, EvenAndOddMeshesAgreeWithinDiscretization)
{
    std::vector<SpeciesBasis> even_species{make_exponential_species(1000, 0.1, 2)};
    std::vector<SpeciesBasis> odd_species{make_exponential_species(999, 0.1, 2)};
    AtomicBasis even(std::move(even_species), 8, 0.1);
    AtomicBasis odd(std::move(odd_species), 8, 0.1);
    EXPECT_NEAR(odd.species(0).channels[0].table[7],
                even.species(0).channels[0].table[7],
                1e-4);
}

TEST(Tb2jAtomicBasis, RadialConstructionPreservesSourceArrays)
{
    std::vector<SpeciesBasis> species{make_exponential_species(11, 0.1)};
    const auto r = species[0].r;
    const auto rab = species[0].rab;
    const auto chi = species[0].channels[0].chi;
    AtomicBasis basis(species, 4, 0.1);
    EXPECT_EQ(r, species[0].r);
    EXPECT_EQ(rab, species[0].rab);
    EXPECT_EQ(chi, species[0].channels[0].chi);
    EXPECT_EQ(chi, basis.species(0).channels[0].chi);
}

/**
 * Harness for fill_pw: a synthetic full-mesh, three-channel basis for two
 * atoms, one k-point with a single G, mocked structure factor sk == 1,
 * following the source_psi/test hand-built PW convention.
 */
class Tb2jFillPwTest : public testing::Test
{
  protected:
    void SetUp() override
    {
        snap_ = std::make_unique<SpeciesBasis>(make_test_species());
        std::vector<SpeciesBasis> species;
        species.push_back(*snap_);
        basis_ = std::make_unique<AtomicBasis>(std::move(species), 1000, 0.01);

        ucell_ = std::make_unique<UnitCell>();
        ucell_->a1 = {10.0, 0.0, 0.0};
        ucell_->a2 = {0.0, 10.0, 0.0};
        ucell_->a3 = {0.0, 0.0, 10.0};
        ucell_->lat0 = 1.0;
        ucell_->omega = 1000.0;
        ucell_->latvec.e11 = 10.0;
        ucell_->latvec.e12 = 0.0;
        ucell_->latvec.e13 = 0.0;
        ucell_->latvec.e21 = 0.0;
        ucell_->latvec.e22 = 10.0;
        ucell_->latvec.e23 = 0.0;
        ucell_->latvec.e31 = 0.0;
        ucell_->latvec.e32 = 0.0;
        ucell_->latvec.e33 = 10.0;
        ucell_->GT = ucell_->latvec.Inverse();
        ucell_->G = ucell_->GT.Transpose();
        ucell_->GGT = ucell_->G * ucell_->GT;
        ucell_->tpiba = 2.0 * M_PI / ucell_->lat0;
        ucell_->tpiba2 = ucell_->tpiba * ucell_->tpiba;
        ucell_->ntype = 1;
        ucell_->nat = 2;
        ucell_->lmax_ppwf = 1; // max l of the captured basis
        ucell_->set_atom_flag = true;
        ucell_->atoms = new Atom[1];
        ucell_->atoms[0].label = "Fe";
        ucell_->atoms[0].na = 2;
        ucell_->atoms[0].angle1.resize(2, 0.0);
        ucell_->atoms[0].angle2.resize(2, 0.0);
        ucell_->atoms[0].tau.resize(2, ModuleBase::Vector3<double>(0.0, 0.0, 0.0));
        ucell_->atoms[0].taud.resize(2, ModuleBase::Vector3<double>(0.25, 0.25, 0.25));
        ucell_->atoms[0].mbl.resize(2, ModuleBase::Vector3<int>(0, 0, 0));
        // pseudopotential data is irrelevant to fill_pw; a later normalization
        // mutation of it must not change the basis output (see Immunity test)
        ucell_->atoms[0].ncpp.chi.create(1, 4);
        ucell_->atoms[0].ncpp.nchi = 1;

        pw_ = std::make_unique<ModulePW::PW_Basis_K>();
        pw_->nks = 1;
        pw_->npwk_max = 1;
        pw_->npwk = new int[1];
        pw_->npwk[0] = 1;
        pw_->igl2isz_k = new int[1];
        pw_->igl2isz_k[0] = 0;
        pw_->igl2ig_k = new int[1];
        pw_->igl2ig_k[0] = 0;
        pw_->gk2 = new double[1];
        pw_->gk2[0] = 0.0;
        pw_->gcar = new ModuleBase::Vector3<double>[1];
        // all three components nonzero so every l = 1 real harmonic is nonzero
        pw_->gcar[0] = {0.01, 0.02, 0.03};
        pw_->kvec_c = new ModuleBase::Vector3<double>[1];
        pw_->kvec_c[0] = {0.0, 0.0, 0.0};
        pw_->latvec = ucell_->latvec;
        pw_->G = ucell_->G;
        pw_->GT = ucell_->GT;
        pw_->GGT = ucell_->GGT;
        pw_->lat0 = ucell_->lat0;
        pw_->tpiba = ucell_->tpiba;
        pw_->tpiba2 = ucell_->tpiba2;
    }

    void TearDown() override
    {
        // pw_/ucell_ are unique_ptrs; PW_Basis_K and its PW_Basis base own
        // every raw array set up here (all pointers are nullptr-initialized
        // and deleted by their destructors), so no manual delete here.
    }

  public:
    std::unique_ptr<ModuleIO::Tb2j::SpeciesBasis> snap_;
    std::unique_ptr<ModuleIO::Tb2j::AtomicBasis> basis_;
    std::unique_ptr<UnitCell> ucell_;
    std::unique_ptr<ModulePW::PW_Basis_K> pw_;
    Structure_Factor sf_;
    ModuleIO::Tb2j::AtomicBasis::FillWorkspace work_;
    // nproj = 10 rows, stride 4 columns
    std::vector<std::complex<double>> out_ = std::vector<std::complex<double>>(10 * 4, {5.0, 5.0});
};

TEST_F(Tb2jFillPwTest, FillsNativeConventionsAllChannelsDeterministicOrder)
{
    ASSERT_EQ(nullptr, basis_->fill_pw(0, *ucell_, sf_, pw_.get(), work_, out_.data(), 4));

    // native reference factors for the single G
    const int npw = pw_->npwk[0];
    const int total_lm = (ucell_->lmax_ppwf + 1) * (ucell_->lmax_ppwf + 1);
    std::vector<ModuleBase::Vector3<double>> gk(npw);
    for (int ig = 0; ig < npw; ++ig)
    {
        gk[ig] = pw_->getgpluskcar(0, ig);
    }
    ModuleBase::matrix ylm(total_lm, npw);
    ModuleBase::YlmReal::Ylm_Real(total_lm, npw, gk.data(), ylm);
    const double pref = ModuleBase::FOUR_PI / std::sqrt(ucell_->omega);
    const double q = gk[0].norm() * ucell_->tpiba;
    const auto& channels = basis_->species(0).channels;
    auto radial = [&](int ic) {
        return ModuleBase::PolyInt::Polynomial_Interpolation(channels[ic].table.data(), 1000, 0.01, q);
    };

    // row 0/1: the two repeated-l s channels are distinct (different tables)
    const std::complex<double> y00 = ylm(0, 0);
    EXPECT_NEAR(std::real(pref * y00 * radial(0)), std::real(out_[0]), 1e-12);
    EXPECT_NEAR(std::imag(pref * y00 * radial(0)), std::imag(out_[0]), 1e-12);
    EXPECT_NEAR(std::real(pref * y00 * radial(1)), std::real(out_[4]), 1e-12);
    EXPECT_NE(out_[0], out_[4]);

    // rows 2..4: l = 1 channel, lm = l*l + m, phase pow(NEG_IMAG_UNIT, 1) = -i
    for (int m = 0; m < 3; ++m)
    {
        const std::complex<double> ref = pow(ModuleBase::NEG_IMAG_UNIT, 1) * pref * ylm(1 + m, 0) * radial(2);
        EXPECT_NEAR(std::real(ref), std::real(out_[(2 + m) * 4]), 1e-12) << "m=" << m;
        EXPECT_NEAR(std::imag(ref), std::imag(out_[(2 + m) * 4]), 1e-12) << "m=" << m;
        // the -i phase keeps the l = 1 rows essentially purely imaginary
        EXPECT_NEAR(0.0, std::real(out_[(2 + m) * 4]), 1e-12) << "m=" << m;
    }

    // rows 5..9: second atom of the same species, identical coefficients
    for (int row = 0; row < 5; ++row)
    {
        EXPECT_EQ(out_[row * 4], out_[(5 + row) * 4]);
    }

    // all channels filled, including oc = 0 and oc = -1 (rows 1..4 nonzero)
    for (int row = 0; row < 10; ++row)
    {
        EXPECT_NE(std::complex<double>(0.0, 0.0), out_[row * 4]) << "row " << row;
    }

    // rows zeroed up to the caller stride beyond npw
    for (int row = 0; row < 10; ++row)
    {
        for (int col = 1; col < 4; ++col)
        {
            EXPECT_EQ(std::complex<double>(0.0, 0.0), out_[row * 4 + col]);
        }
    }
}

TEST_F(Tb2jFillPwTest, AppliesCellOmegaAtFillTime)
{
    ASSERT_EQ(nullptr, basis_->fill_pw(0, *ucell_, sf_, pw_.get(), work_, out_.data(), 4));
    const std::vector<std::complex<double>> out_omega1000 = out_;

    ucell_->omega = 8000.0; // relaxed cell: tables stay, prefactor follows
    ASSERT_EQ(nullptr, basis_->fill_pw(0, *ucell_, sf_, pw_.get(), work_, out_.data(), 4));

    const double ratio = std::sqrt(1000.0 / 8000.0);
    for (int row = 0; row < 10; ++row)
    {
        EXPECT_NEAR(std::real(out_omega1000[row * 4]) * ratio, std::real(out_[row * 4]), 1e-12);
        EXPECT_NEAR(std::imag(out_omega1000[row * 4]) * ratio, std::imag(out_[row * 4]), 1e-12);
    }
}

TEST_F(Tb2jFillPwTest, ReusesWorkspaceAndTracksNewG)
{
    ASSERT_EQ(nullptr, basis_->fill_pw(0, *ucell_, sf_, pw_.get(), work_, out_.data(), 4));
    const std::complex<double> first = out_[0];

    pw_->gcar[0] = {0.03, 0.0, 0.0}; // different G, same reused workspace
    ASSERT_EQ(nullptr, basis_->fill_pw(0, *ucell_, sf_, pw_.get(), work_, out_.data(), 4));

    const double q = pw_->getgpluskcar(0, 0).norm() * ucell_->tpiba;
    ModuleBase::matrix ylm(4, 1);
    std::vector<ModuleBase::Vector3<double>> gk(1, pw_->getgpluskcar(0, 0));
    ModuleBase::YlmReal::Ylm_Real(4, 1, gk.data(), ylm);
    const double pref = ModuleBase::FOUR_PI / std::sqrt(ucell_->omega);
    const double radial = ModuleBase::PolyInt::Polynomial_Interpolation(basis_->species(0).channels[0].table.data(), 1000, 0.01, q);
    const std::complex<double> ref = pref * ylm(0, 0) * radial;

    EXPECT_NEAR(std::real(ref), std::real(out_[0]), 1e-12);
    EXPECT_NEAR(std::imag(ref), std::imag(out_[0]), 1e-12);
    EXPECT_NE(first, out_[0]);
}

TEST_F(Tb2jFillPwTest, SnapshotImmuneToInitializerNormalizationMutation)
{
    ASSERT_EQ(nullptr, basis_->fill_pw(0, *ucell_, sf_, pw_.get(), work_, out_.data(), 4));
    const std::vector<std::complex<double>> before = out_;

    // the native initializer renormalizes atom->ncpp.chi in place during
    // tabulate; the captured basis must be unaffected by any later mutation
    for (int ir = 0; ir < 4; ++ir)
    {
        ucell_->atoms[0].ncpp.chi(0, ir) *= 3.0;
    }
    ucell_->atoms[0].ncpp.chi.create(2, 4); // even a reallocation of ncpp data
    for (int ir = 0; ir < 4; ++ir)
    {
        ucell_->atoms[0].ncpp.chi(0, ir) = -1.0;
        ucell_->atoms[0].ncpp.chi(1, ir) = -1.0;
    }

    ASSERT_EQ(nullptr, basis_->fill_pw(0, *ucell_, sf_, pw_.get(), work_, out_.data(), 4));
    for (std::size_t i = 0; i < before.size(); ++i)
    {
        EXPECT_EQ(before[i], out_[i]);
    }
}

TEST_F(Tb2jFillPwTest, KeepsLastInterpolationCell)
{
    std::vector<ModuleIO::Tb2j::SpeciesBasis> species;
    species.push_back(*snap_);
    basis_ = std::make_unique<ModuleIO::Tb2j::AtomicBasis>(std::move(species), 4, 0.1);
    const double q = 0.09; // floor(q/dq) == nqx-4: all four samples exist
    pw_->gcar[0] = {q / ucell_->tpiba, 0.0, 0.0};
    ASSERT_EQ(nullptr, basis_->fill_pw(0, *ucell_, sf_, pw_.get(), work_, out_.data(), 4));
    const double radial = ModuleBase::PolyInt::Polynomial_Interpolation(
        basis_->species(0).channels[0].table.data(), 4, 0.1, q);
    const double expected = ModuleBase::FOUR_PI / std::sqrt(ucell_->omega)
                            * work_.ylm(0, 0) * radial;
    EXPECT_NEAR(expected, out_[0].real(), 1e-12);
}

TEST_F(Tb2jFillPwTest, RejectsMissingInterpolationSamples)
{
    std::vector<ModuleIO::Tb2j::SpeciesBasis> species;
    species.push_back(*snap_);
    basis_ = std::make_unique<ModuleIO::Tb2j::AtomicBasis>(std::move(species), 4, 0.1);
    // The chosen nonzero G gives q about 0.235, so the four-point stencil is unavailable.
    const auto before = out_;
    const char* error = basis_->fill_pw(0, *ucell_, sf_, pw_.get(), work_, out_.data(), 4);
    ASSERT_NE(nullptr, error);
    EXPECT_NE(std::string::npos, std::string(error).find("radial table"));
    EXPECT_EQ(before, out_);
}

