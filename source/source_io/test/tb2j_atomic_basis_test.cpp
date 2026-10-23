#include "gtest/gtest.h"

/************************************************
 *  unit test of the TB2J immutable atomic basis
 ***********************************************/

/**
 * - Tested Functions:
 *   - ModuleIO::Tb2j::sha256_hex / sha256_file_hex
 *     - known SHA256 vectors including the 55/56/64-byte padding boundaries
 *     - raw-byte hashing of a pinned fixture file
 *   - AtomicBasis::capture_species
 *     - full even UPF mesh retained, ALL channels captured in source order
 *       (repeated-l, zero and negative occupation), channels normalized on
 *       the full mesh with the source shape preserved, raw-file SHA256
 *     - explicit rejection of USPP, SOC, fully-relativistic, zero-norm and
 *       unreadable inputs (never silently dropping a channel)
 *   - AtomicBasis (mapper)
 *     - deterministic native order species -> atom -> source channel ->
 *       m = 0..2l; nproj and per-site l/channel/start/count; deep-copy
 *       ownership immune to later mutation of the captured snapshot
 *   - AtomicBasis::fill_pw
 *     - native conventions: getgpluskcar, Ylm_Real on (lmax+1)^2 real
 *       harmonics with lm = l*l + m, get_sk, pow(NEG_IMAG_UNIT, l) phase,
 *       radial interpolation at |G+k| * tpiba, 4pi/sqrt(omega) from the
 *       current cell, all channels filled including unoccupied/negative
 *       occupation, rows zeroed up to the caller stride, reusable workspace
 *   - capture_species snapshot independence from later pseudopotential
 *     mutations (the initializer normalizes ncpp.chi in place)
 */

#include "source_io/module_tb2j/atomic_projectors.h"
#include "source_io/module_tb2j/sha256.h"

#include "source_base/constants.h"
#include "source_base/math_ylmreal.h"
#include "source_base/math_polyint.h"
#include "source_cell/atom_spec.h"
#include "source_pw/module_pwdft/stru_fac.h"

#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <unistd.h>
#include <string>
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
const char* kEvenMeshFixture = "./support/Fe_tb2j_even_mesh.upf";
const char* kEvenMeshHash = "db81aafc222a8be1f8e87ed8ce543be3bc309e84e9976f2bcd94aa19b1c28310";

/// Independent fine-grid trapezoidal norm for the fixture's linear channels.
/// Subdivision avoids the coarse ten-point trapezoid's quadratic error without
/// reusing either native Simpson implementation.
double trapezoid_norm(const std::vector<double>& chi, const std::vector<double>& r)
{
    constexpr int subdivisions = 100;
    double norm = 0.0;
    for (std::size_t i = 0; i + 1 < chi.size(); ++i)
    {
        const double h = (r[i + 1] - r[i]) / subdivisions;
        for (int j = 0; j < subdivisions; ++j)
        {
            const double a = chi[i] + (chi[i + 1] - chi[i]) * j / subdivisions;
            const double b = chi[i] + (chi[i + 1] - chi[i]) * (j + 1) / subdivisions;
            norm += 0.5 * (a * a + b * b) * h;
        }
    }
    return norm;
}
} // namespace

TEST(Tb2jSha256, KnownVectorsAndPaddingBoundaries)
{
    EXPECT_EQ(std::string("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
              ModuleIO::Tb2j::sha256_hex("", 0));
    EXPECT_EQ(std::string("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
              ModuleIO::Tb2j::sha256_hex("abc", 3));
    // 55 bytes: padding fits the current block
    const std::string a55(55, 'a');
    EXPECT_EQ(std::string("9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318"),
              ModuleIO::Tb2j::sha256_hex(a55.data(), a55.size()));
    // 56 bytes: the 0x80 pad byte spills into the next block
    const std::string msg56 = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    ASSERT_EQ(56u, msg56.size());
    EXPECT_EQ(std::string("248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"),
              ModuleIO::Tb2j::sha256_hex(msg56.data(), msg56.size()));
    // 64 bytes: exact block boundary
    const std::string a64(64, 'a');
    EXPECT_EQ(std::string("ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb"),
              ModuleIO::Tb2j::sha256_hex(a64.data(), a64.size()));
    // multi-block streaming
    const std::string a1000(1000, 'a');
    EXPECT_EQ(std::string("41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3"),
              ModuleIO::Tb2j::sha256_hex(a1000.data(), a1000.size()));
}

TEST(Tb2jSha256, FileHashMatchesPinnedFixture)
{
    EXPECT_EQ(std::string(kEvenMeshHash), ModuleIO::Tb2j::sha256_file_hex(kEvenMeshFixture));
    EXPECT_EQ("", ModuleIO::Tb2j::sha256_file_hex("./support/does_not_exist.upf"));
}

TEST(Tb2jAtomicBasis, CaptureSpeciesKeepsFullMeshAllChannelsAndProvenance)
{
    std::string error;
    const auto snap = ModuleIO::Tb2j::AtomicBasis::capture_species(kEvenMeshFixture, "Fe", "auto", error);
    ASSERT_TRUE(error.empty()) << error;
    ASSERT_NE(snap, nullptr);

    EXPECT_EQ("Fe", snap->label);
    EXPECT_EQ(10, snap->mesh); // full even mesh preserved
    ASSERT_EQ(static_cast<std::size_t>(10), snap->r.size());
    ASSERT_EQ(static_cast<std::size_t>(10), snap->rab.size());
    EXPECT_NEAR(1.0, snap->r[9], 1e-12);
    EXPECT_NEAR(0.1, snap->rab[9], 1e-12);
    EXPECT_EQ(std::string(kEvenMeshHash), snap->sha256);

    // ALL source channels, source order, repeated-l distinct
    ASSERT_EQ(3u, snap->channels.size());
    EXPECT_EQ(0, snap->channels[0].l);
    EXPECT_EQ(0, snap->channels[1].l); // repeated l, distinct channel entry
    EXPECT_EQ(1, snap->channels[2].l);
    EXPECT_NEAR(2.0, snap->channels[0].occupation, 1e-12);
    EXPECT_NEAR(0.0, snap->channels[1].occupation, 1e-12);  // unoccupied survives
    EXPECT_NEAR(-1.0, snap->channels[2].occupation, 1e-12); // negative survives
    EXPECT_EQ("3S", snap->channels[0].label);
    EXPECT_EQ("4S", snap->channels[1].label);
    EXPECT_EQ("2P", snap->channels[2].label);

    // normalized on the full mesh, source shape preserved
    static const std::vector<std::vector<double>> source_shapes = {
        {1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55},
        {0.5, 0.48, 0.46, 0.44, 0.42, 0.40, 0.38, 0.36, 0.34, 0.32},
        {0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50}};
    for (int ic = 0; ic < 3; ++ic)
    {
        ASSERT_EQ(10u, snap->channels[ic].chi.size());
        const double norm = trapezoid_norm(snap->channels[ic].chi, snap->r);
        EXPECT_NEAR(1.0, norm, 1e-6) << "channel " << ic;
        for (int ir = 0; ir < 10; ++ir)
        {
            // scale-invariant shape: chi_norm[ir]/chi_norm[0] == source ratio
            EXPECT_NEAR(source_shapes[ic][ir] / source_shapes[ic][0],
                        snap->channels[ic].chi[ir] / snap->channels[ic].chi[0],
                        1e-12)
                << "channel " << ic << " ir " << ir;
        }
    }
}

TEST(Tb2jAtomicBasis, CaptureSpeciesRejectsInvalidInputs)
{
    std::string error;
    // ultrasoft projectors are out of scope
    EXPECT_EQ(nullptr, ModuleIO::Tb2j::AtomicBasis::capture_species("./support/fe_pbe_v1.5.uspp.F.UPF", "Fe", "auto", error));
    EXPECT_NE(std::string::npos, error.find("scalar norm-conserving"));
    // spin-orbit / fully relativistic source
    error.clear();
    EXPECT_EQ(nullptr, ModuleIO::Tb2j::AtomicBasis::capture_species("./support/Al.pbe-sp-van-so.UPF", "Al", "auto", error));
    EXPECT_NE(std::string::npos, error.find("scalar norm-conserving"));
    error.clear();
    EXPECT_EQ(nullptr, ModuleIO::Tb2j::AtomicBasis::capture_species("./support/Fe_tb2j_fullrel.upf", "Fe", "auto", error));
    EXPECT_NE(std::string::npos, error.find("scalar norm-conserving"));
    // zero-norm channel must fail explicitly, never be silently dropped
    error.clear();
    EXPECT_EQ(nullptr, ModuleIO::Tb2j::AtomicBasis::capture_species("./support/Fe_tb2j_zero_norm.upf", "Fe", "auto", error));
    EXPECT_FALSE(error.empty());
    // unreadable file
    error.clear();
    EXPECT_EQ(nullptr, ModuleIO::Tb2j::AtomicBasis::capture_species("./support/does_not_exist.upf", "Fe", "auto", error));
    EXPECT_FALSE(error.empty());
}

TEST(Tb2jAtomicBasis, MapperIsDeterministicNativeOrderAndOwnsCopies)
{
    std::string error;
    const auto snap = ModuleIO::Tb2j::AtomicBasis::capture_species(kEvenMeshFixture, "Fe", "auto", error);
    ASSERT_TRUE(error.empty()) << error;
    snap->natoms = 2;

    std::vector<ModuleIO::Tb2j::SpeciesBasis> species;
    species.push_back(*snap); // copy of the snapshot

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

    // deep-copy ownership: later mutation of the captured snapshot or of the
    // constructor input must not change the basis (initializer immunity)
    const double chi3 = basis.species(0).channels[0].chi[3];
    snap->channels[0].chi[3] *= 7.0;
    species[0].channels[0].chi[3] *= 7.0;
    EXPECT_EQ(chi3, basis.species(0).channels[0].chi[3]);
}

TEST(Tb2jAtomicBasis, TablesAreCellIndependentQSpaceTransforms)
{
    // hand-made analytic species: chi = e^{-r}, l = 0 on an odd mesh;
    // int e^{-r} j_0(q r) r dr -> 1/(1+q^2) for negligible tail
    ModuleIO::Tb2j::SpeciesBasis s;
    s.label = "H";
    s.natoms = 1;
    s.mesh = 801;
    s.r.resize(801);
    s.rab.assign(801, 0.1);
    ModuleIO::Tb2j::AtomicChannel ch;
    ch.l = 0;
    ch.occupation = 1.0;
    ch.label = "1S";
    ch.chi.resize(801);
    for (int i = 0; i < 801; ++i)
    {
        s.r[i] = 0.1 * i;
        ch.chi[i] = std::exp(-s.r[i]);
    }
    s.channels.push_back(ch);

    const int nqx = 1000;
    const double dq = 0.01;
    std::vector<ModuleIO::Tb2j::SpeciesBasis> species;
    species.push_back(s);
    ModuleIO::Tb2j::AtomicBasis basis(species, nqx, dq);

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

/**
 * Harness for fill_pw: one species ("Fe", captured from the even-mesh
 * fixture), two atoms, one k-point with a single G, mocked structure factor
 * sk == 1, following the source_psi/test hand-built PW convention.
 */
class Tb2jFillPwTest : public testing::Test
{
  protected:
    void SetUp() override
    {
        std::string error;
        snap_ = ModuleIO::Tb2j::AtomicBasis::capture_species(kEvenMeshFixture, "Fe", "auto", error);
        ASSERT_TRUE(error.empty()) << error;
        snap_->natoms = 2;
        std::vector<ModuleIO::Tb2j::SpeciesBasis> species;
        species.push_back(*snap_);
        basis_ = std::make_unique<ModuleIO::Tb2j::AtomicBasis>(species, 1000, 0.01);

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
    // The fixture's q is about 0.235, so its four-point stencil is unavailable.
    const auto before = out_;
    const char* error = basis_->fill_pw(0, *ucell_, sf_, pw_.get(), work_, out_.data(), 4);
    ASSERT_NE(nullptr, error);
    EXPECT_NE(std::string::npos, std::string(error).find("radial table"));
    EXPECT_EQ(before, out_);
}

TEST(Tb2jAtomicBasis, DeclaredUpf201DoesNotDependOnFirstLine)
{
    std::string filename = testing::TempDir() + "tb2j-upf-format-XXXXXX";
    const int fd = mkstemp(filename.data());
    ASSERT_GE(fd, 0);
    close(fd);
    {
        std::ifstream source(kEvenMeshFixture);
        std::ofstream output(filename);
        output << '\n' << source.rdbuf(); // defeats first-line auto detection
    }
    std::string error;
    const auto snapshot = ModuleIO::Tb2j::AtomicBasis::capture_species(
        filename, "Fe", "upf201", error);
    std::remove(filename.c_str());
    ASSERT_TRUE(error.empty()) << error;
    ASSERT_NE(snapshot, nullptr);
    EXPECT_EQ(10, snapshot->mesh);
    EXPECT_EQ(-1.0, snapshot->channels.at(2).occupation);
}
