#include "gtest/gtest.h"
#include "source_io/module_tb2j/native_state.h"
#include "source_base/constants.h"

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace
{
using ModuleIO::Tb2j::GridRecipe;
using ModuleIO::Tb2j::NativeState;
using ModuleIO::Tb2j::SpectralBlock;

GridRecipe recipe(const std::string& convention = "Gamma")
{
    GridRecipe value;
    value.generated = true;
    value.input_count = 0;
    value.original_count = 4;
    value.symmetry = -1;
    value.dimensions = {2, 2, 1};
    value.offsets = {0.5, 0.0, 0.25};
    value.convention = convention;
    return value;
}

struct Rows
{
    std::vector<int> global, spin;
    std::vector<ModuleBase::Vector3<double>> k;
    std::vector<double> weight;
    ModuleBase::matrix energy, wg;

    explicit Rows(const NativeState& state)
        : global(8), spin(8), k(8), weight(8, 0.25), energy(8, 2), wg(8, 2)
    {
        // Intentionally shuffled local rows; global identity is not row index.
        const int order[] = {7, 0, 5, 2, 4, 1, 6, 3};
        for (int row = 0; row < 8; ++row)
        {
            const int g = order[row];
            global[row] = g;
            spin[row] = g / 4;
            k[row] = state.kpoints[g % 4];
            k[row].x += 1.0;
            k[row].y -= 2.0; // reciprocal-period equivalence, not raw equality
            for (int band = 0; band < 2; ++band)
            {
                energy(row, band) = 0.1 * g + 0.03 * band;
                const double occupation = spin[row] == 0 ? (band == 0 ? 0.8 : 0.3)
                                                        : (band == 0 ? 0.6 : 0.1);
                wg(row, band) = weight[row] * occupation;
            }
        }
    }

    SpectralBlock view() const
    {
        return {static_cast<int>(global.size()), global.data(), spin.data(),
                k.data(), weight.data(), &energy, &wg};
    }
};
}

TEST(Tb2jNativeState, GeneratedGammaAndMpPreserveShuffledSpectraAndCounts)
{
    for (const auto& convention : {"Gamma", "MP"})
    {
        std::string error;
        auto state = NativeState::create(recipe(convention), 2, error);
        ASSERT_NE(nullptr, state) << error;
        Rows rows(*state);
        ASSERT_TRUE(state->append(rows.view(), error)) << error;
        ASSERT_TRUE(state->finish(1.8, {0.12, 0.19}, false, error)) << error;
        EXPECT_EQ(4, state->nkpt);
        EXPECT_EQ(2, state->nband);
        EXPECT_DOUBLE_EQ(1.0, std::accumulate(state->weights.begin(), state->weights.end(), 0.0));
        EXPECT_NEAR(1.1, state->native_electrons[0], 1e-14);
        EXPECT_NEAR(0.7, state->native_electrons[1], 1e-14);
        for (int spin = 0; spin < 2; ++spin)
        {
            EXPECT_NEAR(state->native_electrons[spin], state->electrons[spin], 1e-14);
            EXPECT_DOUBLE_EQ((spin == 0 ? 0.12 : 0.19) * ModuleBase::Ry_to_eV,
                             state->efermi_spin[spin]);
            for (int ik = 0; ik < 4; ++ik)
            {
                for (int band = 0; band < 2; ++band)
                {
                    const int index = (spin * 4 + ik) * 2 + band;
                    EXPECT_NEAR((0.1 * (spin * 4 + ik) + 0.03 * band) * ModuleBase::Ry_to_eV,
                                state->eigenvalues[index], 1e-14);
                    EXPECT_NEAR(spin == 0 ? (band == 0 ? 0.8 : 0.3)
                                          : (band == 0 ? 0.6 : 0.1),
                                state->occupations[index], 1e-14);
                }
            }
        }
    }
}

TEST(Tb2jNativeState, CustomWeightedOccupationsAreNotAnIntegerSplitOrCapped)
{
    std::string error;
    auto state = NativeState::create(recipe(), 2, error);
    ASSERT_NE(nullptr, state) << error;
    Rows rows(*state);
    // row 1 is native global 0; custom wg is already weighted.
    rows.wg(1, 0) = 0.4;
    ASSERT_TRUE(state->append(rows.view(), error)) << error;
    ASSERT_TRUE(state->finish(2.0, {0.12, 0.19}, true, error)) << error;
    EXPECT_TRUE(state->custom_occupations);
    EXPECT_NEAR(1.6, state->occupations[0], 1e-14);
    EXPECT_NEAR(1.3, state->electrons[0], 1e-14);
    EXPECT_NEAR(0.7, state->electrons[1], 1e-14);
}

TEST(Tb2jNativeState, RejectsUnprovableReducedAndInvalidRecipes)
{
    for (int which = 0; which < 7; ++which)
    {
        auto value = recipe();
        if (which == 0) value.generated = false;
        if (which == 1) value.input_count = 4;
        if (which == 2) value.symmetry = 0;
        if (which == 3) value.original_count = 3;
        if (which == 4) value.dimensions[0] = 0;
        if (which == 5) value.convention = "Direct";
        if (which == 6) value.offsets[0] = std::numeric_limits<double>::infinity();
        std::string error;
        EXPECT_EQ(nullptr, NativeState::create(value, 2, error));
        EXPECT_FALSE(error.empty());
    }
}

TEST(Tb2jNativeState, RejectsDuplicateMissingAndSpinMismatchedCoverage)
{
    for (int which = 0; which < 3; ++which)
    {
        std::string error;
        auto state = NativeState::create(recipe(), 2, error);
        ASSERT_NE(nullptr, state) << error;
        Rows rows(*state);
        if (which == 0)
        {
            rows.global[7] = rows.global[6];
            rows.spin[7] = rows.spin[6];
            rows.k[7] = rows.k[6];
        }
        if (which == 1) rows.global.pop_back();
        if (which == 2) rows.spin[0] = 0;
        auto block = rows.view();
        // Missing coverage is a valid partial block until finalization.
        if (which == 1)
        {
            ModuleBase::matrix energy(7, 2), wg(7, 2);
            for (int row = 0; row < 7; ++row)
                for (int band = 0; band < 2; ++band)
                {
                    energy(row, band) = rows.energy(row, band);
                    wg(row, band) = rows.wg(row, band);
                }
            block.energies_ry = &energy;
            block.weighted_occupations = &wg;
            ASSERT_TRUE(state->append(block, error)) << error;
            EXPECT_FALSE(state->finish(1.8, {0.12, 0.19}, false, error));
            EXPECT_NE(std::string::npos, error.find("coverage"));
        }
        else
        {
            EXPECT_FALSE(state->append(block, error));
            EXPECT_NE(std::string::npos, error.find(which == 0 ? "duplicate" : "spin"));
        }
    }
}

TEST(Tb2jNativeState, RejectsWrongPointZeroWeightAndNonfiniteSpectrum)
{
    for (int which = 0; which < 4; ++which)
    {
        std::string error;
        auto state = NativeState::create(recipe(), 2, error);
        ASSERT_NE(nullptr, state) << error;
        Rows rows(*state);
        if (which == 0) rows.k[0].x += 0.01;
        if (which == 1) rows.weight[0] = 0.0;
        if (which == 2) rows.energy(0, 0) = std::numeric_limits<double>::quiet_NaN();
        if (which == 3) rows.wg(0, 0) = std::numeric_limits<double>::infinity();
        EXPECT_FALSE(state->append(rows.view(), error));
        EXPECT_FALSE(error.empty());
    }
}

TEST(Tb2jNativeState, RejectsUnequalSpinWeightsNonuniformWeightsAndElectronMismatch)
{
    for (int which = 0; which < 3; ++which)
    {
        std::string error;
        auto state = NativeState::create(recipe(), 2, error);
        ASSERT_NE(nullptr, state) << error;
        Rows rows(*state);
        if (which == 0)
            for (int row = 0; row < 8; ++row)
                if (rows.spin[row] == 1) rows.weight[row] *= 0.5;
        if (which == 1)
        {
            rows.weight[1] += 0.01;
            rows.weight[4] += 0.01; // same physical point, other spin
        }
        ASSERT_TRUE(state->append(rows.view(), error)) << error;
        EXPECT_FALSE(state->finish(which == 2 ? 2.1 : 1.8, {0.12, 0.19}, false, error));
        EXPECT_NE(std::string::npos, error.find(which == 2 ? "electron" : "weight"));
    }
}
