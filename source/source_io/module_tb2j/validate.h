#ifndef ABACUS_TB2J_VALIDATE_H
#define ABACUS_TB2J_VALIDATE_H

#include "source_base/global_variable.h"
#include "source_base/parallel_common.h"
#include "source_base/tool_quit.h"
#include "source_cell/read_pp.h"
#include "source_cell/unitcell.h"

#include <cmath>

namespace ModuleIO
{
namespace Tb2j
{
inline const char* atomic_channel_error(const Atom_pseudo& pp, const bool fully_relativistic = false)
{
    if (pp.tvanp || pp.pp_type != "NC" || pp.has_so || fully_relativistic)
    {
        return "Only scalar norm-conserving atomic projectors are supported";
    }
    if (pp.nchi <= 0 || pp.mesh <= 1 || pp.chi.nr != pp.nchi || pp.chi.nc < pp.mesh
        || pp.r.size() != static_cast<std::size_t>(pp.mesh)
        || pp.rab.size() != static_cast<std::size_t>(pp.mesh)
        || pp.lchi.size() != static_cast<std::size_t>(pp.nchi))
    {
        return "Missing or invalid atomic-wavefunction channels";
    }
    for (int channel = 0; channel < pp.nchi; ++channel)
    {
        double square_sum = 0.0;
        for (int ir = 0; ir < pp.mesh; ++ir)
        {
            const double value = pp.chi(channel, ir);
            square_sum += value * value;
        }
        if (pp.lchi[channel] < 0 || !std::isfinite(square_sum) || square_sum == 0.0)
        {
            return "Atomic-wavefunction channel has an invalid radial norm";
        }
    }
    return nullptr;
}

// Only opt-in requests inspect the original file; native ncpp may already be
// SOC-averaged. Match the native rank-zero-only pseudopotential IO policy.
inline void validate_species(const UnitCell& cell, const std::string& pseudo_dir)
{
    std::string error_message;
    if (GlobalV::MY_RANK == 0)
    {
        for (int it = 0; it < cell.ntype && error_message.empty(); ++it)
        {
            Atom_pseudo original;
            Pseudopot_upf reader;
            std::string format = "auto";
            const std::string filename = pseudo_dir + cell.pseudo_fn[it];
            if (reader.init_pseudo_reader(filename, format, original) != 0
                || (format != "upf" && format != "upf201"))
            {
                error_message = "Atomic projectors require a readable UPF pseudopotential";
            }
            else if (const char* error = atomic_channel_error(original, reader.relativistic == "full"))
            {
                error_message = error;
            }
        }
    }
    Parallel_Common::bcast_string(error_message);
    if (!error_message.empty())
    {
        ModuleBase::WARNING_QUIT("out_tb2j", error_message);
    }
}
} // namespace Tb2j
} // namespace ModuleIO

#endif
