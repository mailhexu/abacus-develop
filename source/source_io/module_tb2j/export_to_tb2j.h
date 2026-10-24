#ifndef ABACUS_TB2J_EXPORT_TO_TB2J_H
#define ABACUS_TB2J_EXPORT_TO_TB2J_H

class UnitCell;
class K_Vectors;
class Charge;
class Structure_Factor;
class Setup_Psi_pw;
struct Input_para;
namespace elecstate { class ElecState; }
namespace ModulePW { class PW_Basis; class PW_Basis_K; }

namespace ModuleIO
{
namespace Tb2j
{
/// All ranks call only after confirmed SCF convergence and before estate teardown.
/// Builds the original-UPF basis locally and aliases CPU-double wavefunctions.
/// Uses native MPI ownership and root-only complete-file publication.
/// Errors after entry abort MPI_COMM_WORLD (serial: throw).
void export_to_tb2j(const UnitCell& cell, const K_Vectors& kv,
                       const elecstate::ElecState& estate, const Charge& charge,
                       const Input_para& input,
                       const Structure_Factor& sf, const ModulePW::PW_Basis_K& wfc,
                       ModulePW::PW_Basis& dense, ModulePW::PW_Basis& smooth,
                       const Setup_Psi_pw& wavefunctions);

} // namespace Tb2j
} // namespace ModuleIO
#endif
