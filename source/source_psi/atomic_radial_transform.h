#ifndef ATOMIC_RADIAL_TRANSFORM_H
#define ATOMIC_RADIAL_TRANSFORM_H

namespace atomic_radial
{

/// Integrate a caller-owned work buffer with the original UPF PP_RAB factors.
/// The even-mesh path folds the measure into the buffer in place. No allocation
/// or normalization policy; both paths reuse the existing native primitives.
double integrate_measure(const int mesh, double* integrand, const double* rab);

/**
 * @brief Transform one immutable radial channel onto a spherical-Bessel q grid.
 *
 * MathDoc: tb2j-dev/qe-atomic-pao-trace
 *
 * Computes
 *   table[iq] = \int chi(ir) * j_l(q[iq] * r(ir)) * r(ir) dr,  iq = 0..nq-1,
 * with the source UPF PP_RAB integration factors and native spherical-Bessel/
 * Simpson mechanics of psi_init_atomic::tabulate. The channel is non-mutating:
 * data are read-only, only the caller-provided work and output buffers are
 * written, and no allocation happens. It decides neither occupations nor
 * normalization policy; the CALLER chooses the mesh (full UPF mesh vs the
 * native truncated msh) and applies any scaling (e.g. 4pi/sqrt(omega)).
 *
 * Quadrature parity, existing primitives only, chosen by mesh parity:
 * - odd mesh: Integral::Simpson_Integral, bit-identical to the pre-change
 *   native initializer arithmetic (the initializer's complete_default policy
 *   forces an odd msh);
 * - even mesh (full preserved UPF mesh): the same Simpson weights on the
 *   rab-weighted integrand via the existing Integral::simpson(n, f, 1.0)
 *   composite 1/3 + 3/8 rule.
 *
 * @param mesh      number of radial points to transform (caller's choice)
 * @param r         radial grid [mesh]
 * @param rab       original UPF PP_RAB discrete integration factors [mesh]
 * @param chi       channel radial values [mesh]
 * @param l         angular momentum of the channel (>= 0)
 * @param q         q points [nq], e.g. PARAM.globalv.dq * iq
 * @param nq        number of q points (>= 0)
 * @param bessel    work buffer, >= mesh entries (caller-owned, reused)
 * @param integrand work buffer, >= mesh entries (caller-owned, reused)
 * @param table     output, >= nq entries: q-space transform per iq
 */
void transform_channel(const int mesh,
                       const double* r,
                       const double* rab,
                       const double* chi,
                       const int l,
                       const double* q,
                       const int nq,
                       double* bessel,
                       double* integrand,
                       double* table);
} // namespace atomic_radial

#endif // ATOMIC_RADIAL_TRANSFORM_H
