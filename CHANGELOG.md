# Changelog

## Unreleased

- Add optional `ENABLE_TB2J_PROJECTOR_NETCDF` / `out_tb2j` support for converged
  two-spin scalar-NC UPF plane-wave calculations. All UPF atomic-wavefunction
  channels, raw primal coefficients, full k-dependent Gram, occupations and
  complete onsite XC-only `delta_total` are exported through the existing
  `tb2j.projector_green/1.0` schema with native/input/UPF provenance.
- Assemble PW, k-pool and distributed-band ownership before bounded one-k
  streaming to a serial NetCDF writer. Publish only a closed complete dataset;
  streaming errors abort the world rather than leaving blocked peers or a
  fallback final. Abrupt peer aborts may leave an unpublished partial sibling.
- Preserve default PW/LCAO behavior and require no exporter dependencies in the
  default-OFF build. Unsupported physics/channels and uncertified grids fail
  actionably instead of being converted to a different projector workflow.
- Validate matched unmasked bccFe against stored ABINIT with separate replay,
  cross-code and strict same-binary MPI budgets for three mesh-safe shells.
  This is not extended-range, full-spectrum, GPU or general convergence proof.
  See [build, runtime and validation boundaries](docs/advanced/install.md#build-with-tb2j-pw-atomic-projector-support).
- Record a secondary same-policy unmasked FM FeO comparison without inventing
  a tolerance or equating generic projector populations with ABINIT-specific
  dual-basis Mulliken side metadata.
