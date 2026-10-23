#include "export.h"
#include "atomic_projectors.h"
#include "native_state.h"
#include "projection.h"
#include "projector_netcdf.h"
#include "source_base/global_variable.h"
#include "source_base/module_device/memory_op.h"
#include "source_base/module_parallel/para_band_output.h"
#include "source_base/module_parallel/para_bridge.h"
#include "source_base/parallel_global.h"
#include "source_basis/module_pw/pw_basis_k.h"
#include "source_cell/klist.h"
#include "source_cell/unitcell.h"
#include "source_estate/elecstate.h"
#include "source_io/module_parameter/input_parameter.h"
#include "source_io/module_parameter/parameter.h"
#include "source_psi/setup_psi_pw.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ModuleIO
{
namespace Tb2j
{
namespace
{
using Complex = std::complex<double>;
constexpr int header_tag = 8300, coefficients_tag = 8301, gram_tag = 8302, sites_tag = 8303;

int count_int(const std::size_t count)
{
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("TB2J one-k message exceeds the native MPI count range");
    return static_cast<int>(count);
}

#ifdef __MPI
void mpi_check(const int status, const char* context)
{
    if (status == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(status, message, &length);
    throw std::runtime_error(std::string(context) + ": " + std::string(message, length));
}

void receive(void* data, const int count, const MPI_Datatype type,
              const int source, const int tag, const char* context)
{
    MPI_Status status;
    mpi_check(MPI_Probe(source, tag, MPI_COMM_WORLD, &status), context);
    int observed = -1;
    mpi_check(MPI_Get_count(&status, type, &observed), context);
    if (observed != count)
        throw std::runtime_error(std::string(context) + ": malformed block extent");
    mpi_check(MPI_Recv(data, count, type, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE), context);
}

void send(const void* data, const int count, const MPI_Datatype type,
           const int tag, const char* context)
{
    mpi_check(MPI_Send(data, count, type, 0, tag, MPI_COMM_WORLD), context);
}
#endif

void preflight(const std::string& local_error)
{
#ifdef __MPI
    int first = local_error.empty() ? std::numeric_limits<int>::max() : GlobalV::MY_RANK;
    mpi_check(MPI_Allreduce(MPI_IN_PLACE, &first, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD), "preflight ownership");
    if (first == std::numeric_limits<int>::max()) return;
    std::string message = local_error;
    int length = count_int(message.size());
    mpi_check(MPI_Bcast(&length, 1, MPI_INT, first, MPI_COMM_WORLD), "preflight diagnostic extent");
    message.resize(length);
    mpi_check(MPI_Bcast(&message[0], length, MPI_CHAR, first, MPI_COMM_WORLD), "preflight diagnostic");
    throw std::runtime_error("TB2J export preflight: " + message);
#else
    if (!local_error.empty()) throw std::runtime_error("TB2J export preflight: " + local_error);
#endif
}

void pool_reduce(Complex* values, const std::size_t size)
{
#ifdef __MPI
    const int count = count_int(2 * size);
    if (count == 0) return;
    if (GlobalV::RANK_IN_POOL == 0)
        mpi_check(MPI_Reduce(MPI_IN_PLACE, reinterpret_cast<double*>(values), count,
                              MPI_DOUBLE, MPI_SUM, 0, POOL_WORLD), "PW-root projection reduction");
    else
        mpi_check(MPI_Reduce(reinterpret_cast<double*>(values), nullptr, count,
                              MPI_DOUBLE, MPI_SUM, 0, POOL_WORLD), "PW projection reduction");
#else
    (void)values;
    (void)size;
#endif
}

struct HostBuffers
{
    std::vector<Complex> converted;
#if defined(__CUDA) || defined(__ROCM)
    std::vector<std::complex<float>> single;
#endif
};
struct HostView
{
    const Complex* values;
    int stride;
};

template<typename T, typename Device>
const T* native_k(const Setup_Psi_pw& source, const int ik)
{
    const auto* psi = source.get_psi_t<T, Device>();
    if (!psi) throw std::runtime_error("Missing native typed PW wavefunction");
    if (psi->get_nbands() == 0) return nullptr;
    // No fix_k cursor mutation and no refresh of the whole CPU mirror.
    return psi->get_pointer() - psi->get_psi_bias()
           + static_cast<std::size_t>(ik) * psi->get_nbands() * psi->get_nbasis();
}

HostView cpu_double(const Setup_Psi_pw& source, const int ik, const int, HostBuffers&)
{
    return {native_k<Complex, base_device::DEVICE_CPU>(source, ik), source.get_nbasis()};
}

HostView cpu_single(const Setup_Psi_pw& source, const int ik, const int npw, HostBuffers& work)
{
    const auto* input = native_k<std::complex<float>, base_device::DEVICE_CPU>(source, ik);
    work.converted.resize(static_cast<std::size_t>(source.get_nbands()) * npw);
    for (int band = 0; band < source.get_nbands(); ++band)
        base_device::memory::cast_memory_op<Complex, std::complex<float>,
            base_device::DEVICE_CPU, base_device::DEVICE_CPU>()(
                work.converted.data() + static_cast<std::size_t>(band) * npw,
                input + static_cast<std::size_t>(band) * source.get_nbasis(), npw);
    return {work.converted.data(), npw};
}

#if defined(__CUDA) || defined(__ROCM)
HostView gpu_double(const Setup_Psi_pw& source, const int ik, const int npw, HostBuffers& work)
{
    const auto* input = native_k<Complex, base_device::DEVICE_GPU>(source, ik);
    work.converted.resize(static_cast<std::size_t>(source.get_nbands()) * npw);
    base_device::memory::synchronize_memory_2d_op<Complex,
        base_device::DEVICE_CPU, base_device::DEVICE_GPU>()(
            work.converted.data(), npw, input, source.get_nbasis(), npw, source.get_nbands());
    return {work.converted.data(), npw};
}

HostView gpu_single(const Setup_Psi_pw& source, const int ik, const int npw, HostBuffers& work)
{
    const auto* input = native_k<std::complex<float>, base_device::DEVICE_GPU>(source, ik);
    const auto size = static_cast<std::size_t>(source.get_nbands()) * npw;
    work.single.resize(size);
    work.converted.resize(size);
    base_device::memory::synchronize_memory_2d_op<std::complex<float>,
        base_device::DEVICE_CPU, base_device::DEVICE_GPU>()(
            work.single.data(), npw, input, source.get_nbasis(), npw, source.get_nbands());
    base_device::memory::cast_memory_op<Complex, std::complex<float>,
        base_device::DEVICE_CPU, base_device::DEVICE_CPU>()(
            work.converted.data(), work.single.data(), size);
    return {work.converted.data(), npw};
}
#endif

using Stage = HostView (*)(const Setup_Psi_pw&, int, int, HostBuffers&);
Stage host_stage(const Setup_Psi_pw& source)
{
    if (source.get_device_type() == base_device::CpuDevice)
    {
        if (source.get_precision_type() == Setup_Psi_pw::PrecisionType::ComplexDouble) return cpu_double;
        if (source.get_precision_type() == Setup_Psi_pw::PrecisionType::ComplexFloat) return cpu_single;
    }
#if defined(__CUDA) || defined(__ROCM)
    if (source.get_device_type() == base_device::GpuDevice)
    {
        if (source.get_precision_type() == Setup_Psi_pw::PrecisionType::ComplexDouble) return gpu_double;
        if (source.get_precision_type() == Setup_Psi_pw::PrecisionType::ComplexFloat) return gpu_single;
    }
#endif
    throw std::runtime_error("Unsupported native PW device/precision for TB2J host staging");
}

struct KPair
{
    int global = -1, up = -1, down = -1;
};
} // namespace

void export_projector(const UnitCell& cell, const K_Vectors& kv,
                       const elecstate::ElecState& estate, const Charge& charge,
                       const Input_para& input, const AtomicBasis& basis,
                       const Structure_Factor& sf, const ModulePW::PW_Basis_K& wfc,
                       ModulePW::PW_Basis& dense, ModulePW::PW_Basis& smooth,
                       const Setup_Psi_pw& wavefunctions)
try
{
    const auto band_world = Parallel::make_band_world();
    const Parallel::ParaBandOutput bands(PARAM.globalv.nbands_l, input.nbands, band_world);
    auto state = NativeState::capture(cell, kv, estate, input, bands);
    const int nkpt = kv.get_nkstot() / 2;
    const int nproj = basis.nproj();
    const auto sites = atomic_sites(basis);
    const auto nsite_values = sites.empty() ? 0 : sites.back().offset
        + static_cast<std::size_t>(sites.back().count) * sites.back().count;
    const auto coefficient_size = 2 * static_cast<std::size_t>(input.nbands) * nproj;
    const auto gram_size = static_cast<std::size_t>(nproj) * nproj;
    const int my_pool = GlobalV::MY_POOL;
    std::vector<int> pool_count(GlobalV::KPAR), pool_start(GlobalV::KPAR), pool_owner(GlobalV::KPAR);
    std::string error;
    if (nkpt <= 0 || nproj <= 0 || sites.size() != static_cast<std::size_t>(cell.nat)
        || my_pool < 0 || my_pool >= GlobalV::KPAR || GlobalV::KPAR <= 0)
        error = "Invalid native unique-k/projector/site/pool dimensions";
    if (GlobalV::KPAR == 1)
    {
        pool_count[0] = nkpt;
        pool_start[0] = pool_owner[0] = 0;
    }
    else if (kv.para_k.nks_pool.size() != pool_count.size() || kv.para_k.startk_pool.size() != pool_count.size())
        error = "Missing native physical-pool k ownership";
    else
        for (int pool = 0; pool < GlobalV::KPAR; ++pool)
        {
            pool_count[pool] = kv.para_k.nks_pool[pool];
            pool_start[pool] = kv.para_k.startk_pool[pool];
            pool_owner[pool] = kv.para_k.get_startpro_pool(pool);
        }
    if (error.empty())
    {
        int end = 0;
        for (int pool=0; pool<GlobalV::KPAR; ++pool)
        {
            if (pool_count[pool] < 0 || pool_start[pool] != end
                || pool_count[pool] > nkpt-end || pool_owner[pool] < 0
                || pool_owner[pool] >= GlobalV::NPROC)
            { error = "Invalid native contiguous physical-pool ownership"; break; }
            end += pool_count[pool];
        }
        if (end != nkpt || pool_owner[0] != 0)
            error = "Native physical pools do not cover the certified full BZ";
    }
    preflight(error);

    const bool representative = GlobalV::MY_RANK == pool_owner[my_pool];
    const bool pw_root = GlobalV::RANK_IN_POOL == 0;
    const bool vertex_group = GlobalV::MY_BNDGROUP == 0;
    const bool distributed = !bands.bands_are_replicated();
    const bool active = vertex_group || distributed;
    const Stage stage = host_stage(wavefunctions);
    if (!wavefunctions.psi_cpu || !wavefunctions.psi_cpu->get_k_first() || !wfc.npwk
        || wfc.npwk_max < 0 || wavefunctions.get_nk() != kv.get_nks()
        || wavefunctions.get_nbands() != bands.local_nbands() || wavefunctions.get_nbasis() < wfc.npwk_max
        || wfc.nks != kv.get_nks() || kv.get_nks() != 2 * pool_count[my_pool]
        || bands.band_group() != GlobalV::MY_BNDGROUP
        || representative != (vertex_group && pw_root))
        error = "Inconsistent native wavefunction, band or representative layout";
    if (coefficient_size > static_cast<std::size_t>(std::numeric_limits<int>::max()/2)
        || gram_size > static_cast<std::size_t>(std::numeric_limits<int>::max()/2)
        || nsite_values > static_cast<std::size_t>(std::numeric_limits<int>::max()/2)
        || static_cast<std::size_t>(nproj) * std::max(1,wfc.npwk_max)
              > static_cast<std::size_t>(std::numeric_limits<int>::max()/2))
        error = "One-k projector/coefficients/site blocks exceed native MPI count bounds";
    std::vector<KPair> pairs(pool_count[my_pool]);
    for (int i = 0; i < pool_count[my_pool]; ++i) pairs[i].global = pool_start[my_pool]+i;
    if (error.empty())
        for (int ik = 0; ik < kv.get_nks(); ++ik)
        {
            const int spin = kv.isk[ik], global = kv.ik2iktot[ik];
            const int unique = global - spin*nkpt;
            if (spin < 0 || spin > 1 || unique < pool_start[my_pool]
                || unique >= pool_start[my_pool]+pool_count[my_pool])
            {
                error = "Native k/spin identity lies outside its physical pool";
                break;
            }
            auto& pair = pairs[unique-pool_start[my_pool]];
            int& slot = spin == 0 ? pair.up : pair.down;
            if (slot >= 0) { error = "Duplicate native local spin/k identity"; break; }
            slot = ik;
        }
    if (error.empty())
        for (const auto& pair : pairs)
        {
            if (pair.up < 0 || pair.down < 0 || wfc.npwk[pair.up] < 0
                || wfc.npwk[pair.up] > wfc.npwk_max || wfc.npwk[pair.up] != wfc.npwk[pair.down])
            { error = "Missing native spin partner or unusable PW layouts"; break; }
            for (int g = 0; g < wfc.npwk[pair.up]; ++g)
                if (wfc.getigl2isz(pair.up,g) != wfc.getigl2isz(pair.down,g)
                    || (wfc.getgpluskcar(pair.up,g)-wfc.getgpluskcar(pair.down,g)).norm2() > 1e-20)
                { error = "Native spin partners use different PW identities"; break; }
            if (!error.empty()) break;
        }
    preflight(error);

#ifdef __MPI
    if (distributed)
    {
        int extent_min = GlobalV::NPROC_IN_POOL, extent_max = extent_min;
        mpi_check(MPI_Allreduce(MPI_IN_PLACE,&extent_min,1,MPI_INT,MPI_MIN,band_world.comm()), "band PW extent minimum");
        mpi_check(MPI_Allreduce(MPI_IN_PLACE,&extent_max,1,MPI_INT,MPI_MAX,band_world.comm()), "band PW extent maximum");
        if (extent_min != extent_max) error = "Native band groups have unequal PW partitions";
        std::vector<int> minimum(pairs.size()), maximum(pairs.size());
        for (std::size_t i=0; i<pairs.size(); ++i) minimum[i]=maximum[i]=wfc.npwk[pairs[i].up];
        mpi_check(MPI_Allreduce(MPI_IN_PLACE,minimum.data(),count_int(minimum.size()),MPI_INT,MPI_MIN,band_world.comm()), "band PW count minimum");
        mpi_check(MPI_Allreduce(MPI_IN_PLACE,maximum.data(),count_int(maximum.size()),MPI_INT,MPI_MAX,band_world.comm()), "band PW count maximum");
        if (minimum != maximum) error = "Native band groups have different local PW identities";
    }
#endif
    std::vector<int> coverage(nkpt,0);
    if (representative) for (const auto& pair : pairs) ++coverage[pair.global];
#ifdef __MPI
    mpi_check(MPI_Allreduce(MPI_IN_PLACE,coverage.data(),nkpt,MPI_INT,MPI_SUM,MPI_COMM_WORLD), "unique-k representative coverage");
#endif
    if (std::find_if(coverage.begin(),coverage.end(),[](const int n){return n!=1;}) != coverage.end())
        error = "Native unique-k ownership is not exactly once";
    preflight(error);
    // Reuse the exact certified divisor, not a guessed uniform weight or an
    // all-rank copy of the entire global weight array.
    double weight_sum = state ? state->native_weight_sum : 0.;
#ifdef __MPI
    mpi_check(MPI_Bcast(&weight_sum,1,MPI_DOUBLE,0,MPI_COMM_WORLD), "certified native weight normalization");
#endif
    if (!std::isfinite(weight_sum) || weight_sum <= 0.)
        throw std::runtime_error("Invalid certified native weight normalization");
    // Preserve the completed SCF/coverage certificate even if a different pool
    // aborts the world while root is blocked in the following streaming phase.
    if (GlobalV::MY_RANK == 0) GlobalV::ofs_running.flush();

    std::vector<double> delta_v;
    if (vertex_group)
    {
        delta_v = xc_difference(charge,cell,dense,smooth,wfc);
        if (GlobalV::MY_RANK == 0)
            GlobalV::ofs_running << " TB2J XC-only potential prepared from converged charge on native smooth FFT\n";
    }
    AtomicBasis::FillWorkspace fill_work;
    ProjectionWorkspace projection_work;
    HostBuffers host_work;
    std::vector<Complex> phi, local_coefficients, coefficients, gram, vertex, accumulated, band_row;
    if (active && !pairs.empty())
    {
        phi.resize(static_cast<std::size_t>(nproj)*std::max(1,wfc.npwk_max));
        if (!representative) local_coefficients.resize(static_cast<std::size_t>(bands.local_nbands())*nproj);
        if (pw_root && distributed && !representative) band_row.resize(nproj);
    }
    if (representative || GlobalV::MY_RANK == 0)
    {
        coefficients.resize(coefficient_size);
        accumulated.resize(nsite_values);
    }
    if ((vertex_group && !pairs.empty()) || GlobalV::MY_RANK == 0)
    {
        gram.resize(gram_size);
        vertex.resize(nsite_values);
    }

    // Every pool computes independently. Remote representatives may block in
    // their indexed sends while root processes pool 0; no world collective is
    // issued inside these unequal local-k loops.
    auto run = [&](ProjectorNetcdf* writer)
    {
        if (active)
            for (const auto& pair : pairs)
            {
                const int npw = wfc.npwk[pair.up], stride = std::max(1,npw);
                if (vertex_group)
                {
                    const char* failure = basis.fill_pw(pair.up,cell,sf,&wfc,fill_work,phi.data(),stride);
                    if (failure) throw std::runtime_error(failure);
                }
#ifdef __MPI
                if (distributed)
                    mpi_check(MPI_Bcast(reinterpret_cast<double*>(phi.data()),count_int(2*static_cast<std::size_t>(nproj)*stride),
                                         MPI_DOUBLE,0,band_world.comm()), "one-k atomic projector band broadcast");
#endif
                if (vertex_group)
                {
                    full_gram(nproj,npw,phi.data(),stride,gram.data());
                    pool_reduce(gram.data(),gram_size);
                    if (pw_root) metric_condition(nproj,gram.data(),projection_work);
                    project_site_xc(pair.up,wfc,sites,phi.data(),stride,delta_v.data(),projection_work,vertex.data());
                    pool_reduce(vertex.data(),nsite_values);
                    if (pw_root)
                        accumulate_site_vertex(kv.wk[pair.up]/weight_sum,nsite_values,vertex.data(),accumulated.data());
                }
                for (int spin=0; spin<2; ++spin)
                {
                    Complex* local = representative
                        ? coefficients.data()+static_cast<std::size_t>(spin)*input.nbands*nproj
                            +static_cast<std::size_t>(bands.local_offset())*nproj
                        : local_coefficients.data();
                    if (bands.local_nbands() > 0 && npw > 0)
                    {
                        const HostView view = stage(wavefunctions,spin==0?pair.up:pair.down,npw,host_work);
                        primal_overlaps(bands.local_nbands(),nproj,npw,view.values,view.stride,phi.data(),stride,local);
                    }
                    else
                        primal_overlaps(bands.local_nbands(),nproj,0,nullptr,0,nullptr,0,local);
                    pool_reduce(local,static_cast<std::size_t>(bands.local_nbands())*nproj);
                    if (pw_root && distributed)
                        for (int band=0; band<input.nbands; ++band)
                        {
                            Complex* row = representative
                                ? coefficients.data()+(static_cast<std::size_t>(spin)*input.nbands+band)*nproj
                                : bands.owner_group(band)==bands.band_group()
                                    ? local+static_cast<std::size_t>(bands.local_index(band))*nproj : band_row.data();
                            bands.bcast_band(band,row,nproj);
                        }
                }
                if (representative)
                {
                    if (writer) writer->write_k(pair.global,coefficients.data(),gram.data());
#ifdef __MPI
                    else
                    {
                        const int header[] = {pair.global,input.nbands,nproj,2};
                        send(header,4,MPI_INT,header_tag,"one-k indexed header send");
                        send(reinterpret_cast<const double*>(coefficients.data()),count_int(2*coefficient_size),MPI_DOUBLE,
                              coefficients_tag,"one-k coefficient send");
                        send(reinterpret_cast<const double*>(gram.data()),count_int(2*gram_size),MPI_DOUBLE,
                              gram_tag,"one-k full Gram send");
                    }
#endif
                }
            }
#ifdef __MPI
        if (representative && !writer && !pairs.empty())
            send(reinterpret_cast<const double*>(accumulated.data()),count_int(2*nsite_values),MPI_DOUBLE,
                  sites_tag,"pool onsite vertex send");
        if (writer)
            for (int pool=1; pool<GlobalV::KPAR; ++pool)
            {
                if (pool_count[pool] == 0) continue;
                for (int index=0; index<pool_count[pool]; ++index)
                {
                    int header[4];
                    receive(header,4,MPI_INT,pool_owner[pool],header_tag,"one-k indexed header receive");
                    if (header[0] != pool_start[pool]+index || header[1] != input.nbands
                        || header[2] != nproj || header[3] != 2)
                        throw std::runtime_error("Malformed TB2J indexed pool/band/projector block");
                    receive(reinterpret_cast<double*>(coefficients.data()),count_int(2*coefficient_size),MPI_DOUBLE,
                             pool_owner[pool],coefficients_tag,"one-k coefficient receive");
                    receive(reinterpret_cast<double*>(gram.data()),count_int(2*gram_size),MPI_DOUBLE,
                             pool_owner[pool],gram_tag,"one-k full Gram receive");
                    writer->write_k(header[0],coefficients.data(),gram.data());
                }
                receive(reinterpret_cast<double*>(vertex.data()),count_int(2*nsite_values),MPI_DOUBLE,
                         pool_owner[pool],sites_tag,"pool onsite vertex receive");
                for (std::size_t i=0; i<nsite_values; ++i) accumulated[i]+=vertex[i];
            }
#endif
        if (writer)
        {
            writer->finish(accumulated.data());
            GlobalV::ofs_running << " TB2J complete projector dataset published: "
                                 << PARAM.globalv.global_out_dir << "tb2j_projector.nc\n";
        }
#ifdef __MPI
        mpi_check(MPI_Barrier(MPI_COMM_WORLD),"complete projector publication");
#endif
    };
    if (GlobalV::MY_RANK == 0)
    {
        ProjectorNetcdf writer(PARAM.globalv.global_out_dir+"tb2j_projector.nc",*state,basis,sites,input);
        run(&writer);
    }
    else
        run(nullptr);
}
catch (const std::exception& error)
{
    std::cerr << "TB2J projector stream rank " << GlobalV::MY_RANK
              << " pool " << GlobalV::MY_POOL << ": " << error.what() << std::endl;
#ifdef __MPI
    MPI_Abort(MPI_COMM_WORLD,1);
#endif
    throw;
}
} // namespace Tb2j
} // namespace ModuleIO
