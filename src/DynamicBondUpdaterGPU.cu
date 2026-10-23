// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file DynamicBondUpdaterGPU.cu
 * \brief Definition of kernel drivers and kernels for DynamicBondUpdaterGPU
 */

#include "DynamicBondUpdaterGPU.cuh"

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

namespace hoomd
    {
namespace azplugins
    {
namespace gpu
    {
namespace kernel
    {
/*!
 * \param d_group_ids Group of each local particle
 * \param d_free_bonds Number of free bonds of each local particle
 * \param d_members Indexes of the particles in the group
 * \param num_members Number of particles in the group
 * \param group_id Group of the particles (1 or 2)
 * \param max_bonds Maximum number of bonds of the particles
 * \param bond_type Type of bonds to count
 * \param d_bond_table Bonds of each particle
 * \param d_n_bonds Number of bonds of each particle
 * \param bond_table_indexer Indexer into \a d_bond_table
 *
 * One thread is used per particle in the group.
 */
__global__ void set_bonding_state(unsigned int* d_group_ids,
                                  unsigned int* d_free_bonds,
                                  const unsigned int* d_members,
                                  const unsigned int num_members,
                                  const unsigned int group_id,
                                  const unsigned int max_bonds,
                                  const unsigned int bond_type,
                                  const group_storage<2>* d_bond_table,
                                  const unsigned int* d_n_bonds,
                                  const Index2D bond_table_indexer)
    {
    const unsigned int member = blockIdx.x * blockDim.x + threadIdx.x;
    if (member >= num_members)
        return;

    const unsigned int idx = d_members[member];
    d_group_ids[idx] = group_id;
    d_free_bonds[idx] = detail::countFreeBonds(idx,
                                               max_bonds,
                                               bond_type,
                                               d_bond_table,
                                               d_n_bonds,
                                               bond_table_indexer);
    }

/*!
 * \param d_counts Counts in entries 1 through N, with entry 0 set to zero (output, or nullptr)
 * \param d_candidates Compact array of candidates (output, or nullptr)
 * \param d_offsets First index of each particle's candidates in \a d_candidates
 * \param find_candidate Finder for candidate bonds
 * \param d_n_neigh Number of neighbors of each particle
 * \param d_nlist Neighbor list
 * \param d_head_list Start of each particle's neighbors in \a d_nlist
 * \param N Number of local particles
 *
 * One thread is used per local particle. The candidates are counted if \a d_counts is not
 * null, and they are written if \a d_candidates is not null. The finder is deterministic, so the
 * same candidates are found when they are counted and when they are written.
 */
__global__ void find_bond_candidates(size_t* d_counts,
                                     detail::BondCandidate* d_candidates,
                                     const size_t* d_offsets,
                                     const detail::BondCandidateFinder find_candidate,
                                     const unsigned int* d_n_neigh,
                                     const unsigned int* d_nlist,
                                     const size_t* d_head_list,
                                     const unsigned int N)
    {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;

    unsigned int n_candidates = 0;
    if (find_candidate.canBond(i))
        {
        const size_t head = d_head_list[i];
        const unsigned int n_neigh = d_n_neigh[i];
        for (unsigned int neigh = 0; neigh < n_neigh; ++neigh)
            {
            detail::BondCandidate candidate;
            if (find_candidate(candidate, i, d_nlist[head + neigh]))
                {
                if (d_candidates)
                    {
                    d_candidates[d_offsets[i] + n_candidates] = candidate;
                    }
                ++n_candidates;
                }
            }
        }

    if (d_counts)
        {
        d_counts[static_cast<size_t>(i) + 1] = n_candidates;
        if (i == 0)
            {
            d_counts[0] = 0;
            }
        }
    }
    } // end namespace kernel

/*!
 * \param d_group_ids Group of each local particle
 * \param d_free_bonds Number of free bonds of each local particle
 * \param N Number of local particles
 */
cudaError_t
zero_bonding_state(unsigned int* d_group_ids, unsigned int* d_free_bonds, const unsigned int N)
    {
    cudaError_t result = cudaMemset(d_group_ids, 0, sizeof(unsigned int) * N);
    if (result != cudaSuccess)
        {
        return result;
        }
    return cudaMemset(d_free_bonds, 0, sizeof(unsigned int) * N);
    }

/*!
 * \param d_group_ids Group of each local particle
 * \param d_free_bonds Number of free bonds of each local particle
 * \param d_members Indexes of the particles in the group
 * \param num_members Number of particles in the group
 * \param group_id Group of the particles (1 or 2)
 * \param max_bonds Maximum number of bonds of the particles
 * \param bond_type Type of bonds to count
 * \param d_bond_table Bonds of each particle
 * \param d_n_bonds Number of bonds of each particle
 * \param bond_table_indexer Indexer into \a d_bond_table
 * \param block_size Number of threads per block
 */
cudaError_t set_bonding_state(unsigned int* d_group_ids,
                              unsigned int* d_free_bonds,
                              const unsigned int* d_members,
                              const unsigned int num_members,
                              const unsigned int group_id,
                              const unsigned int max_bonds,
                              const unsigned int bond_type,
                              const group_storage<2>* d_bond_table,
                              const unsigned int* d_n_bonds,
                              const Index2D& bond_table_indexer,
                              const unsigned int block_size)
    {
    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)kernel::set_bonding_state);
    max_block_size = attr.maxThreadsPerBlock;

    const unsigned int run_block_size = min(block_size, max_block_size);
    kernel::set_bonding_state<<<num_members / run_block_size + 1, run_block_size>>>(
        d_group_ids,
        d_free_bonds,
        d_members,
        num_members,
        group_id,
        max_bonds,
        bond_type,
        d_bond_table,
        d_n_bonds,
        bond_table_indexer);

    return cudaSuccess;
    }

/*!
 * \param d_offsets Array of N+1 entries to fill with a zero followed by the candidate counts
 * \param find_candidate Finder for candidate bonds
 * \param d_n_neigh Number of neighbors of each particle
 * \param d_nlist Neighbor list
 * \param d_head_list Start of each particle's neighbors in \a d_nlist
 * \param N Number of local particles
 * \param block_size Number of threads per block
 */
cudaError_t count_bond_candidates(size_t* d_offsets,
                                  const detail::BondCandidateFinder& find_candidate,
                                  const unsigned int* d_n_neigh,
                                  const unsigned int* d_nlist,
                                  const size_t* d_head_list,
                                  const unsigned int N,
                                  const unsigned int block_size)
    {
    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)kernel::find_bond_candidates);
    max_block_size = attr.maxThreadsPerBlock;

    const unsigned int run_block_size = min(block_size, max_block_size);
    kernel::find_bond_candidates<<<N / run_block_size + 1, run_block_size>>>(d_offsets,
                                                                             nullptr,
                                                                             nullptr,
                                                                             find_candidate,
                                                                             d_n_neigh,
                                                                             d_nlist,
                                                                             d_head_list,
                                                                             N);

    return cudaSuccess;
    }

/*!
 * \param d_offsets Array of N+1 counts, overwritten with offsets and the total in entry N
 * \param N Number of local particles
 * \param alloc Allocator for temporary memory
 * \returns Total number of candidates
 */
size_t scan_bond_candidates(size_t* d_offsets, const unsigned int N, CachedAllocator& alloc)
    {
    if (N == 0)
        {
        return 0;
        }

    thrust::device_ptr<size_t> offsets(d_offsets);
    thrust::inclusive_scan(thrust::cuda::par(alloc), offsets + 1, offsets + N + 1, offsets + 1);
    return offsets[N];
    }

/*!
 * \param d_candidates Compact array of candidates
 * \param d_offsets First index of each particle's candidates in \a d_candidates
 * \param find_candidate Finder for candidate bonds
 * \param d_n_neigh Number of neighbors of each particle
 * \param d_nlist Neighbor list
 * \param d_head_list Start of each particle's neighbors in \a d_nlist
 * \param N Number of local particles
 * \param block_size Number of threads per block
 */
cudaError_t write_bond_candidates(detail::BondCandidate* d_candidates,
                                  const size_t* d_offsets,
                                  const detail::BondCandidateFinder& find_candidate,
                                  const unsigned int* d_n_neigh,
                                  const unsigned int* d_nlist,
                                  const size_t* d_head_list,
                                  const unsigned int N,
                                  const unsigned int block_size)
    {
    unsigned int max_block_size;
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, (const void*)kernel::find_bond_candidates);
    max_block_size = attr.maxThreadsPerBlock;

    const unsigned int run_block_size = min(block_size, max_block_size);
    kernel::find_bond_candidates<<<N / run_block_size + 1, run_block_size>>>(nullptr,
                                                                             d_candidates,
                                                                             d_offsets,
                                                                             find_candidate,
                                                                             d_n_neigh,
                                                                             d_nlist,
                                                                             d_head_list,
                                                                             N);

    return cudaSuccess;
    }

/*!
 * \param d_candidates Candidate bonds
 * \param num_candidates Number of candidate bonds
 * \param alloc Allocator for temporary memory
 */
cudaError_t sort_bond_candidates(detail::BondCandidate* d_candidates,
                                 const size_t num_candidates,
                                 CachedAllocator& alloc)
    {
    thrust::device_ptr<detail::BondCandidate> candidates(d_candidates);
    thrust::sort(thrust::cuda::par(alloc),
                 candidates,
                 candidates + num_candidates,
                 detail::CompareBondCandidatesByDistance());

    return cudaSuccess;
    }

    } // end namespace gpu
    } // end namespace azplugins
    } // end namespace hoomd
