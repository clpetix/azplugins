// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file DynamicBondUpdaterGPU.cuh
 * \brief Declaration of kernel drivers for DynamicBondUpdaterGPU
 */

#ifndef AZPLUGINS_DYNAMIC_BOND_UPDATER_GPU_CUH_
#define AZPLUGINS_DYNAMIC_BOND_UPDATER_GPU_CUH_

#include "DynamicBondCandidate.h"

#include "hoomd/CachedAllocator.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"

#include <cuda_runtime.h>

namespace hoomd
    {
namespace azplugins
    {
namespace gpu
    {

//! Set the group and number of free bonds of all local particles to zero
cudaError_t
zero_bonding_state(unsigned int* d_group_ids, unsigned int* d_free_bonds, const unsigned int N);

//! Set the group and number of free bonds of the particles in a group
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
                              const unsigned int block_size);

//! Count candidates into entries 1 through N of an N+1 offset array, with entry 0 set to zero
cudaError_t count_bond_candidates(size_t* d_offsets,
                                  const detail::BondCandidateFinder& find_candidate,
                                  const unsigned int* d_n_neigh,
                                  const unsigned int* d_nlist,
                                  const size_t* d_head_list,
                                  const unsigned int N,
                                  const unsigned int block_size);

//! Compute where the candidates of each particle start in a compact array
size_t scan_bond_candidates(size_t* d_offsets, const unsigned int N, CachedAllocator& alloc);

//! Write the candidate bonds of each local particle to a compact array
cudaError_t write_bond_candidates(detail::BondCandidate* d_candidates,
                                  const size_t* d_offsets,
                                  const detail::BondCandidateFinder& find_candidate,
                                  const unsigned int* d_n_neigh,
                                  const unsigned int* d_nlist,
                                  const size_t* d_head_list,
                                  const unsigned int N,
                                  const unsigned int block_size);

//! Sort candidate bonds by priority
cudaError_t sort_bond_candidates(detail::BondCandidate* d_candidates,
                                 const size_t num_candidates,
                                 CachedAllocator& alloc);

    } // end namespace gpu
    } // end namespace azplugins
    } // end namespace hoomd

#endif // AZPLUGINS_DYNAMIC_BOND_UPDATER_GPU_CUH_
