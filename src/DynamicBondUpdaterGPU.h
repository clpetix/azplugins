// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file DynamicBondUpdaterGPU.h
 * \brief Declaration of DynamicBondUpdaterGPU
 */

#ifndef AZPLUGINS_DYNAMIC_BOND_UPDATER_GPU_H_
#define AZPLUGINS_DYNAMIC_BOND_UPDATER_GPU_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "DynamicBondUpdater.h"
#include "DynamicBondUpdaterGPU.cuh"

#include "hoomd/Autotuner.h"

namespace hoomd
    {
namespace azplugins
    {

//! Form bonds dynamically between nearby particles on the GPU
/*!
 * See DynamicBondUpdater for details. The bonding state of the particles and the candidate bonds
 * are found on the GPU. The candidates of each particle are counted first so that they can be
 * written to a compact array, which is then sorted by priority on the GPU. The bonds are selected
 * from the sorted candidates on the CPU.
 */
class PYBIND11_EXPORT DynamicBondUpdaterGPU : public DynamicBondUpdater
    {
    public:
    //! Constructor
    DynamicBondUpdaterGPU(std::shared_ptr<SystemDefinition> sysdef,
                          std::shared_ptr<Trigger> trigger,
                          std::shared_ptr<md::NeighborList> nlist,
                          std::shared_ptr<ParticleGroup> group_1,
                          std::shared_ptr<ParticleGroup> group_2,
                          const std::string& bond_type,
                          Scalar r_cut,
                          unsigned int max_bonds_1,
                          unsigned int max_bonds_2,
                          Scalar probability);

    protected:
    //! Compute the group and number of free bonds of each local particle
    void computeBondingState() override;

    //! Find candidate bonds involving local particles, sorted by priority
    void findBondCandidates(uint64_t timestep) override;

    private:
    std::shared_ptr<Autotuner<1>> m_tuner_state; //!< Tuner for setting the bonding state
    std::shared_ptr<Autotuner<1>> m_tuner_count; //!< Tuner for counting candidates
    std::shared_ptr<Autotuner<1>> m_tuner_write; //!< Tuner for writing candidates

    GPUVector<unsigned int> m_n_candidates;      //!< Number of candidates of each particle
    GPUVector<unsigned int> m_candidate_offsets; //!< First index of each particle's candidates
    GPUVector<detail::BondCandidate> m_gpu_candidates; //!< Candidates on the GPU
    };

namespace detail
    {
//! Export DynamicBondUpdaterGPU to python
void export_DynamicBondUpdaterGPU(pybind11::module& m);
    } // end namespace detail

    } // end namespace azplugins
    } // end namespace hoomd

#endif // AZPLUGINS_DYNAMIC_BOND_UPDATER_GPU_H_
