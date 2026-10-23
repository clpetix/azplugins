// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file DynamicBondUpdaterGPU.cc
 * \brief Definition of DynamicBondUpdaterGPU
 */

#include "DynamicBondUpdaterGPU.h"

namespace hoomd
    {
namespace azplugins
    {

/*!
 * \param sysdef System definition
 * \param trigger Trigger to run the updater
 * \param nlist Neighbor list used to find pairs of particles
 * \param group_1 First group of particles that form bonds
 * \param group_2 Second group of particles that form bonds
 * \param bond_type Name of the type of bonds to form
 * \param r_cut Cutoff distance for forming bonds
 * \param max_bonds_1 Maximum number of bonds of particles in the first group
 * \param max_bonds_2 Maximum number of bonds of particles in the second group
 * \param probability Probability to form a bond
 */
DynamicBondUpdaterGPU::DynamicBondUpdaterGPU(std::shared_ptr<SystemDefinition> sysdef,
                                             std::shared_ptr<Trigger> trigger,
                                             std::shared_ptr<md::NeighborList> nlist,
                                             std::shared_ptr<ParticleGroup> group_1,
                                             std::shared_ptr<ParticleGroup> group_2,
                                             const std::string& bond_type,
                                             Scalar r_cut,
                                             unsigned int max_bonds_1,
                                             unsigned int max_bonds_2,
                                             Scalar probability)
    : DynamicBondUpdater(sysdef,
                         trigger,
                         nlist,
                         group_1,
                         group_2,
                         bond_type,
                         r_cut,
                         max_bonds_1,
                         max_bonds_2,
                         probability),
      m_candidate_offsets(m_exec_conf), m_gpu_candidates(m_exec_conf)
    {
    m_tuner_state.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                         m_exec_conf,
                                         "dynamic_bond_state"));
    m_tuner_count.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                         m_exec_conf,
                                         "dynamic_bond_count"));
    m_tuner_write.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                         m_exec_conf,
                                         "dynamic_bond_write"));
    m_autotuners.insert(m_autotuners.end(), {m_tuner_state, m_tuner_count, m_tuner_write});
    }

void DynamicBondUpdaterGPU::computeBondingState()
    {
    const unsigned int N = m_pdata->getN();
    m_group_ids.resize(N);
    m_free_bonds.resize(N);
    if (N == 0)
        {
        return;
        }

    ArrayHandle<unsigned int> d_group_ids(m_group_ids,
                                          access_location::device,
                                          access_mode::overwrite);
    ArrayHandle<unsigned int> d_free_bonds(m_free_bonds,
                                           access_location::device,
                                           access_mode::overwrite);
    gpu::zero_bonding_state(d_group_ids.data, d_free_bonds.data, N);
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    ArrayHandle<BondData::members_t> d_bond_table(m_bond_data->getGPUTable(),
                                                  access_location::device,
                                                  access_mode::read);
    ArrayHandle<unsigned int> d_n_bonds(m_bond_data->getNGroupsArray(),
                                        access_location::device,
                                        access_mode::read);
    const Index2D& bond_table_indexer = m_bond_data->getGPUTableIndexer();

    auto set_group = [&](ParticleGroup& group, unsigned int group_id, unsigned int max_bonds)
    {
        const unsigned int num_members = group.getNumMembers();
        if (num_members == 0)
            {
            return;
            }

        ArrayHandle<unsigned int> d_members(group.getIndexArray(),
                                            access_location::device,
                                            access_mode::read);
        m_tuner_state->begin();
        gpu::set_bonding_state(d_group_ids.data,
                               d_free_bonds.data,
                               d_members.data,
                               num_members,
                               group_id,
                               max_bonds,
                               m_bond_type,
                               d_bond_table.data,
                               d_n_bonds.data,
                               bond_table_indexer,
                               m_tuner_state->getParam()[0]);
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_tuner_state->end();
    };

    set_group(*m_group_1, 1, m_max_bonds_1);
    if (!m_groups_identical)
        {
        set_group(*m_group_2, 2, m_max_bonds_2);
        }
    }

/*!
 * \param timestep Current timestep
 */
void DynamicBondUpdaterGPU::findBondCandidates(uint64_t timestep)
    {
    m_candidates.clear();
    const unsigned int N = m_pdata->getN();
    if (N == 0)
        {
        return;
        }

    ArrayHandle<unsigned int> d_n_neigh(m_nlist->getNNeighArray(),
                                        access_location::device,
                                        access_mode::read);
    ArrayHandle<unsigned int> d_nlist(m_nlist->getNListArray(),
                                      access_location::device,
                                      access_mode::read);
    ArrayHandle<size_t> d_head_list(m_nlist->getHeadList(),
                                    access_location::device,
                                    access_mode::read);

    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_group_ids(m_group_ids, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_free_bonds(m_free_bonds,
                                           access_location::device,
                                           access_mode::read);

    ArrayHandle<BondData::members_t> d_bond_table(m_bond_data->getGPUTable(),
                                                  access_location::device,
                                                  access_mode::read);
    ArrayHandle<unsigned int> d_n_bonds(m_bond_data->getNGroupsArray(),
                                        access_location::device,
                                        access_mode::read);

    const detail::BondCandidateFinder find_candidate(d_pos.data,
                                                     d_tag.data,
                                                     d_group_ids.data,
                                                     d_free_bonds.data,
                                                     d_bond_table.data,
                                                     d_n_bonds.data,
                                                     m_bond_data->getGPUTableIndexer(),
                                                     m_pdata->getBox(),
                                                     m_r_cut,
                                                     m_probability,
                                                     m_groups_identical,
                                                     N,
                                                     timestep,
                                                     m_sysdef->getSeed());

    // count the candidates of each particle to find where to write them
    m_candidate_offsets.resize(static_cast<size_t>(N) + 1);
    size_t num_candidates = 0;
        {
        ArrayHandle<size_t> d_offsets(m_candidate_offsets,
                                      access_location::device,
                                      access_mode::overwrite);
        m_tuner_count->begin();
        gpu::count_bond_candidates(d_offsets.data,
                                   find_candidate,
                                   d_n_neigh.data,
                                   d_nlist.data,
                                   d_head_list.data,
                                   N,
                                   m_tuner_count->getParam()[0]);
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_tuner_count->end();

        num_candidates
            = gpu::scan_bond_candidates(d_offsets.data, N, m_exec_conf->getCachedAllocator());
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    if (num_candidates == 0)
        {
        return;
        }

    // write the candidates to a compact array
    m_gpu_candidates.resize(num_candidates);
        {
        ArrayHandle<detail::BondCandidate> d_candidates(m_gpu_candidates,
                                                        access_location::device,
                                                        access_mode::overwrite);
        ArrayHandle<size_t> d_offsets(m_candidate_offsets,
                                      access_location::device,
                                      access_mode::read);
        m_tuner_write->begin();
        gpu::write_bond_candidates(d_candidates.data,
                                   d_offsets.data,
                                   find_candidate,
                                   d_n_neigh.data,
                                   d_nlist.data,
                                   d_head_list.data,
                                   N,
                                   m_tuner_write->getParam()[0]);
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_tuner_write->end();

        // MPI needs to merge partial candidates before sorting by priority.
        if (!m_sysdef->isDomainDecomposed())
            {
            gpu::sort_bond_candidates(d_candidates.data,
                                      num_candidates,
                                      m_exec_conf->getCachedAllocator());
            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }
        }

    // the bonds are selected on the CPU
    ArrayHandle<detail::BondCandidate> h_candidates(m_gpu_candidates,
                                                    access_location::host,
                                                    access_mode::read);
    m_candidates.assign(h_candidates.data, h_candidates.data + num_candidates);
    }

namespace detail
    {
/*!
 * \param m Python module to export to
 */
void export_DynamicBondUpdaterGPU(pybind11::module& m)
    {
    pybind11::class_<DynamicBondUpdaterGPU,
                     DynamicBondUpdater,
                     std::shared_ptr<DynamicBondUpdaterGPU>>(m, "DynamicBondUpdaterGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<Trigger>,
                            std::shared_ptr<md::NeighborList>,
                            std::shared_ptr<ParticleGroup>,
                            std::shared_ptr<ParticleGroup>,
                            const std::string&,
                            Scalar,
                            unsigned int,
                            unsigned int,
                            Scalar>());
    }
    } // end namespace detail

    } // end namespace azplugins
    } // end namespace hoomd
