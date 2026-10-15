// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file DynamicBondUpdater.cc
 * \brief Definition of DynamicBondUpdater
 */

#include "DynamicBondUpdater.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

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
DynamicBondUpdater::DynamicBondUpdater(std::shared_ptr<SystemDefinition> sysdef,
                                       std::shared_ptr<Trigger> trigger,
                                       std::shared_ptr<md::NeighborList> nlist,
                                       std::shared_ptr<ParticleGroup> group_1,
                                       std::shared_ptr<ParticleGroup> group_2,
                                       const std::string& bond_type,
                                       Scalar r_cut,
                                       unsigned int max_bonds_1,
                                       unsigned int max_bonds_2,
                                       Scalar probability)
    : Updater(sysdef, trigger), m_nlist(nlist), m_group_1(group_1), m_group_2(group_2),
      m_bond_data(sysdef->getBondData()), m_bond_type(0), m_r_cut(0), m_max_bonds_1(0),
      m_max_bonds_2(0), m_probability(0), m_groups_identical(false), m_group_ids(m_exec_conf),
      m_free_bonds(m_exec_conf), m_attached(true), m_particles_changed(false)
    {
    m_exec_conf->msg->notice(5) << "Constructing DynamicBondUpdater" << std::endl;

    setBondType(bond_type);
    setMaxBonds1(max_bonds_1);
    setMaxBonds2(max_bonds_2);
    setProbability(probability);
    checkGroups();

    // the neighbor list must include all pairs of particles within the cutoff
    m_r_cut_nlist
        = std::make_shared<GPUArray<Scalar>>(m_nlist->getTypePairIndexer().getNumElements(),
                                             m_exec_conf);
    setRCut(r_cut);
    m_nlist->addRCutMatrix(m_r_cut_nlist);

    m_pdata->getGlobalParticleNumberChangeSignal()
        .connect<DynamicBondUpdater, &DynamicBondUpdater::slotGlobalParticleNumberChange>(this);

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        auto comm_weak = m_sysdef->getCommunicator();
        assert(comm_weak.lock());
        m_comm = comm_weak.lock();
        }
#endif // ENABLE_MPI
    }

DynamicBondUpdater::~DynamicBondUpdater()
    {
    m_exec_conf->msg->notice(5) << "Destroying DynamicBondUpdater" << std::endl;

    notifyDetach();

    m_pdata->getGlobalParticleNumberChangeSignal()
        .disconnect<DynamicBondUpdater, &DynamicBondUpdater::slotGlobalParticleNumberChange>(this);
    }

/*!
 * \param timestep Current timestep
 */
void DynamicBondUpdater::update(uint64_t timestep)
    {
    // groups and the particle types in them can change with the number of particles
    if (m_particles_changed)
        {
        checkGroups();
        updateRCutMatrix();
        m_particles_changed = false;
        }

    if (m_groups_identical && m_max_bonds_1 != m_max_bonds_2)
        {
        throw std::invalid_argument("DynamicBondUpdater: max_bonds_1 and max_bonds_2 must be "
                                    "equal when the groups contain the same particles.");
        }

    // These conditions are global, so all ranks skip the update together.
    if (m_probability == Scalar(0.0) || m_group_1->getNumMembersGlobal() == 0
        || m_group_2->getNumMembersGlobal() == 0)
        {
        return;
        }

    m_nlist->compute(timestep);
    computeBondingState();
    findBondCandidates(timestep);

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        gatherBondCandidates();
        }
#endif // ENABLE_MPI

    // bonds are selected on the root rank, then shared with all ranks
    m_new_bonds.clear();
    if (m_exec_conf->getRank() == 0)
        {
        selectBonds();
        }

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        bcast(m_new_bonds, 0, m_exec_conf->getMPICommunicator());
        }
#endif // ENABLE_MPI

    formBonds(timestep);
    }

void DynamicBondUpdater::notifyDetach()
    {
    if (m_attached)
        {
        m_nlist->removeRCutMatrix(m_r_cut_nlist);
        m_nlist->notifyRCutMatrixChange();
        }
    m_attached = false;
    }

/*!
 * \param r_cut Cutoff distance for forming bonds
 */
void DynamicBondUpdater::setRCut(Scalar r_cut)
    {
    if (!std::isfinite(r_cut) || r_cut <= Scalar(0.0))
        {
        throw std::invalid_argument("DynamicBondUpdater: r_cut must be finite and positive.");
        }
    m_r_cut = r_cut;
    updateRCutMatrix();
    }

/*!
 * \param probability Probability to form a bond
 */
void DynamicBondUpdater::setProbability(Scalar probability)
    {
    if (!(probability >= Scalar(0.0) && probability <= Scalar(1.0)))
        {
        throw std::invalid_argument("DynamicBondUpdater: probability must be between 0 and 1.");
        }
    m_probability = probability;
    }

/*!
 * The group of each local particle is 1 or 2, or 0 if it is not in either group. When the groups
 * are identical, all particles in them are assigned to group 1. The number of free bonds is the
 * number of additional bonds of the type being formed that the particle can have, which is zero
 * for particles that are not in a group.
 */
void DynamicBondUpdater::computeBondingState()
    {
    const unsigned int N = m_pdata->getN();
    m_group_ids.resize(N);
    m_free_bonds.resize(N);
    if (N == 0)
        {
        return;
        }

    ArrayHandle<unsigned int> h_group_ids(m_group_ids,
                                          access_location::host,
                                          access_mode::overwrite);
    ArrayHandle<unsigned int> h_free_bonds(m_free_bonds,
                                           access_location::host,
                                           access_mode::overwrite);
    std::fill(h_group_ids.data, h_group_ids.data + N, 0);
    std::fill(h_free_bonds.data, h_free_bonds.data + N, 0);

    ArrayHandle<BondData::members_t> h_bond_table(m_bond_data->getGPUTable(),
                                                  access_location::host,
                                                  access_mode::read);
    ArrayHandle<unsigned int> h_n_bonds(m_bond_data->getNGroupsArray(),
                                        access_location::host,
                                        access_mode::read);
    const Index2D& bond_table_indexer = m_bond_data->getGPUTableIndexer();

    auto set_group = [&](ParticleGroup& group, unsigned int group_id, unsigned int max_bonds)
    {
        const unsigned int num_members = group.getNumMembers();
        ArrayHandle<unsigned int> h_members(group.getIndexArray(),
                                            access_location::host,
                                            access_mode::read);
        for (unsigned int member = 0; member < num_members; ++member)
            {
            const unsigned int idx = h_members.data[member];
            h_group_ids.data[idx] = group_id;
            h_free_bonds.data[idx] = detail::countFreeBonds(idx,
                                                            max_bonds,
                                                            m_bond_type,
                                                            h_bond_table.data,
                                                            h_n_bonds.data,
                                                            bond_table_indexer);
            }
    };

    set_group(*m_group_1, 1, m_max_bonds_1);
    if (!m_groups_identical)
        {
        set_group(*m_group_2, 2, m_max_bonds_2);
        }
    }

/*!
 * \param timestep Current timestep
 *
 * Each local particle that can form bonds is tested against its neighbors. The candidates are
 * sorted by priority on one rank. With domain decomposition, sorting is deferred until the
 * candidates have been gathered and merged.
 */
void DynamicBondUpdater::findBondCandidates(uint64_t timestep)
    {
    m_candidates.clear();
    const unsigned int N = m_pdata->getN();
    if (N == 0)
        {
        return;
        }

    ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(),
                                        access_location::host,
                                        access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(),
                                      access_location::host,
                                      access_mode::read);
    ArrayHandle<size_t> h_head_list(m_nlist->getHeadList(),
                                    access_location::host,
                                    access_mode::read);

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_group_ids(m_group_ids, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_free_bonds(m_free_bonds, access_location::host, access_mode::read);

    ArrayHandle<BondData::members_t> h_bond_table(m_bond_data->getGPUTable(),
                                                  access_location::host,
                                                  access_mode::read);
    ArrayHandle<unsigned int> h_n_bonds(m_bond_data->getNGroupsArray(),
                                        access_location::host,
                                        access_mode::read);

    const detail::BondCandidateFinder find_candidate(h_pos.data,
                                                     h_tag.data,
                                                     h_group_ids.data,
                                                     h_free_bonds.data,
                                                     h_bond_table.data,
                                                     h_n_bonds.data,
                                                     m_bond_data->getGPUTableIndexer(),
                                                     m_pdata->getBox(),
                                                     m_r_cut,
                                                     m_probability,
                                                     m_groups_identical,
                                                     N,
                                                     timestep,
                                                     m_sysdef->getSeed());

    for (unsigned int i = 0; i < N; ++i)
        {
        if (!find_candidate.canBond(i))
            {
            continue;
            }

        const size_t head = h_head_list.data[i];
        const unsigned int n_neigh = h_n_neigh.data[i];
        for (unsigned int neigh = 0; neigh < n_neigh; ++neigh)
            {
            detail::BondCandidate candidate;
            if (find_candidate(candidate, i, h_nlist.data[head + neigh]))
                {
                m_candidates.push_back(candidate);
                }
            }
        }

    if (!m_sysdef->isDomainDecomposed())
        {
        std::sort(m_candidates.begin(),
                  m_candidates.end(),
                  detail::CompareBondCandidatesByDistance());
        }
    }

#ifdef ENABLE_MPI
/*!
 * A pair of particles owned by two different ranks is found on both ranks, but each rank only
 * knows the number of free bonds of its own particle. These partial candidates are merged on the
 * root rank. A candidate is discarded if the number of free bonds of either particle is still not
 * known, which means that the rank owning that particle did not find it could form a bond.
 */
void DynamicBondUpdater::gatherBondCandidates()
    {
    const MPI_Comm mpi_comm = m_exec_conf->getMPICommunicator();
    const unsigned int n_ranks = m_exec_conf->getNRanks();
    const bool is_root = (m_exec_conf->getRank() == 0);

    // Check counts on every rank before narrowing to the int counts used by MPI_Gatherv.
    // All ranks must fail together if the gathered array is too large.
    const uint64_t n_candidates = m_candidates.size();
    std::vector<uint64_t> counts(n_ranks);
    MPI_Allgather(&n_candidates, 1, MPI_UINT64_T, counts.data(), 1, MPI_UINT64_T, mpi_comm);

    std::vector<int> n_candidates_per_rank(n_ranks);
    std::vector<int> offsets(n_ranks);
    uint64_t total = 0;
    for (unsigned int rank = 0; rank < n_ranks; ++rank)
        {
        if (counts[rank] > std::numeric_limits<int>::max() - total)
            {
            throw std::overflow_error("DynamicBondUpdater: too many candidates for MPI_Gatherv.");
            }
        offsets[rank] = static_cast<int>(total);
        n_candidates_per_rank[rank] = static_cast<int>(counts[rank]);
        total += counts[rank];
        }

    if (total == 0)
        {
        return;
        }
    if (is_root)
        {
        // Rank 0's candidates already occupy the first receive segment. Gather in place to
        // keep a single candidate buffer on root and reuse its capacity on the next update.
        m_candidates.resize(total);
        }

    // Candidates are plain data, so they can be sent as bytes. The datatype keeps MPI counts
    // in units of candidates instead of bytes.
    MPI_Datatype mpi_candidate;
    MPI_Type_contiguous(sizeof(detail::BondCandidate), MPI_BYTE, &mpi_candidate);
    MPI_Type_commit(&mpi_candidate);

    MPI_Gatherv(is_root ? MPI_IN_PLACE : m_candidates.data(),
                static_cast<int>(n_candidates),
                mpi_candidate,
                is_root ? m_candidates.data() : nullptr,
                n_candidates_per_rank.data(),
                offsets.data(),
                mpi_candidate,
                0,
                mpi_comm);
    MPI_Type_free(&mpi_candidate);

    if (!is_root)
        {
        m_candidates.clear();
        return;
        }

    // partial candidates for the same pair are adjacent after sorting by tags
    std::sort(m_candidates.begin(), m_candidates.end(), detail::CompareBondCandidatesByTags());
    auto first = m_candidates.begin();
    auto output = first;
    while (first != m_candidates.end())
        {
        detail::BondCandidate merged = *first;
        auto next = first + 1;
        while (next != m_candidates.end() && next->tag_1 == merged.tag_1
               && next->tag_2 == merged.tag_2)
            {
            // A known capacity is positive, so it replaces the zero for a remote particle.
            merged.free_bonds_1 = std::max(merged.free_bonds_1, next->free_bonds_1);
            merged.free_bonds_2 = std::max(merged.free_bonds_2, next->free_bonds_2);
            merged.r_sq = std::min(merged.r_sq, next->r_sq);
            ++next;
            }

        if (merged.free_bonds_1 != detail::UNKNOWN_FREE_BONDS
            && merged.free_bonds_2 != detail::UNKNOWN_FREE_BONDS)
            {
            *output++ = merged;
            }
        first = next;
        }

    m_candidates.erase(output, m_candidates.end());
    std::sort(m_candidates.begin(), m_candidates.end(), detail::CompareBondCandidatesByDistance());
    }
#endif // ENABLE_MPI

/*!
 * The candidates are accepted in order of priority if both particles can still form a bond,
 * including the bonds that have already been accepted in this update.
 */
void DynamicBondUpdater::selectBonds()
    {
    if (m_candidates.empty())
        {
        return;
        }

    m_num_new_bonds.resize(m_pdata->getRTags().size(), 0);
    for (const auto& candidate : m_candidates)
        {
        unsigned int& num_new_bonds_1 = m_num_new_bonds[candidate.tag_1];
        unsigned int& num_new_bonds_2 = m_num_new_bonds[candidate.tag_2];
        if (num_new_bonds_1 < candidate.free_bonds_1 && num_new_bonds_2 < candidate.free_bonds_2)
            {
            m_new_bonds.push_back(make_uint2(candidate.tag_1, candidate.tag_2));
            ++num_new_bonds_1;
            ++num_new_bonds_2;
            }
        }

    // reset the counts for the next update
    for (const auto& bond : m_new_bonds)
        {
        m_num_new_bonds[bond.x] = 0;
        m_num_new_bonds[bond.y] = 0;
        }
    }

/*!
 * \param timestep Current timestep
 *
 * All ranks add the new bonds in the same order so that the bonds get the same tags on all ranks.
 */
void DynamicBondUpdater::formBonds(uint64_t timestep)
    {
    for (const auto& bond : m_new_bonds)
        {
        m_bond_data->addBondedGroup(Bond(m_bond_type, bond.x, bond.y));
        }

#ifdef ENABLE_MPI
    // the members of new bonds may need to be communicated as ghost particles
    if (m_sysdef->isDomainDecomposed() && !m_new_bonds.empty())
        {
        m_comm->forceMigrate();
        m_comm->communicate(timestep);
        }
#endif // ENABLE_MPI
    }

void DynamicBondUpdater::checkGroups()
    {
    // Identical filters normally share a ParticleGroup, including when that group is empty.
    if (m_group_1 == m_group_2)
        {
        m_groups_identical = true;
        return;
        }

    const unsigned int n_common = m_group_1->intersectionSize(m_group_2);
    if (n_common == 0)
        {
        m_groups_identical = false;
        }
    else if (n_common == m_group_1->getNumMembersGlobal()
             && n_common == m_group_2->getNumMembersGlobal())
        {
        m_groups_identical = true;
        }
    else
        {
        throw std::invalid_argument("DynamicBondUpdater: groups must contain the same particles "
                                    "or have no particles in common.");
        }
    }

/*!
 * \param max_bonds Maximum number of bonds
 */
void DynamicBondUpdater::checkMaxBonds(unsigned int max_bonds) const
    {
    if (max_bonds == 0)
        {
        throw std::invalid_argument(
            "DynamicBondUpdater: maximum number of bonds must be positive.");
        }
    }

/*!
 * The cutoff is set for each pair of particle types where one type is in the first group and the
 * other type is in the second group. The cutoff is zero for all other pairs of types.
 */
void DynamicBondUpdater::updateRCutMatrix()
    {
    const unsigned int n_types = m_pdata->getNTypes();
    auto find_types = [&](ParticleGroup& group)
    {
        std::vector<unsigned int> has_type(n_types, 0);
        const unsigned int num_members = group.getNumMembers();
        ArrayHandle<unsigned int> h_members(group.getIndexArray(),
                                            access_location::host,
                                            access_mode::read);
        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::read);
        for (unsigned int member = 0; member < num_members; ++member)
            {
            const Scalar4 postype = h_pos.data[h_members.data[member]];
            has_type[__scalar_as_int(postype.w)] = 1;
            }
        return has_type;
    };

    // the groups may be the same object, so their arrays are accessed one at a time
    std::vector<unsigned int> has_type_1 = find_types(*m_group_1);
    std::vector<unsigned int> has_type_2;
    if (!m_groups_identical)
        {
        has_type_2 = find_types(*m_group_2);
        }

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        MPI_Allreduce(MPI_IN_PLACE,
                      has_type_1.data(),
                      n_types,
                      MPI_UNSIGNED,
                      MPI_MAX,
                      m_exec_conf->getMPICommunicator());
        if (!m_groups_identical)
            {
            MPI_Allreduce(MPI_IN_PLACE,
                          has_type_2.data(),
                          n_types,
                          MPI_UNSIGNED,
                          MPI_MAX,
                          m_exec_conf->getMPICommunicator());
            }
        }
#endif // ENABLE_MPI

    if (m_groups_identical)
        {
        has_type_2 = has_type_1;
        }

        {
        const Index2D& typpair_idx = m_nlist->getTypePairIndexer();
        ArrayHandle<Scalar> h_r_cut_nlist(*m_r_cut_nlist,
                                          access_location::host,
                                          access_mode::overwrite);
        for (unsigned int i = 0; i < n_types; ++i)
            {
            for (unsigned int j = 0; j < n_types; ++j)
                {
                const bool can_bond
                    = (has_type_1[i] && has_type_2[j]) || (has_type_1[j] && has_type_2[i]);
                h_r_cut_nlist.data[typpair_idx(i, j)] = (can_bond) ? m_r_cut : Scalar(0.0);
                }
            }
        }
    m_nlist->notifyRCutMatrixChange();
    }

namespace detail
    {
/*!
 * \param m Python module to export to
 */
void export_DynamicBondUpdater(pybind11::module& m)
    {
    pybind11::class_<DynamicBondUpdater, Updater, std::shared_ptr<DynamicBondUpdater>>(
        m,
        "DynamicBondUpdater")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<Trigger>,
                            std::shared_ptr<md::NeighborList>,
                            std::shared_ptr<ParticleGroup>,
                            std::shared_ptr<ParticleGroup>,
                            const std::string&,
                            Scalar,
                            unsigned int,
                            unsigned int,
                            Scalar>())
        .def_property_readonly("filter_1",
                               [](const DynamicBondUpdater& self)
                               { return self.getGroup1()->getFilter(); })
        .def_property_readonly("filter_2",
                               [](const DynamicBondUpdater& self)
                               { return self.getGroup2()->getFilter(); })
        .def_property("bond_type",
                      &DynamicBondUpdater::getBondType,
                      &DynamicBondUpdater::setBondType)
        .def_property("r_cut", &DynamicBondUpdater::getRCut, &DynamicBondUpdater::setRCut)
        .def_property("max_bonds_1",
                      &DynamicBondUpdater::getMaxBonds1,
                      &DynamicBondUpdater::setMaxBonds1)
        .def_property("max_bonds_2",
                      &DynamicBondUpdater::getMaxBonds2,
                      &DynamicBondUpdater::setMaxBonds2)
        .def_property("probability",
                      &DynamicBondUpdater::getProbability,
                      &DynamicBondUpdater::setProbability);
    }
    } // end namespace detail

    } // end namespace azplugins
    } // end namespace hoomd
