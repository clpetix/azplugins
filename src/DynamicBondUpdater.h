// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file DynamicBondUpdater.h
 * \brief Declaration of DynamicBondUpdater
 */

#ifndef AZPLUGINS_DYNAMIC_BOND_UPDATER_H_
#define AZPLUGINS_DYNAMIC_BOND_UPDATER_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "DynamicBondCandidate.h"

#include "hoomd/GPUVector.h"
#include "hoomd/ParticleGroup.h"
#include "hoomd/Updater.h"
#include "hoomd/md/NeighborList.h"

#include <pybind11/pybind11.h>
#include <string>
#include <vector>

namespace hoomd
    {
namespace azplugins
    {

//! Form bonds dynamically between nearby particles
/*!
 * Bonds of one type are formed between particles in two groups that are closer than a cutoff
 * distance. The groups must either be identical or have no particles in common. Each particle can
 * have a maximum number of bonds of this type, which depends on its group. Bonds are formed in
 * order of increasing distance between the particles, and each pair of particles that is close
 * enough is considered for bonding with a fixed probability.
 *
 * Pairs of particles are found using a neighbor list. The updater adds its cutoff to the neighbor
 * list like a pair potential does, so the neighbor list can be shared with pair potentials. Pairs
 * of particles that are excluded from the neighbor list never form bonds.
 *
 * Each update proceeds in the following steps:
 *
 * 1. Each rank finds candidate bonds between its local particles and their neighbors.
 * 2. With MPI, the candidates are gathered on the root rank. A pair of particles owned by two
 *    different ranks is found on both ranks, and these two partial candidates are merged.
 * 3. The root rank selects the bonds to form by accepting the candidates in order of priority
 *    if both particles can still form a bond.
 * 4. With MPI, the new bonds are broadcast. All ranks add the new bonds in the same order.
 *
 * Step 3 is sequential, so it is performed on the CPU even when the other steps run on the GPU.
 */
class PYBIND11_EXPORT DynamicBondUpdater : public Updater
    {
    public:
    //! Constructor
    DynamicBondUpdater(std::shared_ptr<SystemDefinition> sysdef,
                       std::shared_ptr<Trigger> trigger,
                       std::shared_ptr<md::NeighborList> nlist,
                       std::shared_ptr<ParticleGroup> group_1,
                       std::shared_ptr<ParticleGroup> group_2,
                       const std::string& bond_type,
                       Scalar r_cut,
                       unsigned int max_bonds_1,
                       unsigned int max_bonds_2,
                       Scalar probability);

    //! Destructor
    virtual ~DynamicBondUpdater();

    //! Form new bonds
    void update(uint64_t timestep) override;

    //! Notify that the updater has been detached from the simulation
    void notifyDetach() override;

    //! Get the first group of particles
    std::shared_ptr<ParticleGroup> getGroup1() const
        {
        return m_group_1;
        }

    //! Get the second group of particles
    std::shared_ptr<ParticleGroup> getGroup2() const
        {
        return m_group_2;
        }

    //! Get the name of the type of bonds to form
    std::string getBondType() const
        {
        return m_bond_data->getNameByType(m_bond_type);
        }

    //! Set the name of the type of bonds to form
    void setBondType(const std::string& bond_type)
        {
        m_bond_type = m_bond_data->getTypeByName(bond_type);
        }

    //! Get the cutoff distance for forming bonds
    Scalar getRCut() const
        {
        return m_r_cut;
        }

    //! Set the cutoff distance for forming bonds
    void setRCut(Scalar r_cut);

    //! Get the maximum number of bonds of particles in the first group
    unsigned int getMaxBonds1() const
        {
        return m_max_bonds_1;
        }

    //! Set the maximum number of bonds of particles in the first group
    void setMaxBonds1(unsigned int max_bonds)
        {
        checkMaxBonds(max_bonds);
        m_max_bonds_1 = max_bonds;
        }

    //! Get the maximum number of bonds of particles in the second group
    unsigned int getMaxBonds2() const
        {
        return m_max_bonds_2;
        }

    //! Set the maximum number of bonds of particles in the second group
    void setMaxBonds2(unsigned int max_bonds)
        {
        checkMaxBonds(max_bonds);
        m_max_bonds_2 = max_bonds;
        }

    //! Get the probability to form a bond
    Scalar getProbability() const
        {
        return m_probability;
        }

    //! Set the probability to form a bond
    void setProbability(Scalar probability);

    protected:
    std::shared_ptr<md::NeighborList> m_nlist; //!< Neighbor list to find pairs of particles
    std::shared_ptr<ParticleGroup> m_group_1;  //!< First group of particles that form bonds
    std::shared_ptr<ParticleGroup> m_group_2;  //!< Second group of particles that form bonds
    std::shared_ptr<BondData> m_bond_data;     //!< Bonds in the system
    unsigned int m_bond_type;                  //!< Type of bonds to form
    Scalar m_r_cut;                            //!< Cutoff distance for forming bonds
    unsigned int m_max_bonds_1;                //!< Maximum number of bonds in the first group
    unsigned int m_max_bonds_2;                //!< Maximum number of bonds in the second group
    Scalar m_probability;                      //!< Probability to form a bond
    bool m_groups_identical;                   //!< True if the groups have the same particles

    GPUVector<unsigned int> m_group_ids;  //!< Group (1 or 2) of each local particle, 0 if none
    GPUVector<unsigned int> m_free_bonds; //!< Number of bonds each local particle can still form
    std::vector<detail::BondCandidate> m_candidates; //!< Candidate bonds
    std::vector<uint2> m_new_bonds;                  //!< Tags of particles in new bonds

    //! Compute the group and number of free bonds of each local particle
    virtual void computeBondingState();

    //! Find candidate bonds, sorted by priority when not domain decomposed
    virtual void findBondCandidates(uint64_t timestep);

    private:
    std::shared_ptr<GPUArray<Scalar>> m_r_cut_nlist; //!< Cutoff matrix for the neighbor list
    bool m_attached;                                 //!< True if the updater is attached
    bool m_particles_changed;                        //!< True if the number of particles changed
    std::vector<unsigned int> m_num_new_bonds; //!< Number of new bonds of each particle (by tag)

#ifdef ENABLE_MPI
    std::shared_ptr<Communicator> m_comm; //!< Communicator

    //! Gather candidate bonds from all ranks on the root rank
    void gatherBondCandidates();
#endif // ENABLE_MPI

    //! Select bonds to form from the candidates
    void selectBonds();

    //! Form the selected bonds
    void formBonds(uint64_t timestep);

    //! Determine if the groups are identical or have no particles in common
    void checkGroups();

    //! Check that a maximum number of bonds is valid
    void checkMaxBonds(unsigned int max_bonds) const;

    //! Set the cutoff for each pair of particle types in the neighbor list
    void updateRCutMatrix();

    //! Notification that the global number of particles changed
    void slotGlobalParticleNumberChange()
        {
        m_particles_changed = true;
        }
    };

namespace detail
    {
//! Export DynamicBondUpdater to python
void export_DynamicBondUpdater(pybind11::module& m);
    } // end namespace detail

    } // end namespace azplugins
    } // end namespace hoomd

#endif // AZPLUGINS_DYNAMIC_BOND_UPDATER_H_
