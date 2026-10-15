// Copyright (c) 2018-2020, Michael P. Howard
// Copyright (c) 2021-2025, Auburn University
// Part of azplugins, released under the BSD 3-Clause License.

/*!
 * \file DynamicBondCandidate.h
 * \brief Helpers to find pairs of particles that can be bonded by DynamicBondUpdater
 *
 * These helpers are shared by the CPU and GPU implementations.
 */

#ifndef AZPLUGINS_DYNAMIC_BOND_CANDIDATE_H_
#define AZPLUGINS_DYNAMIC_BOND_CANDIDATE_H_

#include "RNGIdentifiers.h"

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/RandomNumbers.h"

// group_storage is only fully defined for host code in BondedGroupData.h
#ifdef __HIPCC__
#include "hoomd/BondedGroupData.cuh"
#define HOSTDEVICE __host__ __device__
#else
#include "hoomd/BondedGroupData.h"
#define HOSTDEVICE
#endif // __HIPCC__

namespace hoomd
    {
namespace azplugins
    {
namespace detail
    {

//! Number of free bonds of a particle that is not local to the rank finding a candidate
const unsigned int UNKNOWN_FREE_BONDS = 0xffffffff;

//! Pair of particles that may be bonded
/*!
 * The particle from the first group is stored first. When the two groups are the same, the
 * particle with the smaller tag is stored first instead.
 *
 * The number of free bonds is the number of additional bonds a particle can form. It is only
 * known for particles that are local to the rank that found the candidate. Otherwise, it is
 * UNKNOWN_FREE_BONDS and must be filled in from the candidate found by the rank that owns the
 * particle.
 */
struct BondCandidate
    {
    unsigned int tag_1;        //!< Tag of the first particle
    unsigned int tag_2;        //!< Tag of the second particle
    unsigned int free_bonds_1; //!< Number of free bonds of the first particle
    unsigned int free_bonds_2; //!< Number of free bonds of the second particle
    Scalar r_sq;               //!< Squared distance between the particles
    };

//! Order candidates by distance, using the tags to break ties
struct CompareBondCandidatesByDistance
    {
    HOSTDEVICE bool operator()(const BondCandidate& a, const BondCandidate& b) const
        {
        if (a.r_sq != b.r_sq)
            {
            return a.r_sq < b.r_sq;
            }
        else if (a.tag_1 != b.tag_1)
            {
            return a.tag_1 < b.tag_1;
            }
        else
            {
            return a.tag_2 < b.tag_2;
            }
        }
    };

//! Order candidates by tags
struct CompareBondCandidatesByTags
    {
    HOSTDEVICE bool operator()(const BondCandidate& a, const BondCandidate& b) const
        {
        if (a.tag_1 != b.tag_1)
            {
            return a.tag_1 < b.tag_1;
            }
        else
            {
            return a.tag_2 < b.tag_2;
            }
        }
    };

//! Count the number of additional bonds a particle can form
/*!
 * \param idx Index of the particle
 * \param max_bonds Maximum number of bonds of type \a bond_type the particle can have
 * \param bond_type Type of bonds to count
 * \param bond_table Bonds of each particle (see BondData::getGPUTable)
 * \param n_bonds Number of bonds of each particle
 * \param bond_table_indexer Indexer into \a bond_table
 * \returns Number of bonds of type \a bond_type the particle can still form
 *
 * Bonds of other types do not count toward \a max_bonds.
 */
HOSTDEVICE inline unsigned int countFreeBonds(unsigned int idx,
                                              unsigned int max_bonds,
                                              unsigned int bond_type,
                                              const group_storage<2>* bond_table,
                                              const unsigned int* n_bonds,
                                              const Index2D& bond_table_indexer)
    {
    unsigned int num_bonds = 0;
    for (unsigned int bond = 0; bond < n_bonds[idx]; ++bond)
        {
        // the second entry in the table is the type of the bond
        if (bond_table[bond_table_indexer(idx, bond)].idx[1] == bond_type)
            {
            ++num_bonds;
            }
        }
    return (num_bonds < max_bonds) ? max_bonds - num_bonds : 0;
    }

//! Test if particles in a neighbor list can be bonded
/*!
 * Particle \a i must be local, while particle \a j can be local or a ghost. A candidate is formed
 * if the particles are in different groups (or the groups are the same), can both form more bonds
 * (ghosts are checked later by the rank that owns them), are closer than the cutoff distance, are
 * not already bonded, and pass a random test with the bond formation probability.
 *
 * The random number is drawn using the tags of the particles, so all ranks make the same
 * decision for a pair of particles independently of the domain decomposition.
 */
class BondCandidateFinder
    {
    public:
    //! Constructor
    /*!
     * \param positions Particle positions
     * \param tags Particle tags
     * \param group_ids Group of each local particle (1 or 2, or 0 if not in a group)
     * \param free_bonds Number of additional bonds each local particle can form
     * \param bond_table Bonds of each particle (see BondData::getGPUTable)
     * \param n_bonds Number of bonds of each particle
     * \param bond_table_indexer Indexer into \a bond_table
     * \param box Local simulation box
     * \param r_cut Cutoff distance for forming a bond
     * \param probability Probability to form a bond
     * \param groups_identical If true, the two groups contain the same particles
     * \param N Number of local particles
     * \param timestep Current timestep
     * \param seed Simulation seed
     */
    HOSTDEVICE BondCandidateFinder(const Scalar4* positions,
                                   const unsigned int* tags,
                                   const unsigned int* group_ids,
                                   const unsigned int* free_bonds,
                                   const group_storage<2>* bond_table,
                                   const unsigned int* n_bonds,
                                   const Index2D& bond_table_indexer,
                                   const BoxDim& box,
                                   Scalar r_cut,
                                   Scalar probability,
                                   bool groups_identical,
                                   unsigned int N,
                                   uint64_t timestep,
                                   uint16_t seed)
        : m_positions(positions), m_tags(tags), m_group_ids(group_ids), m_free_bonds(free_bonds),
          m_bond_table(bond_table), m_n_bonds(n_bonds), m_bond_table_indexer(bond_table_indexer),
          m_box(box), m_r_cut_sq(r_cut * r_cut), m_probability(probability),
          m_groups_identical(groups_identical), m_N(N), m_timestep(timestep), m_seed(seed)
        {
        }

    //! Test if a local particle can form more bonds
    HOSTDEVICE bool canBond(unsigned int i) const
        {
        // particles that are not in a group have no free bonds
        return m_free_bonds[i] > 0;
        }

    //! Test if two particles can be bonded
    /*!
     * \param candidate Candidate that is filled in if the particles can be bonded
     * \param i Index of a local particle that can form more bonds
     * \param j Index of a local or ghost particle
     * \returns True if the particles are a candidate to be bonded
     */
    HOSTDEVICE bool operator()(BondCandidate& candidate, unsigned int i, unsigned int j) const
        {
        // check a local particle j now. A half neighbor list stores a pair of local particles with
        // the particle that has the smaller index, while a full neighbor list stores it with both
        // particles, so only test the pair from the particle with the smaller index.
        const bool j_is_local = (j < m_N);
        if (j_is_local
            && (j < i || m_free_bonds[j] == 0
                || (!m_groups_identical && m_group_ids[i] == m_group_ids[j])))
            {
            return false;
            }

        // particles must be closer than the cutoff
        const Scalar4 postype_i = m_positions[i];
        const Scalar4 postype_j = m_positions[j];
        Scalar3 dr = make_scalar3(postype_j.x - postype_i.x,
                                  postype_j.y - postype_i.y,
                                  postype_j.z - postype_i.z);
        dr = m_box.minImage(dr);
        const Scalar r_sq = dot(dr, dr);
        if (r_sq >= m_r_cut_sq)
            {
            return false;
            }

        // particles must not already be bonded. All bonds of the local particle i are listed.
        for (unsigned int bond = 0; bond < m_n_bonds[i]; ++bond)
            {
            if (m_bond_table[m_bond_table_indexer(i, bond)].idx[0] == j)
                {
                return false;
                }
            }

        // put the particle from the first group (or with the smaller tag) first
        const unsigned int tag_i = m_tags[i];
        const unsigned int tag_j = m_tags[j];
        const unsigned int free_bonds_i = m_free_bonds[i];
        const unsigned int free_bonds_j = (j_is_local) ? m_free_bonds[j] : UNKNOWN_FREE_BONDS;
        const bool i_is_first = (m_groups_identical) ? tag_i < tag_j : m_group_ids[i] == 1;
        if (i_is_first)
            {
            candidate.tag_1 = tag_i;
            candidate.tag_2 = tag_j;
            candidate.free_bonds_1 = free_bonds_i;
            candidate.free_bonds_2 = free_bonds_j;
            }
        else
            {
            candidate.tag_1 = tag_j;
            candidate.tag_2 = tag_i;
            candidate.free_bonds_1 = free_bonds_j;
            candidate.free_bonds_2 = free_bonds_i;
            }
        candidate.r_sq = r_sq;

        // accept the candidate with the bond formation probability. The random number can be
        // exactly 1, so the draw is skipped when bonds always form.
        if (m_probability < Scalar(1.0))
            {
            hoomd::RandomGenerator rng(
                hoomd::Seed(RNGIdentifier::DynamicBondUpdater, m_timestep, m_seed),
                hoomd::Counter(candidate.tag_1, candidate.tag_2));
            if (hoomd::UniformDistribution<Scalar>()(rng) >= m_probability)
                {
                return false;
                }
            }

        return true;
        }

    private:
    const Scalar4* const m_positions;           //!< Particle positions
    const unsigned int* const m_tags;           //!< Particle tags
    const unsigned int* const m_group_ids;      //!< Group of each local particle
    const unsigned int* const m_free_bonds;     //!< Free bonds of each local particle
    const group_storage<2>* const m_bond_table; //!< Bonds of each particle
    const unsigned int* const m_n_bonds;        //!< Number of bonds of each particle
    const Index2D m_bond_table_indexer;         //!< Indexer into the bond table
    const BoxDim m_box;                         //!< Local simulation box
    const Scalar m_r_cut_sq;                    //!< Squared cutoff distance
    const Scalar m_probability;                 //!< Probability to form a bond
    const bool m_groups_identical;              //!< True if the groups contain the same particles
    const unsigned int m_N;                     //!< Number of local particles
    const uint64_t m_timestep;                  //!< Current timestep
    const uint16_t m_seed;                      //!< Simulation seed
    };

    } // end namespace detail
    } // end namespace azplugins
    } // end namespace hoomd

#undef HOSTDEVICE

#endif // AZPLUGINS_DYNAMIC_BOND_CANDIDATE_H_
