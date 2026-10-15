# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2025, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

"""Dynamic bond updater unit tests."""

import collections

import hoomd
import numpy

import pytest


def make_snapshot(
    positions,
    typeids,
    L=20,
    particle_types=("A", "B"),
    bond_types=("A-A",),
    bonds=None,
    bond_typeids=None,
):
    """Make a snapshot with particles and optional existing bonds."""
    snap = hoomd.Snapshot()
    if snap.communicator.rank == 0:
        snap.configuration.box = [L, L, L, 0, 0, 0]
        snap.particles.N = len(positions)
        snap.particles.types = list(particle_types)
        snap.particles.position[:] = positions
        snap.particles.typeid[:] = typeids
        snap.bonds.types = list(bond_types)
        if bonds is not None:
            snap.bonds.N = len(bonds)
            snap.bonds.group[:] = bonds
            snap.bonds.typeid[:] = bond_typeids
    return snap


def make_updater(**kwargs):
    """Make an updater that runs every step, with defaults for unit tests."""
    params = dict(
        trigger=hoomd.trigger.Periodic(1),
        nlist=hoomd.md.nlist.Cell(buffer=0.4),
        bond_type="A-A",
        r_cut=1.1,
        filter_1=hoomd.filter.All(),
        filter_2=hoomd.filter.All(),
        max_bonds_1=2,
        max_bonds_2=2,
    )
    params.update(kwargs)
    return hoomd.azplugins.update.DynamicBond(**params)


def get_bonds(sim, bond_type=None):
    """Get the bonds (in order of their tags) on rank 0, optionally of one type."""
    snap = sim.state.get_snapshot()
    if snap.communicator.rank != 0:
        return None
    bonds = numpy.array(snap.bonds.group, dtype=int).reshape(-1, 2)
    if bond_type is not None:
        typeid = snap.bonds.types.index(bond_type)
        bonds = bonds[numpy.array(snap.bonds.typeid) == typeid]
    return bonds


class TestParameters:
    def test_attach_detach(self, simulation_factory):
        nlist = hoomd.md.nlist.Cell(buffer=0.4)
        filter_A = hoomd.filter.Type(["A"])
        filter_B = hoomd.filter.Type(["B"])
        updater = hoomd.azplugins.update.DynamicBond(
            trigger=hoomd.trigger.Periodic(10),
            nlist=nlist,
            bond_type="A-B",
            r_cut=1.0,
            filter_1=filter_A,
            filter_2=filter_B,
            max_bonds_1=1,
            max_bonds_2=2,
        )

        def check_parameters():
            assert updater.trigger == hoomd.trigger.Periodic(10)
            assert updater.nlist is nlist
            assert updater.bond_type == "A-B"
            assert updater.r_cut == pytest.approx(1.0)
            assert updater.filter_1 == filter_A
            assert updater.filter_2 == filter_B
            assert updater.max_bonds_1 == 1
            assert updater.max_bonds_2 == 2
            assert updater.probability == pytest.approx(1.0)

        check_parameters()

        snap = make_snapshot(
            positions=[[0, 0, 0], [3, 0, 0]],
            typeids=[0, 1],
            bond_types=["A-B", "B-B"],
        )
        sim = simulation_factory(snap)
        sim.operations.updaters.append(updater)
        sim.run(0)
        check_parameters()

        # mutable parameters can be changed after attaching
        updater.bond_type = "B-B"
        updater.r_cut = 1.5
        updater.max_bonds_1 = 3
        updater.max_bonds_2 = 4
        updater.probability = 0.5
        assert updater.bond_type == "B-B"
        assert updater.r_cut == pytest.approx(1.5)
        assert updater.max_bonds_1 == 3
        assert updater.max_bonds_2 == 4
        assert updater.probability == pytest.approx(0.5)

        # the neighbor list and groups cannot be changed after attaching
        with pytest.raises(hoomd.error.MutabilityError):
            updater.nlist = hoomd.md.nlist.Cell(buffer=0.4)
        with pytest.raises(hoomd.error.MutabilityError):
            updater.filter_1 = filter_B
        with pytest.raises(hoomd.error.MutabilityError):
            updater.filter_2 = filter_A

        # invalid assignments leave the attached updater's parameters unchanged
        for probability in (-0.5, 1.5, numpy.nan, numpy.inf):
            with pytest.raises(ValueError):
                updater.probability = probability
            assert updater.probability == pytest.approx(0.5)
        for r_cut in (numpy.nan, numpy.inf):
            with pytest.raises(ValueError):
                updater.r_cut = r_cut
            assert updater.r_cut == pytest.approx(1.5)

        sim.operations.updaters.remove(updater)
        assert updater.r_cut == pytest.approx(1.5)
        sim.run(1)

    def test_invalid_parameters(self):
        with pytest.raises(hoomd.error.TypeConversionError):
            make_updater(r_cut=0.0)
        with pytest.raises(hoomd.error.TypeConversionError):
            make_updater(max_bonds_1=0)
        with pytest.raises(hoomd.error.TypeConversionError):
            make_updater(max_bonds_2=-1)

    @pytest.mark.parametrize(
        "params,error",
        [
            (dict(probability=1.5), ValueError),
            (dict(probability=-0.5), ValueError),
            (dict(probability=numpy.nan), ValueError),
            (dict(probability=numpy.inf), ValueError),
            (dict(r_cut=numpy.nan), ValueError),
            (dict(r_cut=numpy.inf), ValueError),
            (dict(bond_type="B-B"), RuntimeError),
            (dict(filter_1=hoomd.filter.Tags([0, 1])), ValueError),
        ],
        ids=[
            "probability_high",
            "probability_low",
            "probability_nan",
            "probability_inf",
            "cutoff_nan",
            "cutoff_inf",
            "bond_type",
            "partial_overlap",
        ],
    )
    def test_invalid_attach(self, simulation_factory, params, error):
        snap = make_snapshot(
            positions=[[0, 0, 0], [3, 0, 0], [6, 0, 0]], typeids=[0, 0, 0]
        )
        sim = simulation_factory(snap)
        sim.operations.updaters.append(make_updater(**params))
        with pytest.raises(error):
            sim.run(0)

    def test_max_bonds_identical_groups(self, simulation_factory):
        snap = make_snapshot(positions=[[0, 0, 0], [3, 0, 0]], typeids=[0, 0])
        sim = simulation_factory(snap)
        updater = make_updater(max_bonds_1=1, max_bonds_2=2)
        sim.operations.updaters.append(updater)
        with pytest.raises(ValueError):
            sim.run(1)


def test_form_bonds_same_group(simulation_factory):
    # dynamic bonding operates on groups, so the particle types should not matter
    snap = make_snapshot(
        positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [6, 0, 0]],
        typeids=[0, 0, 1, 1, 0, 0],
    )
    sim = simulation_factory(snap)
    group = hoomd.filter.Tags([0, 1, 2, 3, 4, 5])
    sim.operations.updaters.append(make_updater(filter_1=group, filter_2=group))

    # bonds form between 0-1, 1-2, 2-3, and 3-4, but particle 5 is too far away
    expected_bonds = [[0, 1], [1, 2], [2, 3], [3, 4]]
    sim.run(1)
    bonds = get_bonds(sim)
    if bonds is not None:
        numpy.testing.assert_array_equal(bonds, expected_bonds)

    # no new bonds form on the next step
    sim.run(1)
    bonds = get_bonds(sim)
    if bonds is not None:
        numpy.testing.assert_array_equal(bonds, expected_bonds)

    # put particle 5 within range of particles 2 and 3, but they already have the
    # maximum number of bonds
    snap = sim.state.get_snapshot()
    if snap.communicator.rank == 0:
        snap.particles.position[5] = [2.5, 0, 0]
    sim.state.set_snapshot(snap)
    sim.run(1)
    bonds = get_bonds(sim)
    if bonds is not None:
        numpy.testing.assert_array_equal(bonds, expected_bonds)


def test_form_bonds_same_group_priority(simulation_factory):
    snap = make_snapshot(
        positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [2.5, 0, 0]],
        typeids=[0, 0, 1, 1, 0, 0],
    )
    sim = simulation_factory(snap)
    sim.operations.updaters.append(make_updater())
    sim.run(1)

    # particle 5 is close to particles 2 and 3, so bonds 2-5 and 3-5 (length 0.5)
    # form first. Then, bonds 0-1, 1-2, 2-3, and 3-4 (length 1) are formed in order
    # of their tags. Particle 2 has two bonds after 1-2 is formed, so 2-3 is not
    # formed, but 3-4 is.
    bonds = get_bonds(sim)
    if bonds is not None:
        numpy.testing.assert_array_equal(
            bonds, [[2, 5], [3, 5], [0, 1], [1, 2], [3, 4]]
        )


def test_form_bonds_two_groups(simulation_factory):
    snap = make_snapshot(
        positions=[[0, 0, 0], [0.9, 0, 0], [1.1, 0, 0], [2, 0, 0]],
        typeids=[1, 0, 1, 0],
    )
    sim = simulation_factory(snap)
    updater = make_updater(
        r_cut=1.0,
        filter_1=hoomd.filter.Tags([0, 1]),
        filter_2=hoomd.filter.Tags([2, 3]),
        max_bonds_1=1,
        max_bonds_2=2,
    )
    sim.operations.updaters.append(updater)

    # only 1-2 can be bonded: 0-1 and 2-3 are close, but they are in the same group
    sim.run(1)
    bonds = get_bonds(sim)
    if bonds is not None:
        numpy.testing.assert_array_equal(bonds, [[1, 2]])

    # no duplicate bond is formed as the simulation continues
    sim.run(10)
    bonds = get_bonds(sim)
    if bonds is not None:
        numpy.testing.assert_array_equal(bonds, [[1, 2]])

    # increasing the cutoff allows particles 0-2 and 1-3 to be bonded, but 1 can only
    # have one bond
    updater.r_cut = 1.2
    sim.run(1)
    bonds = get_bonds(sim)
    if bonds is not None:
        numpy.testing.assert_array_equal(bonds, [[1, 2], [0, 2]])


@pytest.mark.parametrize("exclusions", [("bond",), ()], ids=["excluded", "included"])
def test_existing_bonds(simulation_factory, exclusions):
    # 0-1 have a bond of a different type, and 3-4 have a dynamic bond already
    snap = make_snapshot(
        positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]],
        typeids=[0, 0, 0, 0, 0],
        bond_types=["dynamic", "backbone"],
        bonds=[[0, 1], [3, 4]],
        bond_typeids=[1, 0],
    )
    sim = simulation_factory(snap)
    updater = make_updater(
        nlist=hoomd.md.nlist.Cell(buffer=0.4, exclusions=exclusions),
        bond_type="dynamic",
        max_bonds_1=1,
        max_bonds_2=1,
    )
    sim.operations.updaters.append(updater)
    sim.run(1)

    # bonds of other types do not count toward the maximum, so only 1-2 is formed:
    # 0-1 and 3-4 are already bonded, and 3 already has its one dynamic bond
    bonds = get_bonds(sim, "dynamic")
    backbone = get_bonds(sim, "backbone")
    if bonds is not None:
        numpy.testing.assert_array_equal(bonds, [[3, 4], [1, 2]])
        numpy.testing.assert_array_equal(backbone, [[0, 1]])


def make_dimer_snapshot(num_dimers):
    """Make a snapshot of dimers that are far away from each other."""
    num_side = int(numpy.ceil(num_dimers ** (1 / 3)))
    spacing = 3.0
    L = num_side * spacing
    lattice = numpy.arange(num_side) * spacing - 0.5 * L
    sites = numpy.array(numpy.meshgrid(lattice, lattice, lattice)).reshape(3, -1).T
    sites = sites[:num_dimers]
    # Some dimers straddle the periodic and MPI domain boundaries.
    offset = numpy.array([0.25, 0.0, 0.0])
    positions = numpy.concatenate((sites - offset, sites + offset))
    positions -= L * numpy.floor((positions + 0.5 * L) / L)
    return make_snapshot(positions=positions, typeids=[0] * len(positions), L=L)


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_probability_limits(simulation_factory, probability):
    sim = simulation_factory(make_dimer_snapshot(64))
    updater = make_updater(probability=probability)
    sim.operations.updaters.append(updater)
    sim.run(1)
    bonds = get_bonds(sim)
    if bonds is not None:
        assert len(bonds) == (64 if probability == 1.0 else 0)

    # Switching off and back on does not reuse candidates from an earlier update.
    updater.probability = 0.0
    sim.run(1)
    paused_bonds = get_bonds(sim)
    if paused_bonds is not None:
        numpy.testing.assert_array_equal(paused_bonds, bonds)
    updater.probability = 1.0
    sim.run(1)
    bonds = get_bonds(sim)
    if bonds is not None:
        assert len(bonds) == 64


def test_probability(simulation_factory):
    num_dimers = 64
    snap = make_dimer_snapshot(num_dimers)
    decomposition = (2, 1, 1) if snap.communicator.num_ranks == 2 else None
    sim = simulation_factory(snap, domain_decomposition=decomposition)
    updater = make_updater(probability=0.5)
    sim.operations.updaters.append(updater)

    # about half of the dimers form a bond on one step
    sim.run(1)
    first_bonds = get_bonds(sim)
    if first_bonds is not None:
        assert 12 < len(first_bonds) < 52

    # the random numbers change each step, so eventually all dimers are bonded
    sim.run(40)
    bonds = get_bonds(sim)
    if bonds is not None:
        assert len(bonds) == num_dimers

    # The same seed gives the same bonds even with a different decomposition.
    decomposition = (1, 2, 1) if snap.communicator.num_ranks == 2 else None
    sim = simulation_factory(
        make_dimer_snapshot(num_dimers), domain_decomposition=decomposition
    )
    sim.operations.updaters.append(make_updater(probability=0.5))
    sim.run(1)
    repeat_bonds = get_bonds(sim)
    if repeat_bonds is not None:
        numpy.testing.assert_array_equal(repeat_bonds, first_bonds)


def test_bonds_across_boundaries(simulation_factory):
    # a line of particles across the box forms a ring of bonds through the periodic
    # boundary, and the domain boundary when there are two ranks
    L = 10
    num_particles = 10
    x = numpy.arange(num_particles) - 0.5 * L + 0.5
    positions = [[xi, 0, 0] for xi in x]
    snap = make_snapshot(positions=positions, typeids=[0] * num_particles, L=L)
    if snap.communicator.num_ranks == 2:
        sim = simulation_factory(snap, domain_decomposition=(2, 1, 1))
    else:
        sim = simulation_factory(snap)
    sim.operations.updaters.append(make_updater())
    sim.run(1)

    # all bonds have the same length, so they are formed in order of their tags
    expected_bonds = [[0, 1], [0, num_particles - 1]]
    expected_bonds += [[i, i + 1] for i in range(1, num_particles - 1)]
    bonds = get_bonds(sim)
    if bonds is not None:
        numpy.testing.assert_array_equal(bonds, expected_bonds)


@pytest.mark.parametrize("nlist_cls", [hoomd.md.nlist.Cell, hoomd.md.nlist.Tree])
@pytest.mark.parametrize("storage_mode", ["half", "full"])
def test_neighbor_list_storage(simulation_factory, device, nlist_cls, storage_mode):
    if isinstance(device, hoomd.device.GPU) and storage_mode == "half":
        pytest.skip("GPU neighbor lists require full storage")

    # Tags and spatial order differ, and both groups contain the same particle type.
    snap = make_snapshot(
        positions=[[1, 0, 0], [-0.5, 0, 0], [0.5, 0, 0], [-1, 0, 0]],
        typeids=[0, 0, 0, 0],
    )
    sim = simulation_factory(snap)
    nlist = nlist_cls(buffer=0.4, exclusions=())
    sim.operations.updaters.append(
        make_updater(
            nlist=nlist,
            filter_1=hoomd.filter.Tags([0, 1]),
            filter_2=hoomd.filter.Tags([2, 3]),
        )
    )
    sim.run(0)
    nlist._cpp_obj.setStorageMode(
        getattr(hoomd.md._md.NeighborList.storageMode, storage_mode)
    )
    for _ in range(2):
        sim.run(1)
        bonds = get_bonds(sim)
        if bonds is not None:
            numpy.testing.assert_array_equal(bonds, [[0, 2], [1, 3], [1, 2]])


def test_cross_domain_candidates(simulation_factory):
    # Each row straddles x=0. Only the first row can form a bond: the others have
    # two members of group 1, a nonmember, or a saturated member of group 2.
    snap = make_snapshot(
        positions=[
            [-0.25, -6, 0],
            [0.25, -6, 0],
            [-0.25, -2, 0],
            [0.25, -2, 0],
            [-0.25, 2, 0],
            [0.25, 2, 0],
            [-0.25, 6, 0],
            [0.25, 6, 0],
            [0.75, 6, 0],
        ],
        typeids=[0, 1, 0, 0, 0, 2, 0, 1, 1],
        particle_types=["A", "B", "C"],
        bonds=[[7, 8]],
        bond_typeids=[0],
    )
    decomposition = (2, 1, 1) if snap.communicator.num_ranks == 2 else None
    sim = simulation_factory(snap, domain_decomposition=decomposition)
    sim.operations.updaters.append(
        make_updater(
            # Include pairs of all types, including already bonded pairs.
            nlist=hoomd.md.nlist.Cell(buffer=0.4, default_r_cut=1.0, exclusions=()),
            filter_1=hoomd.filter.Type(["A"]),
            filter_2=hoomd.filter.Type(["B"]),
            r_cut=0.75,
            max_bonds_1=1,
            max_bonds_2=1,
        )
    )
    for _ in range(2):
        sim.run(1)
        bonds = get_bonds(sim)
        if bonds is not None:
            numpy.testing.assert_array_equal(bonds, [[7, 8], [0, 1]])


@pytest.mark.parametrize("x", [-3.0, 3.0], ids=["empty_nonroot", "empty_root"])
def test_empty_rank(simulation_factory, x):
    # With two x domains, one rank has neither local nor ghost particles.
    snap = make_snapshot(positions=[[x, 0, 0], [x + 0.5, 0, 0]], typeids=[0, 0])
    decomposition = (2, 1, 1) if snap.communicator.num_ranks == 2 else None
    sim = simulation_factory(snap, domain_decomposition=decomposition)
    sim.operations.updaters.append(make_updater())
    for _ in range(2):
        sim.run(1)
        bonds = get_bonds(sim)
        if bonds is not None:
            numpy.testing.assert_array_equal(bonds, [[0, 1]])


def test_large_bond_capacity(simulation_factory):
    # The largest unsigned capacity must not be mistaken for unknown ghost data.
    snap = make_snapshot(positions=[[-0.25, 0, 0], [0.25, 0, 0]], typeids=[0, 0])
    decomposition = (2, 1, 1) if snap.communicator.num_ranks == 2 else None
    sim = simulation_factory(snap, domain_decomposition=decomposition)
    sim.operations.updaters.append(
        make_updater(max_bonds_1=2**32 - 1, max_bonds_2=2**32 - 1)
    )
    sim.run(1)
    bonds = get_bonds(sim)
    if bonds is not None:
        numpy.testing.assert_array_equal(bonds, [[0, 1]])


@pytest.mark.parametrize(
    "filters",
    [
        (hoomd.filter.Null(), hoomd.filter.Null()),
        (hoomd.filter.All(), hoomd.filter.Null()),
        (hoomd.filter.Null(), hoomd.filter.All()),
    ],
    ids=["both_empty", "second_empty", "first_empty"],
)
def test_empty_groups(simulation_factory, filters):
    snap = make_snapshot(positions=[[-0.25, 0, 0], [0.25, 0, 0]], typeids=[0, 0])
    sim = simulation_factory(snap)
    sim.operations.updaters.append(
        make_updater(filter_1=filters[0], filter_2=filters[1])
    )
    sim.run(2)
    bonds = get_bonds(sim)
    if bonds is not None:
        assert len(bonds) == 0


def test_change_bond_capacity_and_type(simulation_factory):
    snap = make_snapshot(
        positions=[[-1, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0.5, 0]],
        typeids=[0, 0, 0, 0],
        bond_types=["dynamic", "other"],
        bonds=[[0, 1], [1, 2]],
        bond_typeids=[0, 0],
    )
    sim = simulation_factory(snap)
    updater = make_updater(
        bond_type="dynamic", r_cut=0.75, max_bonds_1=1, max_bonds_2=1
    )
    sim.operations.updaters.append(updater)

    # Particle 1 already exceeds the limit. Its remaining capacity must be zero.
    sim.run(1)
    bonds = get_bonds(sim)
    if bonds is not None:
        numpy.testing.assert_array_equal(bonds, [[0, 1], [1, 2]])

    updater.max_bonds_1 = updater.max_bonds_2 = 3
    sim.run(1)
    bonds = get_bonds(sim)
    if bonds is not None:
        numpy.testing.assert_array_equal(bonds, [[0, 1], [1, 2], [1, 3]])

    # The new type has fresh capacity, but already bonded pairs stay excluded.
    updater.bond_type = "other"
    updater.r_cut = 1.2
    updater.max_bonds_1 = updater.max_bonds_2 = 1
    sim.run(1)
    bonds = get_bonds(sim, "other")
    if bonds is not None:
        numpy.testing.assert_array_equal(bonds, [[0, 3]])


def test_cutoff(simulation_factory):
    snap = make_snapshot(
        positions=[[0, 0, 0], [1, 0, 0], [4, 0, 0], [4.75, 0, 0]],
        typeids=[0, 0, 0, 0],
    )
    sim = simulation_factory(snap)
    # The neighbor list also contains pairs at and beyond the bonding cutoff.
    sim.operations.updaters.append(
        make_updater(
            nlist=hoomd.md.nlist.Cell(buffer=0.4, default_r_cut=2.0), r_cut=1.0
        )
    )
    sim.run(1)
    bonds = get_bonds(sim)
    if bonds is not None:
        numpy.testing.assert_array_equal(bonds, [[2, 3]])


def test_shared_neighbor_list_detach(simulation_factory):
    snap = make_snapshot(
        positions=[[0, 0, 0], [1.5, 0, 0]], typeids=[0, 0], particle_types=["A"]
    )
    sim = simulation_factory(snap)
    nlist = hoomd.md.nlist.Cell(buffer=0.0)
    pair = hoomd.md.pair.DPDConservative(nlist=nlist, default_r_cut=1.0)
    pair.params[("A", "A")] = dict(A=0.0)
    sim.operations.integrator = hoomd.md.Integrator(dt=0.005, forces=[pair])
    updater = make_updater(nlist=nlist, r_cut=2.0, probability=0.0)
    sim.operations.updaters.append(updater)

    sim.run(0)
    pairs = nlist.pair_list
    if sim.device.communicator.rank == 0:
        numpy.testing.assert_array_equal(pairs, [[0, 1]])

    sim.operations.updaters.remove(updater)
    sim.run(0)
    pairs = nlist.pair_list
    if sim.device.communicator.rank == 0:
        assert len(pairs) == 0

    sim.operations.updaters.append(updater)
    sim.run(0)
    pairs = nlist.pair_list
    if sim.device.communicator.rank == 0:
        numpy.testing.assert_array_equal(pairs, [[0, 1]])


def find_reference_bonds(
    positions, L, group_1, group_2, existing_bonds, r_cut, max_bonds_1, max_bonds_2
):
    """Form bonds with the same greedy algorithm as the updater, with probability 1.

    The groups are arrays of flags for each particle tag.
    """
    identical = numpy.array_equal(group_1, group_2)
    existing = {tuple(sorted(bond)) for bond in existing_bonds}

    # find all pairs of particles closer than the cutoff
    candidates = []
    N = len(positions)
    for i in range(N - 1):
        dr = positions[i + 1 :] - positions[i]
        dr -= L * numpy.round(dr / L)
        r_sq = numpy.sum(dr * dr, axis=1)
        for j in numpy.flatnonzero(r_sq < r_cut**2) + i + 1:
            if (i, j) in existing:
                continue
            if identical and group_1[i] and group_1[j]:
                candidates.append((r_sq[j - i - 1], i, j))
            elif group_1[i] and group_2[j]:
                candidates.append((r_sq[j - i - 1], i, j))
            elif group_2[i] and group_1[j]:
                candidates.append((r_sq[j - i - 1], j, i))

    # accept the closest pairs first if both particles can still form a bond
    max_bonds = numpy.where(group_1, max_bonds_1, max_bonds_2)
    num_bonds = numpy.zeros(N, dtype=int)
    bonds = []
    for _, tag_1, tag_2 in sorted(candidates):
        if num_bonds[tag_1] < max_bonds[tag_1] and num_bonds[tag_2] < max_bonds[tag_2]:
            bonds.append([tag_1, tag_2])
            num_bonds[tag_1] += 1
            num_bonds[tag_2] += 1
    return numpy.array(bonds, dtype=int).reshape(-1, 2)


@pytest.mark.parametrize("identical_groups", [True, False], ids=["same", "different"])
def test_random_system(simulation_factory, identical_groups):
    # random particles with some existing bonds of another type
    L = 12
    N = 600
    rng = numpy.random.default_rng(42)
    positions = rng.uniform(-0.5 * L, 0.5 * L, size=(N, 3))
    typeids = rng.integers(0, 2, size=N)
    existing_bonds = numpy.reshape(rng.choice(N, size=100, replace=False), (50, 2))
    snap = make_snapshot(
        positions=positions,
        typeids=typeids,
        L=L,
        bond_types=["A-B", "backbone"],
        bonds=existing_bonds,
        bond_typeids=[1] * len(existing_bonds),
    )
    sim = simulation_factory(snap)

    if identical_groups:
        filters = [hoomd.filter.All(), hoomd.filter.All()]
        group_1 = numpy.ones(N, dtype=bool)
        group_2 = group_1
        max_bonds = (2, 2)
    else:
        filters = [hoomd.filter.Type(["A"]), hoomd.filter.Type(["B"])]
        group_1 = typeids == 0
        group_2 = typeids == 1
        max_bonds = (2, 3)
    r_cut = 1.0
    updater = make_updater(
        bond_type="A-B",
        r_cut=r_cut,
        filter_1=filters[0],
        filter_2=filters[1],
        max_bonds_1=max_bonds[0],
        max_bonds_2=max_bonds[1],
    )
    sim.operations.updaters.append(updater)
    sim.run(1)

    bonds = get_bonds(sim, "A-B")
    if bonds is not None:
        expected_bonds = find_reference_bonds(
            positions, L, group_1, group_2, existing_bonds, r_cut, *max_bonds
        )
        assert len(expected_bonds) > 0
        numpy.testing.assert_array_equal(bonds, expected_bonds)


def test_dynamics(simulation_factory, lattice_snapshot_factory):
    # particles on a lattice that are free to move and form bonds while they do
    snap = lattice_snapshot_factory(
        particle_types=["A", "B"], n=8, a=1.1, r=0.1, dimensions=3
    )
    if snap.communicator.rank == 0:
        snap.particles.typeid[:] = numpy.arange(snap.particles.N) % 2
        snap.bonds.types = ["A-B"]
    sim = simulation_factory(snap)

    nlist = hoomd.md.nlist.Cell(buffer=0.4)
    dpd = hoomd.md.pair.DPDConservative(nlist=nlist, default_r_cut=1.0)
    dpd.params[("A", "A")] = dict(A=25.0)
    dpd.params[("A", "B")] = dict(A=25.0)
    dpd.params[("B", "B")] = dict(A=25.0)
    harmonic = hoomd.md.bond.Harmonic()
    harmonic.params["A-B"] = dict(k=10.0, r0=1.0)
    integrator = hoomd.md.Integrator(dt=0.005, forces=[dpd, harmonic])
    integrator.methods.append(
        hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=1.0)
    )
    sim.operations.integrator = integrator

    max_bonds = (1, 3)
    updater = hoomd.azplugins.update.DynamicBond(
        trigger=hoomd.trigger.Periodic(5),
        nlist=nlist,
        bond_type="A-B",
        r_cut=1.2,
        filter_1=hoomd.filter.Type(["A"]),
        filter_2=hoomd.filter.Type(["B"]),
        max_bonds_1=max_bonds[0],
        max_bonds_2=max_bonds[1],
        probability=0.2,
    )
    sim.operations.updaters.append(updater)
    sim.run(200)

    bonds = get_bonds(sim)
    pairs = nlist.pair_list
    snap = sim.state.get_snapshot()
    if snap.communicator.rank == 0:
        assert len(bonds) > 0

        # bonds are between particles of type A and B, and are not duplicated
        typeids = numpy.array(snap.particles.typeid)
        numpy.testing.assert_array_equal(typeids[bonds[:, 0]], 0)
        numpy.testing.assert_array_equal(typeids[bonds[:, 1]], 1)
        assert len({tuple(bond) for bond in bonds}) == len(bonds)

        # no particle has more than the maximum number of bonds
        counts = collections.Counter(bonds.flatten())
        for tag, count in counts.items():
            assert count <= max_bonds[typeids[tag]]

        # bonded particles are excluded from the neighbor list
        bonded = {tuple(sorted(bond)) for bond in bonds}
        assert not bonded & {tuple(sorted(pair)) for pair in pairs}
