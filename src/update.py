# Copyright (c) 2018-2020, Michael P. Howard
# Copyright (c) 2021-2025, Auburn University
# Part of azplugins, released under the BSD 3-Clause License.

"""Updaters."""

import copy
import warnings

import hoomd
from hoomd.azplugins import _azplugins
from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import positive_int, positive_real
from hoomd.md import _md


class DynamicBond(hoomd.operation.Updater):
    r"""Form bonds dynamically between nearby particles.

    Args:
        trigger (hoomd.trigger.trigger_like): Select the timesteps to form
            bonds.
        nlist (hoomd.md.nlist.NeighborList): Neighbor list used to find pairs
            of particles.
        bond_type (str): Type of bond to form.
        r_cut (float): Finite, positive cutoff distance for forming a bond
            :math:`[\mathrm{length}]`.
        filter_1 (hoomd.filter.ParticleFilter): First group of particles.
        filter_2 (hoomd.filter.ParticleFilter): Second group of particles.
        max_bonds_1 (int): Maximum number of bonds of type ``bond_type`` for
            a particle in the first group.
        max_bonds_2 (int): Maximum number of bonds of type ``bond_type`` for
            a particle in the second group.
        probability (float): Probability between 0 and 1 to form a bond between
            a pair of particles that are close enough.

    `DynamicBond` forms bonds of type ``bond_type`` between particles in the
    first group and particles in the second group that are closer than
    ``r_cut``. Each time the updater runs, it finds the pairs of particles that
    are close enough and are not already bonded to each other by a bond of any
    type. Each pair is considered for bonding with ``probability``. The pairs
    are processed in order of increasing distance, using the particle tags to
    break ties, and a bond is formed if both particles have fewer bonds of type
    ``bond_type`` than their maximum. Bonds of other types do not count toward
    the maximum number of bonds.

    The two groups must either contain the same particles or have no particles
    in common. When the groups contain the same particles, bonds are formed
    between particles in that group, and ``max_bonds_1`` and ``max_bonds_2``
    must be equal.

    `DynamicBond` adds ``r_cut`` to the cutoff of ``nlist`` for the pairs of
    particle types in the two groups, so the neighbor list can be shared with a
    pair potential. Pairs of particles that are excluded from the neighbor list
    are never bonded. If ``"bond"`` is one of the neighbor list exclusions,
    new bonds are also excluded from the neighbor list.

    The random number that decides if a pair of particles forms a bond depends
    on the timestep, `hoomd.Simulation.seed`, and the tags of the particles.
    These random draws are independent of the device and MPI domain
    decomposition. Floating point roundoff can affect the order of pairs with
    nearly equal distances or pairs very close to ``r_cut``.

    Example::

        nlist = hoomd.md.nlist.Cell(buffer=0.4)
        dynamic_bond = hoomd.azplugins.update.DynamicBond(
            trigger=hoomd.trigger.Periodic(100),
            nlist=nlist,
            bond_type="A-B",
            r_cut=1.2,
            filter_1=hoomd.filter.Type(["A"]),
            filter_2=hoomd.filter.Type(["B"]),
            max_bonds_1=1,
            max_bonds_2=2,
            probability=0.5,
        )
        simulation.operations.updaters.append(dynamic_bond)

    Bonds between particles of the same type::

        dynamic_bond = hoomd.azplugins.update.DynamicBond(
            trigger=hoomd.trigger.Periodic(100),
            nlist=nlist,
            bond_type="A-A",
            r_cut=1.2,
            filter_1=hoomd.filter.Type(["A"]),
            filter_2=hoomd.filter.Type(["A"]),
            max_bonds_1=2,
            max_bonds_2=2,
        )

    .. warning::

        The filter selections and the particle types in each group must remain
        fixed while the updater is attached. Remove the updater before changing
        them (for example, using `hoomd.update.FilterUpdater`), then reattach it
        to refresh the group checks and neighbor list cutoffs.

    Attributes:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list used to find pairs
            of particles (*read only*).

        bond_type (str): Type of bond to form.

        r_cut (float): Cutoff distance for forming a bond
            :math:`[\mathrm{length}]`.

        filter_1 (hoomd.filter.ParticleFilter): First group of particles
            (*read only*).

        filter_2 (hoomd.filter.ParticleFilter): Second group of particles
            (*read only*).

        max_bonds_1 (int): Maximum number of bonds of type ``bond_type`` for a
            particle in the first group.

        max_bonds_2 (int): Maximum number of bonds of type ``bond_type`` for a
            particle in the second group.

        probability (float): Probability to form a bond between a pair of
            particles that are close enough.

    """

    def __init__(
        self,
        trigger,
        nlist,
        bond_type,
        r_cut,
        filter_1,
        filter_2,
        max_bonds_1,
        max_bonds_2,
        probability=1.0,
    ):
        super().__init__(trigger)

        param_dict = ParameterDict(
            nlist=hoomd.md.nlist.NeighborList,
            bond_type=str,
            r_cut=positive_real,
            filter_1=hoomd.filter.ParticleFilter,
            filter_2=hoomd.filter.ParticleFilter,
            max_bonds_1=positive_int,
            max_bonds_2=positive_int,
            probability=float,
        )
        param_dict.update(
            dict(
                nlist=nlist,
                bond_type=bond_type,
                r_cut=r_cut,
                filter_1=filter_1,
                filter_2=filter_2,
                max_bonds_1=max_bonds_1,
                max_bonds_2=max_bonds_2,
                probability=probability,
            )
        )
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        sim = self._simulation

        # attach the neighbor list in the same way as the pair potentials
        if self.nlist._attached and sim != self.nlist._simulation:
            warnings.warn(
                f"{self} object is creating a new equivalent neighbor list."
                f" This is happening since the updater is moving to a new "
                f"simulation. Set a new nlist to suppress this warning.",
                RuntimeWarning,
            )
            self.nlist = copy.deepcopy(self.nlist)
        self.nlist._attach(sim)

        if isinstance(sim.device, hoomd.device.GPU):
            cpp_class = _azplugins.DynamicBondUpdaterGPU
            self.nlist._cpp_obj.setStorageMode(_md.NeighborList.storageMode.full)
        else:
            cpp_class = _azplugins.DynamicBondUpdater

        # bonds are formed randomly using the simulation seed
        sim._warn_if_seed_unset()

        self._cpp_obj = cpp_class(
            sim.state._cpp_sys_def,
            self.trigger,
            self.nlist._cpp_obj,
            sim.state._get_group(self.filter_1),
            sim.state._get_group(self.filter_2),
            self.bond_type,
            self.r_cut,
            self.max_bonds_1,
            self.max_bonds_2,
            self.probability,
        )

        super()._attach_hook()

    def _detach_hook(self):
        self.nlist._detach()
