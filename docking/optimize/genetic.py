from __future__ import annotations

import copy
import math
import random
from typing import Dict, List, Optional

from docking.optimize.monte_carlo import _within_sphere
from docking.optimize.moves import biased_move, random_move
from docking.optimize.pose import Pose
from docking.scoring.total_score import score_complex_chemplp


def _tournament(pop: List[Pose], k: int = 3) -> Pose:
    """Tournament selection (minimization)."""
    sel = random.sample(pop, k)
    sel.sort(key=lambda p: float(p.score))
    return sel[0]


def _crossover(p1: Pose, p2: Pose, alpha: float) -> Pose:
    """Blend crossover in DOF space."""
    c = p1.copy()
    c.tx = alpha * p1.tx + (1.0 - alpha) * p2.tx
    c.ty = alpha * p1.ty + (1.0 - alpha) * p2.ty
    c.tz = alpha * p1.tz + (1.0 - alpha) * p2.tz
    c.rx = alpha * p1.rx + (1.0 - alpha) * p2.rx
    c.ry = alpha * p1.ry + (1.0 - alpha) * p2.ry
    c.rz = alpha * p1.rz + (1.0 - alpha) * p2.rz
    c.score = None
    return c


def _score_pose(
    receptor_atoms,
    ligand,
    pose: Pose,
    chemplp_params: Dict,
    *,
    precision: str,
    engine: str,
    prepared_common: Optional[Dict] = None,
) -> float:
    """Score pose using ChemPLP. Uses pose.transformed_xyz() to avoid object churn."""
    L_xyz = pose.transformed_xyz()
    prepared = dict(prepared_common or {})
    prepared["L_xyz"] = L_xyz
    bd = score_complex_chemplp(
        receptor_atoms,
        ligand_atoms=ligand.atoms,
        ligand_rdkit_mol=ligand.rdkit_mol,
        params=chemplp_params,
        precision=precision,
        engine=engine,
        prepared=prepared,
    )
    return float(bd.total)


def genetic_rigid(
    receptor_atoms,
    ligand,
    chemplp_params: Dict,
    *,
    precision: str = "SP",
    engine: str = "auto",
    prepared_common: Optional[Dict] = None,
    seed_poses: Optional[List[Pose]] = None,
    # GA params
    pop_size: int = 40,
    generations: int = 40,
    elite_frac: float = 0.15,
    tournament_k: int = 3,
    # mutation control
    t_step: float = 1.5,
    r_step: float = 20.0,
    mutation_rate: float = 0.8,
    # biased mutation
    biased: bool = True,
    bias_strength: float = 0.4,
    rotate_prob: float = 0.35,
    noise_frac: float = 0.5,
    # constraint
    pocket_center=None,
    pocket_radius=None,
    verbose: bool = True,
) -> Pose:
    """Genetic algorithm for rigid ligand docking.

    Individual = Pose(tx,ty,tz,rx,ry,rz)
    Fitness    = ChemPLP total score (lower is better)
    """

    assert pop_size >= 4
    elite_n = max(1, int(pop_size * elite_frac))

    # --- initialize population
    population: List[Pose] = []

    if seed_poses:
        for sp in seed_poses:
            population.append(sp.copy())
            if len(population) >= pop_size:
                break

    # fill the rest randomly
    while len(population) < pop_size:
        p = Pose(ligand.atoms)

        if pocket_center is not None and pocket_radius is not None:
            r = pocket_radius * random.random()
            theta = random.random() * 2.0 * math.pi
            phi = math.acos(2.0 * random.random() - 1.0)
            p.tx = pocket_center[0] + r * math.sin(phi) * math.cos(theta)
            p.ty = pocket_center[1] + r * math.sin(phi) * math.sin(theta)
            p.tz = pocket_center[2] + r * math.cos(phi)
        else:
            p.tx = random.uniform(-10.0, 10.0)
            p.ty = random.uniform(-10.0, 10.0)
            p.tz = random.uniform(-10.0, 10.0)

        p.rx = random.uniform(-math.pi, math.pi)
        p.ry = random.uniform(-math.pi, math.pi)
        p.rz = random.uniform(-math.pi, math.pi)

        p.score = _score_pose(
            receptor_atoms,
            ligand,
            p,
            chemplp_params,
            precision=precision,
            engine=engine,
            prepared_common=prepared_common,
        )
        population.append(p)

    population.sort(key=lambda x: float(x.score))
    best = population[0].copy()

    if verbose:
        print(f"[GA] init best = {best.score:.3f}")

    # --- evolution loop
    for g in range(1, generations + 1):
        new_pop: List[Pose] = []

        # elitism
        elites = sorted(population, key=lambda p: float(p.score))[:elite_n]
        new_pop.extend(copy.deepcopy(elites))

        # reproduction
        while len(new_pop) < pop_size:
            p1 = _tournament(population, tournament_k)
            p2 = _tournament(population, tournament_k)
            child = _crossover(p1, p2, alpha=random.random())

            # mutation
            if random.random() < mutation_rate:
                if biased and pocket_center is not None:
                    biased_move(
                        child,
                        pocket_center=pocket_center,
                        t_step=t_step,
                        r_step=r_step,
                        bias_strength=bias_strength,
                        rotate_prob=rotate_prob,
                        noise_frac=noise_frac,
                    )
                else:
                    random_move(child, t_step, r_step)

            # constraint
            if pocket_center is not None and pocket_radius is not None:
                c = child.transformed_centroid()
                if not _within_sphere(c, pocket_center, pocket_radius):
                    continue

            child.score = _score_pose(
                receptor_atoms,
                ligand,
                child,
                chemplp_params,
                precision=precision,
                engine=engine,
                prepared_common=prepared_common,
            )
            new_pop.append(child)

        population = sorted(new_pop, key=lambda p: float(p.score))
        if float(population[0].score) < float(best.score):
            best = population[0].copy()

        if verbose:
            print(
                f"[GA] gen {g:3d}  best={population[0].score: .3f}  global_best={best.score: .3f}"
            )

    return best
