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
    crossover_prob: float = 0.35,
    # mutation control (aliases supported)
    mutation_rate: float = 0.8,
    mutate_prob: Optional[float] = None,
    t_step: float = 1.5,                 # Å
    mutate_sigma_t: Optional[float] = None,
    r_step: float = 20.0,                # degrees (moves.py expects degrees)
    mutate_sigma_r: Optional[float] = None,  # radians (if provided; will be converted)
    # biased mutation
    biased: bool = True,
    bias_strength: float = 0.4,
    rotate_prob: float = 0.35,
    noise_frac: float = 0.5,
    # multi-point mutation + escape from local minima
    multipoint_prob: float = 0.35,
    multipoint_max: int = 3,
    kick_after: int = 80,
    kick_fraction: float = 0.60,
    kick_scale_t: float = 3.0,
    kick_scale_r: float = 2.0,
    immigrants_per_gen: int = 0,
    # constraint
    pocket_center=None,
    pocket_radius=None,
    # reporting / hooks
    print_every: int = 25,
    on_checkpoint=None,
    verbose: bool = True,
) -> Pose:
    """Genetic algorithm for rigid ligand docking.

    Individual = Pose(tx,ty,tz,rx,ry,rz)
    Fitness    = ChemPLP total score (lower is better)
    """

    assert pop_size >= 4
    elite_n = max(1, int(pop_size * elite_frac))

    # --- aliases / unit handling ---
    if mutate_prob is not None:
        mutation_rate = float(mutate_prob)
    if mutate_sigma_t is not None:
        t_step = float(mutate_sigma_t)
    if mutate_sigma_r is not None:
        # CLI provides radians; moves.py expects degrees
        r_step = math.degrees(float(mutate_sigma_r))
    print_every = max(1, int(print_every))

    if on_checkpoint is not None:
        try:
            on_checkpoint(None, 0)
        except Exception:
            pass

    # --- initialize population
    population: List[Pose] = []

    if seed_poses:
        for sp in seed_poses:
            population.append(sp.copy())
            if len(population) >= pop_size:
                break

    def _spawn_random_pose() -> Pose:
        """Create a fresh random pose inside the pocket sphere (or near origin if no pocket)."""
        p = Pose(ligand)
        if pocket_center is not None and pocket_radius is not None and pocket_radius > 0:
            # uniform-in-volume radius: r = R * U^(1/3)
            u = random.random()
            r = float(pocket_radius) * (u ** (1.0 / 3.0))
            theta = math.acos(2.0 * random.random() - 1.0)
            phi = 2.0 * math.pi * random.random()
            dx = r * math.sin(theta) * math.cos(phi)
            dy = r * math.sin(theta) * math.sin(phi)
            dz = r * math.cos(theta)
            p.t = np.array([pocket_center[0] + dx, pocket_center[1] + dy, pocket_center[2] + dz], dtype=float)
        else:
            p.t = np.array([0.0, 0.0, 0.0], dtype=float)
        p.rx = random.uniform(-math.pi, math.pi)
        p.ry = random.uniform(-math.pi, math.pi)
        p.rz = random.uniform(-math.pi, math.pi)
        p.score = _score_pose(
            receptor_atoms, ligand, p, chemplp_params,
            precision=precision, engine=engine, prepared_common=prepared_common,
        )
        return p

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
    last_best_score = float(best.score)
    stagnation = 0

    if verbose:
        print(f"[GA] init best = {best.score:.3f}")

    if on_checkpoint is not None:
        try:
            on_checkpoint(best, 0)
        except Exception:
            pass


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
            child = _crossover(p1, p2, alpha=random.random()) if random.random() < crossover_prob else p1.copy()

            # mutation
            # Mutation (single or multi-point)
            if random.random() < mutation_rate:
                n_points = 1
                if int(multipoint_max) >= 2 and random.random() < float(multipoint_prob):
                    n_points = random.randint(2, int(multipoint_max))
                for _ in range(int(n_points)):
                    child.random_move(sigma_t=mutate_sigma_t, sigma_r=mutate_sigma_r)

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

        # Report only periodically (and at start/end) to avoid huge logs.
        do_report = (g == 1) or (g == generations) or (print_every > 0 and (g % print_every == 0))
        if verbose and do_report:
            print(
                f"[GA] gen {g:3d}  best={population[0].score: .3f}  global_best={best.score: .3f}"

        # --- stagnation detection + escape kick ---
        cur_best = float(population[0].score)
        if cur_best < float(last_best_score) - 1e-9:
            last_best_score = cur_best
            stagnation = 0
        else:
            stagnation += 1

        if int(kick_after) > 0 and stagnation >= int(kick_after):
            elite_n = max(1, int(round(float(elite_frac) * float(pop_size))))
            n_kicked = 0
            for i in range(elite_n, int(pop_size)):
                if random.random() >= float(kick_fraction):
                    continue
                # 50/50: full re-spawn vs. strong multi-step mutation
                if random.random() < 0.5:
                    population[i] = _spawn_random_pose()
                else:
                    p2 = population[i].copy()
                    npts = random.randint(2, max(2, int(multipoint_max)))
                    for _ in range(int(npts)):
                        p2.random_move(
                            sigma_t=float(mutate_sigma_t) * float(kick_scale_t),
                            sigma_r=float(mutate_sigma_r) * float(kick_scale_r),
                        )
                    if pocket_center is not None and pocket_radius is not None:
                        p2.ensure_within_sphere(pocket_center, pocket_radius)
                    p2.score = _score_pose(
                        receptor_atoms, ligand, p2, chemplp_params,
                        precision=precision, engine=engine, prepared_common=prepared_common,
                    )
                    population[i] = p2
                n_kicked += 1

            population = sorted(population, key=lambda p: float(p.score))
            if float(population[0].score) < float(best.score):
                best = population[0].copy()
            last_best_score = float(population[0].score)
            stagnation = 0
            if verbose:
                print(f"[GA] stagnation kick: mutated/reseeded {n_kicked} individuals; best={population[0].score:.3f}")

            )

        # Checkpoint callback: write the current global best pose to trajectory.
        if on_checkpoint is not None and do_report:
            try:
                on_checkpoint(best, g)
            except Exception:
                pass

    # Ensure we always write the final generation (if not already written)
    if on_checkpoint is not None and not (print_every > 0 and (generations % print_every == 0)):
        try:
            on_checkpoint(best, generations)
        except Exception:
            pass
    return best