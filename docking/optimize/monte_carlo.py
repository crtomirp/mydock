from __future__ import annotations
import random
import math

from docking.scoring.total_score import score_complex_chemplp


def _random_unit_vector():
    u = random.random()
    v = random.random()
    theta = 2.0 * math.pi * u
    z = 2.0 * v - 1.0
    r = math.sqrt(max(0.0, 1.0 - z * z))
    return (r * math.cos(theta), r * math.sin(theta), z)


def _within_sphere(xyz, center_xyz, radius) -> bool:
    dx = xyz[0] - center_xyz[0]
    dy = xyz[1] - center_xyz[1]
    dz = xyz[2] - center_xyz[2]
    return (dx * dx + dy * dy + dz * dz) <= (radius * radius)


def _lin_interp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def monte_carlo_rigid(
    receptor_atoms,
    ligand,
    chemplp_params,
    n_steps=2000,

    # If anneal=False, these are constant:
    temperature=1.0,
    t_step=1.0,
    r_step=15.0,

    # Initial pose
    init_mode: str = "current",     # "current" or "outside"
    init_distance: float = 30.0,    # Å (used for init_mode="outside")
    init_spin_deg: float = 180.0,   # initial random rotation range

    # Pocket constraint: centroid must stay within sphere
    pocket_center=None,             # (x,y,z)
    pocket_radius=None,             # float Å

    # Annealing schedule
    anneal: bool = True,
    stages: int = 5,
    temp_start: float = 2.5,
    temp_end: float = 0.8,
    tstep_start: float = 3.0,
    tstep_end: float = 0.6,
    rstep_start: float = 35.0,
    rstep_end: float = 10.0,

    # NEW: biased move toward pocket center
    biased: bool = True,
    bias_strength: float = 0.35,
    rotate_prob: float = 0.35,
    noise_frac: float = 0.5,

    verbose=True,
):
    """
    Rigid Monte Carlo docking with:
    - optional annealing
    - optional pocket centroid constraint
    - optional biased translation toward pocket center

    Constraint:
      if pocket_center and pocket_radius are provided, enforce:
        dist(ligand_centroid, pocket_center) <= pocket_radius
      Moves violating constraint are rejected without scoring.

    Biased move:
      If biased=True and pocket_center provided, translation steps are a mix of:
        - component toward pocket center
        - random noise
      plus rotations with probability rotate_prob.
    """
    from .pose import Pose
    from .moves import random_move, biased_move

    prep_common = prepared_common or {}

    pose = Pose(ligand.atoms)
    if seed_pose is not None:
        pose = seed_pose.copy()

    # --- initialize pose
    mode = (init_mode or "current").strip().lower()
    if mode == "outside":
        ux, uy, uz = _random_unit_vector()
        pose.tx += ux * init_distance
        pose.ty += uy * init_distance
        pose.tz += uz * init_distance

        a = math.radians(float(init_spin_deg))
        pose.rx = random.uniform(-a, a)
        pose.ry = random.uniform(-a, a)
        pose.rz = random.uniform(-a, a)
    elif mode == "current":
        pass
    else:
        raise ValueError(f"Unknown init_mode={init_mode!r}. Use current|outside")

    # If constrained, project initial centroid into sphere (slightly inside)
    if pocket_center is not None and pocket_radius is not None:
        c = pose.transformed_centroid()
        if not _within_sphere(c, pocket_center, pocket_radius):
            dx = c[0] - pocket_center[0]
            dy = c[1] - pocket_center[1]
            dz = c[2] - pocket_center[2]
            d = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-12
            scale = (float(pocket_radius) * 0.98) / d
            new_c = (
                pocket_center[0] + dx * scale,
                pocket_center[1] + dy * scale,
                pocket_center[2] + dz * scale,
            )
            pose.tx += (new_c[0] - c[0])
            pose.ty += (new_c[1] - c[1])
            pose.tz += (new_c[2] - c[2])

    best = pose.copy()

    # --- initial score
    ligand_atoms = pose.transformed_atoms()
    best.score = score_complex_chemplp(
        receptor_atoms,
        ligand_atoms,
        ligand_rdkit_mol=ligand.rdkit_mol,
        params=chemplp_params,
    ).total
    pose.score = best.score

    if verbose:
        msg = f"[MC] init_mode={mode}  init_score={best.score:.3f}"
        if pocket_center is not None and pocket_radius is not None:
            msg += f"  constraint=sphere(r={float(pocket_radius):.1f}Å)"
        if anneal:
            msg += f"  anneal=True stages={int(stages)}"
        else:
            msg += f"  anneal=False T={float(temperature):.3f}"
        if biased and pocket_center is not None:
            msg += f"  biased=True(bias={float(bias_strength):.2f}, rot_p={float(rotate_prob):.2f})"
        print(msg)

    # --- stage loop
    if anneal:
        stages = max(1, int(stages))
        steps_per_stage = max(1, int(n_steps) // stages)
        total_steps = steps_per_stage * stages
    else:
        total_steps = int(n_steps)
        steps_per_stage = total_steps
        stages = 1

    step_global = 0
    for s in range(stages):
        if anneal:
            t = 0.0 if stages == 1 else (s / (stages - 1))
            T = _lin_interp(float(temp_start), float(temp_end), t)
            ts = _lin_interp(float(tstep_start), float(tstep_end), t)
            rs = _lin_interp(float(rstep_start), float(rstep_end), t)
            if verbose:
                print(f"[MC] stage {s+1}/{stages}  T={T:.3f}  t_step={ts:.2f}Å  r_step={rs:.1f}°")
        else:
            T = float(temperature)
            ts = float(t_step)
            rs = float(r_step)

        for _ in range(steps_per_stage):
            step_global += 1
            if step_global > total_steps:
                break

            trial = pose.copy()

            # --- propose move (biased if enabled and pocket_center is known)
            if biased and (pocket_center is not None):
                biased_move(
                    trial,
                    pocket_center=pocket_center,
                    t_step=ts,
                    r_step=rs,
                    bias_strength=float(bias_strength),
                    rotate_prob=float(rotate_prob),
                    noise_frac=float(noise_frac),
                )
            else:
                random_move(trial, ts, rs)

            # --- hard constraint: centroid must remain within pocket sphere
            if pocket_center is not None and pocket_radius is not None:
                ctrial = trial.transformed_centroid()
                if not _within_sphere(ctrial, pocket_center, float(pocket_radius)):
                    continue  # reject without scoring

            # --- score
            ligand_atoms = trial.transformed_atoms()
            res = score_complex_chemplp(
                receptor_atoms,
                ligand_atoms,
                ligand_rdkit_mol=ligand.rdkit_mol,
                params=chemplp_params,
            )
            Enew = res.total
            Eold = pose.score
            dE = Enew - Eold

            if dE <= 0:
                accept = True
            else:
                accept = (random.random() < math.exp(-dE / T))

            if accept:
                trial.score = Enew
                pose = trial
                if Enew < best.score:
                    best = trial.copy()

            if verbose and (step_global % 200 == 0):
                print(f"[MC] step {step_global:6d}  curr={pose.score: .3f}  best={best.score: .3f}")

    return best
