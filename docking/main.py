"""docking.main

Rigid docking driver with ChemPLP-like scoring + pocket restriction + optimizers.

Modes:
  --opt none   : score input pose only
  --opt mc     : Monte Carlo rigid docking (anneal + constraint + optional bias)
  --opt ga     : Genetic algorithm rigid docking (constraint + optional bias) + optional MC refine

Examples:
  python -m docking.main --receptor receptor.pdb --ligand ligand.sdf --pocket-radius 12 --pocket-mode atoms --scoring chemplp --chemplp-weights chemplp.json --opt mc --mc-anneal --mc-biased
  python -m docking.main --receptor receptor.pdb --ligand ligand.sdf --pocket-radius 12 --pocket-mode atoms --scoring chemplp --chemplp-weights chemplp.json --opt ga --ga-pop 50 --ga-gen 60 --refine-mc-after-ga
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Optional

from docking.io.read_pdb import read_pdb_atoms
from docking.io.read_sdf import read_sdf_first_mol
from docking.geometry.distances import distance
from docking.geometry.pocket import define_pocket_from_ligand

from docking.scoring.total_score import score_complex, score_complex_chemplp
from docking.scoring.electrostatics import top_electrostatic_pairs
from docking.scoring.chemplp import top_hbond_contacts, top_clash_pairs

from docking.optimize.monte_carlo import monte_carlo_rigid
from docking.optimize.genetic import genetic_rigid


def _closest_distance(receptor_atoms, ligand_atoms) -> float:
    min_r = 1e9
    for ra in receptor_atoms:
        for la in ligand_atoms:
            r = distance(ra.x, ra.y, ra.z, la.x, la.y, la.z)
            if r < min_r:
                min_r = r
    return float(min_r)


def _load_json_maybe(json_or_path: Optional[str]) -> Dict[str, Any]:
    """
    Accept:
      - None => {}
      - JSON string (starts with { or [) => parsed
      - path to .json => loaded
    """
    if not json_or_path:
        return {}

    s = json_or_path.strip()

    if s.startswith("{") or s.startswith("["):
        return json.loads(s)

    looks_like_path = s.lower().endswith(".json") or ("\\" in s) or ("/" in s)
    if looks_like_path:
        if not (os.path.exists(s) and os.path.isfile(s)):
            raise FileNotFoundError(f"ChemPLP weights file not found: {s}")
        with open(s, "r", encoding="utf-8") as f:
            return json.load(f)

    return json.loads(s)


def _print_ele_pairs(title: str, pairs) -> None:
    print(title)
    if not pairs:
        print("  (no pairs within cutoff)")
        return
    for i, p in enumerate(pairs, 1):
        rlab = f"{p.ra_resname} {p.ra_chain}{p.ra_resseq} {p.ra_name} [{p.ra_element}]"
        llab = f"lig#{p.la_idx} [{p.la_element}]"
        print(
            f"  {i:>2d}) {rlab:<20s} q={p.ra_charge:+.3f}  <->  {llab:<10s} q={p.la_charge:+.3f}  "
            f"r={p.r:5.2f} Å  E={p.e: .3f}"
        )


def _print_hbond_debug(hb_list) -> None:
    if not hb_list:
        print("  (no H-bond contacts found)")
        return
    for i, h in enumerate(hb_list, 1):
        rlab = f"{h.ra_resname} {h.ra_chain}{h.ra_resseq} {h.ra_name} [{h.ra_element}]"
        llab = f"lig#{h.la_idx} [{h.la_element}]"
        ang = "n/a" if h.angle is None else f"{h.angle:6.1f}°"
        print(
            f"  {i:>2d}) {rlab:<20s} <-> {llab:<10s}  r={h.r:5.2f} Å  angle={ang}  "
            f"E_dist={h.e_dist: .3f}  E={h.e_total: .3f}"
        )


def _print_clash_debug(cl_list) -> None:
    if not cl_list:
        print("  (no clashes above hard-cap threshold)")
        return
    for i, c in enumerate(cl_list, 1):
        rlab = f"{c.ra_resname} {c.ra_chain}{c.ra_resseq} {c.ra_name} [{c.ra_element}]"
        llab = f"lig#{c.la_idx} [{c.la_element}]"
        print(
            f"  {i:>2d}) {rlab:<20s} <-> {llab:<10s}  r={c.r:5.2f} Å  r0={c.r0:5.2f} Å  "
            f"overlap={c.overlap:5.2f} Å  penalty={c.penalty: .1f}"
        )


def _score_and_print(
    scoring: str,
    receptor_atoms,
    ligand_atoms,
    ligand_rdkit_mol,
    chem_params: Dict[str, Any],
) -> float:
    if scoring == "simple":
        b = score_complex(receptor_atoms, ligand_atoms)
        print("---- Score breakdown (simple) ----")
        print(f"vdW:   {b.vdw: .4f}")
        print(f"Ele:   {b.ele: .4f}")
        print(f"Hbond: {b.hbond: .4f}")
        print(f"TOTAL: {b.total: .4f}")
        return float(b.total)

    cb = score_complex_chemplp(
        receptor_atoms,
        ligand_atoms,
        ligand_rdkit_mol=ligand_rdkit_mol,
        params=chem_params,
    )
    print("---- Score breakdown (ChemPLP-like, advanced) ----")
    print(f"Steric:  {cb.steric: .4f}")
    print(f"Hbond:   {cb.hbond: .4f}")
    print(f"Lipo:    {cb.lipo: .4f}")
    print(f"Clashes: {cb.clashes: .4f}")
    print(f"TOTAL:   {cb.total: .4f}")
    return float(cb.total)


def _write_pose_to_sdf(writer, pose, rdkit_mol, name: str, props: dict):
    from rdkit import Chem
    mol = Chem.Mol(rdkit_mol)
    conf = mol.GetConformer()
    L_xyz = pose.transformed_xyz()
    for aidx in range(mol.GetNumAtoms()):
        x, y, z = map(float, L_xyz[aidx])
        conf.SetAtomPosition(aidx, Chem.rdGeometry.Point3D(x, y, z))
    mol.SetProp("_Name", name)
    for k, v in (props or {}).items():
        mol.SetProp(str(k), str(v))
    writer.write(mol)



def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--receptor", required=True, help="Receptor PDB file")
    ap.add_argument("--ligand", required=True, help="Ligand SDF file (pose)")

    ap.add_argument("--pocket-radius", type=float, default=12.0, help="Pocket radius (Å). 0 disables.")
    ap.add_argument("--pocket-mode", choices=["centroid", "atoms"], default="centroid")

    ap.add_argument("--scoring", choices=["simple", "chemplp"], default="chemplp")
    ap.add_argument(
        "--chemplp-weights",
        default=None,
        help='ChemPLP parameters JSON: JSON string \'{"w_hbond":2.0}\' or path to .json file.',
    )

    # Scoring engine: python fallback vs numba
    ap.add_argument(
        "--engine",
        choices=["auto", "numba", "python"],
        default="auto",
        help="Scoring backend. auto=use numba if available else python.",
    )

    # Precision modes (Glide-like)
    ap.add_argument(
        "--precision",
        choices=["HTVS", "SP", "XP"],
        default="SP",
        help="Scoring strictness preset (default: SP).",
    )

    # Initial placement toggle (default ON; this flag disables it)
    ap.add_argument(
        "--no-initial-placement",
        action="store_true",
        help="Disable Glide-like initial placement (default is enabled).",
    )





    # Choose optimizer
    ap.add_argument("--opt", choices=["none", "mc", "ga"], default="none", help="Optimization method (rigid ligand)")
    ap.add_argument("--refine-mc-after-ga", action="store_true", help="After GA, refine best pose with MC annealing")

    # Electrostatics debug
    ap.add_argument("--debug-ele", action="store_true")
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--ele-cutoff", type=float, default=12.0)
    ap.add_argument("--ele-eps", type=float, default=10.0)

    # ChemPLP debug (applies to final pose if optimized)
    ap.add_argument("--chemplp-debug-hbond", action="store_true")
    ap.add_argument("--topn-hb", type=int, default=20)
    ap.add_argument("--chemplp-debug-clash", action="store_true")
    ap.add_argument("--topn-clash", type=int, default=20)

    # -------------------- MC params --------------------
    ap.add_argument("--mc-steps", type=int, default=30000)
    ap.add_argument("--mc-temp", type=float, default=1.5)        # used when anneal is off
    ap.add_argument("--mc-tstep", type=float, default=1.0)       # used when anneal is off
    ap.add_argument("--mc-rstep", type=float, default=20.0)      # used when anneal is off

    ap.add_argument("--mc-init", choices=["current", "outside"], default="outside")
    ap.add_argument("--mc-init-distance", type=float, default=30.0)
    ap.add_argument("--mc-init-spin", type=float, default=180.0)

    ap.add_argument("--mc-anneal", action="store_true")
    ap.add_argument("--mc-stages", type=int, default=6)

    ap.add_argument("--mc-temp-start", type=float, default=3.0)
    ap.add_argument("--mc-temp-end", type=float, default=0.6)

    ap.add_argument("--mc-tstep-start", type=float, default=4.0)
    ap.add_argument("--mc-tstep-end", type=float, default=0.6)

    ap.add_argument("--mc-rstep-start", type=float, default=45.0)
    ap.add_argument("--mc-rstep-end", type=float, default=10.0)

    ap.add_argument("--mc-biased", action="store_true")
    ap.add_argument("--mc-bias-strength", type=float, default=0.35)
    ap.add_argument("--mc-rotate-prob", type=float, default=0.35)
    ap.add_argument("--mc-noise-frac", type=float, default=0.5)

    # -------------------- GA params --------------------
    ap.add_argument("--ga-pop", type=int, default=40, help="GA population size")
    ap.add_argument("--ga-gen", type=int, default=40, help="GA generations")
    ap.add_argument("--ga-elite-frac", type=float, default=0.15, help="Fraction of elites kept each generation")
    ap.add_argument("--ga-crossover-prob", type=float, default=0.35, help="Crossover probability")
    ap.add_argument("--ga-mutate-prob", type=float, default=0.80, help="Mutation probability")
    ap.add_argument("--ga-mutate-sigma-t", type=float, default=0.75, help="Translation mutation sigma (Å)")
    ap.add_argument("--ga-mutate-sigma-r", type=float, default=0.35, help="Rotation mutation sigma (rad)")

    ap.add_argument("--ga-print-every", type=int, default=25, help="Print GA progress every N generations")
    ap.add_argument("--ga-dump-best-sdf", default=None, help="Write best individual every --ga-print-every generations to this SDF (multi-mol)")

    # Backward-compat/legacy args (kept; not used directly)
    ap.add_argument("--ga-mutation-rate", type=float, default=0.8, help=argparse.SUPPRESS)
    ap.add_argument("--ga-tournament-k", type=int, default=3, help=argparse.SUPPRESS)
    args = ap.parse_args()

    # Read inputs
    receptor_atoms_full = read_pdb_atoms(args.receptor, keep_hetatm=True, drop_waters=True)
    ligand = read_sdf_first_mol(args.ligand, add_hs=True, compute_gasteiger=True)

    # Pocket selection + pocket center
    pocket_center = None
    if args.pocket_radius and args.pocket_radius > 0:
        pocket_atoms, center = define_pocket_from_ligand(
            receptor_atoms_full,
            ligand.atoms,
            pocket_radius=args.pocket_radius,
            pocket_mode=args.pocket_mode,
        )
        receptor_atoms = pocket_atoms
        pocket_center = center
        print(f"Pocket mode: {args.pocket_mode}")
        print(f"Pocket center (ligand centroid): ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
        print(f"Pocket atoms within {args.pocket_radius:.1f} Å: {len(receptor_atoms)} / {len(receptor_atoms_full)}")
    else:
        receptor_atoms = receptor_atoms_full
        print("Pocket selection disabled (using full receptor).")

    print(f"Receptor atoms (full):   {len(receptor_atoms_full)}")
    print(f"Receptor atoms (scored): {len(receptor_atoms)}")
    print(f"Ligand atoms:            {len(ligand.atoms)}  ({ligand.name})")

    # Load ChemPLP params if needed
    chem_params: Dict[str, Any] = {}
    if args.scoring == "chemplp" or args.opt in ("mc", "ga"):
        chem_params = _load_json_maybe(args.chemplp_weights)
        if chem_params:
            print("ChemPLP params loaded:", chem_params)


    # -------------------- speed caches (Numba) --------------------
    prepared_common: Dict[str, Any] = {}
    if args.scoring == "chemplp" or args.opt in ("mc", "ga"):
        if args.engine in ("auto", "numba"):
            try:
                from docking.scoring.numba import prepare_receptor_arrays, prepare_ligand_static, prepare_pi_features
                prepared_common["R_arrays"] = prepare_receptor_arrays(receptor_atoms)
                prepared_common["L_static"] = prepare_ligand_static(ligand.atoms, ligand.rdkit_mol)
                # PI features can be computed once (cheap) and reused
                prepared_common["PI"] = prepare_pi_features(receptor_atoms, ligand.rdkit_mol)
            except Exception as e:
                if args.engine == "numba":
                    print("[WARN] --engine numba requested but numba-prep failed; falling back to python:", e)
                # leave prepared_common empty; engine 'auto' will choose python
                prepared_common = {}

    # -------------------- initial placement seeding (Glide-like) --------------------
    seed_poses = None
    seed_pose_mc = None
    if (args.opt in ("mc", "ga")) and (not args.no_initial_placement):
        try:
            from docking.placement import generate_initial_poses
            seed_poses = generate_initial_poses(
                receptor_atoms=receptor_atoms,
                ligand=ligand,
                pocket_center=pocket_center,
                pocket_radius=float(args.pocket_radius),
                max_poses=max(int(args.ga_pop), 80),
                rotations_per_match=24 if args.precision != "XP" else 40,
            )
            # fast-score seeds and sort
            for p in seed_poses:
                L_xyz = p.transformed_xyz()
                p.score = score_complex_chemplp(
                    receptor_atoms,
                    ligand.atoms,
                    ligand_rdkit_mol=ligand.rdkit_mol,
                    params=chem_params,
                    precision=args.precision,
                    engine=args.engine,
                    prepared={**prepared_common, "L_xyz": L_xyz},
                ).total
            seed_poses.sort(key=lambda p: p.score)
            seed_pose_mc = seed_poses[0].copy() if seed_poses else None
            if seed_poses:
                print(f"Initial placement: generated {len(seed_poses)} seed poses; best score={seed_poses[0].score:.3f}")
        except Exception as e:
            print("[WARN] Initial placement failed; continuing without seeding:", e)
            seed_poses = None
            seed_pose_mc = None

    # Final pose atoms for debug prints
    final_atoms = ligand.atoms
    final_score = None

    # -------------------- run mode --------------------

    final_atoms = ligand.atoms
    final_score = None

    if args.opt == "none":
        final_score = _score_and_print(args.scoring, receptor_atoms, ligand.atoms, ligand.rdkit_mol, chem_params)

    elif args.opt == "mc":
        if pocket_center is None or not (args.pocket_radius and args.pocket_radius > 0):
            raise ValueError("MC requires --pocket-radius > 0 (pocket sphere needed for centroid constraint).")

        print("\n=== Monte Carlo rigid docking (ChemPLP) ===")
        best_pose = monte_carlo_rigid(
            receptor_atoms,
            ligand,
            chemplp_params=chem_params,
            steps=int(args.mc_steps),
            seed_pose=seed_pose_mc if 'seed_pose_mc' in locals() else None,
            pocket_center=pocket_center,
            pocket_radius=float(args.pocket_radius),
            anneal=bool(args.mc_anneal),
            stages=int(args.mc_stages),
            temp_start=float(args.mc_temp_start),
            temp_end=float(args.mc_temp_end),
            tstep_start=float(args.mc_tstep_start),
            tstep_end=float(args.mc_tstep_end),
            rstep_start=float(args.mc_rstep_start),
            rstep_end=float(args.mc_rstep_end),
            biased=bool(args.mc_biased),
            bias_strength=float(args.mc_bias_strength),
            rotate_prob=float(args.mc_rotate_prob),
            noise_frac=float(args.mc_noise_frac),
            verbose=True,
        )
        final_atoms = best_pose.transformed_atoms()
        print("\n=== Best pose after MC ===")
        print(f"Best MC total score: {best_pose.score:.4f}")
        final_score = _score_and_print("chemplp", receptor_atoms, final_atoms, ligand.rdkit_mol, chem_params)

    elif args.opt == "ga":
        if pocket_center is None or not (args.pocket_radius and args.pocket_radius > 0):
            raise ValueError("GA requires --pocket-radius > 0 (pocket sphere needed for centroid constraint).")

        print("\n=== Genetic Algorithm rigid docking (ChemPLP) ===")

        ga_state = {"writer": None, "last_gen": None}

        def _ga_checkpoint_cb(best_pose, gen):
            """Write the current best pose to an SDF trajectory (APPEND).

            NOTE: Some GA implementations may call this callback with a constant `gen`
            value (e.g., total generations). We therefore track our own write counter
            to ensure we don't accidentally write only a single record.
            """
            if not args.ga_dump_best_sdf:
                return
            if best_pose is None:
                return

            # monotonically increasing counter of how many poses we wrote
            ga_state["n_written"] = int(ga_state.get("n_written", 0)) + 1
            n_written = ga_state["n_written"]

            try:
                if ga_state["writer"] is None:
                    from rdkit import Chem
                    ga_state["writer"] = Chem.SDWriter(args.ga_dump_best_sdf)

                # use both `gen` and the write index in the name so it's always unique
                safe_gen = int(gen) if gen is not None else -1
                _write_pose_to_sdf(
                    ga_state["writer"],
                    best_pose,
                    ligand.rdkit_mol,
                    name=f"ga_best_step_{n_written:05d}_gen_{safe_gen:04d}",
                    props={
                        "gen": safe_gen,
                        "write_idx": n_written,
                        "score": getattr(best_pose, "score", None),
                        "precision": args.precision,
                        "engine": args.engine,
                    },
                )
            except Exception as e:
                print(f"[WARN] Failed to dump best SDF at callback #{n_written} (gen={gen}): {e}")

        best_pose = genetic_rigid(
            receptor_atoms,
            ligand,
            chemplp_params=chem_params,
            pop_size=int(args.ga_pop),
            generations=int(args.ga_gen),
            elite_frac=float(args.ga_elite_frac),
            tournament_k=int(args.ga_tournament_k),
            mutate_prob=float(args.ga_mutate_prob),
            mutate_sigma_t=float(args.ga_mutate_sigma_t),
            mutate_sigma_r=float(args.ga_mutate_sigma_r),
            crossover_prob=float(args.ga_crossover_prob),
            seed_poses=seed_poses if 'seed_poses' in locals() else None,
            precision=args.precision,
            engine=args.engine,
            prepared_common=prepared_common if 'prepared_common' in locals() else None,
            verbose=True,
            print_every=int(args.ga_print_every),
            on_checkpoint=_ga_checkpoint_cb,
            pocket_center=pocket_center,
            pocket_radius=float(args.pocket_radius),
        )

        # close trajectory writer (RDKit)
        try:
            if ga_state.get("writer", None) is not None:
                ga_state["writer"].close()
        except Exception:
            pass

        if args.refine_mc_after_ga:
            print("\n=== Refinement: MC annealing from GA best (ChemPLP) ===")
            saved_atoms = ligand.atoms
            try:
                ligand.atoms = best_pose.transformed_atoms()
                best_pose2 = monte_carlo_rigid(
                    receptor_atoms,
                    ligand,
                    chemplp_params=chem_params,
                    steps=int(args.mc_steps),
                    seed_pose=None,
                    pocket_center=pocket_center,
                    pocket_radius=float(args.pocket_radius),
                    anneal=True,
                    stages=int(args.mc_stages),
                    temp_start=float(args.mc_temp_start),
                    temp_end=float(args.mc_temp_end),
                    tstep_start=float(args.mc_tstep_start),
                    tstep_end=float(args.mc_tstep_end),
                    rstep_start=float(args.mc_rstep_start),
                    rstep_end=float(args.mc_rstep_end),
                    biased=bool(args.mc_biased),
                    bias_strength=float(args.mc_bias_strength),
                    rotate_prob=float(args.mc_rotate_prob),
                    noise_frac=float(args.mc_noise_frac),
                    verbose=True,
                )
                best_pose = best_pose2
            finally:
                ligand.atoms = saved_atoms

        final_atoms = best_pose.transformed_atoms()
        print("\n=== Best pose after GA ===")
        print(f"Best GA total score: {best_pose.score:.4f}")
        final_score = _score_and_print("chemplp", receptor_atoms, final_atoms, ligand.rdkit_mol, chem_params)

        if ga_state["writer"] is not None:
            ga_state["writer"].close()
            print(f"Wrote GA best-per-checkpoint poses to: {args.ga_dump_best_sdf}")

    else:
        raise ValueError(f"Unknown --opt {args.opt}")

    # Distances
    min_r = _closest_distance(receptor_atoms, final_atoms)
    print(f"Closest receptor-ligand atom distance (final pose): {min_r:.3f} Å")

    # ChemPLP debug on final pose
    if args.chemplp_debug_hbond:
        hb = top_hbond_contacts(
            receptor_atoms,
            final_atoms,
            ligand_rdkit_mol=ligand.rdkit_mol,
            params=chem_params,
            top_n=args.topn_hb,
        )
        print(f"\n---- TOP {args.topn_hb} ChemPLP H-bond contacts (final pose) ----")
        _print_hbond_debug(hb)

    if args.chemplp_debug_clash:
        cl = top_clash_pairs(
            receptor_atoms,
            final_atoms,
            params=chem_params,
            top_n=args.topn_clash,
        )
        print(f"\n---- TOP {args.topn_clash} ChemPLP clash pairs (final pose) ----")
        _print_clash_debug(cl)

    # Electrostatics debug on final pose
    if args.debug_ele:
        attractive, repulsive = top_electrostatic_pairs(
            receptor_atoms,
            final_atoms,
            cutoff=args.ele_cutoff,
            eps=args.ele_eps,
            top_n=args.topn,
        )
        print()
        _print_ele_pairs(f"---- TOP {args.topn} attractive electrostatic pairs ----", attractive)
        print()
        _print_ele_pairs(f"---- TOP {args.topn} repulsive electrostatic pairs ----", repulsive)


if __name__ == "__main__":
    main()
