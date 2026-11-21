"""Repeatable synthetic timing probe; this is NOT a full-capsid benchmark.

Run from an environment where mastercap is installed:
    python benchmarks/search_shell.py --residues 600 --iterations 30 --repeats 3
"""
import argparse
import json
import statistics

import numpy as np

from mastercap import __version__
from mastercap.analysis import analyze, target_profiles
from mastercap.generate import search, select_inventory
from mastercap.model import Atom, Structure


def synthetic_shell(residues):
    k = np.arange(residues)
    z = 1-2*(k+0.5)/residues
    phi = k*np.pi*(3-np.sqrt(5))
    directions = np.column_stack([np.sqrt(1-z*z)*np.cos(phi), np.sqrt(1-z*z)*np.sin(phi), z])
    radius = 6*(residues/60)**(1/3)*(1+0.05*np.sin(k))
    centers = directions*radius[:, None]
    tangent = np.cross(directions, [0, 0, 1.0])
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    xyz = np.array([p for i in range(residues) for p in
                    (centers[i]-0.7*tangent[i], centers[i]+0.7*tangent[i])])
    atoms = [Atom("CA" if i % 2 == 0 else "C", "C", "GLY", i//2) for i in range(2*residues)]
    return Structure(atoms, xyz)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residues", type=int, default=600)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-limit", type=float, default=1800, help="Search seconds; use push-limited runs for seed reproducibility")
    parser.add_argument("--refine-updates", type=int, default=50)
    parser.add_argument("--refine-iterations", type=int, default=None)
    parser.add_argument("--refine-patience", type=int, default=10)
    parser.add_argument("--selection-objective", choices=("total", "worst-element"), default="total")
    args = parser.parse_args()
    if args.residues < 3 or args.iterations < 0 or args.repeats < 1 or args.seed < 0 or not np.isfinite(args.time_limit) or args.time_limit <= 0:
        parser.error("require residues >= 3, iterations >= 0, repeats >= 1 and seed >= 0")
    source = synthetic_shell(args.residues)
    analysis = analyze(source, sigma=0.4)
    edges, profiles = target_profiles(analysis)
    runs = []
    for _ in range(args.repeats):
        rng = np.random.default_rng(args.seed)
        inventory, _ = select_inventory(analysis, edges, profiles, rng)
        _, details = search(inventory, analysis, edges, profiles, analysis.center, rng,
                            max_iterations=args.iterations, time_limit=args.time_limit,
                            refine_updates=args.refine_updates, refine_iterations=args.refine_iterations,
                            refine_patience=args.refine_patience, selection_objective=args.selection_objective)
        runs.append({k: details.get(k) for k in
                     ("iterations", "preparation_seconds", "elapsed_seconds", "termination", "geometry", "errors_at_push", "first_feasible_iteration",
                      "candidate_iteration", "profile_updates", "rollback_count", "median_recovery_iterations",
                      "feasible_iterations", "push_history", "best_history", "initial_metrics", "metrics",
                      "refinement_stop_limit", "profile_error_objective", "refinement_settings",
                      "refinement_updates", "last_search_geometry")})
    # Before 0.1.1 elapsed_seconds already included initialization; the fallback
    # allows the same probe to compare that revision without subtracting work.
    elapsed = [run["elapsed_seconds"]+(run["preparation_seconds"] or 0.0) for run in runs]
    print(json.dumps({"mastercap_version": __version__, "case": "synthetic_carbon_shell",
                      "parameters": vars(args), "atoms": len(source.atoms),
                      "median_initialization_and_search_seconds": statistics.median(elapsed),
                      "runs": runs}, indent=2))


if __name__ == "__main__":
    main()
