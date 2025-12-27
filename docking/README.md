# Docking MVP scaffold

This ZIP contains the requested project structure and starter modules.

## Run (requires RDKit)
```bash
python -m docking.main --receptor receptor.pdb --ligand ligand.sdf
```

Notes:
- This MVP only *scores* an existing ligand pose (no pose search yet).
- Receptor charges are assumed 0.0 for now.
