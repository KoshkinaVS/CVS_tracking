# CVS_tracking
Python library for tracking Coherent Vortical Structures (CVS) based on Eulerian criteria fields in atmospheric and oceanographic datasets.

## Overview
This repository provides tools for identifying and tracking trajectories of Coherent Vortical Structures (CVS) — using Eulerian vortex identification methods applied to velocity fields. The implementation builds upon vortex detection algorithms (see [vortex_identification](https://github.com/KoshkinaVS/vortex_identification)) and adds temporal tracking capabilities through step‑by‑step trajectory association.


Итогом работы скрипта является совокупность траекторий Ц/АЦ КВС в формате csv.


- **Data Support**:
  - NETCDF files from [vortex_identification lib](https://github.com/KoshkinaVS/vortex_identification) with DBSCAN-separated CVSs with additional info.
 
- **Output Format**:
The script produces a collection of cyclonic (C) and anticyclonic (AC) CVS trajectories in CSV format.

## Repository Structure

```
CVS_tracking/ 
└── vortex_dir/
    ├── DBSCAN_tracking_multiprocessed.py     # Example
    ├── step_of_tracking.py                   # Core tracking functions: track_init (initializes new trajectories) and auxiliary utilities.
    ├── func_for_global_only.py               # func step_of_tracking (performs one tracking step for the case with global extrema)
    ├── func_for_local_2_phase.py             # func step_of_tracking (performs one tracking step for the case with local extrema with local radius info)
    └── func_for_local_global.py              # func step_of_tracking (performs one tracking step for the case with global extrema initialization and local extrema tracking)
```
