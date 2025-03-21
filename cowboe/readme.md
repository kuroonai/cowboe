# COWBOE
## _Construction Of Windows Based on free Energy_

[![N|Solid](https://xiresearch.org/wp-content/uploads/2019/11/xiresearch-withcolor5-300x96.png)](https://xiresearch.org/)

COWBOE is a Python 3 tool for parameter selection in Umbrella Sampling, a free energy calculation method used in Molecular dynamics simulations.

## Features

- Parameter optimization for Umbrella Sampling windows and force constants
- Object-oriented architecture with specialized classes for different tasks
- Nelder-Mead and Restricted Nelder-Mead simplex optimization algorithms
- Comprehensive visualization tools for PMF curves and optimization progress
- Analysis tools for trajectory distributions and overlap coefficients

## Installation

COWBOE requires Python 3.7+ to run.

### Creating a virtual environment

**Using venv**

For Linux/macOS:
```sh
python3 -m venv /path/to/new/virtual/environment
source <venv>/bin/activate
```

For Windows:
```sh
python -m venv c:\path\to\myenv
<venv>\Scripts\activate.bat
```

**Using conda**
```sh
conda create -n cowboe python=3.9
conda activate cowboe
```

### Installing COWBOE

Using pip:
```sh
python3 -m pip install --upgrade pip
pip install cowboe
```

Building from source:
```sh
cd <location of extracted files with setup.py file>
pip3 install numpy scipy matplotlib seaborn shapely imageio pandas
python setup.py install
```

Verify installation:
```sh
python -c "import cowboe"
```

## Usage

COWBOE features a modular, object-oriented architecture for parameter optimization and analysis:

```python
from cowboe import (
    pmf_to_points, cowboe, cowboe_wham, ComparisonAnalyzer,
    Visualizer, WindowCalculator, KSAnalyzer, OverlapAnalyzer,
    TrajectoryProcessor, PMFPlotter, NelderMead, NMProgressAnalyzer,
    SurfacePlotter, create_progress_file, SETTINGS, update_settings
)
import numpy as np
import os

# Update settings
update_settings('common', {"param_B": 2.0})

# Process test PMF file
pmf_to_points(
    location='./testpmf',
    test_pmf='test_pmf.txt',
    order=12
)

# Generate windows using COWBOE algorithm
cowboe(
    param_a=3.5000,
    param_v=0.8000,
    sampling_considered=8,
    name='3',
    subtype='100mc',
    location='./testpmf',
    equal_sampling=True
)

# Calculate PMF using WHAM
cowboe_wham(
    name='benchmark.txt',
    location='./benchmark',
    mc_trials=0,
    hist_min=2.0,
    hist_max=14.5
)

cowboe_wham(
    name='3.txt',
    location='./3',
    mc_trials=0,
    hist_min=2.0,
    hist_max=14.5
)

# Plot and analyze PMFs
plotter = PMFPlotter()
plotter.plot_pmf(pmf_file='benchmark.txt', name='benchmark', splice=0)
plotter.plot_pmf(pmf_file='3.txt', name='3', splice=0)

# Compare PMFs
analyzer = ComparisonAnalyzer()
analyzer.calculate_pmf_difference(test_file='3.txt', bench_file='benchmark.txt')
analyzer.compare_pmfs(
    pmf_files=['benchmark.txt', '3.txt'],
    name='comparison',
    splices=[8, 10]
)

# Nelder-Mead optimization
nm = NelderMead()
a_values = [2.0, 2.9, 3.5]
v_values = [0.75, 0.87, 0.80]
fitness_values = [1.9834, 1.3844, 4.7587]
moves = nm.optimize(a_values, v_values, fitness_values)

# Analyze optimization progress
progress_data = np.array([
    [[2.0, 0.75, 1.9943], [2.9, 0.87, 1.8232], [3.5, 0.8, 4.7636]],
    [[2.0, 0.75, 1.9943], [2.9, 0.87, 1.8232], [1.6571, 0.82, 0.9899]],
    [[2.4028, 0.94, 2.0045], [2.9, 0.87, 1.8232], [1.6571, 0.82, 0.9899]]
])
create_progress_file(progress_data)

progress_analyzer = NMProgressAnalyzer()
progress_analyzer.analyze_progress(progress_file='progress.txt')

# Create 3D surface plot of parameter space
plotter3d = SurfacePlotter()
plotter3d.create_surface_plot(progress_file='progress.txt')

# Analyze trajectory distribution
ks_analyzer = KSAnalyzer()
ks_analyzer.analyze_distributions(
    location='./benchmark',
    listfile='list.txt',
    percentage=85
)

# Calculate overlap coefficients
overlap_analyzer = OverlapAnalyzer()
overlap_analyzer.calculate_overlap(
    location='./benchmark',
    name='benchmark',
    listfile='list.txt'
)

# Process trajectories
processor = TrajectoryProcessor()
processor.slice_trajectories(
    location='./benchmark',
    name='benchmark',
    percentage=50.0,
    start=0
)
```

## Citation

Please cite the COWBOE algorithm as follows:
```
@article{vasudevanpotential,
  title={Potential of Mean Force of Short-Chain Surface Adsorption using Non-Uniform Sampling Windows for Optimal Computational Efficiency},
  author={Vasudevan, Naveen Kumar and Li, Dongyang and Xi, Li},
  journal={Macromolecular Theory and Simulations},
  pages={2300057},
  publisher={Wiley Online Library}
}
```

## Nelder-Mead Simplex Algorithm

The primary difference between the NM and restricted NM is that the RNM doesn't include the expansion step after reflection. The NelderMead class provides possible NM steps for a given simplex based on this algorithm:

```
Initialize simplex with n+1 points where n is the number of parameters.

While not converged:
    1. Order points by fitness
    2. Calculate centroid of all points except worst
    3. Calculate reflection point
    4. If reflection better than best:
       - (NM only) Calculate expansion point and use better of reflection/expansion
    5. Else if reflection worse than all but worst:
       - Calculate outward contraction
       - If outward better than reflection, use it
       - Else, shrink the simplex
    6. Else if reflection worse than all:
       - Calculate inward contraction
       - If inward better than worst, use it
       - Else, shrink the simplex
    7. Else:
       - Use reflection
    8. Check convergence
```

The library provides both standard and restricted implementations through the `restricted` parameter.

## License

GNU General Public License v3.0
