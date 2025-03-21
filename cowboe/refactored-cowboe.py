"""
COWBOE - Construction Of Windows Based On free Energy. 
Package for optimization and selection of parameters for umbrella sampling.
"""

import os
import sys
import math
import glob
import time
import random
import shutil
import pickle
import imageio
import inspect
import json
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from datetime import datetime
from scipy.stats import ks_2samp
from matplotlib import animation, cm
from matplotlib import pyplot as plt
from math import sqrt, fabs, erf, log
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema
from shapely.geometry import Polygon
import matplotlib.ticker as mticker

# Configuration settings
SETTINGS = {
    "common": {
        "PMF_unit": "PMF (kcal / mol)",
        "reaction_coordinate_unit": r"$\xi$ - reaction coordinate ($\AA$)",
        "polynomial_fit_order": 12,
        "param_B": 2.0,
        "datapoints": 10**5,
        "conventional_force_constant": 7,
        "conventional_window_width": 0.5,
        "conventional_window_count": 24,
        "equal_sampling": True,
        "conv_min_first_window": 2.5,
        "conv_min_last_window": 14.5,
        "fill_color": "r",
        "NM_alpha": 1,
        "NM_gamma": 2,
        "NM_beta": 0.5,
        "NM_delta": 0.5,
        "error_every": 3,
        "error_bar": False,
        "fig_extension": "jpg",
        "KS_coefficient_D": 1.36,
        "markers": ['^', '|', 'v', '*', 'x', 's', '2', 'D', 'o', 'p'],
        "colors": ['b', 'g', 'r', 'k', 'c', 'y', 'darkorange', 'darkviolet', 'saddlebrown', 'slategray'],
        "linestyles": ['-', '--', '-.', ':'],
        "mark_every": 3,
        "marker_size": 10,
        "xlim": (2, 16),
        "ylim": (-0.5, 16)
    },
    "wham": {
        "metadatafile": "list.txt",
        "hist_min": 2.0,
        "hist_max": 14.5,
        "num_bins": 100,
        "tol": 0.0001,
        "temp": 300.0,
        "numpad": 0,
        "rand_seed": random.randint(9999, 10000000)
    }
}

# Set up matplotlib defaults
def setup_matplotlib():
    """Configure matplotlib with consistent settings."""
    font = {
        'weight': 'bold',
        'size': 12
    }
    matplotlib.rc('font', **font)

# Initialize matplotlib settings
setup_matplotlib()

class FileHandler:
    """Class to handle file operations in a platform-independent way."""
    
    @staticmethod
    def ensure_directory(directory):
        """Create directory if it doesn't exist."""
        if not os.path.exists(directory):
            os.makedirs(directory)
        return directory
    
    @staticmethod
    def save_pickle(file_path, data):
        """Save data to pickle file."""
        with open(file_path, 'wb') as f:
            pickle.dump(data, f, protocol=-1)
    
    @staticmethod
    def load_pickle(file_path):
        """Load data from pickle file."""
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def save_numpy(file_path, data, header=None):
        """Save numpy array to file."""
        if header:
            np.savetxt(file_path, data, header=header)
        else:
            np.savetxt(file_path, data)
    
    @staticmethod
    def load_numpy(file_path):
        """Load numpy array from file."""
        return np.loadtxt(file_path)

class Visualizer:
    """Class to handle visualization tasks."""
    
    def __init__(self, settings=None):
        """Initialize with specified settings or default settings."""
        self.settings = settings or SETTINGS["common"]
    
    def save_figure(self, filename, dpi=300, tight=True):
        """Save current figure with consistent parameters."""
        if tight:
            plt.savefig(filename, bbox_inches='tight', dpi=dpi)
        else:
            plt.savefig(filename, dpi=dpi)
    
    def plot_pmf_and_fit(self, x_data, y_data, y_fit, output_path, title=None):
        """Plot PMF data and its polynomial fit."""
        plt.figure(figsize=(10, 6))
        plt.plot(x_data, y_data, c='r', label='original', marker='^', 
                ms=self.settings["marker_size"], markevery=self.settings["mark_every"])
        plt.plot(x_data, y_fit, c='g', label='fitted', marker='s', 
                ms=self.settings["marker_size"], markevery=self.settings["mark_every"])
        
        plt.xlabel(self.settings["reaction_coordinate_unit"], fontsize=14, weight='bold')
        plt.ylabel(r'PMF F($\xi$) (kcal/mol)', fontsize=14, weight='bold')
        plt.xlim(self.settings["xlim"])
        plt.ylim(self.settings["ylim"])
        plt.yticks(range(int(self.settings["ylim"][0]), int(self.settings["ylim"][1]+2.0), 2))
        
        if title:
            plt.title(title)
            
        plt.legend(loc='best')
        self.save_figure(output_path)
        plt.close()
    
    def plot_gradient(self, x_data, y_original, y_fitted, output_path, title=None):
        """Plot gradient data and its smoothed version."""
        plt.figure(figsize=(10, 6))
        plt.plot(x_data, y_original, c='r', label='original', marker='^', 
                ms=self.settings["marker_size"], markevery=self.settings["mark_every"])
        plt.plot(x_data, y_fitted, c='g', label='fitted', marker='s', 
                ms=self.settings["marker_size"], markevery=self.settings["mark_every"])
        
        plt.xlabel(self.settings["reaction_coordinate_unit"], fontsize=14, weight='bold')
        plt.ylabel(r'dF($\xi$)/d$\xi$ (kcal/mol/$\AA$)', fontsize=14, weight='bold')
        plt.xlim(self.settings["xlim"])
        plt.ylim(self.settings["ylim"])
        plt.yticks(range(int(self.settings["ylim"][0]), int(self.settings["ylim"][1]+2.0), 2))
        
        if title:
            plt.title(title)
            
        plt.legend(loc='best')
        self.save_figure(output_path)
        plt.close()
        
    def plot_window_guess(self, x, y, extremes, crest, trough, output_path):
        """Plot the initial window guess based on extrema."""
        plt.figure(figsize=(10, 6))
        plt.plot(x[::-1], y[::-1])
        plt.plot(x[extremes], y[extremes], '*', c='k')
        plt.ylabel(r'$\Delta$ PMF', fontsize=14, weight='bold')
        plt.xlabel(self.settings["reaction_coordinate_unit"], fontsize=14, weight='bold')
        plt.xlim(self.settings["xlim"])
        plt.ylim(self.settings["ylim"])
        plt.yticks(range(int(self.settings["ylim"][0]), int(self.settings["ylim"][1]+2.0), 2))
        
        for exr in x[trough]:
            plt.axvline(exr, ls='-.', c='r')    
        for exr in x[crest]:
            plt.axvline(exr, ls='--', c='g')
            
        self.save_figure(output_path)
        plt.close()

def pmf_to_points(location, test_pmf, order=None):
    """
    Process a test PMF file to generate gradient and initial window guesses.
    
    Args:
        location: Directory to save the results
        test_pmf: Filename of the test PMF data
        order: Polynomial fit order (optional)
    
    Returns:
        A tuple of (x, y, extremes, extreme_values, crest, trough, bounds)
    """
    # Save current directory
    original_dir = os.getcwd()
    
    # Create directory if it doesn't exist
    file_handler = FileHandler()
    file_handler.ensure_directory(location)
    
    # Change to target directory
    os.chdir(location)
    
    # Get settings
    settings = SETTINGS["common"]
    poly_order = order if order is not None else settings["polynomial_fit_order"]
    data_points = settings["datapoints"]
    
    try:
        # Load free energy data
        data = np.loadtxt(test_pmf)
        location_data = data[:, 0]
        energy_data = data[:, 1]
        
        # Remove inf values from free energy
        for i, value in enumerate(energy_data):
            if not np.isinf(value):
                splice_index = i
                break
        
        energy_data_filtered = energy_data[splice_index:]
        x_values = location_data[len(energy_data) - len(energy_data_filtered):]
        
        # Apply polynomial fitting
        poly_fit = np.poly1d(np.polyfit(x_values, energy_data_filtered, poly_order))
        energy_fit = poly_fit(x_values)
        
        # Create visualizer and plot PMF and fit
        viz = Visualizer(settings)
        viz.plot_pmf_and_fit(
            x_values, energy_data_filtered, energy_fit,
            "PMF-actual+polyfit." + settings["fig_extension"]
        )
        
        # Calculate and smooth gradient
        gradient = np.gradient(energy_data_filtered, x_values[1] - x_values[0])
        gradient_fit = np.gradient(energy_fit, x_values[1] - x_values[0])
        
        # Save gradient data
        np.savetxt('pol_smooth-grad.txt', np.c_[x_values, gradient_fit], fmt='%.4f')
        
        # Prepare gradient data for visualization
        pos = np.loadtxt('pol_smooth-grad.txt')[:, 0]
        abs_gradient = np.abs(gradient)
        abs_gradient_fit = np.abs(gradient_fit)
        
        # Plot gradient data
        viz.plot_gradient(
            pos, abs_gradient, abs_gradient_fit,
            "gradient-actual+polyfit." + settings["fig_extension"]
        )
        
        # Flip the reaction coordinate
        flipped_gradient = np.flip(abs_gradient_fit)
        flipped_x = np.flip(pos)
        
        # Interpolate to get more points
        f = interp1d(flipped_x, flipped_gradient, kind='cubic')
        x = np.linspace(flipped_x[0], flipped_x[-1], data_points)
        y = np.array([f(i) for i in x])
        
        # Find crests and troughs
        order = 1
        crest = extrema(y, np.greater_equal, order=order)[0]
        trough = extrema(y, np.less_equal, order=order)[0]
        
        # Process extremes for visualization
        extremes = np.sort(np.concatenate((crest, trough)))
        extreme_values = y[extremes].astype(float)
        
        # Plot initial window guess
        viz.plot_window_guess(
            x, y, extremes, crest, trough,
            "guess." + settings["fig_extension"]
        )
        
        # Generate window bounds
        bounds = []
        for i in range(len(extremes) - 1):
            newpair = np.arange(extremes[i], extremes[i+1] + 1)
            bounds.append(tuple(newpair))
        
        bounds = tuple(bounds)
        
        # Save variables for later use
        result = [x, y, extremes, extreme_values, crest, trough, bounds]
        file_handler.save_pickle(os.path.join(location, 'variables.pkl'), result)
        
        return result
        
    finally:
        # Restore original directory
        os.chdir(original_dir)

class WindowCalculator:
    """Calculate optimal windows based on free energy profiles."""
    
    def __init__(self, settings=None):
        """Initialize with specified settings or default settings."""
        self.settings = settings or SETTINGS["common"]
        
    def calculate_width(self, max_gradient, param_a, param_b):
        """Calculate window width based on gradient."""
        return round(1 / ((max_gradient / param_a) + (1 / param_b)), 6)
    
    def find_windows(self, x, y, param_a, param_b, initial_point=0):
        """Find windows using the COWBOE algorithm."""
        # Initialize
        windows = []
        indices = []
        
        # Add first window position
        start = x[initial_point]
        windows.append(start)
        indices.append(initial_point)
        
        current_index = initial_point
        
        # Loop until we reach the end
        while current_index < len(x):
            # Find extremes to the left of current position
            extremes_left = [i for i, val in enumerate(x) if val < x[current_index]]
            
            if not extremes_left:
                break
                
            # Calculate window width based on maximum gradient
            max_grad = max(y[current_index:extremes_left[-1]+1])
            width = self.calculate_width(max_grad, param_a, param_b)
            
            # Calculate next position
            next_pos = x[current_index] - width
            
            # Find closest index
            next_index = self._find_closest_index(x, next_pos, current_index, extremes_left[-1])
            
            if next_index == current_index:
                break
                
            windows.append(x[next_index])
            indices.append(next_index)
            current_index = next_index
        
        # Process windows
        return np.flip(np.unique(np.array(windows))), indices
    
    def _find_closest_index(self, x, target, left_bound, right_bound):
        """Find the closest index to target value within bounds."""
        # Binary search to find closest point
        left, right = left_bound, right_bound
        
        while right - left > 1:
            mid = (left + right) // 2
            if x[mid] > target:
                left = mid
            else:
                right = mid
                
        # Return the closer of the two points
        if abs(x[left] - target) < abs(x[right] - target):
            return left
        else:
            return right
    
    def calculate_force_constants(self, windows, energy_barrier, k_given=None):
        """Calculate force constants for the windows."""
        windows_copy = windows.copy()
        
        # Apply boundary constraints
        min_last = self.settings["conv_min_last_window"]
        min_first = self.settings["conv_min_first_window"]
        windows_copy[0], windows_copy[-1] = min_last, min_first
        
        # Calculate force constants using the formula K = 2*V/(width/2)^2
        window_widths = np.diff(windows_copy[::-1])
        
        if k_given is None:
            k_given = 2 * energy_barrier / self.settings["conventional_window_width"]**2
            
        force_constants = [2.0 * energy_barrier / (width/2.0)**2 for width in window_widths]
        barrier_heights = [0.5 * k * (width/2.0)**2 for k, width in zip(force_constants, window_widths)]
        
        # Calculate middle points
        midpoints = []
        for left, right in zip(windows_copy[::-1][:-1], windows_copy[::-1][1:]):
            midpoints.append((left + right) / 2)
            
        return force_constants, midpoints, barrier_heights

def cowboe(param_a, param_v, sampling_considered, name, subtype, location, 
           param_b=None, equal_sampling=None, rc_start=None, rc_stop=None):
    """
    COWBOE algorithm for iteration and window selection.
    
    Args:
        param_a: Optimization parameter 'A' for window width calculation
        param_v: Energy barrier parameter 'V'
        sampling_considered: Sampling time in ns
        name: Name of the point being evaluated
        subtype: Subtype of the system
        location: Location of the pickled variable file
        param_b: Optional parameter 'B' for window width calculation
        equal_sampling: Whether to use equal sampling across windows
        rc_start: Start position for reaction coordinate
        rc_stop: Stop position for reaction coordinate
    
    Returns:
        Dictionary containing window information
    """
    # Set up parameters
    settings = SETTINGS["common"]
    param_b = param_b or settings["param_B"]
    equal_sampling = equal_sampling if equal_sampling is not None else settings["equal_sampling"]
    rc_start = rc_start or settings["conv_min_first_window"]
    rc_stop = rc_stop or settings["conv_min_last_window"]
    
    # Create output directory
    point_dir = os.path.join(os.getcwd(), name)
    FileHandler.ensure_directory(point_dir)
    os.chdir(point_dir)
    
    # Load variables
    try:
        variables = FileHandler.load_pickle(os.path.join(location, 'variables.pkl'))
        x, y, extremes, extreme_values, crest, trough, bounds = variables
        
        # Create visualization
        viz = Visualizer(settings)
        
        # Plot initial state
        plt.figure(figsize=(10, 6))
        plt.plot(x[::-1], y[::-1])
        plt.xlim((x[-1]-1, x[0]+1))
        plt.plot(x[extremes], y[extremes], '*', c='k')
        plt.ylabel(r'$\Delta$ PMF')
        plt.xlabel(settings["reaction_coordinate_unit"])
        
        for exr in x[trough]:
            plt.axvline(exr, ls='--', c='r')
        for exr in x[crest]:
            plt.axvline(exr, ls='--', c='g')
            
        plt.xlim(settings["xlim"])
        plt.ylim(settings["ylim"])
        viz.save_figure(f'up and down_{param_a:.4f}_{param_b:.4f}.{settings["fig_extension"]}')
        plt.title(f'A = {param_a:.4f} & B = {param_b:.4f} - initial guess')
        plt.show()
        plt.close()
        
        # Calculate windows
        window_calculator = WindowCalculator(settings)
        windows, indices = window_calculator.find_windows(x, y, param_a, param_b)
        
        # Plot windows
        plt.figure(figsize=(10, 6))
        plt.plot(x[::-1], y[::-1])
        
        for window in windows:
            plt.axvline(window, ls='--', c='r')
            
        plt.xlim(settings["xlim"])
        plt.ylim(settings["ylim"])
        plt.yticks(range(int(settings["ylim"][0]), int(settings["ylim"][1]+2.0), 2))
        plt.ylabel(r'$\Delta$ PMF', fontsize=14, weight='bold')
        plt.xlabel(settings["reaction_coordinate_unit"], fontsize=14, weight='bold')
        viz.save_figure(f'{location}/windowdist_{param_a:.4f}_{param_b:.4f}.{settings["fig_extension"]}')
        plt.show()
        plt.close()
        
        # Calculate force constants
        k_given = 2.0 * param_v / settings["conventional_window_width"]**2
        force_constants, midpoints, _ = window_calculator.calculate_force_constants(
            windows, param_v, k_given
        )
        
        # Calculate sampling times
        if equal_sampling:
            window_count = len(windows) - 1
            sampling_times = list(np.full((window_count,), float(sampling_considered / window_count)))
        else:
            window_widths = np.diff(windows)
            fraction_times = [w / sum(window_widths) for w in window_widths]
            sampling_times = [(f * sampling_considered) for f in fraction_times]
        
        # Plot sampling times
        plt.figure(figsize=(10, 6))
        plt.plot(sampling_times[::-1], 'r^--')
        plt.xlim(-0.25, len(windows) - 1.25)
        plt.ylabel('ns', fontsize=14, weight='bold')
        plt.xlabel('Windows', fontsize=14, weight='bold')
        plt.title(f'A = {param_a:.4f} & B = {param_b:.4f} - sampling/window')
        plt.show()
        plt.close()
        
        # Create output data
        output_data = np.c_[
            np.flip(windows)[:-1], 
            np.array(midpoints),
            np.flip(windows)[1:], 
            force_constants, 
            np.array(sampling_times[::-1])
        ]
        
        # Save output data
        header = f'Total number of windows = {len(windows)-1}\n' + \
                 f'Sampling time considered = {sampling_considered} ns - for {settings["conventional_window_count"]} windows\n' + \
                 'left\tmiddle\tright\tforce constant\tSampling time'
                 
        FileHandler.save_numpy(f'A={param_a}_B={param_b}_V={param_v}.txt', output_data, header)
        np.save(f'A={param_a}_B={param_b}_V={param_v}', output_data)
        
        # Create output report
        window_df = pd.DataFrame(
            output_data, 
            columns=['Left', 'Middle', 'Right', 'Force constant', 'Sampling time']
        )
        
        print(f'\nTotal number of windows = {len(windows)-1}\n')
        print(window_df)
        
        # Return to original directory
        os.chdir('..')
        
        return {
            'windows': windows,
            'force_constants': force_constants,
            'midpoints': midpoints,
            'sampling_times': sampling_times
        }
        
    except Exception as e:
        os.chdir('..')
        raise Exception(f"Error in cowboe algorithm: {str(e)}")

def cowboe_wham(location, name="cowboe_pmf_output.txt", mc_trials=0, hist_min=None, hist_max=None, 
                num_bins=None, tol=None, temp=None, numpad=None, metadatafile=None):
    """
    WHAM wrapper for PMF generation using trajectory files.
    
    Args:
        location: Directory containing trajectory files
        name: Output filename
        mc_trials: Number of Monte Carlo trials for bootstrapping
        hist_min: Minimum value for histogram
        hist_max: Maximum value for histogram
        num_bins: Number of bins for histogram
        tol: Tolerance for convergence
        temp: Temperature in Kelvin
        numpad: Numpad value for WHAM calculation
        metadatafile: Name of metadata file
        
    Returns:
        Path to output PMF file
    """
    # Save current directory
    current_dir = os.getcwd()
    
    # Set parameters
    wham_settings = SETTINGS["wham"]
    hist_min = hist_min if hist_min is not None else wham_settings["hist_min"]
    hist_max = hist_max if hist_max is not None else wham_settings["hist_max"]
    num_bins = num_bins if num_bins is not None else wham_settings["num_bins"]
    tol = tol if tol is not None else wham_settings["tol"]
    temp = temp if temp is not None else wham_settings["temp"]
    numpad = numpad if numpad is not None else wham_settings["numpad"]
    metadatafile = metadatafile if metadatafile is not None else wham_settings["metadatafile"]
    rand_seed = random.randint(9999, 10000000)
    
    try:
        # Change to target directory
        os.chdir(location)
        
        # Print WHAM information
        print("\nCalling WHAM for PMF generation using trajectory files.")
        print("Must have wham installed in system path")
        print("cite: Grossfield, Alan, "WHAM: the weighted histogram analysis method",")
        print("version 2.0.10, http://membrane.urmc.rochester.edu/wordpress/?page_id=126")
        
        # Construct command
        if mc_trials > 0:
            command = f'wham {hist_min:.4f} {hist_max:.4f} {num_bins} {tol:.6f} {temp:.4f} ' + \
                      f'{numpad} {metadatafile} {name} {mc_trials} {rand_seed} > wham_output.txt'
        else:
            command = f'wham {hist_min:.4f} {hist_max:.4f} {num_bins} {tol:.6f} {temp:.4f} ' + \
                      f'{numpad} {metadatafile} {name} > wham_output.txt'
                  
        print(f"\nUsing: {command}\n")
        
        # Execute WHAM
        exit_code = os.system(command)
        
        if exit_code != 0:
            raise Exception("WHAM execution failed")
            
        print("\nPMF calculation done!\n\nCopying PMF file to the current directory.")
        
        # Copy output file to original directory
        output_path = os.path.join(current_dir, name)
        shutil.copy(os.path.join(location, name), output_path)
        
        return output_path
        
    except Exception as e:
        print(f"\nError in WHAM calculation: {str(e)}")
        return None
        
    finally:
        # Return to original directory
        os.chdir(current_dir)

class ComparisonAnalyzer:
    """Analyze and compare PMF profiles."""
    
    def __init__(self, settings=None):
        """Initialize with specified settings or default settings."""
        self.settings = settings or SETTINGS["common"]
        self.viz = Visualizer(settings)
        
    def compare_pmfs(self, pmf_files, name, splices, markzero=False, markers=None, 
                     colors=None, linestyles=None, mfc="none", legend_loc="outside"):
        """
        Plot and compare multiple PMF profiles.
        
        Args:
            pmf_files: List of PMF filenames
            name: Base name for output files
            splices: List of indices to start from for each PMF
            markzero: Whether to mark y=0 with a line
            markers: List of marker styles
            colors: List of colors
            linestyles: List of line styles
            mfc: Marker face color
            legend_loc: Legend location ("inside" or "outside")
            
        Returns:
            Path to output figure
        """
        # Set up parameters
        markers = markers or self.settings["markers"]
        colors = colors or self.settings["colors"]
        linestyles = linestyles or self.settings["linestyles"]
        
        # Limit to available markers, colors, linestyles
        markers = markers[:len(pmf_files)]
        colors = colors[:len(pmf_files)]
        linestyles = list(np.resize(linestyles, len(pmf_files)))
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Plot each PMF
        for pmf_file, splice, marker, color, linestyle in zip(pmf_files, splices, markers, colors, linestyles):
            # Get base filename without extension
            base_name = os.path.splitext(Path(pmf_file).name)[0]
            
            # Load data
            pmf_data = np.loadtxt(pmf_file)[splice:]
            x_values = pmf_data[:, 0]
            y_values = pmf_data[:, 1]
            errors = pmf_data[:, 2]
            
            # Plot with or without error bars
            if self.settings["error_bar"]:
                plt.errorbar(
                    x_values, y_values, yerr=errors, marker=marker, c=color,
                    markevery=self.settings["mark_every"], ls=linestyle,
                    lw=1.5, capsize=2, errorevery=self.settings["error_every"],
                    elinewidth=1.5, mfc=mfc, ms=self.settings["marker_size"],
                    label=base_name
                )
            else:
                plt.plot(
                    x_values, y_values, lw=1.5, marker=marker, c=color, 
                    ls=linestyle, markevery=self.settings["mark_every"], 
                    mfc=mfc, ms=self.settings["marker_size"], label=base_name
                )
                
        # Add zero line if requested
        if markzero:
            plt.axhline(y=0.0, ls='--', c='r')
            
        # Set axis limits
        plt.xlim(self.settings["xlim"])
        plt.ylim(self.settings["ylim"])
        plt.yticks(range(int(self.settings["ylim"][0]), int(self.settings["ylim"][1]+2.0), 2))
        
        # Add labels
        plt.xlabel(self.settings["reaction_coordinate_unit"], fontsize=14, weight='bold')
        plt.ylabel(self.settings["PMF_unit"], fontsize=14, weight='bold')
        
        # Set legend position
        if legend_loc == "inside":
            plt.legend(loc='best')
        else:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            
        # Save figure
        output_path = f"{name}.{self.settings['fig_extension']}"
        self.viz.save_figure(output_path)
        plt.close()
        
        return output_path
    
    def calculate_pmf_difference(self, test_file, bench_file, from_min=True):
        """
        Calculate difference between test and benchmark PMF profiles.
        
        Args:
            test_file: Test PMF filename
            bench_file: Benchmark PMF filename
            from_min: Whether to start comparison from minimum (where PMF=0)
            
        Returns:
            Dictionary with difference metrics
        """
        # Clean and align files
        clean_test, clean_bench = self._clean_files(test_file, bench_file, from_min)
        
        # Calculate differences
        max_diff, area_diff, x_min, x_max, max_pos = self._calculate_area_difference(
            clean_test, clean_bench
        )
        
        # Normalize area by range
        area_norm = area_diff / (x_max - x_min)
        
        # Create output dictionary
        results = {
            'absolute_maximum_deviation': round(max_diff, 4),
            'maximum_position': round(max_pos, 4),
            'absolute_integral_error': round(area_diff, 4),
            'x_range': [x_min, x_max],
            'normalized_area': round(area_norm, 4)
        }
        
        # Print summary
        print(f"\nAbsolute maximum deviation: {max_diff:.4f}")
        print(f"\tIt is at x={max_pos:.4f}")
        print(f"Absolute integral error: {area_diff:.4f}")
        print(f"\tIt is between x=[{x_min}, {x_max}]")
        print(f"The normalized area: {area_norm:.4f}\n")
        
        return results
    
    def _clean_files(self, test_file, bench_file, from_zero):
        """Prepare files for comparison by cleaning and aligning data."""
        # Load data
        test_data = np.loadtxt(test_file, dtype=np.float32, delimiter='\t', comments='#')
        bench_data = np.loadtxt(bench_file, dtype=np.float32, delimiter='\t', comments='#')
        
        # Find common x values
        common_indices = np.intersect1d(test_data[:, 0], bench_data[:, 0], assume_unique=True)
        test_data = test_data[np.isin(test_data[:, 0], common_indices)]
        bench_data = bench_data[np.isin(bench_data[:, 0], common_indices)]
        
        # Start from zero if requested
        if from_zero:
            start_index_test = np.argmax(test_data[:, 1] == 0.0)
            start_index_bench = np.argmax(bench_data[:, 1] == 0.0)
            start_index = max(start_index_test, start_index_bench)
            
            test_data = test_data[start_index:]
            bench_data = bench_data[start_index:]
        
        # Remove NaN or Inf values
        mask = np.isfinite(test_data[:, 1]) & np.isfinite(bench_data[:, 1])
        test_data = test_data[mask]
        bench_data = bench_data[mask]
        
        # Save cleaned data
        test_output = f"clean_{Path(test_file).name}"
        bench_output = f"clean_{Path(bench_file).name}"
        
        np.savetxt(test_output, test_data, fmt='%.6f\t%.6f\t%.6f\t%.6f\t%.6f', delimiter='\t')
        np.savetxt(bench_output, bench_data, fmt='%.6f\t%.6f\t%.6f\t%.6f\t%.6f', delimiter='\t')
        
        return test_output, bench_output
    
    def _calculate_area_difference(self, test_file, bench_file):
        """Calculate area and maximum difference between two PMF profiles."""
        # Load data
        test_data = np.loadtxt(test_file, delimiter='\t')
        bench_data = np.loadtxt(bench_file, delimiter='\t')
        
        # Extract x and y values
        x = test_data[:, 0]
        test_y = test_data[:, 1]
        bench_y = bench_data[:, 1]
        
        # Find maximum vertical distance
        vertical_diff = np.abs(test_y - bench_y)
        max_vertical_distance = np.max(vertical_diff)
        max_pos = x[np.argmax(vertical_diff)]
        
        # Calculate area between curves
        absolute_area = np.trapz(vertical_diff, x)
        
        # Create visualization of maximum difference
        plt.figure(figsize=(10, 6))
        plt.plot(
            x, test_y, color='blue', label='Test', lw=1.5, marker='s',
            markevery=self.settings['mark_every'], ms=self.settings['marker_size'], mfc="none"
        )
        plt.plot(
            x, bench_y, color='k', label='Benchmark', lw=1.5, marker='o',
            markevery=self.settings['mark_every'], ms=self.settings['marker_size'], mfc="none"
        )
        
        # Annotate maximum difference
        max_index = np.argmax(vertical_diff)
        plt.annotate(
            "", xy=(x[max_index], test_y[max_index]),
            xytext=(x[max_index], bench_y[max_index]),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5),
            annotation_clip=False
        )
        plt.annotate(
            f'max. deviation = {max_vertical_distance:.2f} kcal/mol',
            (max(x)-0.5, min(bench_y)+2),
            textcoords="axes fraction",
            xytext=(0.6, 0.04),
            ha='center', bbox=dict(boxstyle='round,pad=0.2', fc='red', alpha=0.25),
            fontsize=11
        )
        
        plt.legend()
        self.viz.save_figure('maximum_local_deviation.{}'.format(self.settings['fig_extension']))
        plt.close()
        
        # Create visualization of area difference
        plt.figure(figsize=(10, 6))
        plt.plot(
            x, test_y, color='blue', label='Test', lw=1.5, marker='s',
            markevery=self.settings['mark_every'], ms=self.settings['marker_size'], mfc="none"
        )
        plt.plot(
            x, bench_y, color='k', label='Benchmark', lw=1.5, marker='o',
            markevery=self.settings['mark_every'], ms=self.settings['marker_size'], mfc="none"
        )
        
        # Fill between curves
        plt.fill_between(x, test_y, bench_y, color='red', alpha=0.25, interpolate=True)
        
        # Annotate area
        plt.annotate(
            f'integral_of_deviation = {absolute_area:.2f} kcal $\AA$/mol',
            (max(x)-0.5, min(bench_y)+2),
            textcoords="axes fraction",
            xytext=(0.6, 0.04),
            ha='center', bbox=dict(boxstyle='round,pad=0.2', fc='red', alpha=0.25),
            fontsize=11
        )
        
        self.viz.save_figure('integral_of_deviation.{}'.format(self.settings['fig_extension']))
        plt.close()
        
        return max_vertical_distance, absolute_area, min(x), max(x), max_pos

class KSAnalyzer:
    """Analyze trajectory distributions using Kolmogorov-Smirnov tests."""
    
    def __init__(self, settings=None):
        """Initialize with specified settings or default settings."""
        self.settings = settings or SETTINGS["common"]
        self.viz = Visualizer(settings)
        
    def analyze_distributions(self, location, listfile, percentage):
        """
        Compute Kolmogorov-Smirnov statistics for trajectory distributions.
        
        Args:
            location: Directory containing trajectory files
            listfile: File containing list of trajectory files
            percentage: Percentage of data to compare (e.g. 80 means compare full data to 80% of data)
            
        Returns:
            Dictionary with KS statistics and p-values
        """
        # Check trajectory files
        trajectory_files = glob.glob1(location, "*.traj")
        window_count = len(trajectory_files)
        print(f'\nFound {window_count} individual window\'s trajectories in the folder')
        
        # Process list file
        with open(os.path.join(location, listfile)) as file:
            filenames = [line.split()[0] for line in file.readlines()]
        
        # Initialize result arrays
        p_values = []
        ks_stats = []
        
        # Process each trajectory file
        for window_index, trajfile in enumerate(filenames):
            # Load trajectory data
            data = np.loadtxt(os.path.join(location, trajfile))[:, 1]
            
            # Get subset of data for comparison
            subset_length = int(round(len(data) * percentage / 100.0))
            data_subset = data[:subset_length]
            
            # Calculate KS test critical value
            l1, l2 = len(data), len(data_subset)
            critical_value = self.settings["KS_coefficient_D"] * np.sqrt((l1 + l2) / (l1 * l2))
            
            # Perform KS test
            ks_stat, p_value = ks_2samp(data, data_subset)
            p_values.append(p_value)
            ks_stats.append(ks_stat)
            
            # Print results
            print(f'\n Window {window_index} :\n\tp:\t{p_value}\n\tKS:\t{ks_stat}\n')
            
            # Plot distributions
            self._plot_distributions(window_index, data, data_subset, location)
        
        # Plot summary statistics
        self._plot_p_values(p_values, filenames, location)
        self._plot_ks_statistics(ks_stats, filenames, critical_value, location)
        
        return {
            'p_values': p_values,
            'ks_statistics': ks_stats
        }
        
    def _plot_distributions(self, window_index, data1, data2, output_dir):
        """Plot probability distributions for a window."""
        plt.figure(figsize=(10, 6))
        sns.distplot(sorted(data1), hist=False)
        sns.distplot(sorted(data2), hist=False)
        plt.xlabel(self.settings["reaction_coordinate_unit"])
        plt.ylabel('pdf')
        self.viz.save_figure(os.path.join(output_dir, f'dist_{window_index}.{self.settings["fig_extension"]}'))
        plt.close()
        
    def _plot_p_values(self, p_values, filenames, output_dir):
        """Plot p-values for all windows."""
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(filenames)), p_values)
        plt.axhline(y=0.05, ls='--', c='g')
        plt.xlabel('windows')
        plt.xticks(range(len(filenames)))
        self.viz.save_figure(os.path.join(output_dir, f'pvalues.{self.settings["fig_extension"]}'))
        plt.close()
        
    def _plot_ks_statistics(self, ks_stats, filenames, critical_value, output_dir):
        """Plot KS statistics for all windows."""
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(filenames)), ks_stats)
        plt.axhline(y=critical_value, ls='--', c='r')
        plt.xlabel('windows')
        plt.xticks(range(len(filenames)))
        self.viz.save_figure(os.path.join(output_dir, f'KS-statistics.{self.settings["fig_extension"]}'))
        plt.close()


class OverlapAnalyzer:
    """Analyze overlap between trajectory distributions."""
    
    def __init__(self, settings=None):
        """Initialize with specified settings or default settings."""
        self.settings = settings or SETTINGS["common"]
        self.viz = Visualizer(settings)
        
    def calculate_overlap(self, location, name, listfile, distplot=True):
        """
        Calculate overlap coefficients between distributions.
        
        Args:
            location: Directory containing trajectory files
            name: Name for output files
            listfile: File containing list of trajectory files
            distplot: Whether to plot distributions
            
        Returns:
            Overlap coefficient matrix
        """
        # Check trajectory files
        win_count = len(glob.glob1(location, "*.traj"))
        print(f'\nFound {win_count} individual window\'s trajectories in the folder')
        
        # Read list file
        with open(os.path.join(location, listfile)) as file:
            filenames = [line.split()[0] for line in file.readlines()]
        
        # Plot all distributions
        plt.figure(figsize=(10, 6))
        for trajfile in filenames:
            dist_data = np.loadtxt(os.path.join(location, trajfile))[:, 1]
            sns.distplot(dist_data, hist=False, kde=True, kde_kws={'linewidth': 2})
            
        plt.xlabel(self.settings["reaction_coordinate_unit"], fontsize=14, weight='bold')
        plt.ylabel('probability density function', fontsize=14, weight='bold')
        self.viz.save_figure(os.path.join(location, f'distribution-kde-{name}.{self.settings["fig_extension"]}'))
        plt.close()
        
        # Initialize overlap matrix
        ovl_matrix = np.zeros((win_count, win_count))
        
        # Calculate overlap between adjacent windows
        for i, trajfile1 in enumerate(filenames):
            data1 = np.loadtxt(os.path.join(location, trajfile1))[:, 1]
            print('\n')
            
            for j in range(i, win_count):
                # Set diagonal to 1.0 (self-overlap)
                if i == j:
                    ovl_matrix[i][j] = 1.0
                    continue
                
                # Only calculate overlap for adjacent windows
                if abs(i - j) != 1:
                    continue
                    
                # Load second distribution
                data2 = np.loadtxt(os.path.join(location, filenames[j]))[:, 1]
                
                # Calculate overlap
                ovl = self._calculate_distribution_overlap(
                    np.array(sorted(data1)), 
                    np.array(sorted(data2)), 
                    i, j, distplot, location
                )
                
                # Store overlap value (symmetric matrix)
                ovl_matrix[i][j] = ovl
                ovl_matrix[j][i] = ovl
                
                print(f'\n\tOverlap coefficient between Window {i} and {j} is: {ovl:.6f}')
                
                # Stop if no overlap
                if ovl == 0.0:
                    break
        
        # Save and plot overlap matrix
        np.savetxt(f'OVL-{name}.txt', ovl_matrix, fmt='%.6f')
        np.save(f'OVL-{name}.npy', ovl_matrix)
        
        plt.figure(figsize=(10, 8))
        plt.matshow(ovl_matrix, cmap='plasma', interpolation='nearest')
        plt.colorbar()
        plt.clim(0, 1)
        self.viz.save_figure(os.path.join(location, f'OVL-{name}.jpg'))
        plt.close()
        
        return ovl_matrix
    
    def _calculate_distribution_overlap(self, hist1, hist2, i, j, distplot, output_dir):
        """Calculate statistical overlap between two distributions."""
        # Set up helper functions
        def cdf(s, x):
            """Calculate cumulative distribution function."""
            return 0.5 * (1.0 + erf((x - np.mean(s)) / (np.std(s) * sqrt(2.0))))
        
        # Get parameters
        x, y = hist1, hist2
        m1, m2 = np.mean(x), np.mean(y)
        s1, s2 = np.std(x), np.std(y)
        v1, v2 = np.var(x), np.var(y)
        
        # Ensure ordering
        if (s2, m2) < (s1, m1):
            x, y = y, x
            
        # Calculate intersection points
        dv = v2 - v1
        dm = fabs(m2 - m1)
        a = m1 * v2 - m2 * v1
        b = s1 * s2 * sqrt(dm**2.0 + dv * log(v2 / v1))
        x1 = (a + b) / dv
        x2 = (a - b) / dv
        
        # Plot distributions if requested
        if distplot:
            plt.figure(figsize=(10, 6))
            sns.distplot(hist1, label=f'window {i}', color='r')
            sns.distplot(hist2, label=f'window {j}', color='b')
            plt.xlabel(self.settings["reaction_coordinate_unit"], fontsize=14, weight='bold')
            plt.ylabel('probability density function', fontsize=14, weight='bold')
            self.viz.save_figure(os.path.join(output_dir, f'windows-{i}&{j}.{self.settings["fig_extension"]}'))
            plt.close()
            
        # Calculate overlap coefficient
        output = 1.0 - (fabs(cdf(y, x1) - cdf(x, x1)) + fabs(cdf(y, x2) - cdf(x, x2)))
        return round(output, 4)


class TrajectoryProcessor:
    """Process trajectory files for analysis."""
    
    @staticmethod
    def slice_trajectories(location, name, percentage, start, listfile=None):
        """
        Create new trajectory files with a subset of data.
        
        Args:
            location: Directory containing trajectory files
            name: Name for output directory
            percentage: Percentage of data to extract
            start: Starting index
            listfile: Name of list file (optional)
            
        Returns:
            Path to output directory
        """
        # Use default list file if not specified
        listfile = listfile or SETTINGS["wham"]["metadatafile"]
        
        # Create output directory
        output_dir = os.path.join(location, f'{name}_{percentage}_percentage_{start}')
        FileHandler.ensure_directory(output_dir)
        
        # Process all trajectory files
        traj_files = glob.glob1(location, "*.traj")
        
        for traj_file in traj_files:
            # Load trajectory data
            data = np.genfromtxt(os.path.join(location, traj_file))
            
            # Calculate slice indices
            stop = start + int(round(len(data) * percentage / 100.0))
            data_slice = data[start:stop]
            
            # Save sliced data
            output_file = os.path.join(output_dir, traj_file)
            np.savetxt(output_file, data_slice)
            
        # Copy list file to output directory
        shutil.copy(os.path.join(location, listfile), output_dir)
        
        return output_dir


class PMFPlotter:
    """Plot PMF profiles."""
    
    def __init__(self, settings=None):
        """Initialize with specified settings or default settings."""
        self.settings = settings or SETTINGS["common"]
        self.viz = Visualizer(settings)
        
    def plot_pmf(self, pmf_file, name, splice):
        """
        Plot a PMF curve.
        
        Args:
            pmf_file: PMF data file
            name: Output filename base
            splice: Index to start from
            
        Returns:
            Path to output figure
        """
        # Load PMF data
        pmf_data = np.loadtxt(pmf_file)[splice:]
        x_values = pmf_data[:, 0]
        y_values = pmf_data[:, 1]
        errors = pmf_data[:, 2]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Plot with or without error bars
        if self.settings['error_bar']:
            plt.errorbar(
                x_values, y_values, yerr=errors, lw=1.5, capsize=2,
                errorevery=self.settings['error_every'], elinewidth=1.5, label='PMF-I'
            )
        else:
            plt.plot(x_values, y_values, lw=1.5, label='PMF-I')
            
        # Set labels
        plt.xlabel(self.settings["reaction_coordinate_unit"], fontsize=14, weight='bold')
        plt.ylabel(self.settings["PMF_unit"], fontsize=14, weight='bold')
        
        # Add legend and adjust ticks
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        
        # Save figure
        output_path = f'{name}.{self.settings["fig_extension"]}'
        self.viz.save_figure(output_path)
        plt.close()
        
        return output_path


class NelderMead:
    """Nelder-Mead optimization for window parameters."""
    
    def __init__(self, settings=None):
        """Initialize with specified settings or default settings."""
        self.settings = settings or SETTINGS["common"]
        self.restricted = False
        
    def optimize(self, a_values, v_values, fitness_values, restricted=False):
        """
        Run Nelder-Mead optimization algorithm.
        
        Args:
            a_values: Array of A parameter values for simplex vertices
            v_values: Array of V parameter values for simplex vertices
            fitness_values: Array of fitness values for simplex vertices
            restricted: Whether to use restricted Nelder-Mead (no expansion)
            
        Returns:
            Dictionary with possible moves and their parameters
        """
        self.restricted = restricted
        
        # Convert parameters to logarithmic scale
        log_a = {1: np.log(a_values[0]), 2: np.log(a_values[1]), 3: np.log(a_values[2])}
        v_dict = {1: v_values[0], 2: v_values[1], 3: v_values[2]}
        f_dict = {1: fitness_values[0], 2: fitness_values[1], 3: fitness_values[2]}
        
        # Get algorithm parameters
        alpha = self.settings["NM_alpha"]
        beta = self.settings["NM_beta"] 
        delta = self.settings["NM_delta"]
        gamma = self.settings["NM_gamma"] if not restricted else None
        
        # Calculate reflection point and identify vertices
        centroid, reflected, best, worst, other, best_vertex, worst_vertex, other_vertex = \
            self._reflection(log_a, v_dict, f_dict, alpha)
        
        # Calculate other possible moves
        if not restricted:
            expanded = self._expansion(centroid, reflected, best, worst, other, gamma)
        else:
            expanded = None
            
        inner_contracted = self._inner_contraction(centroid, worst_vertex, best, worst, other, beta)
        outer_contracted = self._outer_contraction(reflected, centroid, worst_vertex, best, worst, other, beta)
        s_worst, s_other = self._shrink(worst_vertex, other_vertex, best, worst, other, delta)
        
        # Convert results back to original scale
        if not restricted:
            moves = self._convert_moves(reflected, expanded, inner_contracted, outer_contracted, s_worst, s_other)
        else:
            moves = self._convert_moves(reflected, None, inner_contracted, outer_contracted, s_worst, s_other)
            
        # Print summary
        print('\n')
        print(f'{"Move"}\t\t{"A"}\t{"V"}')
        print('==============================')
        for move, params in moves.items():
            params_rounded = [round(p, 4) for p in params]
            print(f'{move}\t{params_rounded[0]}\t{params_rounded[1]}')
        print('==============================')
        
        return moves
    
    def _centroid(self, point1, point2):
        """Calculate centroid of two points."""
        x1, y1 = point1
        x2, y2 = point2
        return [(x1 + x2) / 2, (y1 + y2) / 2]
    
    def _plot_simplex(self, best, worst, other, centroid=None, new_point=None, 
                     new_point_name=None, log_a=None, v_dict=None, f_dict=None):
        """Plot the current simplex and possible moves."""
        # Extract vertex coordinates
        if log_a and v_dict and f_dict:
            a_coords = [log_a[1], log_a[2], log_a[3]]
            v_coords = [v_dict[1], v_dict[2], v_dict[3]]
            f_values = [f_dict[1], f_dict[2], f_dict[3]]
        else:
            a_coords = None
            v_coords = None
            f_values = None
            
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Plot simplex
        if a_coords and v_coords:
            plt.plot(a_coords, v_coords, 'k.', markersize=5)
            plt.plot([a_coords[0], a_coords[1]], [v_coords[0], v_coords[1]], 'k-')
            plt.plot([a_coords[1], a_coords[2]], [v_coords[1], v_coords[2]], 'k-')
            plt.plot([a_coords[2], a_coords[0]], [v_coords[2], v_coords[0]], 'k-')
            
            # Label vertices
            for i, (x, y, f) in enumerate(zip(a_coords, v_coords, f_values)):
                idx = i + 1
                if idx == best:
                    label = f'{f:.4f}'
                    color = 'g'
                    plt.plot(x, y, 'g^', markersize=10, 
                            label=f'best ({np.exp(x):.4f},{y:.4f})')
                elif idx == worst:
                    label = f'{f:.4f}'
                    color = 'r'
                    plt.plot(x, y, 'r^', markersize=10,
                            label=f'worst ({np.exp(x):.4f},{y:.4f})')
                else:
                    label = f'{f:.4f}'
                    color = 'y'
                    plt.plot(x, y, 'y^', markersize=10,
                            label=f'other ({np.exp(x):.4f},{y:.4f})')
                    
                plt.annotate(
                    label, (x, y), textcoords="offset points",
                    xytext=(0, 5), color=color, ha='center'
                )
        
        # Plot centroid
        if centroid:
            plt.plot(centroid[0], centroid[1], 'b^', markersize=10,
                    label=f'centroid ({np.exp(centroid[0]):.4f},{centroid[1]:.4f})')
        
        # Plot new point
        if new_point and new_point_name:
            if new_point_name == 'Reflected':
                plt.plot(new_point[0], new_point[1], 'g*', markersize=10,
                        label=f'{new_point_name} ({np.exp(new_point[0]):.4f},{new_point[1]:.4f})')
            elif new_point_name == 'expanded':
                plt.plot(new_point[0], new_point[1], 'b*', markersize=10,
                        label=f'{new_point_name} ({np.exp(new_point[0]):.4f},{new_point[1]:.4f})')
            elif new_point_name == 'inner_c':
                plt.plot(new_point[0], new_point[1], 'k*', markersize=10,
                        label=f'{new_point_name} ({np.exp(new_point[0]):.4f},{new_point[1]:.4f})')
            elif new_point_name == 'outer_c':
                plt.plot(new_point[0], new_point[1], 'c*', markersize=10,
                        label=f'{new_point_name} ({np.exp(new_point[0]):.4f},{new_point[1]:.4f})')
            elif new_point_name == 's_worst':
                plt.plot(new_point[0], new_point[1], 'm*', markersize=10,
                        label=f'{new_point_name} ({np.exp(new_point[0]):.4f},{new_point[1]:.4f})')
            elif new_point_name == 's_other':
                plt.plot(new_point[0], new_point[1], 'k*', markersize=10,
                        label=f'{new_point_name} ({np.exp(new_point[0]):.4f},{new_point[1]:.4f})')
                
        # Add labels and legend
        plt.xlabel('ln A')
        plt.ylabel(r'$\Delta$ U')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Set consistent limits for all plots
        if log_a and v_dict:
            worst_v = [log_a[worst], v_dict[worst]]
            
            if new_point:
                plt.xlim(
                    min(a_coords + [new_point[0]]) - 0.1,
                    max(a_coords + [new_point[0]]) + 0.1
                )
                plt.ylim(
                    min(v_coords + [new_point[1]]) - 0.1,
                    max(v_coords + [new_point[1]]) + 0.1
                )
                
                if new_point_name not in ['s_worst', 's_other']:
                    plt.plot([new_point[0], worst_v[0]], [new_point[1], worst_v[1]], 'k--')
            
        # Save and show figure
        if new_point_name:
            plt.savefig(f'{new_point_name}.{self.settings["fig_extension"]}', bbox_inches='tight', dpi=300)
        plt.close()
    
    def _reflection(self, log_a, v_dict, f_dict, alpha):
        """Calculate reflection point."""
        # Find best, worst and other vertices
        best = min(f_dict, key=f_dict.get)
        worst = max(f_dict, key=f_dict.get)
        other = 6 - (best + worst)  # Clever way to find third vertex in a triangle
        
        # Extract vertex coordinates
        best_vertex = [log_a[best], v_dict[best]]
        worst_vertex = [log_a[worst], v_dict[worst]]
        other_vertex = [log_a[other], v_dict[other]]
        
        # Plot initial simplex
        self._plot_simplex(best, worst, other, log_a=log_a, v_dict=v_dict, f_dict=f_dict)
        
        # Calculate centroid of best and other vertices
        centroid = self._centroid(best_vertex, other_vertex)
        
        # Calculate reflection point
        reflected = [
            centroid[0] + alpha * (centroid[0] - log_a[worst]),
            centroid[1] + alpha * (centroid[1] - v_dict[worst])
        ]
        
        # Plot reflection
        self._plot_simplex(
            best, worst, other, centroid, reflected, 'Reflected', log_a, v_dict, f_dict
        )
        
        return centroid, reflected, best, worst, other, best_vertex, worst_vertex, other_vertex
    
    def _expansion(self, centroid, reflected, best, worst, other, gamma):
        """Calculate expansion point."""
        # Calculate expansion point
        expanded = [
            centroid[0] + gamma * (reflected[0] - centroid[0]),
            centroid[1] + gamma * (reflected[1] - centroid[1])
        ]
        
        # Plot expansion
        self._plot_simplex(best, worst, other, centroid, expanded, 'expanded')
        
        return expanded
    
    def _inner_contraction(self, centroid, worst_vertex, best, worst, other, beta):
        """Calculate inner contraction point."""
        # Calculate inner contraction
        inner_contracted = [
            centroid[0] + beta * (worst_vertex[0] - centroid[0]),
            centroid[1] + beta * (worst_vertex[1] - centroid[1])
        ]
        
        # Plot inner contraction
        self._plot_simplex(best, worst, other, centroid, inner_contracted, 'inner_c')
        
        return inner_contracted
    
    def _outer_contraction(self, reflected, centroid, worst_vertex, best, worst, other, beta):
        """Calculate outer contraction point."""
        # Calculate outer contraction
        outer_contracted = [
            centroid[0] + beta * (reflected[0] - centroid[0]),
            centroid[1] + beta * (reflected[1] - centroid[1])
        ]
        
        # Plot outer contraction
        self._plot_simplex(best, worst, other, centroid, outer_contracted, 'outer_c')
        
        return outer_contracted
    
    def _shrink(self, worst_vertex, other_vertex, best, worst, other, delta):
        """Calculate shrink points."""
        # Get best vertex coordinates (from class variables)
        best_vertex = [log_a[best], v_dict[best]]
        
        # Calculate shrink points
        shrink_worst = [
            best_vertex[0] + delta * (worst_vertex[0] - best_vertex[0]),
            best_vertex[1] + delta * (worst_vertex[1] - best_vertex[1])
        ]
        
        shrink_other = [
            best_vertex[0] + delta * (other_vertex[0] - best_vertex[0]),
            best_vertex[1] + delta * (other_vertex[1] - best_vertex[1])
        ]
        
        # Plot shrink
        self._plot_simplex(
            best, worst, other, new_point=shrink_worst, new_point_name='s_worst'
        )
        self._plot_simplex(
            best, worst, other, new_point=shrink_other, new_point_name='s_other'
        )
        
        return shrink_worst, shrink_other
    
    def _convert_point(self, point):
        """Convert point from log scale back to original scale."""
        return [np.exp(point[0]), point[1]]
    
    def _convert_moves(self, reflected, expanded, inner_contracted, outer_contracted, shrink_worst, shrink_other):
        """Convert all moves from log scale back to original scale."""
        moves = {
            'reflection': self._convert_point(reflected),
            'in_contract': self._convert_point(inner_contracted),
            'out_contract': self._convert_point(outer_contracted),
            'shrink_worst': self._convert_point(shrink_worst),
            'shrink_other': self._convert_point(shrink_other)
        }
        
        if expanded is not None:
            moves['expansion'] = self._convert_point(expanded)
            
        return moves


class NMProgressAnalyzer:
    """Analyze Nelder-Mead optimization progress."""
    
    def __init__(self, settings=None):
        """Initialize with specified settings or default settings."""
        self.settings = settings or SETTINGS["common"]
        
    def analyze_progress(self, progress_file, fps=1):
        """
        Analyze and visualize Nelder-Mead optimization progress.
        
        Args:
            progress_file: File containing simplex vertices at each step
            fps: Frames per second for animation
            
        Returns:
            Dictionary with analysis results
        """
        # Load progress data
        progress_data = np.loadtxt(progress_file)
        simplexes = progress_data.reshape((int(len(progress_data) / 3), 3, 3))
        
        # Create output directory
        output_dir = f'cowboe_NM_steps_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        os.mkdir(output_dir)
        os.chdir(output_dir)
        
        # Analyze each simplex
        areas = []
        stopping_criteria = []
        
        for idx, simplex in enumerate(simplexes):
            # Calculate area and stopping criterion
            area, criterion = self._analyze_simplex(simplex, idx, simplexes)
            areas.append(area)
            stopping_criteria.append(criterion)
            
            # Get detailed properties
            a, c, r, f, flatness = self._calculate_simplex_properties(simplex)
            
            print(f'\n\nThe area of the simplex is:\t\t{a}')
            print(f'The flatness of the simplex is:\t\t{flatness}')
            print(f'The centroid of the simplex is:\t\t{c}')
            print(f'The circum circle radius of the simplex is:\t\t{r}')
            print(f'The distance between centroid and the best point of the simplex is:\t\t{f}\n\n')
        
        # Plot area progress
        plt.figure(figsize=(10, 6))
        plt.plot(areas, 'r^--')
        plt.ylim((0, np.max(areas) * 1.10))
        plt.xlabel('simplexes', weight='bold')
        plt.ylabel('area', weight='bold')
        plt.savefig('Area of simplexes.pdf', bbox_inches='tight', dpi=300)
        plt.close()
        
        # Plot stopping criteria
        plt.figure(figsize=(10, 6))
        plt.plot(stopping_criteria, 'bs--')
        plt.ylim((0, np.max(stopping_criteria) * 1.10))
        plt.axhline(y=np.min(stopping_criteria) + 0.1, c='g', ls='-.')
        plt.xlabel('simplexes', weight='bold')
        plt.ylabel('RMSD - stopping criteria', weight='bold')
        plt.savefig('RMSF-fit().pdf', bbox_inches='tight', dpi=300)
        plt.close()
        
        # Create animation
        print('\nConstructing GIF image of the NM steps ...')
        self._create_animation(fps=fps)
        print('\nDone ...!')
        
        # Return to parent directory
        os.chdir('..')
        
        return {
            'areas': areas,
            'stopping_criteria': stopping_criteria,
            'output_dir': output_dir
        }
        
    def _analyze_simplex(self, simplex, index, all_simplexes):
        """Analyze a single simplex."""
        # Extract vertex coordinates and fitness values
        a_values = simplex[:, 0]
        v_values = simplex[:, 1]
        fitness = simplex[:, 2]
        
        # Create formatter for scientific notation
        formatter = mticker.ScalarFormatter(useOffset=False, useMathText=True)
        format_func = lambda x, pos: "${}$".format(formatter._formatSciNotation('%1.4e' % x))
        formatter_func = mticker.FuncFormatter(format_func)
        
        # Calculate area
        area_value = self._calculate_area(a_values, v_values)
        
        # Calculate stopping criterion
        stopping_criterion = self._calculate_stopping_criterion(fitness)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Get log-transformed A values
        log_a = np.array([np.log(a) for a in a_values])
        
        # Plot previous simplex if available
        if index > 0:
            prev_simplex = all_simplexes[index - 1]
            prev_log_a = np.array([np.log(a) for a in prev_simplex[:, 0]])
            prev_v = prev_simplex[:, 1]
            
            plt.plot([prev_log_a[0], prev_log_a[1]], [prev_v[0], prev_v[1]], 'k--')
            plt.plot([prev_log_a[1], prev_log_a[2]], [prev_v[1], prev_v[2]], 'k--')
            plt.plot([prev_log_a[2], prev_log_a[0]], [prev_v[2], prev_v[0]], 'k--')
        
        # Plot current simplex
        plt.plot([log_a[0], log_a[1]], [v_values[0], v_values[1]], 'k-')
        plt.plot([log_a[1], log_a[2]], [v_values[1], v_values[2]], 'k-')
        plt.plot([log_a[2], log_a[0]], [v_values[2], v_values[0]], 'k-')
        
        # Set labels
        plt.xlabel('ln A', fontsize=14, weight='bold')
        plt.ylabel(r'$\Delta$ U', fontsize=14, weight='bold')
        plt.xticks(fontsize=14, weight='bold')
        plt.yticks(fontsize=14, weight='bold')
        
        # Get best, worst vertices
        best_idx = np.argmin(fitness)
        worst_idx = np.argmax(fitness)
        
        # Set limits based on all simplexes
        all_log_a = np.array([np.log(a) for a in all_simplexes[:, :, 0].flatten()])
        all_v = all_simplexes[:, :, 1].flatten()
        plt.xlim(min(all_log_a) - 0.15, max(all_log_a) + 0.15)
        plt.ylim(min(all_v) - 0.15, max(all_v) + 0.15)
        
        # Plot vertices with labels
        for i, (x, y, f) in enumerate(zip(log_a, v_values, fitness)):
            if i == best_idx:
                plt.plot(x, y, 'g^', markersize=15, label=f'{f:.4f}')
                color = 'g'
            elif i == worst_idx:
                plt.plot(x, y, 'rX', markersize=15, label=f'{f:.4f}')
                color = 'r'
            else:
                plt.plot(x, y, 'yo', markersize=15, label=f'{f:.4f}')
                color = 'y'
                
        # Add area annotation
        plt.annotate(
            f'area = {formatter_func(area_value)}',
            (max(all_log_a) - 0.35, max(all_v) + 0.1),
            textcoords="axes fraction",
            xytext=(0.5, 0.93),
            fontsize=18,
            ha='center',
            bbox=dict(boxstyle='round,pad=0.2', fc='red', alpha=0.3),
            size=14
        )
        
        # Add step number
        plt.annotate(
            f'{index + 1}',
            xy=(1.35, 0.95),
            textcoords="axes fraction",
            xytext=(0.96, 0.9),
            color='k',
            fontsize=18,
            ha='center',
            bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3)
        )
        
        # Add legend and save figure
        plt.legend(loc='lower right')
        plt.savefig(f'{index + 1}.jpg', dpi=300)
        plt.close()
        
        return area_value, stopping_criterion
    
    def _calculate_area(self, x, y):
        """Calculate area of triangle."""
        # Calculate side lengths
        a = np.sqrt((x[0] - x[1])**2 + (y[0] - y[1])**2)
        b = np.sqrt((x[1] - x[2])**2 + (y[1] - y[2])**2)
        c = np.sqrt((x[0] - x[2])**2 + (y[0] - y[2])**2)
        
        # Calculate semi-perimeter
        s = (a + b + c) / 2
        
        # Calculate area using Heron's formula
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        
        return round(area, 8)
    
    def _calculate_stopping_criterion(self, fitness):
        """Calculate RMSD of fitness values for stopping criterion."""
        mean_fitness = np.mean(fitness)
        squared_diffs = [(f - mean_fitness)**2 for f in fitness]
        rmsd = np.sqrt(sum(squared_diffs) / 3.0)
        
        return rmsd
    
    def _calculate_simplex_properties(self, simplex):
        """Calculate various properties of the simplex."""
        # Extract coordinates and fitness
        x = np.array([np.log(a) for a in simplex[:, 0]])
        y = simplex[:, 1]
        fitness = simplex[:, 2]
        
        # Find best vertex
        best_idx = np.argmin(fitness)
        
        # Calculate area
        area = self._calculate_area(x, y)
        
        # Calculate centroid
        cx = np.mean(x)
        cy = np.mean(y)
        centroid = (round(cx, 4), round(cy, 4))
        
        # Calculate circumcircle radius
        a = np.sqrt((x[0] - x[1])**2 + (y[0] - y[1])**2)
        b = np.sqrt((x[1] - x[2])**2 + (y[1] - y[2])**2)
        c = np.sqrt((x[0] - x[2])**2 + (y[0] - y[2])**2)
        
        try:
            radius = (a * b * c) / np.sqrt((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c))
        except:
            radius = 0
            
        # Calculate distance from best point to centroid
        func_distance = np.sqrt((x[best_idx] - cx)**2 + (y[best_idx] - cy)**2)
        
        # Calculate flatness (area to max side^3 ratio)
        max_side = max(a, b, c)
        flatness = area / (max_side**3) if max_side > 0 else 0
        
        return area, centroid, round(radius, 6), round(func_distance, 6), round(flatness, 6)
    
    def _create_animation(self, fps=1):
        """Create animated GIF from step images."""
        # Get all JPG files
        filenames = sorted([f for f in os.listdir('.') if f.endswith('.jpg')], 
                          key=lambda x: int(x.split('.')[0]))
        
        # Read images
        images = [imageio.imread(filename) for filename in filenames]
        
        # Save animation
        imageio.mimsave(f'cowboe_NM_steps_FPS_{fps}.gif', images, fps=fps)


class SurfacePlotter:
    """Create 3D surface plots for optimization results."""
    
    def __init__(self, settings=None):
        """Initialize with specified settings or default settings."""
        self.settings = settings or SETTINGS["common"]
        
    def create_surface_plot(self, progress_file, fps=15, dpi=300, name='cowboe3Dsurface.mp4'):
        """
        Create a 3D surface plot animation of optimization progress.
        
        Args:
            progress_file: File containing simplex vertices at each step
            fps: Frames per second for animation
            dpi: Resolution in dots per inch
            name: Output filename
            
        Returns:
            Path to output animation
        """
        # Load progress data
        progress_data = np.loadtxt(progress_file)
        simplexes = progress_data.reshape((int(len(progress_data) / 3), 3, 3))
        
        # Extract all vertex coordinates and fitness values
        x = np.array([np.log(a) for a in simplexes[:, :, 0].flatten()])
        y = simplexes[:, :, 1].flatten()
        z = simplexes[:, :, 2].flatten()
        
        # Set up animation writer
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
        
        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.gca(projection='3d')
        
        # Create surface
        surface = ax.plot_trisurf(
            x, y, z, linewidth=0.2, antialiased=False,
            cmap=cm.plasma, edgecolor='none'
        )
        
        # Add colorbar and set limits
        fig.colorbar(surface)
        ax.set_zlim(min(z) - 0.25, max(z) + 0.25)
        
        # Set labels
        ax.set_xlabel('ln A', fontsize=12)
        ax.set_ylabel('V', fontsize=12)
        ax.set_zlabel('Fitness', fontsize=12)
        
        # Create rotation function for animation
        def rotate(angle):
            ax.view_init(azim=angle)
        
        # Create animation
        print('\nConstructing 3D surface plot for the points and fit() values')
        animation_obj = animation.FuncAnimation(
            fig, rotate, frames=np.arange(0, 360, 1), interval=50
        )
        
        # Save animation
        animation_obj.save(name, writer=writer, dpi=dpi)
        plt.close()
        
        return name


def create_progress_file(points, filename='progress.txt'):
    """
    Create a progress file from simplex data.
    
    Args:
        points: Array of simplexes (shape n×3×3)
        filename: Output filename
        
    Returns:
        Path to output file
    """
    with open(filename, 'w') as progress_file:
        for simplex in points:
            progress_file.write('\n# New row/simplex/step\n')
            for vertex in simplex:
                progress_file.write(str(vertex).strip('[]'))
                progress_file.write('\n')
                
    return filename

class GeneralizedNelderMead:
    """Nelder-Mead optimization for n-dimensional problems."""
    
    def __init__(self, settings=None):
        self.settings = settings or SETTINGS["common"]
        
    def optimize(self, vertices, fitness_values, restricted=False, parameter_transforms=None):
        """
        Run Nelder-Mead optimization algorithm in n dimensions.
        
        Args:
            vertices: Array of shape (n+1, n) representing simplex vertices
            fitness_values: Array of shape (n+1,) with fitness values
            restricted: Whether to use restricted Nelder-Mead (no expansion)
            parameter_transforms: Optional dict of functions to transform parameters
                                 (e.g., {'param_0': np.log, 'param_0_inverse': np.exp})
        """
        n_dim = vertices.shape[1]  # Problem dimensionality
        n_vertices = vertices.shape[0]
        
        if n_vertices != n_dim + 1:
            raise ValueError(f"For {n_dim}D optimization, need {n_dim+1} vertices, got {n_vertices}")
        
        # Sort vertices by fitness
        indices = np.argsort(fitness_values)
        sorted_vertices = vertices[indices]
        sorted_fitness = fitness_values[indices]
        
        # Get algorithm parameters
        alpha = self.settings.get("NM_alpha", 1.0)
        beta = self.settings.get("NM_beta", 0.5)
        gamma = self.settings.get("NM_gamma", 2.0) if not restricted else None
        delta = self.settings.get("NM_delta", 0.5)
        
        # Calculate centroid of all vertices except worst
        centroid = np.mean(sorted_vertices[:-1], axis=0)
        worst_vertex = sorted_vertices[-1]
        
        # Calculate reflection
        reflection = centroid + alpha * (centroid - worst_vertex)
        
        # Calculate other possible moves
        moves = {'reflection': reflection}
        
        if not restricted:
            # Calculate expansion
            expansion = centroid + gamma * (reflection - centroid)
            moves['expansion'] = expansion
            
        # Calculate contractions
        inner_contraction = centroid + beta * (worst_vertex - centroid)
        outer_contraction = centroid + beta * (reflection - centroid)
        moves['inner_contraction'] = inner_contraction
        moves['outer_contraction'] = outer_contraction
        
        # Calculate shrink points (all vertices move toward best)
        best_vertex = sorted_vertices[0]
        shrink_vertices = [best_vertex]
        for i in range(1, n_vertices):
            shrink_vertex = best_vertex + delta * (sorted_vertices[i] - best_vertex)
            shrink_vertices.append(shrink_vertex)
        moves['shrink'] = np.array(shrink_vertices)
        
        # Apply inverse transformations if needed
        if parameter_transforms:
            for move_name, move_value in moves.items():
                if move_name != 'shrink':
                    for i, (param_name, transform) in enumerate(parameter_transforms.items()):
                        if f"{param_name}_inverse" in parameter_transforms:
                            inverse_func = parameter_transforms[f"{param_name}_inverse"]
                            moves[move_name][i] = inverse_func(moves[move_name][i])
                else:
                    for vertex in move_value:
                        for i, (param_name, transform) in enumerate(parameter_transforms.items()):
                            if f"{param_name}_inverse" in parameter_transforms:
                                inverse_func = parameter_transforms[f"{param_name}_inverse"]
                                vertex[i] = inverse_func(vertex[i])
        
        return moves
    
    def visualize_simplex(self, vertices, fitness_values, move=None, move_name=None):
        """Visualize simplex according to dimensionality."""
        n_dim = vertices.shape[1]
        
        if n_dim == 1:
            self._visualize_1d(vertices, fitness_values, move, move_name)
        elif n_dim == 2:
            self._visualize_2d(vertices, fitness_values, move, move_name)
        elif n_dim == 3:
            self._visualize_3d(vertices, fitness_values, move, move_name)
        else:
            print(f"Cannot visualize {n_dim}-dimensional simplex")
    
    # Visualization methods for 1D, 2D, 3D would be implemented here
    
# Set up parameter transformations (log transform for A)
parameter_transforms = {
    'param_0': np.log,       # For A parameter
    'param_0_inverse': np.exp,
    'param_1': lambda x: x,  # Identity function for V parameter
    'param_1_inverse': lambda x: x
}

# Create vertices matrix from A and V arrays
vertices = np.array([
    [a_values[0], v_values[0]],
    [a_values[1], v_values[1]],
    [a_values[2], v_values[2]]
])

# Transform parameters for optimization
transformed_vertices = vertices.copy()
transformed_vertices[:, 0] = np.log(vertices[:, 0])  # Apply log to A values

nm = GeneralizedNelderMead(settings)
moves = nm.optimize(transformed_vertices, fitness_values, restricted=False, 
                   parameter_transforms=parameter_transforms)

# Main function to export
__all__ = [
    'pmf_to_points',
    'cowboe', 
    'cowboe_wham',
    'ComparisonAnalyzer',
    'Visualizer',
    'WindowCalculator',
    'KSAnalyzer',
    'OverlapAnalyzer',
    'TrajectoryProcessor',
    'PMFPlotter',
    'NelderMead',
    'NMProgressAnalyzer',
    'SurfacePlotter',
    'create_progress_file',
    'SETTINGS'
]

# Settings update function
def update_settings(section, new_settings):
    """
    Update settings dictionary with new values.
    
    Args:
        section: Section to update ('common' or 'wham')
        new_settings: Dictionary with new settings
        
    Returns:
        Updated settings dictionary
    """
    if section in SETTINGS:
        SETTINGS[section].update(new_settings)
        return SETTINGS[section]
    else:
        raise ValueError(f"Unknown settings section: {section}")
