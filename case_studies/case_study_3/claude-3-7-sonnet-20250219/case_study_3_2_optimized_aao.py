"""
Carousel Simulation
==================

A module for simulating and analyzing the dynamics of a carousel with gondolas.

This module provides classes to model a carousel with gondolas, simulate its dynamics,
and analyze the results to find optimal spring stiffness values that meet displacement
criteria.

Examples
--------
>>> from carousel_simulation import CarouselSimulation, CarouselParameters
>>> simulator = CarouselSimulation()
>>> results = simulator.find_optimal_spring_stiffness({
...     "start_n_per_m": 0,
...     "step_size_n_per_m": 100_000,
...     "num_results": 1,
...     "show_results": True
... })
"""


# Standard library imports
import math
from dataclasses import dataclass
from typing import Any, NamedTuple

# Third party imports
import colorama
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp  # type: ignore
from termcolor import colored


colorama.init()


@dataclass
class CarouselParameters:
    """
    Store parameters for the carousel simulation.

    This class holds all the physical parameters needed to define a carousel system,
    including dimensions, masses, and rotation speeds.

    Attributes
    ----------
    gravity: float
        Gravitational acceleration in m/s².
    tilt_angle_deg: float
        Tilt angle of the carousel in degrees.
    carousel_radius_m: float
        Radius of the carousel in meters.
    gondola_mass_kg: float
        Mass of each gondola in kilograms.
    carousel_rot_speed_rps: float
        Rotational speed of the carousel in rotations per second.
    gondola_width_m: float
        Width of each gondola in meters.
    gondola_height_m: float
        Height of each gondola in meters.
    """

    gravity: float = 9.81
    tilt_angle_deg: float = 30
    carousel_radius_m: float = 6
    gondola_mass_kg: float = 300
    carousel_rot_speed_rps: float = 0.2333
    gondola_width_m: float = 1.5
    gondola_height_m: float = 1.5


class DisplacementAnalysisResult(NamedTuple):
    """
    Store the results of a displacement analysis.

    This class holds the minimum and maximum displacement values, the total
    displacement, and whether the displacement criteria are met.

    Attributes
    ----------
    global_minima: float | None
        Minimum displacement value, or None if not available.
    global_maxima: float | None
        Maximum displacement value, or None if not available.
    displacement_m: float | None
        Total displacement in meters, or None if not available.
    criteria_met: bool
        Whether the displacement criteria are met.
    """

    global_minima: float | None
    global_maxima: float | None
    displacement_m: float | None
    criteria_met: bool


class SimulationResult(NamedTuple):
    """
    Store the results of a simulation run.

    This class holds the spring stiffness, displacement values, and the raw
    simulation data for a single simulation run.

    Attributes
    ----------
    spring_stiffness_n_per_m: float
        Spring stiffness in Newtons per meter.
    global_maxima_m: float | None
        Maximum displacement in meters, or None if not available.
    global_minima_m: float | None
        Minimum displacement in meters, or None if not available.
    displacement_m: float | None
        Total displacement in meters, or None if not available.
    simulation_data: tuple
        Raw simulation data.
    """

    spring_stiffness_n_per_m: float
    global_maxima_m: float | None
    global_minima_m: float | None
    displacement_m: float | None
    simulation_data: tuple


class CarouselSimulation:
    """
    Simulate and analyze the dynamics of a carousel with gondolas.

    This class provides methods to simulate the dynamics of a carousel with gondolas,
    analyze the results, and find optimal spring stiffness values that meet
    displacement criteria.

    Attributes
    ----------
    params: CarouselParameters
        Parameters for the carousel simulation.
    simulation_time_s: float
        Duration of the simulation in seconds.
    max_radial_displacement_m: float
        Maximum allowed radial displacement in meters.
    tilt_angle_rad: float
        Tilt angle of the carousel in radians.
    spring_stiffness_n_per_m: float
        Spring stiffness in Newtons per meter.
    initial_conditions: list[float]
        Initial conditions for the simulation.

    Methods
    -------
    run_simulation():
        Run the simulation and return the results.
    analyze_displacement(position_data):
        Analyze the displacement data and determine if criteria are met.
    find_optimal_spring_stiffness(search_config):
        Find optimal spring stiffness values that meet displacement criteria.
    """

    def __init__(
        self,
        params: CarouselParameters | None = None,
        simulation_time_s: float = 10,
        max_radial_displacement_m: float = 0.005,
    ) -> None:
        """
        Initialize the carousel simulation with given parameters.

        Parameters
        ----------
        params: CarouselParameters | None
            Parameters for the carousel simulation. If None, default parameters are used.
        simulation_time_s: float
            Duration of the simulation in seconds.
        max_radial_displacement_m: float
            Maximum allowed radial displacement in meters.
        """
        if params is None:
            params = CarouselParameters()

        self.params = params
        self.simulation_time_s = simulation_time_s
        self.max_radial_displacement_m = max_radial_displacement_m
        self.tilt_angle_rad = math.radians(params.tilt_angle_deg)
        self.spring_stiffness_n_per_m: float = 0

        self.initial_conditions: list[float] = [
            self.params.carousel_radius_m,
            0,
            0,
            2 * np.pi * self.params.carousel_rot_speed_rps,
        ]

    def system_dynamics(self, _: float, state: list[float]) -> list[float]:
        """
        Define the system dynamics for the carousel simulation.

        This method implements the differential equations that describe the motion
        of the gondolas on the carousel.

        Parameters
        ----------
        _: float
            Time variable (unused but required by solve_ivp).
        state: list[float]
            Current state of the system [position, velocity, angle, angular_velocity].

        Returns
        -------
        list[float]
            Derivatives of the state variables.
        """
        position_m, velocity_ms, angle_rad, angular_velocity_rads = state

        dposition_dt = velocity_ms
        dvelocity_dt = (
            position_m * (angular_velocity_rads**2)
            + self.params.gravity *
            np.sin(self.tilt_angle_rad) * np.cos(angle_rad)
            + (
                (self.spring_stiffness_n_per_m / self.params.gondola_mass_kg)
                * (self.params.carousel_radius_m - position_m)
            )
        )

        dangle_dt = angular_velocity_rads
        dangular_velocity_dt = -(
            2 * angular_velocity_rads * velocity_ms * position_m
            + self.params.gravity
            * np.sin(self.tilt_angle_rad)
            * np.sin(angle_rad)
            * position_m
        ) / (
            position_m**2
            + (5 / 3)
            * (self.params.gondola_width_m**2 + self.params.gondola_height_m**2)
            + 20 * self.params.carousel_radius_m**2
        )

        return [dposition_dt, dvelocity_dt, dangle_dt, dangular_velocity_dt]

    def run_simulation(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the simulation and return the results.

        This method solves the differential equations that describe the motion
        of the gondolas on the carousel.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Time array, position, velocity, angle, and angular velocity arrays.
        """
        time_span_s = [0, self.simulation_time_s]
        solution = solve_ivp(
            self.system_dynamics,
            time_span_s,
            self.initial_conditions,
            method="RK45",
            rtol=1e-3,
            atol=1e-6,
        )

        time_array_s = solution.t
        position_m, velocity_ms, angle_rad, angular_velocity_rads = solution.y
        return time_array_s, position_m, velocity_ms, angle_rad, angular_velocity_rads

    def plot_results(self, simulation_data: tuple) -> None:
        """
        Plot the simulation results.

        This method creates a figure with four subplots showing the position,
        velocity, angle, and angular velocity over time.

        Parameters
        ----------
        simulation_data: tuple
            Simulation data to plot.
        """
        time_array_s, position_m, velocity_ms, angle_rad, angular_velocity_rads = (
            simulation_data
        )
        fig, axes = plt.subplots(4, 1, figsize=(10, 12))

        axes[0].plot(time_array_s, position_m, label=r"$x(t)$")
        axes[0].set_ylabel(r"$x$ [m]")
        axes[0].legend(loc="upper right")
        axes[0].grid()
        axes[0].set_xlim(min(time_array_s), max(time_array_s))

        axes[1].plot(time_array_s, velocity_ms, label=r"$\dot{x}(t)$")
        axes[1].set_ylabel(r"$\dot{x}$ [m/s]")
        axes[1].legend(loc="upper right")
        axes[1].grid()
        axes[1].set_xlim(min(time_array_s), max(time_array_s))

        angle_deg = np.degrees(angle_rad)
        axes[2].plot(time_array_s, angle_deg, label=r"$\alpha(t)$")
        axes[2].set_ylabel(r"$\alpha$ [deg]")
        axes[2].legend(loc="upper right")
        axes[2].grid()
        axes[2].set_xlim(min(time_array_s), max(time_array_s))

        angular_velocity_degs = np.degrees(angular_velocity_rads)
        axes[3].plot(time_array_s, angular_velocity_degs,
                     label=r"$\dot{\alpha}(t)$")
        axes[3].set_ylabel(r"$\dot{\alpha}$ [deg/s]")
        axes[3].legend(loc="upper right")
        axes[3].grid()
        axes[3].set_xlim(min(time_array_s), max(time_array_s))

        for i in range(1, int(max(angle_deg) / 360) + 1):
            idx = np.where(np.isclose(angle_deg, 360 * i, atol=1))[0]
            if len(idx) > 0:
                axes[2].plot(time_array_s[idx[0]], angle_deg[idx[0]], "ro")
                for axis in axes:
                    axis.axvline(
                        x=time_array_s[idx[0]], color="r", linestyle="--", linewidth=1
                    )

        y_ticks = np.arange(0, max(angle_deg) + 360, 360)
        axes[2].set_yticks(y_ticks)
        axes[2].set_yticklabels([f"{int(y)}" for y in y_ticks])

        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        fig.text(0.5, 0.01, "Time (t)", ha="center", fontsize=12)
        fig.suptitle(
            f"Spring Stiffness={self.spring_stiffness_n_per_m} N/m", fontsize=16
        )
        plt.show()

    @staticmethod
    def get_local_extremes(data_series: np.ndarray) -> tuple[list[float], list[float]]:
        """
        Find local minima and maxima in the data series.

        Parameters
        ----------
        data_series: np.ndarray
            Data series to analyze.

        Returns
        -------
        tuple[list[float], list[float]]
            Lists of local minima and maxima.
        """
        local_maxima: list[float] = []
        local_minima: list[float] = []

        for i in range(1, len(data_series) - 1):
            if (
                data_series[i] > data_series[i - 1]
                and data_series[i] > data_series[i + 1]
            ):
                local_maxima.append(data_series[i])
            if (
                data_series[i] < data_series[i - 1]
                and data_series[i] < data_series[i + 1]
            ):
                local_minima.append(data_series[i])

        return local_minima, local_maxima

    @staticmethod
    def get_global_extremes(
        local_minima: list[float], local_maxima: list[float]
    ) -> tuple[float | None, float | None, float | None]:
        """
        Find global minima, maxima, and their difference from local extremes.

        Parameters
        ----------
        local_minima: list[float]
            List of local minima.
        local_maxima: list[float]
            List of local maxima.

        Returns
        -------
        tuple[float | None, float | None, float | None]
            Global minima, maxima, and their difference, or None if not available.
        """
        try:
            global_minima = round(min(local_minima), 5)
            global_maxima = round(max(local_maxima), 5)
            difference = round(global_maxima - global_minima, 5)
            return global_minima, global_maxima, difference

        except ValueError:
            return None, None, None

    def analyze_displacement(
        self, position_data: np.ndarray
    ) -> DisplacementAnalysisResult:
        """
        Analyze the displacement data and determine if criteria are met.

        Parameters
        ----------
        position_data: np.ndarray
            Position data to analyze.

        Returns
        -------
        DisplacementAnalysisResult
            Results of the displacement analysis.
        """
        local_minima, local_maxima = self.get_local_extremes(position_data)
        extremes = self.get_global_extremes(local_minima, local_maxima)
        global_minima, global_maxima, displacement_m = extremes

        criteria_met = False
        if (
            displacement_m is not None
            and displacement_m <= self.max_radial_displacement_m
        ):
            criteria_met = True

        return DisplacementAnalysisResult(
            global_minima=global_minima,
            global_maxima=global_maxima,
            displacement_m=displacement_m,
            criteria_met=criteria_met,
        )

    def print_simulation_result(
        self, analysis_result: DisplacementAnalysisResult
    ) -> None:
        """
        Print the simulation results.

        Parameters
        ----------
        analysis_result: DisplacementAnalysisResult
            Results of the displacement analysis.
        """
        global_minima = analysis_result.global_minima
        global_maxima = analysis_result.global_maxima
        displacement_m = analysis_result.displacement_m
        criteria_met = analysis_result.criteria_met

        if displacement_m is not None:
            message = (
                f"[Spring Stiffness: {self.spring_stiffness_n_per_m:<8} N/m] "
                f"Global maxima: {global_maxima:<8} m | "
                f"Global minima: {global_minima:<8} m | "
                f"Displacement: {displacement_m:<8} m"
            )
            if criteria_met:
                print(colored(message, "green"))
            else:
                print(colored(message, "red"))

    def process_stiffness_result(
        self,
        simulation_data: tuple,
        analysis_result: DisplacementAnalysisResult,
        show_results: bool,
    ) -> SimulationResult | None:
        """
        Process the results of a spring stiffness simulation.

        Parameters
        ----------
        simulation_data: tuple
            Raw simulation data.
        analysis_result: DisplacementAnalysisResult
            Results of the displacement analysis.
        show_results: bool
            Whether to show the results.

        Returns
        -------
        SimulationResult | None
            Simulation results, or None if criteria are not met.
        """
        if not analysis_result.criteria_met:
            return None

        result = SimulationResult(
            spring_stiffness_n_per_m=self.spring_stiffness_n_per_m,
            global_maxima_m=analysis_result.global_maxima,
            global_minima_m=analysis_result.global_minima,
            displacement_m=analysis_result.displacement_m,
            simulation_data=simulation_data,
        )

        if show_results:
            self.plot_results(simulation_data)

        return result

    def _run_single_simulation(
        self, stiffness_value_n_per_m: float, show_results: bool
    ) -> SimulationResult | None:
        """
        Run a single simulation with the given spring stiffness.

        Parameters
        ----------
        stiffness_value_n_per_m: float
            Spring stiffness in Newtons per meter.
        show_results: bool
            Whether to show the results.

        Returns
        -------
        SimulationResult | None
            Simulation results, or None if an error occurs or criteria are not met.
        """
        self.spring_stiffness_n_per_m = stiffness_value_n_per_m

        try:
            simulation_data = self.run_simulation()
            _, position_data, _, _, _ = simulation_data

            analysis_result = self.analyze_displacement(position_data)

            self.print_simulation_result(analysis_result)

            return self.process_stiffness_result(
                simulation_data,
                analysis_result,
                show_results,
            )

        except ValueError as error:
            error_message = (
                f"[Spring Stiffness: {self.spring_stiffness_n_per_m:<8} N/m] "
                f"Value Error: {str(error)}"
            )
            print(colored(error_message, "red"))
            return None

        except RuntimeError as error:
            error_message = (
                f"[Spring Stiffness: {self.spring_stiffness_n_per_m:<8} N/m] "
                f"Runtime Error: {str(error)}"
            )
            print(colored(error_message, "red"))
            return None

    def find_optimal_spring_stiffness(
        self, search_config: dict[str, Any]
    ) -> list[SimulationResult]:
        """
        Find optimal spring stiffness values that meet displacement criteria.

        Parameters
        ----------
        search_config: dict[str, Any]
            Configuration for the search.

        Returns
        -------
        list[SimulationResult]
            List of simulation results that meet the criteria.
        """
        start_n_per_m = search_config.get("start_n_per_m", 0)
        step_size_n_per_m = search_config.get("step_size_n_per_m", 100_000)
        num_results = search_config.get("num_results", 1)
        show_results = search_config.get("show_results", True)

        consecutive_valid_results = 0
        results_list: list[SimulationResult] = []
        max_stiffness = 999_999_999_999

        for stiffness in np.arange(start_n_per_m, max_stiffness, step_size_n_per_m):
            result = self._run_single_simulation(stiffness, show_results)

            if result:
                results_list.append(result)
                consecutive_valid_results += 1
                if consecutive_valid_results >= num_results:
                    break
            else:
                consecutive_valid_results = 0

        return results_list


def main() -> list[SimulationResult]:
    """
    Execute the main functionality of the script.

    This function creates a carousel simulator and finds the optimal spring stiffness
    that meets the displacement criteria.

    Returns
    -------
    list[SimulationResult]
        List of simulation results that meet the criteria.
    """
    simulator = CarouselSimulation()

    results = simulator.find_optimal_spring_stiffness(
        {
            "start_n_per_m": 0,
            "step_size_n_per_m": 100_000,
            "num_results": 1,
            "show_results": True,
        }
    )
    return results


if __name__ == "__main__":
    main()
