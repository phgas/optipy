"""
Carousel Simulation
==================

A module for simulating and analyzing the dynamics of a carousel with gondolas.

This module provides classes and functions to simulate the motion of gondolas on a
tilted carousel, analyze the displacement of the gondolas, and find optimal spring
stiffness values to control the radial displacement within specified limits.

Examples
--------
>>> from carousel_simulation import CarouselSimulation, CarouselParameters
>>> params = CarouselParameters(tilt_angle_deg=30, carousel_radius_m=6)
>>> simulator = CarouselSimulation(params)
>>> config = {"start_n_per_m": 0, "step_size_n_per_m": 100_000, "num_results": 1}
>>> results = simulator.find_optimal_spring_stiffness(config)
"""


# Standard library imports
import logging
import math
from dataclasses import dataclass
from typing import Any, NamedTuple

# Third party imports
import colorama
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp  # type: ignore
from termcolor import colored

logging.basicConfig(
    level=logging.INFO,
    format=(
        "[%(asctime)s.%(msecs)03d][%(levelname)s]"
        "[%(filename)s:%(lineno)d - %(funcName)s()]: %(message)s"
    ),
    datefmt="%d-%m-%Y %H:%M:%S",
)

colorama.init()


@dataclass
class CarouselParameters:
    """
    Parameters for configuring a carousel simulation.

    This class holds all the physical parameters needed to define a carousel with
    gondolas, including dimensions, masses, and rotation speeds.

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
        Rotation speed of the carousel in rotations per second.
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
    Results from analyzing the displacement of a gondola.

    This class stores the results of analyzing the radial displacement of a gondola
    during simulation, including minimum and maximum positions and whether the
    displacement meets specified criteria.

    Attributes
    ----------
    global_minima: float | None
        Minimum radial position of the gondola in meters, or None if not available.
    global_maxima: float | None
        Maximum radial position of the gondola in meters, or None if not available.
    displacement_m: float | None
        Total displacement (max - min) in meters, or None if not available.
    criteria_met: bool
        Whether the displacement meets the specified criteria.
    """

    global_minima: float | None
    global_maxima: float | None
    displacement_m: float | None
    criteria_met: bool


class SimulationResult(NamedTuple):
    """
    Results from a complete carousel simulation.

    This class stores the results of a carousel simulation with a specific spring
    stiffness, including the displacement analysis and the raw simulation data.

    Attributes
    ----------
    spring_stiffness_n_per_m: float
        Spring stiffness used in the simulation in N/m.
    global_maxima_m: float | None
        Maximum radial position of the gondola in meters, or None if not available.
    global_minima_m: float | None
        Minimum radial position of the gondola in meters, or None if not available.
    displacement_m: float | None
        Total displacement (max - min) in meters, or None if not available.
    simulation_data: tuple
        Raw data from the simulation (time, position, velocity, angle, angular velocity).
    """

    spring_stiffness_n_per_m: float
    global_maxima_m: float | None
    global_minima_m: float | None
    displacement_m: float | None
    simulation_data: tuple


class CarouselSimulation:
    """
    Simulates the dynamics of a gondola on a tilted carousel.

    This class provides methods to simulate the motion of a gondola on a tilted
    carousel, analyze the radial displacement, and find optimal spring stiffness
    values to control the displacement within specified limits.

    Attributes
    ----------
    params: CarouselParameters
        Parameters defining the carousel and gondola properties.
    simulation_time_s: float
        Duration of the simulation in seconds.
    max_radial_displacement_m: float
        Maximum allowed radial displacement in meters.
    tilt_angle_rad: float
        Tilt angle of the carousel in radians.
    spring_stiffness_n_per_m: float
        Spring stiffness in N/m.
    initial_conditions: list[float]
        Initial conditions for the simulation [position, velocity, angle, angular
        velocity].

    Methods
    -------
    run_simulation()
        Runs the simulation and returns the results.
    plot_results(simulation_data)
        Plots the simulation results.
    analyze_displacement(position_data)
        Analyzes the displacement of the gondola.
    find_optimal_spring_stiffness(search_config)
        Finds optimal spring stiffness values.
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
            Parameters defining the carousel and gondola properties.
            If None, default parameters will be used.
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

        This function defines the differential equations that govern the motion
        of a gondola on a tilted carousel.

        Parameters
        ----------
        _: float
            Time (not used directly in the equations).
        state: list[float]
            Current state of the system [position, velocity, angle, angular velocity].

        Returns
        -------
        list[float]
            Derivatives of the state variables.
        """
        position_m, velocity_ms, angle_rad, angular_velocity_rads = state

        dposition_dt = velocity_ms
        dvelocity_dt = self._calculate_velocity_derivative(
            position_m, angular_velocity_rads, angle_rad
        )
        dangle_dt = angular_velocity_rads
        dangular_velocity_dt = self._calculate_angular_velocity_derivative(
            position_m, velocity_ms, angular_velocity_rads, angle_rad
        )

        return [dposition_dt, dvelocity_dt, dangle_dt, dangular_velocity_dt]

    def _calculate_velocity_derivative(
        self, position_m: float, angular_velocity_rads: float, angle_rad: float
    ) -> float:
        """
        Calculate the derivative of the radial velocity.

        Parameters
        ----------
        position_m: float
            Current radial position in meters.
        angular_velocity_rads: float
            Current angular velocity in radians per second.
        angle_rad: float
            Current angle in radians.

        Returns
        -------
        float
            Derivative of the radial velocity.
        """
        return (
            position_m * (angular_velocity_rads**2)
            + self.params.gravity *
            np.sin(self.tilt_angle_rad) * np.cos(angle_rad)
            + (
                (self.spring_stiffness_n_per_m / self.params.gondola_mass_kg)
                * (self.params.carousel_radius_m - position_m)
            )
        )

    def _calculate_angular_velocity_derivative(
        self,
        position_m: float,
        velocity_ms: float,
        angular_velocity_rads: float,
        angle_rad: float,
    ) -> float:
        """
        Calculate the derivative of the angular velocity.

        Parameters
        ----------
        position_m: float
            Current radial position in meters.
        velocity_ms: float
            Current radial velocity in meters per second.
        angular_velocity_rads: float
            Current angular velocity in radians per second.
        angle_rad: float
            Current angle in radians.

        Returns
        -------
        float
            Derivative of the angular velocity.
        """
        numerator = -(
            2 * angular_velocity_rads * velocity_ms * position_m
            + self.params.gravity
            * np.sin(self.tilt_angle_rad)
            * np.sin(angle_rad)
            * position_m
        )

        denominator = (
            position_m**2
            + (5 / 3)
            * (self.params.gondola_width_m**2 + self.params.gondola_height_m**2)
            + 20 * self.params.carousel_radius_m**2
        )

        return numerator / denominator

    def run_simulation(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the carousel simulation.

        Solves the system of differential equations to simulate the motion of a gondola
        on a tilted carousel over the specified time period.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Time array, position array, velocity array, angle array, and angular
            velocity array.
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

        Creates a figure with four subplots showing the position, velocity, angle,
        and angular velocity of the gondola over time.

        Parameters
        ----------
        simulation_data: tuple
            Tuple containing time, position, velocity, angle, and angular velocity
            arrays.
        """
        time_array_s, position_m, velocity_ms, angle_rad, angular_velocity_rads = (
            simulation_data
        )
        fig, axes = plt.subplots(4, 1, figsize=(10, 12))

        self._plot_position(axes[0], time_array_s, position_m)
        self._plot_velocity(axes[1], time_array_s, velocity_ms)
        self._plot_angle(axes[2], time_array_s, angle_rad)
        self._plot_angular_velocity(
            axes[3], time_array_s, angular_velocity_rads)

        self._mark_full_rotations(axes, time_array_s, np.degrees(angle_rad))
        self._finalize_plot(fig, axes, time_array_s)

    def _plot_position(
        self, ax: plt.Axes, time_array_s: np.ndarray, position_m: np.ndarray
    ) -> None:
        """
        Plot the radial position over time.

        Parameters
        ----------
        ax: plt.Axes
            Matplotlib axes to plot on.
        time_array_s: np.ndarray
            Time array in seconds.
        position_m: np.ndarray
            Position array in meters.
        """
        ax.plot(time_array_s, position_m, label=r"$x(t)$")
        ax.set_ylabel(r"$x$ [m]")
        ax.legend(loc="upper right")
        ax.grid()
        ax.set_xlim(min(time_array_s), max(time_array_s))

    def _plot_velocity(
        self, ax: plt.Axes, time_array_s: np.ndarray, velocity_ms: np.ndarray
    ) -> None:
        """
        Plot the radial velocity over time.

        Parameters
        ----------
        ax: plt.Axes
            Matplotlib axes to plot on.
        time_array_s: np.ndarray
            Time array in seconds.
        velocity_ms: np.ndarray
            Velocity array in meters per second.
        """
        ax.plot(time_array_s, velocity_ms, label=r"$\dot{x}(t)$")
        ax.set_ylabel(r"$\dot{x}$ [m/s]")
        ax.legend(loc="upper right")
        ax.grid()
        ax.set_xlim(min(time_array_s), max(time_array_s))

    def _plot_angle(
        self, ax: plt.Axes, time_array_s: np.ndarray, angle_rad: np.ndarray
    ) -> None:
        """
        Plot the angle over time.

        Parameters
        ----------
        ax: plt.Axes
            Matplotlib axes to plot on.
        time_array_s: np.ndarray
            Time array in seconds.
        angle_rad: np.ndarray
            Angle array in radians.
        """
        angle_deg = np.degrees(angle_rad)
        ax.plot(time_array_s, angle_deg, label=r"$\alpha(t)$")
        ax.set_ylabel(r"$\alpha$ [deg]")
        ax.legend(loc="upper right")
        ax.grid()
        ax.set_xlim(min(time_array_s), max(time_array_s))

        y_ticks = np.arange(0, max(angle_deg) + 360, 360)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{int(y)}" for y in y_ticks])

    def _plot_angular_velocity(
        self, ax: plt.Axes, time_array_s: np.ndarray, angular_velocity_rads: np.ndarray
    ) -> None:
        """
        Plot the angular velocity over time.

        Parameters
        ----------
        ax: plt.Axes
            Matplotlib axes to plot on.
        time_array_s: np.ndarray
            Time array in seconds.
        angular_velocity_rads: np.ndarray
            Angular velocity array in radians per second.
        """
        angular_velocity_deg = np.degrees(angular_velocity_rads)
        ax.plot(time_array_s, angular_velocity_deg, label=r"$\dot{\alpha}(t)$")
        ax.set_ylabel(r"$\dot{\alpha}$ [deg/s]")
        ax.legend(loc="upper right")
        ax.grid()
        ax.set_xlim(min(time_array_s), max(time_array_s))

    def _mark_full_rotations(
        self, axes: list[plt.Axes], time_array_s: np.ndarray, angle_deg: np.ndarray
    ) -> None:
        """
        Mark full rotations on the plots.

        Parameters
        ----------
        axes: list[plt.Axes]
            List of Matplotlib axes to mark on.
        time_array_s: np.ndarray
            Time array in seconds.
        angle_deg: np.ndarray
            Angle array in degrees.
        """
        for i in range(1, int(max(angle_deg) / 360) + 1):
            idx = np.where(np.isclose(angle_deg, 360 * i, atol=1))[0]
            if len(idx) > 0:
                axes[2].plot(time_array_s[idx[0]], angle_deg[idx[0]], "ro")
                for axis in axes:
                    axis.axvline(
                        x=time_array_s[idx[0]], color="r", linestyle="--", linewidth=1
                    )

    def _finalize_plot(
        self, fig: plt.Figure, axes: list[plt.Axes], time_array_s: np.ndarray
    ) -> None:
        """
        Finalize the plot with labels and titles.

        Parameters
        ----------
        fig: plt.Figure
            Matplotlib figure to finalize.
        axes: list[plt.Axes]
            List of Matplotlib axes.
        time_array_s: np.ndarray
            Time array in seconds.
        """
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        fig.text(0.5, 0.01, "Time (t)", ha="center", fontsize=12)
        fig.suptitle(
            f"Spring Stiffness={self.spring_stiffness_n_per_m} N/m", fontsize=16
        )
        plt.show()

    @staticmethod
    def get_local_extremes(data_series: np.ndarray) -> tuple[list[float], list[float]]:
        """
        Find local minima and maxima in a data series.

        Parameters
        ----------
        data_series: np.ndarray
            Data series to analyze.

        Returns
        -------
        tuple[list[float], list[float]]
            Lists of local minima and maxima.
        """
        local_minima: list[float] = []
        local_maxima: list[float] = []

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
        Find global minima, maxima, and displacement from local extremes.

        Parameters
        ----------
        local_minima: list[float]
            List of local minima.
        local_maxima: list[float]
            List of local maxima.

        Returns
        -------
        tuple[float | None, float | None, float | None]
            Global minima, global maxima, and displacement, or None if not available.
        """
        try:
            global_minima = round(min(local_minima), 5)
            global_maxima = round(max(local_maxima), 5)
            displacement = round(global_maxima - global_minima, 5)
            return global_minima, global_maxima, displacement

        except ValueError:
            return None, None, None

    def analyze_displacement(
        self, position_data: np.ndarray
    ) -> DisplacementAnalysisResult:
        """
        Analyze the displacement of the gondola.

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
        Print the results of a simulation.

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
                logging.info(colored(message, "green"))
            else:
                logging.info(colored(message, "red"))

    def process_stiffness_result(
        self,
        simulation_data: tuple,
        analysis_result: DisplacementAnalysisResult,
        show_results: bool,
    ) -> SimulationResult | None:
        """
        Process the results of a simulation with a specific spring stiffness.

        Parameters
        ----------
        simulation_data: tuple
            Raw data from the simulation.
        analysis_result: DisplacementAnalysisResult
            Results of the displacement analysis.
        show_results: bool
            Whether to show plots of the results.

        Returns
        -------
        SimulationResult | None
            Processed simulation results, or None if criteria not met.
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
        Run a single simulation with a specific spring stiffness.

        Parameters
        ----------
        stiffness_value_n_per_m: float
            Spring stiffness to use in N/m.
        show_results: bool
            Whether to show plots of the results.

        Returns
        -------
        SimulationResult | None
            Processed simulation results, or None if criteria not met or errors occurred.
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
            logging.error(colored(error_message, "red"))
            return None

        except RuntimeError as error:
            error_message = (
                f"[Spring Stiffness: {self.spring_stiffness_n_per_m:<8} N/m] "
                f"Runtime Error: {str(error)}"
            )
            logging.error(colored(error_message, "red"))
            return None

    def find_optimal_spring_stiffness(
        self, search_config: dict[str, Any]
    ) -> list[SimulationResult]:
        """
        Find optimal spring stiffness values.

        Searches for spring stiffness values that keep the radial displacement
        within the specified limits.

        Parameters
        ----------
        search_config: dict[str, Any]
            Configuration for the search, including start value, step size,
            number of results to find, and whether to show plots.

        Returns
        -------
        list[SimulationResult]
            List of simulation results with optimal spring stiffness values.
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


def run_simulation_with_config(config: dict[str, Any]) -> list[SimulationResult]:
    """
    Run the carousel simulation with the given configuration.

    Parameters
    ----------
    config: dict[str, Any]
        Configuration for the simulation search.

    Returns
    -------
    list[SimulationResult]
        List of simulation results with optimal spring stiffness values.
    """
    simulator = CarouselSimulation()
    return simulator.find_optimal_spring_stiffness(config)


def main() -> None:
    """
    Execute the main functionality of the script.

    This function sets up a default configuration and runs the carousel simulation
    to find optimal spring stiffness values.
    """
    config = {
        "start_n_per_m": 0,
        "step_size_n_per_m": 100_000,
        "num_results": 1,
        "show_results": True,
    }
    results = run_simulation_with_config(config)
    return results


if __name__ == "__main__":
    main()
