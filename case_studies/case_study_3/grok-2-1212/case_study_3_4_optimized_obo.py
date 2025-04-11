"""
Carousel Simulation Module
==========================

This module provides functionality for simulating and analyzing the dynamics of a carousel system. It includes classes and methods to model the carousel's behavior, run simulations, and analyze the results.

The module uses numerical integration to solve the system of differential equations describing the carousel's motion. It also includes tools for plotting the results and finding optimal spring stiffness values.

Examples
--------
>>> from carousel_simulation import CarouselSimulation, CarouselParameters
>>> params = CarouselParameters()
>>> simulator = CarouselSimulation(params)
>>> results = simulator.find_optimal_spring_stiffness({"start_n_per_m": 0, "step_size_n_per_m": 100_000, "num_results": 1, "show_results": False})
>>> print(results[0].spring_stiffness_n_per_m if results else "No valid results")
"""


# Standard library imports
import logging
import math

# Third party imports
import colorama
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from termcolor import colored

# Local imports
from dataclasses import dataclass
from typing import Any, NamedTuple

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
    A class to hold parameters for the carousel simulation.

    This class encapsulates all the physical parameters needed to model the carousel system.

    Attributes
    ----------
    gravity: float
        Acceleration due to gravity in m/s^2.
    tilt_angle_deg: float
        Tilt angle of the carousel in degrees.
    carousel_radius_m: float
        Radius of the carousel in meters.
    gondola_mass_kg: float
        Mass of the gondola in kilograms.
    carousel_rot_speed_rps: float
        Rotational speed of the carousel in revolutions per second.
    gondola_width_m: float
        Width of the gondola in meters.
    gondola_height_m: float
        Height of the gondola in meters.
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
    A class to represent the results of displacement analysis.

    This class holds the results of analyzing the displacement of the carousel system.

    Attributes
    ----------
    global_minima_m: float | None
        The global minimum displacement in meters.
    global_maxima_m: float | None
        The global maximum displacement in meters.
    displacement_m: float | None
        The total displacement in meters.
    criteria_met: bool
        Indicates whether the displacement criteria were met.
    """
    global_minima_m: float | None
    global_maxima_m: float | None
    displacement_m: float | None
    criteria_met: bool


class SimulationResult(NamedTuple):
    """
    A class to represent the results of a simulation.

    This class encapsulates the results of a single simulation run.

    Attributes
    ----------
    spring_stiffness_n_per_m: float
        The spring stiffness used in the simulation in N/m.
    global_maxima_m: float | None
        The global maximum displacement in meters.
    global_minima_m: float | None
        The global minimum displacement in meters.
    displacement_m: float | None
        The total displacement in meters.
    simulation_data: tuple
        The raw data from the simulation.
    """
    spring_stiffness_n_per_m: float
    global_maxima_m: float | None
    global_minima_m: float | None
    displacement_m: float | None
    simulation_data: tuple


class CarouselSimulation:
    """
    A class to simulate and analyze the dynamics of a carousel system.

    This class models the behavior of a carousel, runs simulations, and analyzes the results. It uses numerical integration to solve the system of differential equations describing the carousel's motion.

    Attributes
    ----------
    params: CarouselParameters
        The parameters of the carousel system.
    simulation_time_s: float
        The duration of the simulation in seconds.
    max_radial_displacement_m: float
        The maximum allowed radial displacement in meters.
    tilt_angle_rad: float
        The tilt angle of the carousel in radians.
    spring_stiffness_n_per_m: float
        The spring stiffness used in the simulation in N/m.
    initial_conditions: list[float]
        The initial conditions for the simulation.

    Methods
    -------
    __init__(params, simulation_time_s, max_radial_displacement_m)
        Initialize the CarouselSimulation object.
    system_dynamics(_, state)
        Calculate the system dynamics for the given state.
    run_simulation()
        Run the simulation and return the results.
    plot_results(simulation_data)
        Plot the results of the simulation.
    get_local_extremes(data_series)
        Find local minima and maxima in the given data series.
    get_global_extremes(local_minima, local_maxima)
        Find global minima and maxima from local extremes.
    analyze_displacement(position_data)
        Analyze the displacement data and return the results.
    print_simulation_result(analysis_result)
        Print the results of the simulation.
    process_stiffness_result(simulation_data, analysis_result, show_results)
        Process the results of a simulation run and return a SimulationResult if criteria are met.
    _run_single_simulation(stiffness_value_n_per_m, show_results)
        Run a single simulation with the given stiffness value.
    find_optimal_spring_stiffness(search_config)
        Find the optimal spring stiffness based on the given search configuration.
    """

    def __init__(
        self,
        params: CarouselParameters | None = None,
        simulation_time_s: float = 10,
        max_radial_displacement_m: float = 0.005,
    ) -> None:
        """
        Initialize the CarouselSimulation object.

        This method sets up the simulation with the given parameters or default values.

        Parameters
        ----------
        params: CarouselParameters | None
            The parameters of the carousel system. If None, default parameters are used.
        simulation_time_s: float
            The duration of the simulation in seconds.
        max_radial_displacement_m: float
            The maximum allowed radial displacement in meters.

        Returns
        -------
        None
        """
        if params is None:
            params = CarouselParameters()

        self.params: CarouselParameters = params
        self.simulation_time_s: float = simulation_time_s
        self.max_radial_displacement_m: float = max_radial_displacement_m
        self.tilt_angle_rad: float = math.radians(params.tilt_angle_deg)
        self.spring_stiffness_n_per_m: float = 0

        self.initial_conditions: list[float] = [
            self.params.carousel_radius_m,
            0,
            0,
            2 * np.pi * self.params.carousel_rot_speed_rps,
        ]

    def system_dynamics(self, _: float, state: list[float]) -> list[float]:
        """
        Calculate the system dynamics for the given state.

        This method computes the derivatives of the state variables based on the current state.

        Parameters
        ----------
        _: float
            Time, not used in the calculation.
        state: list[float]
            The current state of the system.

        Returns
        -------
        list[float]
            The derivatives of the state variables.
        """
        position_m: float
        velocity_ms: float
        angle_rad: float
        angular_velocity_rads: float
        position_m, velocity_ms, angle_rad, angular_velocity_rads = state

        dposition_dt: float = velocity_ms
        dvelocity_dt: float = (
            position_m * (angular_velocity_rads**2)
            + self.params.gravity * np.sin(self.tilt_angle_rad) * np.cos(angle_rad)
            + (
                (self.spring_stiffness_n_per_m / self.params.gondola_mass_kg)
                * (self.params.carousel_radius_m - position_m)
            )
        )

        dangle_dt: float = angular_velocity_rads
        dangular_velocity_dt: float = -(
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

    def run_simulation(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the simulation and return the results.

        This method uses numerical integration to solve the system of differential equations and returns the simulation data.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            The time array and the state variables (position, velocity, angle, angular velocity) over time.
        """
        time_span_s: list[float] = [0, self.simulation_time_s]
        solution = solve_ivp(
            self.system_dynamics,
            time_span_s,
            self.initial_conditions,
            method="RK45",
            rtol=1e-3,
            atol=1e-6,
        )

        time_array_s: np.ndarray = solution.t
        position_m: np.ndarray
        velocity_ms: np.ndarray
        angle_rad: np.ndarray
        angular_velocity_rads: np.ndarray
        position_m, velocity_ms, angle_rad, angular_velocity_rads = solution.y
        return time_array_s, position_m, velocity_ms, angle_rad, angular_velocity_rads

    def plot_results(self, simulation_data: tuple) -> None:
        """
        Plot the results of the simulation.

        This method creates plots for position, velocity, angle, and angular velocity over time.

        Parameters
        ----------
        simulation_data: tuple
            The simulation data containing time and state variables.

        Returns
        -------
        None
        """
        time_array_s: np.ndarray
        position_m: np.ndarray
        velocity_ms: np.ndarray
        angle_rad: np.ndarray
        angular_velocity_rads: np.ndarray
        time_array_s, position_m, velocity_ms, angle_rad, angular_velocity_rads = simulation_data
        fig, axes = plt.subplots(4, 1, figsize=(10, 12))

        axes[0].plot(time_array_s, position_m)
        axes[0].set_ylabel(r"$x$ [m]")
        axes[0].legend([r"$x(t)$"], loc="upper right")
        axes[0].grid()
        axes[0].set_xlim(min(time_array_s), max(time_array_s))

        axes[1].plot(time_array_s, velocity_ms)
        axes[1].set_ylabel(r"$\dot{x}$ [m/s]")
        axes[1].legend([r"$\dot{x}(t)$"], loc="upper right")
        axes[1].grid()
        axes[1].set_xlim(min(time_array_s), max(time_array_s))

        angle_deg: np.ndarray = np.degrees(angle_rad)
        axes[2].plot(time_array_s, angle_deg)
        axes[2].set_ylabel(r"$\alpha$ [deg]")
        axes[2].legend([r"$\alpha(t)$"], loc="upper right")
        axes[2].grid()
        axes[2].set_xlim(min(time_array_s), max(time_array_s))

        angular_velocity_degs: np.ndarray = np.degrees(angular_velocity_rads)
        axes[3].plot(time_array_s, angular_velocity_degs)
        axes[3].set_ylabel(r"$\dot{\alpha}$ [deg/s]")
        axes[3].legend([r"$\dot{\alpha}(t)$"], loc="upper right")
        axes[3].grid()
        axes[3].set_xlim(min(time_array_s), max(time_array_s))

        for i in range(1, int(max(angle_deg) / 360) + 1):
            idx: np.ndarray = np.where(np.isclose(angle_deg, 360 * i, atol=1))[0]
            if len(idx) > 0:
                axes[2].plot(time_array_s[idx[0]], angle_deg[idx[0]], "ro")
                for axis in axes:
                    axis.axvline(x=time_array_s[idx[0]], color="r", linestyle="--", linewidth=1)

        y_ticks: np.ndarray = np.arange(0, max(angle_deg) + 360, 360)
        axes[2].set_yticks(y_ticks)
        axes[2].set_yticklabels([f"{int(y)}" for y in y_ticks])

        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        fig.text(0.5, 0.01, "Time (t)", ha="center", fontsize=12)
        fig.suptitle(f"Spring Stiffness={self.spring_stiffness_n_per_m} N/m", fontsize=16)
        plt.show()

    @staticmethod
    def get_local_extremes(data_series: np.ndarray) -> tuple[list[float], list[float]]:
        """
        Find local minima and maxima in the given data series.

        This method identifies local extrema in the data series.

        Parameters
        ----------
        data_series: np.ndarray
            The data series to analyze.

        Returns
        -------
        tuple[list[float], list[float]]
            Lists of local minima and maxima.
        """
        local_maxima: list[float] = []
        local_minima: list[float] = []

        for i in range(1, len(data_series) - 1):
            if data_series[i] > data_series[i - 1] and data_series[i] > data_series[i + 1]:
                local_maxima.append(data_series[i])
            if data_series[i] < data_series[i - 1] and data_series[i] < data_series[i + 1]:
                local_minima.append(data_series[i])

        return local_minima, local_maxima

    @staticmethod
    def get_global_extremes(local_minima: list[float], local_maxima: list[float]) -> tuple[float | None, float | None, float | None]:
        """
        Find global minima and maxima from local extremes.

        This method calculates the global extrema from the list of local extrema.

        Parameters
        ----------
        local_minima: list[float]
            List of local minima.
        local_maxima: list[float]
            List of local maxima.

        Returns
        -------
        tuple[float | None, float | None, float | None]
            Global minima, global maxima, and the difference between them.
        """
        try:
            global_minima: float = round(min(local_minima), 5)
            global_maxima: float = round(max(local_maxima), 5)
            difference: float = round(global_maxima - global_minima, 5)
            return global_minima, global_maxima, difference
        except ValueError:
            return None, None, None

    def analyze_displacement(self, position_data: np.ndarray) -> DisplacementAnalysisResult:
        """
        Analyze the displacement data and return the results.

        This method analyzes the position data to find global extrema and check if the displacement criteria are met.

        Parameters
        ----------
        position_data: np.ndarray
            The position data to analyze.

        Returns
        -------
        DisplacementAnalysisResult
            The results of the displacement analysis.
        """
        local_minima: list[float]
        local_maxima: list[float]
        local_minima, local_maxima = self.get_local_extremes(position_data)
        extremes: tuple[float | None, float | None, float | None] = self.get_global_extremes(local_minima, local_maxima)
        global_minima_m: float | None
        global_maxima_m: float | None
        displacement_m: float | None
        global_minima_m, global_maxima_m, displacement_m = extremes

        criteria_met: bool = displacement_m is not None and displacement_m <= self.max_radial_displacement_m

        return DisplacementAnalysisResult(
            global_minima_m=global_minima_m,
            global_maxima_m=global_maxima_m,
            displacement_m=displacement_m,
            criteria_met=criteria_met,
        )

    def print_simulation_result(self, analysis_result: DisplacementAnalysisResult) -> None:
        """
        Print the results of the simulation.

        This method logs the simulation results with color-coded output based on whether the criteria were met.

        Parameters
        ----------
        analysis_result: DisplacementAnalysisResult
            The results of the displacement analysis.

        Returns
        -------
        None
        """
        global_minima_m: float | None = analysis_result.global_minima_m
        global_maxima_m: float | None = analysis_result.global_maxima_m
        displacement_m: float | None = analysis_result.displacement_m
        criteria_met: bool = analysis_result.criteria_met

        if displacement_m is not None:
            message: str = (
                f"[Spring Stiffness: {self.spring_stiffness_n_per_m:<8} N/m] "
                f"Global maxima: {global_maxima_m:<8} m | "
                f"Global minima: {global_minima_m:<8} m | "
                f"Displacement: {displacement_m:<8} m"
            )
            color: str = "green" if criteria_met else "red"
            logging.info(colored(message, color))

    def process_stiffness_result(self, simulation_data: tuple, analysis_result: DisplacementAnalysisResult, show_results: bool) -> SimulationResult | None:
        """
        Process the results of a simulation run and return a SimulationResult if criteria are met.

        This method processes the simulation data and analysis results, and optionally plots the results.

        Parameters
        ----------
        simulation_data: tuple
            The raw data from the simulation.
        analysis_result: DisplacementAnalysisResult
            The results of the displacement analysis.
        show_results: bool
            Whether to show the plot of the results.

        Returns
        -------
        SimulationResult | None
            The processed simulation result if criteria are met, otherwise None.
        """
        if not analysis_result.criteria_met:
            return None

        result: SimulationResult = SimulationResult(
            spring_stiffness_n_per_m=self.spring_stiffness_n_per_m,
            global_maxima_m=analysis_result.global_maxima_m,
            global_minima_m=analysis_result.global_minima_m,
            displacement_m=analysis_result.displacement_m,
            simulation_data=simulation_data,
        )

        if show_results:
            self.plot_results(simulation_data)

        return result

    def _run_single_simulation(self, stiffness_value_n_per_m: float, show_results: bool) -> SimulationResult | None:
        """
        Run a single simulation with the given stiffness value.

        This method runs a simulation with the specified spring stiffness and processes the results.

        Parameters
        ----------
        stiffness_value_n_per_m: float
            The spring stiffness to use in the simulation in N/m.
        show_results: bool
            Whether to show the plot of the results.

        Returns
        -------
        SimulationResult | None
            The processed simulation result if criteria are met, otherwise None.

        Raises
        ------
        ValueError
            If there is an error in the simulation data.
        RuntimeError
            If there is an error during the simulation run.
        """
        self.spring_stiffness_n_per_m: float = stiffness_value_n_per_m

        try:
            simulation_data: tuple = self.run_simulation()
            _, position_data, _, _, _ = simulation_data

            analysis_result: DisplacementAnalysisResult = self.analyze_displacement(position_data)

            self.print_simulation_result(analysis_result)

            return self.process_stiffness_result(simulation_data, analysis_result, show_results)
        except ValueError as error:
            error_message: str = (
                f"[Spring Stiffness: {self.spring_stiffness_n_per_m:<8} N/m] "
                f"Value Error: {str(error)}"
            )
            logging.error(colored(error_message, "red"))
            return None
        except RuntimeError as error:
            error_message: str = (
                f"[Spring Stiffness: {self.spring_stiffness_n_per_m:<8} N/m] "
                f"Runtime Error: {str(error)}"
            )
            logging.error(colored(error_message, "red"))
            return None

    def find_optimal_spring_stiffness(self, search_config: dict[str, Any]) -> list[SimulationResult]:
        """
        Find the optimal spring stiffness based on the given search configuration.

        This method iterates through a range of spring stiffness values to find those that meet the displacement criteria.

        Parameters
        ----------
        search_config: dict[str, Any]
            A dictionary containing the search configuration parameters.

        Returns
        -------
        list[SimulationResult]
            A list of SimulationResult objects for the stiffness values that meet the criteria.
        """
        start_n_per_m: float = search_config.get("start_n_per_m", 0)
        step_size_n_per_m: float = search_config.get("step_size_n_per_m", 100_000)
        num_results: int = search_config.get("num_results", 1)
        show_results: bool = search_config.get("show_results", True)

        results_list: list[SimulationResult] = []
        max_stiffness: float = 999_999_999_999

        for stiffness in np.arange(start_n_per_m, max_stiffness, step_size_n_per_m):
            result: SimulationResult | None = self._run_single_simulation(stiffness, show_results)

            if result:
                results_list.append(result)
                if len(results_list) >= num_results:
                    break

        return results_list


def create_simulator() -> CarouselSimulation:
    """
    Create and return a new CarouselSimulation object.

    This function initializes a CarouselSimulation with default parameters.

    Returns
    -------
    CarouselSimulation
        A new CarouselSimulation object.
    """
    return CarouselSimulation()


def run_simulation(simulator: CarouselSimulation) -> list[SimulationResult]:
    """
    Run a simulation using the given simulator and return the results.

    This function sets up the search configuration and runs the simulation to find optimal spring stiffness values.

    Parameters
    ----------
    simulator: CarouselSimulation
        The CarouselSimulation object to use for the simulation.

    Returns
    -------
    list[SimulationResult]
        A list of SimulationResult objects for the stiffness values that meet the criteria.
    """
    search_config: dict[str, Any] = {
        "start_n_per_m": 0,
        "step_size_n_per_m": 100_000,
        "num_results": 1,
        "show_results": True,
    }
    return