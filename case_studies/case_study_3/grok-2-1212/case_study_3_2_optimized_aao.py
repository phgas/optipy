"""
Carousel Simulation
===================

This module simulates the dynamics of a carousel with a gondola, focusing on the radial displacement under various spring stiffness conditions. It includes classes and methods to set up the simulation, run it, analyze the results, and visualize the data.

The simulation calculates the position, velocity, angle, and angular velocity of the gondola over time, and determines the optimal spring stiffness to minimize radial displacement within specified criteria.

Examples
--------
>>> from carousel_simulation import CarouselSimulation, CarouselParameters
>>> params = CarouselParameters()
>>> simulator = CarouselSimulation(params)
>>> results = simulator.find_optimal_spring_stiffness({"start_n_per_m": 0, "step_size_n_per_m": 100_000, "num_results": 1, "show_results": True})
>>> print(results)
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp  # type: ignore
import math
from termcolor import colored
import colorama
from dataclasses import dataclass
from typing import Any, NamedTuple


colorama.init()


@dataclass
class CarouselParameters:
    gravity: float = 9.81
    tilt_angle_deg: float = 30
    carousel_radius_m: float = 6
    gondola_mass_kg: float = 300
    carousel_rot_speed_rps: float = 0.2333
    gondola_width_m: float = 1.5
    gondola_height_m: float = 1.5


class DisplacementAnalysisResult(NamedTuple):
    global_minima: float | None
    global_maxima: float | None
    displacement_m: float | None
    criteria_met: bool


class SimulationResult(NamedTuple):
    spring_stiffness_n_per_m: float
    global_maxima_m: float | None
    global_minima_m: float | None
    displacement_m: float | None
    simulation_data: tuple


class CarouselSimulation:
    """
    A class to simulate the dynamics of a carousel with a gondola.

    This class models the behavior of a carousel under different spring stiffness conditions, calculates the radial displacement, and provides methods for analysis and visualization.

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
        Current spring stiffness used in the simulation in N/m.
    initial_conditions: list[float]
        Initial conditions for the simulation.

    Methods
    -------
    system_dynamics(t, state)
        Calculates the derivatives of the state variables.
    run_simulation()
        Runs the simulation and returns the results.
    plot_results(simulation_data)
        Plots the simulation results.
    get_local_extremes(data_series)
        Finds local minima and maxima in a data series.
    get_global_extremes(local_minima, local_maxima)
        Calculates global minima, maxima, and displacement.
    analyze_displacement(position_data)
        Analyzes the displacement data and returns the result.
    print_simulation_result(analysis_result)
        Prints the simulation result with color coding.
    process_stiffness_result(simulation_data, analysis_result, show_results)
        Processes the simulation result and optionally shows the plot.
    _run_single_simulation(stiffness_value_n_per_m, show_results)
        Runs a single simulation with a given stiffness value.
    find_optimal_spring_stiffness(search_config)
        Finds the optimal spring stiffness based on the given search configuration.
    """

    def __init__(
        self,
        params: CarouselParameters | None = None,
        simulation_time_s: float = 10,
        max_radial_displacement_m: float = 0.005,
    ) -> None:
        """
        Initializes the CarouselSimulation with given parameters.

        Parameters
        ----------
        params: CarouselParameters | None
            Parameters defining the carousel and gondola properties. If None, default parameters are used.
        simulation_time_s: float
            Duration of the simulation in seconds.
        max_radial_displacement_m: float
            Maximum allowed radial displacement in meters.
        """
        if params is None:
            params = CarouselParameters()

        self.params: CarouselParameters = params
        self.simulation_time_s: float = simulation_time_s
        self.max_radial_displacement_m: float = max_radial_displacement_m
        self.tilt_angle_rad: float = math.radians(self.params.tilt_angle_deg)
        self.spring_stiffness_n_per_m: float = 0

        self.initial_conditions: list[float] = [
            self.params.carousel_radius_m,
            0,
            0,
            2 * np.pi * self.params.carousel_rot_speed_rps,
        ]

    def system_dynamics(self, _: float, state: list[float]) -> list[float]:
        """
        Calculates the derivatives of the state variables for the carousel system.

        Parameters
        ----------
        _: float
            Time, not used in the calculation.
        state: list[float]
            Current state of the system [position_m, velocity_ms, angle_rad, angular_velocity_rads].

        Returns
        -------
        list[float]
            Derivatives of the state variables [dposition_dt, dvelocity_dt, dangle_dt, dangular_velocity_dt].
        """
        position_m, velocity_ms, angle_rad, angular_velocity_rads = state

        dposition_dt: float = velocity_ms
        dvelocity_dt: float = (
            position_m * (angular_velocity_rads**2)
            + self.params.gravity *
            np.sin(self.tilt_angle_rad) * np.cos(angle_rad)
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

    def run_simulation(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs the simulation using the initial conditions and system dynamics.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Time array and state variables (position, velocity, angle, angular velocity).
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
        position_m, velocity_ms, angle_rad, angular_velocity_rads = solution.y
        return time_array_s, position_m, velocity_ms, angle_rad, angular_velocity_rads

    def plot_results(self, simulation_data: tuple) -> None:
        """
        Plots the simulation results.

        Parameters
        ----------
        simulation_data: tuple
            Tuple containing time array and state variables.
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

        angle_deg: np.ndarray = np.degrees(angle_rad)
        axes[2].plot(time_array_s, angle_deg, label=r"$\alpha(t)$")
        axes[2].set_ylabel(r"$\alpha$ [deg]")
        axes[2].legend(loc="upper right")
        axes[2].grid()
        axes[2].set_xlim(min(time_array_s), max(time_array_s))

        angular_velocity_degs: np.ndarray = np.degrees(angular_velocity_rads)
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

        y_ticks: np.ndarray = np.arange(0, max(angle_deg) + 360, 360)
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
        Finds local minima and maxima in a data series.

        Parameters
        ----------
        data_series: np.ndarray
            Array of data points to analyze.

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
        Calculates global minima, maxima, and displacement from local extremes.

        Parameters
        ----------
        local_minima: list[float]
            List of local minima.
        local_maxima: list[float]
            List of local maxima.

        Returns
        -------
        tuple[float | None, float | None, float | None]
            Global minima, global maxima, and displacement.
        """
        try:
            global_minima: float = round(min(local_minima), 5)
            global_maxima: float = round(max(local_maxima), 5)
            difference: float = round(global_maxima - global_minima, 5)
            return global_minima, global_maxima, difference

        except ValueError:
            return None, None, None

    def analyze_displacement(
        self, position_data: np.ndarray
    ) -> DisplacementAnalysisResult:
        """
        Analyzes the displacement data and returns the result.

        Parameters
        ----------
        position_data: np.ndarray
            Array of position data.

        Returns
        -------
        DisplacementAnalysisResult
            Result of the displacement analysis.
        """
        local_minima, local_maxima = self.get_local_extremes(position_data)
        extremes = self.get_global_extremes(local_minima, local_maxima)
        global_minima, global_maxima, displacement_m = extremes

        criteria_met: bool = False
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
        Prints the simulation result with color coding.

        Parameters
        ----------
        analysis_result: DisplacementAnalysisResult
            Result of the displacement analysis.
        """
        global_minima = analysis_result.global_minima
        global_maxima = analysis_result.global_maxima
        displacement_m = analysis_result.displacement_m
        criteria_met = analysis_result.criteria_met

        if displacement_m is not None:
            message: str = (
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
        Processes the simulation result and optionally shows the plot.

        Parameters
        ----------
        simulation_data: tuple
            Tuple containing simulation data.
        analysis_result: DisplacementAnalysisResult
            Result of the displacement analysis.
        show_results: bool
            Flag to show the plot.

        Returns
        -------
        SimulationResult | None
            Simulation result if criteria are met, otherwise None.
        """
        if not analysis_result.criteria_met:
            return None

        result: SimulationResult = SimulationResult(
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
        Runs a single simulation with a given stiffness value.

        Parameters
        ----------
        stiffness_value_n_per_m: float
            Spring stiffness value in N/m.
        show_results: bool
            Flag to show the plot.

        Returns
        -------
        SimulationResult | None
            Simulation result if successful, otherwise None.
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
            error_message: str = (
                f"[Spring Stiffness: {self.spring_stiffness_n_per_m:<8} N/m] "
                f"Value Error: {str(error)}"
            )
            print(colored(error_message, "red"))
            return None

        except RuntimeError as error:
            error_message: str = (
                f"[Spring Stiffness: {self.spring_stiffness_n_per_m:<8} N/m] "
                f"Runtime Error: {str(error)}"
            )
            print(colored(error_message, "red"))
            return None

    def find_optimal_spring_stiffness(
        self, search_config: dict[str, Any]
    ) -> list[SimulationResult]:
        """
        Finds the optimal spring stiffness based on the given search configuration.

        Parameters
        ----------
        search_config: dict[str, Any]
            Configuration dictionary for the search.

        Returns
        -------
        list[SimulationResult]
            List of simulation results meeting the criteria.
        """
        start_n_per_m: float = search_config.get("start_n_per_m", 0)
        step_size_n_per_m: float = search_config.get(
            "step_size_n_per_m", 100_000)
        num_results: int = search_config.get("num_results", 1)
        show_results: bool = search_config.get("show_results", True)

        consecutive_valid_results: int = 0
        results_list: list[SimulationResult] = []
        max_stiffness: float = 999_999_999_999

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


def get_simulator() -> CarouselSimulation:
    """
    Creates and returns a CarouselSimulation instance.

    Returns
    -------
    CarouselSimulation
        A new instance of CarouselSimulation.
    """
    return CarouselSimulation()


def run_simulation(search_config: dict[str, Any]) -> list[SimulationResult]:
    """
    Runs the simulation with the given search configuration.

    Parameters
    ----------
    search_config: dict[str, Any]
        Configuration dictionary for the search.

    Returns
    -------
    list[SimulationResult]
        List of simulation results meeting the criteria.
    """
    simulator: CarouselSimulation = get_simulator()
    return simulator.find_optimal_spring_stiffness(search_config)


def main() -> list[SimulationResult]:
    """
    Executes the main functionality of the script.

    Returns
    -------
    list[SimulationResult]
        List of simulation results meeting the criteria.
    """
    search_config: dict[str, Any] = {
        "start_n_per_m": 0,
        "step_size_n_per_m": 100_000,
        "num_results": 1,
        "show_results": True,
    }
    return run_simulation(search_config)


if __name__ == "__main__":
    main()
