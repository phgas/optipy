"""
Carousel Simulation
===================

This module simulates the dynamics of a carousel with a gondola, focusing on the radial displacement under various spring stiffness conditions. It provides tools to analyze and visualize the displacement and find optimal spring stiffness.

The simulation uses the Runge-Kutta method to solve the differential equations describing the system's dynamics. The results are analyzed to determine if the displacement criteria are met, and the optimal spring stiffness is identified.

Examples
--------
>>> from carousel_simulation import CarouselSimulation
>>> simulator = CarouselSimulation()
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
from typing import Any, NamedTuple, list, tuple

colorama.init()


@dataclass
class CarouselParameters:
    """
    A class to hold parameters for the carousel simulation.

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

    Attributes
    ----------
    global_minima: float | None
        The global minimum displacement in meters.
    global_maxima: float | None
        The global maximum displacement in meters.
    displacement_m: float | None
        The total displacement in meters.
    criteria_met: bool
        Indicates whether the displacement criteria are met.
    """
    global_minima: float | None
    global_maxima: float | None
    displacement_m: float | None
    criteria_met: bool


class SimulationResult(NamedTuple):
    """
    A class to represent the results of a simulation.

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
        The raw simulation data.
    """
    spring_stiffness_n_per_m: float
    global_maxima_m: float | None
    global_minima_m: float | None
    displacement_m: float | None
    simulation_data: tuple


class CarouselSimulation:
    """
    A class to simulate and analyze the dynamics of a carousel with a gondola.

    This class models the behavior of a carousel under different spring stiffness conditions, runs simulations, and analyzes the results to find optimal spring stiffness.

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
        Current spring stiffness in N/m.
    initial_conditions: list[float]
        Initial conditions for the simulation.

    Methods
    -------
    system_dynamics(time: float, state: list[float]) -> list[float]
        Calculates the derivatives of the system state.
    run_simulation() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Runs the simulation and returns the results.
    plot_results(simulation_data: tuple) -> None
        Plots the simulation results.
    get_local_extremes(data_series: np.ndarray) -> tuple[list[float], list[float]]
        Finds local minima and maxima in a data series.
    get_global_extremes(local_minima: list[float], local_maxima: list[float]) -> tuple[float | None, float | None, float | None]
        Calculates global minima, maxima, and displacement.
    analyze_displacement(position_data: np.ndarray) -> DisplacementAnalysisResult
        Analyzes the displacement data.
    print_simulation_result(analysis_result: DisplacementAnalysisResult) -> None
        Prints the simulation results.
    process_stiffness_result(simulation_data: tuple, analysis_result: DisplacementAnalysisResult, show_results: bool) -> SimulationResult | None
        Processes the simulation results for a given stiffness.
    _run_single_simulation(stiffness_value_n_per_m: float, show_results: bool) -> SimulationResult | None
        Runs a single simulation for a given stiffness.
    find_optimal_spring_stiffness(search_config: dict[str, Any]) -> list[SimulationResult]
        Finds the optimal spring stiffness based on the search configuration.
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
            Parameters for the carousel simulation. If None, default parameters are used.
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
        Calculates the derivatives of the system state.

        Parameters
        ----------
        _: float
            Time, not used in the calculation.
        state: list[float]
            Current state of the system [position_m, velocity_ms, angle_rad, angular_velocity_rads].

        Returns
        -------
        list[float]
            Derivatives of the system state [dposition_dt, dvelocity_dt, dangle_dt, dangular_velocity_dt].
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
        Runs the simulation and returns the results.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Time array and state variables [time_array_s, position_m, velocity_ms, angle_rad, angular_velocity_rads].
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
            Simulation data containing [time_array_s, position_m, velocity_ms, angle_rad, angular_velocity_rads].
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
            The data series to analyze.

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
        Calculates global minima, maxima, and displacement.

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
        Analyzes the displacement data.

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
        Prints the simulation results.

        Parameters
        ----------
        analysis_result: DisplacementAnalysisResult
            Result of the displacement analysis.
        """
        global_minima: float | None = analysis_result.global_minima
        global_maxima: float | None = analysis_result.global_maxima
        displacement_m: float | None = analysis_result.displacement_m
        criteria_met: bool = analysis_result.criteria_met

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
        Processes the simulation results for a given stiffness.

        Parameters
        ----------
        simulation_data: tuple
            Simulation data containing [time_array_s, position_m, velocity_ms, angle_rad, angular_velocity_rads].
        analysis_result: DisplacementAnalysisResult
            Result of the displacement analysis.
        show_results: bool
            Flag to indicate whether to show the results.

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
        Runs a single simulation for a given stiffness.

        Parameters
        ----------
        stiffness_value_n_per_m: float
            Spring stiffness value in N/m.
        show_results: bool
            Flag to indicate whether to show the results.

        Returns
        -------
        SimulationResult | None
            Simulation result if successful, otherwise None.
        """
        self.spring_stiffness_n_per_m = stiffness_value_n_per_m

        try:
            simulation_data: tuple = self.run_simulation()
            _, position_data, _, _, _ = simulation_data

            analysis_result: DisplacementAnalysisResult = self.analyze_displacement(
                position_data)

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
        Finds the optimal spring stiffness based on the search configuration.

        Parameters
        ----------
        search_config: dict[str, Any]
            Configuration for the search, including start_n_per_m, step_size_n_per_m, num_results, and show_results.

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
            result: SimulationResult | None = self._run_single_simulation(
                stiffness, show_results)

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
    Executes the main functionality of the script.

    Returns
    -------
    list[SimulationResult]
        List of simulation results.
    """
    simulator: CarouselSimulation = CarouselSimulation()

    results: list[SimulationResult] = simulator.find_optimal_spring_stiffness(
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
