"""
Carousel Simulation Module
===========================

This module provides a simulation of a carousel with a focus on
analyzing the radial displacement of the gondolas due to varying
spring stiffness. It includes classes and functions to model the
carousel's dynamics, run simulations, analyze results, and find the
optimal spring stiffness to minimize radial displacement.

The simulation considers parameters such as gravity, tilt angle,
carousel radius, gondola mass, rotation speed, and gondola
dimensions. It uses numerical methods to solve the system's
differential equations and provides visualizations of the results.

Examples
--------
>>> from carousel_simulation import CarouselSimulation, CarouselParameters
>>> simulator = CarouselSimulation(CarouselParameters())
>>> search_config = {
...     "start_n_per_m": 0,
...     "step_size_n_per_m": 100_000,
...     "num_results": 1,
...     "show_results": True,
... }
>>> results = simulator.find_optimal_spring_stiffness(search_config)
>>> # The simulation will run, display results, and the 'results' list
>>> # will contain details of the optimal spring stiffness found.
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, NamedTuple

import colorama
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from termcolor import colored


colorama.init()
logging.basicConfig(
    level=logging.INFO,
    format=(
        "[%(asctime)s.%(msecs)03d][%(levelname)s]"
        "[%(filename)s:%(lineno)d - %(funcName)s()]: %(message)s"
    ),
    datefmt="%d-%m-%Y %H:%M:%S",
)


@dataclass
class CarouselParameters:
    """
    Represents the physical parameters of the carousel.

    Attributes
    ----------
    gravity: float
        The gravitational acceleration (default is 9.81 m/s^2).
    tilt_angle_in_deg: float
        The tilt angle of the carousel in degrees (default is 30
        degrees).
    carousel_radius_in_m: float
        The radius of the carousel in meters (default is 6 meters).
    gondola_mass_in_kg: float
        The mass of a gondola in kilograms (default is 300 kg).
    carousel_rot_speed_in_rps: float
        The rotational speed of the carousel in revolutions per second
        (default is 0.2333 rps).
    gondola_width_in_m: float
        The width of a gondola in meters (default is 1.5 meters).
    gondola_height_in_m: float
        The height of a gondola in meters (default is 1.5 meters).
    """

    gravity: float = 9.81
    tilt_angle_in_deg: float = 30
    carousel_radius_in_m: float = 6
    gondola_mass_in_kg: float = 300
    carousel_rot_speed_in_rps: float = 0.2333
    gondola_width_in_m: float = 1.5
    gondola_height_in_m: float = 1.5


class DisplacementAnalysisResult(NamedTuple):
    """
    Represents the result of the displacement analysis.

    Attributes
    ----------
    global_minima: float | None
        The global minimum displacement in meters.
    global_maxima: float | None
        The global maximum displacement in meters.
    displacement_in_m: float | None
        The difference between the global maximum and minimum
        displacements in meters.
    criteria_was_met: bool
        Indicates whether the displacement criteria were met.
    """

    global_minima: float | None
    global_maxima: float | None
    displacement_in_m: float | None
    criteria_was_met: bool


class SimulationResult(NamedTuple):
    """
    Represents the result of a single simulation run.

    Attributes
    ----------
    spring_stiffness_in_n_per_m: float
        The spring stiffness used in the simulation in N/m.
    global_maxima_in_m: float | None
        The global maximum displacement observed in the simulation in
        meters.
    global_minima_in_m: float | None
        The global minimum displacement observed in the simulation in
        meters.
    displacement_in_m: float | None
        The difference between the global maximum and minimum
        displacements in meters.
    simulation_data: tuple
        The raw data from the simulation (time, position, velocity,
        angle, angular velocity).
    """

    spring_stiffness_in_n_per_m: float
    global_maxima_in_m: float | None
    global_minima_in_m: float | None
    displacement_in_m: float | None
    simulation_data: tuple


class CarouselSimulation:
    """
    A class to simulate the dynamics of a carousel.

    This class models the motion of a carousel, considering factors
    like gravity, tilt angle, carousel radius, gondola mass, rotation
    speed, and gondola dimensions. It allows for simulating the
    system's behavior over time and analyzing the displacement of the
    gondolas to find an optimal spring stiffness that minimizes radial
    displacement.

    Attributes
    ----------
    params: CarouselParameters
        The physical parameters of the carousel.
    simulation_time_in_s: float
        The total time for which the simulation will run, in seconds.
    max_radial_displacement_in_m: float
        The maximum acceptable radial displacement of the gondolas, in
        meters.
    tilt_angle_in_rad: float
        The tilt angle of the carousel in radians, calculated from the
        tilt angle in degrees.
    spring_stiffness_in_n_per_m: float
        The stiffness of the spring connecting the gondola to the
        carousel, in N/m.
    initial_conditions: list[float]
        The initial conditions for the simulation, consisting of the
        initial position, velocity, angle, and angular velocity of the
        gondola.

    Methods
    -------
    run_simulation() -> tuple[np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray]:
        Runs the simulation and returns the time, position, velocity,
        angle, and angular velocity data.
    plot_results(simulation_data: tuple) -> None:
        Plots the simulation results, showing the position, velocity,
        angle, and angular velocity over time.
    find_optimal_spring_stiffness(search_config: dict[str, Any]) ->
    list[SimulationResult]:
        Searches for the optimal spring stiffness that minimizes radial
        displacement, within given constraints.
    """

    def __init__(
        self,
        params: CarouselParameters | None = None,
        simulation_time_in_s: float = 10,
        max_radial_displacement_in_m: float = 0.005,
    ) -> None:
        """
        Constructs all the necessary attributes for the
        CarouselSimulation object.

        Parameters
        ----------
        params: CarouselParameters | None
            The physical parameters of the carousel. If None, default
            parameters are used.
        simulation_time_in_s: float
            The total time for which the simulation will run, in
            seconds.
        max_radial_displacement_in_m: float
            The maximum acceptable radial displacement of the gondolas,
            in meters.
        """
        if params is None:
            params = CarouselParameters()

        self.params = params
        self.simulation_time_in_s = simulation_time_in_s
        self.max_radial_displacement_in_m = max_radial_displacement_in_m
        self.tilt_angle_in_rad = math.radians(params.tilt_angle_in_deg)
        self.spring_stiffness_in_n_per_m: float = 0

        self.initial_conditions: list[float] = [
            self.params.carousel_radius_in_m,
            0,
            0,
            2 * np.pi * self.params.carousel_rot_speed_in_rps,
        ]

    def _calculate_dvelocity_dt(
        self,
        position_in_m: float,
        angular_velocity_in_rads: float,
        angle_in_rad: float,
    ) -> float:
        """
        Calculate the rate of change of velocity.

        Parameters
        ----------
        position_in_m: float
            The current position of the gondola in meters.
        angular_velocity_in_rads: float
            The current angular velocity of the gondola in radians per
            second.
        angle_in_rad: float
            The current angle of the gondola in radians.

        Returns
        -------
        float
            The rate of change of velocity.
        """
        return (
            position_in_m * (angular_velocity_in_rads**2)
            + self.params.gravity
            * np.sin(self.tilt_angle_in_rad)
            * np.cos(angle_in_rad)
            + (
                (
                    self.spring_stiffness_in_n_per_m
                    / self.params.gondola_mass_in_kg
                )
                * (self.params.carousel_radius_in_m - position_in_m)
            )
        )

    def _calculate_dangular_velocity_dt(
        self,
        position_in_m: float,
        velocity_in_ms: float,
        angle_in_rad: float,
        angular_velocity_in_rads: float,
    ) -> float:
        """
        Calculate the rate of change of angular velocity.

        Parameters
        ----------
        position_in_m: float
            The current position of the gondola in meters.
        velocity_in_ms: float
            The current velocity of the gondola in meters per second.
        angle_in_rad: float
            The current angle of the gondola in radians.
        angular_velocity_in_rads: float
            The current angular velocity of the gondola in radians per
            second.

        Returns
        -------
        float
            The rate of change of angular velocity.
        """
        return -(
            2
            * angular_velocity_in_rads
            * velocity_in_ms
            * position_in_m
            + self.params.gravity
            * np.sin(self.tilt_angle_in_rad)
            * np.sin(angle_in_rad)
            * position_in_m
        ) / (
            position_in_m**2
            + (5 / 3)
            * (
                self.params.gondola_width_in_m**2
                + self.params.gondola_height_in_m**2
            )
            + 20 * self.params.carousel_radius_in_m**2
        )

    def system_dynamics(self, _: float, state: list[float]) -> list[float]:
        """
        Define the system dynamics as a set of differential equations.

        Parameters
        ----------
        _: float
            The current time (not used in the calculations but required
            by solve_ivp).
        state: list[float]
            The current state of the system, containing position,
            velocity, angle, and angular velocity.

        Returns
        -------
        list[float]
            The rates of change of position, velocity, angle, and
            angular velocity.
        """
        (
            position_in_m,
            velocity_in_ms,
            angle_in_rad,
            angular_velocity_in_rads,
        ) = state

        dposition_dt: float = velocity_in_ms
        dvelocity_dt: float = self._calculate_dvelocity_dt(
            position_in_m, angular_velocity_in_rads, angle_in_rad
        )

        dangle_dt: float = angular_velocity_in_rads
        dangular_velocity_dt: float = self._calculate_dangular_velocity_dt(
            position_in_m, velocity_in_ms, angle_in_rad, angular_velocity_in_rads
        )

        return [dposition_dt, dvelocity_dt, dangle_dt, dangular_velocity_dt]

    def run_simulation(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the simulation using the defined system dynamics.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            A tuple containing the time, position, velocity, angle, and
            angular velocity data from the simulation.
        """
        time_span_in_s: list[float, float] = [0, self.simulation_time_in_s]
        solution: Any = solve_ivp(
            self.system_dynamics,
            time_span_in_s,
            self.initial_conditions,
            method="RK45",
            rtol=1e-3,
            atol=1e-6,
        )

        time_array_in_s: np.ndarray = solution.t
        (
            position_in_m,
            velocity_in_ms,
            angle_in_rad,
            angular_velocity_in_rads,
        ) = solution.y
        return (
            time_array_in_s,
            position_in_m,
            velocity_in_ms,
            angle_in_rad,
            angular_velocity_in_rads,
        )

    def _plot_single_result(
        self,
        time_array_in_s: np.ndarray,
        data: np.ndarray,
        ylabel: str,
        label: str,
        axis: plt.Axes,
    ) -> None:
        """
        Plot a single data series on a given axis.

        Parameters
        ----------
        time_array_in_s: np.ndarray
            The time data for the x-axis.
        data: np.ndarray
            The data series to plot.
        ylabel: str
            The label for the y-axis.
        label: str
            The label for the data series.
        axis: plt.Axes
            The matplotlib axis on which to plot the data.
        """
        axis.plot(time_array_in_s, data, label=label)
        axis.set_ylabel(ylabel)
        axis.legend(loc="upper right")
        axis.grid()
        axis.set_xlim(min(time_array_in_s), max(time_array_in_s))

    def _add_vertical_lines(
        self, angle_in_deg: np.ndarray, axes: list[plt.Axes]
    ) -> None:
        """
        Add vertical lines to the plot at each full rotation (360
        degrees).

        Parameters
        ----------
        angle_in_deg: np.ndarray
            The angle data in degrees.
        axes: list[plt.Axes]
            The list of matplotlib axes to which the vertical lines will
            be added.
        """
        for i in range(1, int(max(angle_in_deg) / 360) + 1):
            idx: np.ndarray = np.where(np.isclose(
                angle_in_deg, 360 * i, atol=1))[0]
            if len(idx) > 0:
                axes[2].plot(time_array_in_s[idx[0]],
                             angle_in_deg[idx[0]], "ro")
                for axis in axes:
                    axis.axvline(
                        x=time_array_in_s[idx[0]],
                        color="r",
                        linestyle="--",
                        linewidth=1,
                    )

    def plot_results(self, simulation_data: tuple) -> None:
        """
        Plot the simulation results.

        Parameters
        ----------
        simulation_data: tuple
            The simulation data containing time, position, velocity,
            angle, and angular velocity.
        """
        (
            time_array_in_s,
            position_in_m,
            velocity_in_ms,
            angle_in_rad,
            angular_velocity_in_rads,
        ) = simulation_data
        fig, axes = plt.subplots(4, 1, figsize=(10, 12))

        self._plot_single_result(
            time_array_in_s, position_in_m, r"$x$ [m]", r"$x(t)$", axes[0]
        )
        self._plot_single_result(
            time_array_in_s,
            velocity_in_ms,
            r"$\dot{x}$ [m/s]",
            r"$\dot{x}(t)$",
            axes[1],
        )

        angle_in_deg: np.ndarray = np.degrees(angle_in_rad)
        self._plot_single_result(
            time_array_in_s,
            angle_in_deg,
            r"$\alpha$ [deg]",
            r"$\alpha(t)$",
            axes[2],
        )

        angular_velocity_in_degs: np.ndarray = np.degrees(
            angular_velocity_in_rads)
        self._plot_single_result(
            time_array_in_s,
            angular_velocity_in_degs,
            r"$\dot{\alpha}$ [deg/s]",
            r"$\dot{\alpha}(t)$",
            axes[3],
        )

        self._add_vertical_lines(angle_in_deg, axes)

        y_ticks: np.ndarray = np.arange(0, max(angle_in_deg) + 360, 360)
        axes[2].set_yticks(y_ticks)
        axes[2].set_yticklabels([f"{int(y)}" for y in y_ticks])

        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        fig.text(0.5, 0.01, "Time (t)", ha="center", fontsize=12)
        fig.suptitle(
            f"Spring Stiffness={self.spring_stiffness_in_n_per_m} N/m",
            fontsize=16,
        )
        plt.show()

    @staticmethod
    def get_local_extremes(
        data_series: np.ndarray
    ) -> tuple[list[float], list[float]]:
        """
        Find the local minima and maxima in a data series.

        Parameters
        ----------
        data_series: np.ndarray
            The data series to analyze.

        Returns
        -------
        tuple[list[float], list[float]]
            A tuple containing the local minima and local maxima.
        """
        local_maxima: list[float] = []
        local_minima: list[float] = []

        for i, element in enumerate(data_series):
            if i == 0 or i == len(data_series) - 1:
                continue
            is_maxima: bool = (
                data_series[i] > data_series[i - 1]
                and data_series[i] > data_series[i + 1]
            )
            is_minima: bool = (
                data_series[i] < data_series[i - 1]
                and data_series[i] < data_series[i + 1]
            )
            if is_maxima:
                local_maxima.append(element)
            if is_minima:
                local_minima.append(element)

        return local_minima, local_maxima

    @staticmethod
    def get_global_extremes(
        local_minima: list[float], local_maxima: list[float]
    ) -> tuple[float | None, float | None, float | None]:
        """
        Calculate the global minima, maxima, and their difference from
        local extremes.

        Parameters
        ----------
        local_minima: list[float]
            The local minima in the data series.
        local_maxima: list[float]
            The local maxima in the data series.

        Returns
        -------
        tuple[float | None, float | None, float | None]
            A tuple containing the global minima, global maxima, and
            their difference. Returns (None, None, None) if the input
            lists are empty.
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
        Analyze the displacement of the gondola from the position data.

        Parameters
        ----------
        position_data: np.ndarray
            The position data from the simulation.

        Returns
        -------
        DisplacementAnalysisResult
            The result of the displacement analysis, including global
            minima, maxima, their difference, and whether the
            displacement criteria were met.
        """
        local_minima, local_maxima = self.get_local_extremes(position_data)
        extremes: tuple[float | None, float | None, float | None] = (
            self.get_global_extremes(local_minima, local_maxima)
        )
        global_minima, global_maxima, displacement_in_m = extremes

        criteria_was_met: bool = False
        if (
            displacement_in_m is not None
            and displacement_in_m <= self.max_radial_displacement_in_m
        ):
            criteria_was_met = True

        return DisplacementAnalysisResult(
            global_minima=global_minima,
            global_maxima=global_maxima,
            displacement_in_m=displacement_in_m,
            criteria_was_met=criteria_was_met,
        )

    def print_simulation_result(
        self, analysis_result: DisplacementAnalysisResult
    ) -> None:
        """
        Print the simulation result, highlighting whether the
        displacement criteria were met.

        Parameters
        ----------
        analysis_result: DisplacementAnalysisResult
            The result of the displacement analysis.
        """
        global_minima: float | None = analysis_result.global_minima
        global_maxima: float | None = analysis_result.global_maxima
        displacement_in_m: float | None = analysis_result.displacement_in_m
        criteria_was_met: bool = analysis_result.criteria_was_met

        if displacement_in_m is not None:
            message: str = (
                f"[Spring Stiffness: {self.spring_stiffness_in_n_per_m:<8} N/m] "
                f"Global maxima: {global_maxima:<8} m | "
                f"Global minima: {global_minima:<8} m | "
                f"Displacement: {displacement_in_m:<8} m"
            )
            if criteria_was_met:
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
        Process the result of a single simulation run for a given spring
        stiffness.

        Parameters
        ----------
        simulation_data: tuple
            The raw data from the simulation.
        analysis_result: DisplacementAnalysisResult
            The result of the displacement analysis.
        show_results: bool
            Whether to plot the simulation results.

        Returns
        -------
        SimulationResult | None
            The simulation result if the displacement criteria were met,
            otherwise None.
        """
        if not analysis_result.criteria_was_met:
            return None

        result: SimulationResult = SimulationResult(
            spring_stiffness_in_n_per_m=self.spring_stiffness_in_n_per_m,
            global_maxima_in_m=analysis_result.global_maxima,
            global_minima_in_m=analysis_result.global_minima,
            displacement_in_m=analysis_result.displacement_in_m,
            simulation_data=simulation_data,
        )

        if show_results:
            self.plot_results(simulation_data)

        return result

    def _run_single_simulation(
        self, stiffness_value_in_n_per_m: float, show_results: bool
    ) -> SimulationResult | None:
        """
        Run a single simulation with a given spring stiffness.

        Parameters
        ----------
        stiffness_value_in_n_per_m: float
            The spring stiffness to use for the simulation, in N/m.
        show_results: bool
            Whether to plot the simulation results.

        Returns
        -------
        SimulationResult | None
            The simulation result if the displacement criteria were met,
            otherwise None.

        Raises
        ------
        ValueError
            If the numerical integration fails due to invalid input
            parameters.
        RuntimeError
            If the numerical integration fails to converge.
        """
        self.spring_stiffness_in_n_per_m = stiffness_value_in_n_per_m

        try:
            simulation_data: tuple = self.run_simulation()
            _, position_data, _, _, _ = simulation_data

            analysis_result: DisplacementAnalysisResult = (
                self.analyze_displacement(position_data)
            )

            self.print_simulation_result(analysis_result)

            return self.process_stiffness_result(
                simulation_data,
                analysis_result,
                show_results,
            )

        except ValueError as error:
            error_message: str = (
                f"[Spring Stiffness: {self.spring_stiffness_in_n_per_m:<8} N/m] "
                f"Value Error: {error}"
            )
            logging.error(colored(error_message, "red"))
            return None

        except RuntimeError as error:
            error_message = (
                f"[Spring Stiffness: {self.spring_stiffness_in_n_per_m:<8} N/m] "
                f"Runtime Error: {error}"
            )
            logging.error(colored(error_message, "red"))
            return None

    def find_optimal_spring_stiffness(
        self, search_config: dict[str, Any]
    ) -> list[SimulationResult]:
        """
        Search for the optimal spring stiffness that minimizes radial
        displacement.

        Parameters
        ----------
        search_config: dict[str, Any]
            Configuration for the search, including start value, step
            size, number of results, and whether to show results.

        Returns
        -------
        list[SimulationResult]
            A list of simulation results that meet the displacement
            criteria, sorted by spring stiffness.
        """
        start_in_n_per_m: float = search_config.get("start_n_per_m", 0)
        step_size_in_n_per_m: float = search_config.get(
            "step_size_n_per_m", 100_000)
        number_of_results: int = search_config.get("num_results", 1)
        show_results: bool = search_config.get("show_results", True)

        number_of_consecutive_valid_results: int = 0
        results: list[SimulationResult] = []
        max_stiffness: float = 999_999_999_999

        for stiffness in np.arange(
            start_in_n_per_m, max_stiffness, step_size_in_n_per_m
        ):
            result: SimulationResult | None = self._run_single_simulation(
                stiffness, show_results
            )

            if result:
                results.append(result)
                number_of_consecutive_valid_results += 1
                if number_of_consecutive_valid_results >= number_of_results:
                    break
            else:
                number_of_consecutive_valid_results = 0

        return results


def run_simulation(simulator: CarouselSimulation, search_config: dict) -> list:
    """
    Run the carousel simulation and find the optimal spring stiffness.

    Parameters
    ----------
    simulator: CarouselSimulation
        The carousel simulation object.
    search_config: dict
        Configuration for the search, including start value, step size,
        number of results, and whether to show results.

    Returns
    -------
    list
        A list of simulation results that meet the displacement
        criteria, sorted by spring stiffness.
    """
    return simulator.find_optimal_spring_stiffness(search_config)


def main() -> None:
    """
    Execute the main functionality of the script:
    - Create a CarouselSimulation object.
    - Run the simulation with a specified search configuration.
    - Print and plot the results.
    """
    simulator: CarouselSimulation = CarouselSimulation()

    results: list = run_simulation(
        simulator,
        {
            "start_n_per_m": 0,
            "step_size_n_per_m": 100_000,
            "num_results": 1,
            "show_results": True,
        },
    )
    return results


if __name__ == "__main__":
    main()
