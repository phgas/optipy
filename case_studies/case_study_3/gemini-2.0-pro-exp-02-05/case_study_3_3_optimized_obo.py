"""
Carousel Simulation Module
===========================

This module provides a simulation of a carousel, focusing on the
dynamics of gondola displacement.

It includes classes and functions to model the carousel's behavior
under various conditions, calculate the optimal spring stiffness for
the gondolas, and visualize the simulation results.

Examples
--------
>>> from carousel_simulation import CarouselSimulation, \
CarouselParameters, run_optimal_stiffness_search
>>> simulator = CarouselSimulation()
>>> results = run_optimal_stiffness_search(simulator)
Start simulation...
...

>>> params = CarouselParameters(gravity=9.81, tilt_angle_in_deg=25)
>>> simulator_with_params = CarouselSimulation(params=params)
>>> results_with_params = run_optimal_stiffness_search(
...     simulator_with_params
... )
Start simulation...
...
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp  # type: ignore
from termcolor import colored  # type: ignore

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
        The gravitational acceleration in m/s^2. Default is 9.81 m/s^2.
    tilt_angle_in_deg: float
        The tilt angle of the carousel in degrees. Default is 30 degrees.
    carousel_radius_in_m: float
        The radius of the carousel in meters. Default is 6 meters.
    gondola_mass_in_kg: float
        The mass of a gondola in kilograms. Default is 300 kg.
    carousel_rot_speed_in_rps: float
        The rotational speed of the carousel in revolutions per second.
        Default is 0.2333 rps.
    gondola_width_in_m: float
        The width of a gondola in meters. Default is 1.5 meters.
    gondola_height_in_m: float
        The height of a gondola in meters. Default is 1.5 meters.
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
        The global minimum displacement in meters, if available.
    global_maxima: float | None
        The global maximum displacement in meters, if available.
    displacement_in_m: float | None
        The total displacement (difference between global maxima and
        minima) in meters, if available.
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
        meters, if available.
    global_minima_in_m: float | None
        The global minimum displacement observed in the simulation in
        meters, if available.
    displacement_in_m: float | None
        The total displacement (difference between global maxima and
        minima) observed in the simulation in meters, if available.
    simulation_data: Tuple[np.ndarray, np.ndarray, np.ndarray, \
np.ndarray, np.ndarray]
        The raw data from the simulation, including time, position,
        velocity, angle, and angular velocity.
    """

    spring_stiffness_in_n_per_m: float
    global_maxima_in_m: float | None
    global_minima_in_m: float | None
    displacement_in_m: float | None
    simulation_data: Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]


class CarouselSimulation:
    """
    A class to simulate the dynamics of a carousel.

    This class models the motion of a carousel, considering factors
    like gravity, tilt angle, carousel radius, gondola mass, rotational
    speed, and gondola dimensions. It allows for simulating the system
    dynamics, analyzing displacement, and finding the optimal spring
    stiffness to meet specific displacement criteria.

    Attributes
    ----------
    params: CarouselParameters
        The physical parameters of the carousel.
    simulation_time_in_s: float
        The total time for which the simulation will run, in seconds.
    max_radial_displacement_in_m: float
        The maximum allowed radial displacement for the gondola, in
        meters.
    tilt_angle_in_rad: float
        The tilt angle of the carousel in radians, calculated from
        `params.tilt_angle_in_deg`.
    spring_stiffness_in_n_per_m: float
        The stiffness of the spring attached to the gondola, in N/m.
    initial_conditions: List[float]
        The initial conditions for the simulation, including initial
        position, velocity, angle, and angular velocity.

    Methods
    -------
    run_simulation() -> Tuple[np.ndarray, np.ndarray, np.ndarray, \
np.ndarray, np.ndarray]:
        Runs the simulation and returns the time, position, velocity,
        angle, and angular velocity data.
    plot_results(simulation_data: Tuple[np.ndarray, np.ndarray, \
np.ndarray, np.ndarray, np.ndarray]) -> None:
        Plots the simulation results, showing the position, velocity,
        angle, and angular velocity over time.
    analyze_displacement(position_data: np.ndarray) -> \
DisplacementAnalysisResult:
        Analyzes the displacement data from the simulation to find
        global extremes and determine if the displacement criteria are
        met.
    find_optimal_spring_stiffness(search_config: Dict[str, Any]) -> \
List[SimulationResult]:
        Searches for the optimal spring stiffness that meets the
        specified displacement criteria.
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
        params: CarouselParameters | None, optional
            The physical parameters of the carousel. If None, default
            parameters are used.
        simulation_time_in_s: float, optional
            The total time for which the simulation will run, in
            seconds. Default is 10 seconds.
        max_radial_displacement_in_m: float, optional
            The maximum allowed radial displacement for the gondola, in
            meters. Default is 0.005 meters.
        """
        if params is None:
            params = CarouselParameters()

        self.params = params
        self.simulation_time_in_s = simulation_time_in_s
        self.max_radial_displacement_in_m = max_radial_displacement_in_m
        self.tilt_angle_in_rad = math.radians(params.tilt_angle_in_deg)
        self.spring_stiffness_in_n_per_m: float = 0

        self.initial_conditions: List[float] = [
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
            The current angle of the carousel in radians.

        Returns
        -------
        float
            The rate of change of velocity in m/s^2.
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
            The current angle of the carousel in radians.
        angular_velocity_in_rads: float
            The current angular velocity of the gondola in radians per
            second.

        Returns
        -------
        float
            The rate of change of angular velocity in rad/s^2.
        """
        numerator: float = -(
            2
            * angular_velocity_in_rads
            * velocity_in_ms
            * position_in_m
            + self.params.gravity
            * np.sin(self.tilt_angle_in_rad)
            * np.sin(angle_in_rad)
            * position_in_m
        )
        denominator: float = (
            position_in_m**2
            + (5 / 3)
            * (
                self.params.gondola_width_in_m**2
                + self.params.gondola_height_in_m**2
            )
            + 20 * self.params.carousel_radius_in_m**2
        )
        return numerator / denominator

    def system_dynamics(self, _: float, state: List[float]) -> List[float]:
        """
        Define the system dynamics for the carousel.

        Parameters
        ----------
        _: float
            The current time (not used in the calculation but required
            by solve_ivp).
        state: List[float]
            The current state of the system, including position,
            velocity, angle, and angular velocity.

        Returns
        -------
        List[float]
            The rate of change of the state, including dposition_dt,
            dvelocity_dt, dangle_dt, and dangular_velocity_dt.
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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the simulation using the defined system dynamics.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            A tuple containing the time, position, velocity, angle, and
            angular velocity data from the simulation.
        """
        time_span_in_s: List[float] = [0, self.simulation_time_in_s]
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
        Plot a single result on the given axis.

        Parameters
        ----------
        time_array_in_s: np.ndarray
            The time data for the x-axis.
        data: np.ndarray
            The data to plot on the y-axis.
        ylabel: str
            The label for the y-axis.
        label: str
            The label for the plotted data.
        axis: plt.Axes
            The matplotlib axis to plot on.
        """
        axis.plot(time_array_in_s, data, label=label)
        axis.set_ylabel(ylabel)
        axis.legend(loc="upper right")
        axis.grid()
        axis.set_xlim(min(time_array_in_s), max(time_array_in_s))

    def _plot_angle_results(
        self, time_array_in_s: np.ndarray, angle_in_deg: np.ndarray, axis: plt.Axes
    ) -> None:
        """
        Plot the angle results, including markers for full rotations.

        Parameters
        ----------
        time_array_in_s: np.ndarray
            The time data for the x-axis.
        angle_in_deg: np.ndarray
            The angle data in degrees to plot on the y-axis.
        axis: plt.Axes
            The matplotlib axis to plot on.
        """
        self._plot_single_result(
            time_array_in_s, angle_in_deg, r"$\alpha$ [deg]", r"$\alpha(t)$", axis
        )

        for i in range(1, int(max(angle_in_deg) / 360) + 1):
            idx: np.ndarray = np.where(np.isclose(
                angle_in_deg, 360 * i, atol=1))[0]
            if len(idx) > 0:
                axis.plot(time_array_in_s[idx[0]], angle_in_deg[idx[0]], "ro")
                axis.axvline(
                    x=time_array_in_s[idx[0]],
                    color="r",
                    linestyle="--",
                    linewidth=1,
                )

        y_ticks: np.ndarray = np.arange(0, max(angle_in_deg) + 360, 360)
        axis.set_yticks(y_ticks)
        axis.set_yticklabels([f"{int(y)}" for y in y_ticks])

    def plot_results(
        self,
        simulation_data: Tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
        ],
    ) -> None:
        """
        Plot the simulation results.

        Parameters
        ----------
        simulation_data: Tuple[np.ndarray, np.ndarray, np.ndarray, \
np.ndarray, np.ndarray]
            The simulation data to plot, including time, position,
            velocity, angle, and angular velocity.
        """
        (
            time_array_in_s,
            position_in_m,
            velocity_in_ms,
            angle_in_rad,
            angular_velocity_in_rads,
        ) = simulation_data
        figure, axes = plt.subplots(4, 1, figsize=(10, 12))

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
        self._plot_angle_results(time_array_in_s, angle_in_deg, axes[2])

        angular_velocity_in_degs: np.ndarray = np.degrees(
            angular_velocity_in_rads)
        self._plot_single_result(
            time_array_in_s,
            angular_velocity_in_degs,
            r"$\dot{\alpha}$ [deg/s]",
            r"$\dot{\alpha}(t)$",
            axes[3],
        )

        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        figure.text(0.5, 0.01, "Time (t)", ha="center", fontsize=12)
        figure.suptitle(
            f"Spring Stiffness={self.spring_stiffness_in_n_per_m} N/m",
            fontsize=16,
        )
        plt.show()

    @staticmethod
    def get_local_extremes(data_series: np.ndarray) -> Tuple[List[float], List[float]]:
        """
        Find the local extremes (minima and maxima) in a data series.

        Parameters
        ----------
        data_series: np.ndarray
            The data series to analyze.

        Returns
        -------
        Tuple[List[float], List[float]]
            A tuple containing the local minima and local maxima.
        """
        local_maxima: List[float] = []
        local_minima: List[float] = []

        for i, element in enumerate(data_series[1:-1], 1):
            is_maxima: bool = (
                element > data_series[i - 1] and element > data_series[i + 1]
            )
            is_minima: bool = (
                element < data_series[i - 1] and element < data_series[i + 1]
            )
            if is_maxima:
                local_maxima.append(element)
            if is_minima:
                local_minima.append(element)

        return local_minima, local_maxima

    @staticmethod
    def get_global_extremes(
        local_minima: List[float], local_maxima: List[float]
    ) -> Tuple[float | None, float | None, float | None]:
        """
        Calculate the global extremes (minima, maxima, and difference)
        from the local extremes.

        Parameters
        ----------
        local_minima: List[float]
            The local minima.
        local_maxima: List[float]
            The local maxima.

        Returns
        -------
        Tuple[float | None, float | None, float | None]
            A tuple containing the global minima, global maxima, and
            their difference.
            Returns (None, None, None) if the input lists are empty.
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
        Analyze the displacement data to find global extremes and
        determine if the displacement criteria are met.

        Parameters
        ----------
        position_data: np.ndarray
            The position data from the simulation.

        Returns
        -------
        DisplacementAnalysisResult
            The result of the displacement analysis.
        """
        local_minima, local_maxima = self.get_local_extremes(position_data)
        extremes: Tuple[
            float | None, float | None, float | None
        ] = self.get_global_extremes(local_minima, local_maxima)
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
                print(colored(message, "green"))
            else:
                print(colored(message, "red"))

    def process_stiffness_result(
        self,
        simulation_data: Tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
        ],
        analysis_result: DisplacementAnalysisResult,
        show_results: bool,
    ) -> SimulationResult | None:
        """
        Process the result of a single simulation run for a given spring
        stiffness.

        Parameters
        ----------
        simulation_data: Tuple[np.ndarray, np.ndarray, np.ndarray, \
np.ndarray, np.ndarray]
            The raw data from the simulation.
        analysis_result: DisplacementAnalysisResult
            The result of analyzing the displacement from the simulation
            data.
        show_results: bool
            Whether to plot the results of the simulation.

        Returns
        -------
        SimulationResult | None
            The result of the simulation, or None if the displacement
            criteria were not met.
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
        Run a single simulation with the given spring stiffness.

        Parameters
        ----------
        stiffness_value_in_n_per_m: float
            The spring stiffness to use for the simulation, in N/m.
        show_results: bool
            Whether to plot the results of the simulation.

        Returns
        -------
        SimulationResult | None
            The result of the simulation, or None if an error occurred or
            the displacement criteria were not met.
        """
        self.spring_stiffness_in_n_per_m = stiffness_value_in_n_per_m

        try:
            simulation_data: Tuple[
                np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
            ] = self.run_simulation()
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
                f"Value Error: {str(error)}"
            )
            print(colored(error_message, "red"))
            return None

        except RuntimeError as error:
            error_message: str = (
                f"[Spring Stiffness: {self.spring_stiffness_in_n_per_m:<8} N/m] "
                f"Runtime Error: {str(error)}"
            )
            print(colored(error_message, "red"))
            return None

    def find_optimal_spring_stiffness(
        self, search_config: Dict[str, Any]
    ) -> List[SimulationResult]:
        """
        Find the optimal spring stiffness that meets the specified
        displacement criteria.

        Parameters
        ----------
        search_config: Dict[str, Any]
            A dictionary containing the search configuration, including:
            - start_n_per_m: The starting value for the spring stiffness
              search, in N/m.
            - step_size_n_per_m: The step size for incrementing the
              spring stiffness, in N/m.
            - num_results: The number of consecutive valid results to
              find.
            - show_results: Whether to plot the results of each
              simulation.

        Returns
        -------
        List[SimulationResult]
            A list of SimulationResult objects representing the valid
            simulation results.
        """
        start_in_n_per_m: float = search_config.get("start_n_per_m", 0)
        step_size_in_n_per_m: float = search_config.get(
            "step_size_n_per_m", 100_000)
        number_of_results: int = search_config.get("num_results", 1)
        show_results: bool = search_config.get("show_results", True)

        number_of_consecutive_valid_results: int = 0
        results: List[SimulationResult] = []
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


def run_optimal_stiffness_search(
    simulator: CarouselSimulation,
) -> List[SimulationResult]:
    """
    Runs the search for the optimal spring stiffness.

    Parameters
    ----------
    simulator: CarouselSimulation
        The CarouselSimulation object to use for the search.

    Returns
    -------
    List[SimulationResult]
        A list of SimulationResult objects representing the valid
        simulation results.
    """
    return simulator.find_optimal_spring_stiffness(
        {
            "start_n_per_m": 0,
            "step_size_n_per_m": 100_000,
            "num_results": 1,
            "show_results": True,
        }
    )


def main() -> None:
    """Executes the main functionality of the script."""
    simulator: CarouselSimulation = CarouselSimulation()
    run_optimal_stiffness_search(simulator)


if __name__ == "__main__":
    main()
