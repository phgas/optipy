"""
Carousel Simulation Module
===========================

This module provides a simulation of a carousel, focusing on the dynamics of
the gondola's radial displacement.

The simulation calculates the optimal spring stiffness required to keep the
gondola's radial displacement within specified limits. It uses numerical
integration to solve the system's differential equations and provides
visualizations of the results.

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
>>> for result in results:
...     print(f"Optimal Spring Stiffness: {result.spring_stiffness_in_n_per_m} N/m")
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import colorama
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp  # type: ignore
from termcolor import colored  # type: ignore


colorama.init()


@dataclass
class CarouselParameters:
    """
    Represents the physical parameters of the carousel.

    Attributes
    ----------
    gravity : float
        Acceleration due to gravity (default is 9.81 m/s^2).
    tilt_angle_in_deg : float
        Tilt angle of the carousel in degrees (default is 30 degrees).
    carousel_radius_in_m : float
        Radius of the carousel in meters (default is 6 meters).
    gondola_mass_in_kg : float
        Mass of a single gondola in kilograms (default is 300 kg).
    carousel_rot_speed_in_rps : float
        Rotation speed of the carousel in revolutions per second (default is
        0.2333 rps).
    gondola_width_in_m : float
        Width of a gondola in meters (default is 1.5 meters).
    gondola_height_in_m : float
        Height of a gondola in meters (default is 1.5 meters).
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
    global_minima : Optional[float]
        Global minimum displacement in meters.
    global_maxima : Optional[float]
        Global maximum displacement in meters.
    displacement_in_m : Optional[float]
        Total displacement (difference between global maxima and minima) in
        meters.
    criteria_met : bool
        Indicates whether the displacement criteria are met.
    """

    global_minima: Optional[float]
    global_maxima: Optional[float]
    displacement_in_m: Optional[float]
    criteria_met: bool


class SimulationResult(NamedTuple):
    """
    Represents the result of a single simulation run.

    Attributes
    ----------
    spring_stiffness_in_n_per_m : float
        Spring stiffness used in the simulation in N/m.
    global_maxima_in_m : Optional[float]
        Global maximum displacement observed in the simulation in meters.
    global_minima_in_m : Optional[float]
        Global minimum displacement observed in the simulation in meters.
    displacement_in_m : Optional[float]
        Total displacement (difference between global maxima and minima) in
        meters.
    simulation_data : Tuple
        Raw data from the simulation.
    """

    spring_stiffness_in_n_per_m: float
    global_maxima_in_m: Optional[float]
    global_minima_in_m: Optional[float]
    displacement_in_m: Optional[float]
    simulation_data: Tuple


class CarouselSimulation:
    """
    Simulates the motion of a carousel gondola.

    This class models the dynamics of a carousel gondola, considering factors
    like gravity, tilt angle, carousel radius, gondola mass, rotation speed,
    and spring stiffness. It performs simulations to determine the optimal
    spring stiffness that keeps the gondola's radial displacement within
    specified limits.

    Attributes
    ----------
    params : CarouselParameters
        Physical parameters of the carousel.
    simulation_time_in_s : float
        Total simulation time in seconds.
    max_radial_displacement_in_m : float
        Maximum allowed radial displacement of the gondola in meters.
    tilt_angle_in_rad : float
        Tilt angle of the carousel in radians.
    spring_stiffness_in_n_per_m : float
        Spring stiffness of the gondola suspension in N/m.
    initial_conditions : List[float]
        Initial conditions for the simulation, [position, velocity, angle,
        angular velocity].

    Methods
    -------
    run_simulation() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray]:
        Runs the simulation and returns the time series data.
    plot_results(simulation_data: Tuple) -> None:
        Plots the simulation results.
    analyze_displacement(position_data: np.ndarray) ->
    DisplacementAnalysisResult:
        Analyzes the displacement data to find global extremes and check
        criteria.
    find_optimal_spring_stiffness(search_config: Dict[str, Any]) ->
    List[SimulationResult]:
        Searches for the optimal spring stiffness based on the given
        configuration.
    """

    def __init__(
        self,
        params: Optional[CarouselParameters] = None,
        simulation_time_in_s: float = 10,
        max_radial_displacement_in_m: float = 0.005,
    ) -> None:
        """
        Initializes the CarouselSimulation object.

        Parameters
        ----------
        params : Optional[CarouselParameters]
            Physical parameters of the carousel. If None, default parameters
            are used.
        simulation_time_in_s : float
            Total simulation time in seconds (default is 10 seconds).
        max_radial_displacement_in_m : float
            Maximum allowed radial displacement of the gondola in meters
            (default is 0.005 meters).
        """
        self.params = params if params else CarouselParameters()
        self.simulation_time_in_s = simulation_time_in_s
        self.max_radial_displacement_in_m = max_radial_displacement_in_m
        self.tilt_angle_in_rad = math.radians(self.params.tilt_angle_in_deg)
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
        position_in_m : float
            Current radial position of the gondola in meters.
        angular_velocity_in_rads : float
            Current angular velocity of the carousel in radians per second.
        angle_in_rad : float
            Current angle of the carousel in radians.

        Returns
        -------
        float
            The rate of change of velocity (dvelocity_dt).
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
        position_in_m : float
            Current radial position of the gondola in meters.
        velocity_in_ms : float
            Current velocity of the gondola in meters per second.
        angle_in_rad : float
            Current angle of the carousel in radians.
        angular_velocity_in_rads : float
            Current angular velocity of the carousel in radians per second.

        Returns
        -------
        float
            The rate of change of angular velocity (dangular_velocity_dt).
        """
        numerator: float = -(
            2 * angular_velocity_in_rads * velocity_in_ms * position_in_m
            + self.params.gravity
            * np.sin(self.tilt_angle_in_rad)
            * np.sin(angle_in_rad)
            * position_in_m
        )
        denominator: float = (
            position_in_m**2
            + (5 / 3)
            * (self.params.gondola_width_in_m**2 + self.params.gondola_height_in_m**2)
            + 20 * self.params.carousel_radius_in_m**2
        )
        return numerator / denominator

    def system_dynamics(self, _: float, state: List[float]) -> List[float]:
        """
        Define the system dynamics as a set of differential equations.

        Parameters
        ----------
        _ : float
            Current time (not used in the calculations, but required by
            solve_ivp).
        state : List[float]
            Current state of the system [position, velocity, angle, angular
            velocity].

        Returns
        -------
        List[float]
            The derivatives of the state variables [dposition_dt, dvelocity_dt,
            dangle_dt, dangular_velocity_dt].
        """
        position_in_m, velocity_in_ms, angle_in_rad, angular_velocity_in_rads = (
            state
        )

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
            A tuple containing the time series data:
            - time_array_in_s: Time points of the simulation.
            - position_in_m: Radial position of the gondola at each time point.
            - velocity_in_ms: Velocity of the gondola at each time point.
            - angle_in_rad: Angle of the carousel at each time point.
            - angular_velocity_in_rads: Angular velocity of the carousel at
              each time point.
        """
        time_span_in_s: List[float] = [0, self.simulation_time_in_s]
        solution = solve_ivp(
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
        Plot a single data series on the given axis.

        Parameters
        ----------
        time_array_in_s : np.ndarray
            Time points of the simulation.
        data : np.ndarray
            Data series to plot.
        ylabel : str
            Label for the y-axis.
        label : str
            Label for the data series.
        axis : plt.Axes
            Matplotlib axis to plot on.
        """
        axis.plot(time_array_in_s, data, label=label)
        axis.set_ylabel(ylabel)
        axis.legend(loc="upper right")
        axis.grid()
        axis.set_xlim(min(time_array_in_s), max(time_array_in_s))

    def _plot_vertical_lines(
        self,
        time_array_in_s: np.ndarray,
        angle_in_deg: np.ndarray,
        axes: List[plt.Axes],
    ) -> None:
        """
        Plot vertical lines indicating full rotations of the carousel.

        Parameters
        ----------
        time_array_in_s : np.ndarray
            Time points of the simulation.
        angle_in_deg : np.ndarray
            Angle of the carousel in degrees.
        axes : List[plt.Axes]
            List of Matplotlib axes to plot on.
        """
        for i in range(1, int(max(angle_in_deg) / 360) + 1):
            idx = np.where(np.isclose(angle_in_deg, 360 * i, atol=1))[0]
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

    def plot_results(self, simulation_data: Tuple) -> None:
        """
        Plot the simulation results.

        Parameters
        ----------
        simulation_data : Tuple
            Raw data from the simulation.
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

        self._plot_vertical_lines(time_array_in_s, angle_in_deg, axes)

        y_ticks: np.ndarray = np.arange(0, max(angle_in_deg) + 360, 360)
        axes[2].set_yticks(y_ticks)
        axes[2].set_yticklabels([f"{int(y)}" for y in y_ticks])

        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        figure.text(0.5, 0.01, "Time (t)", ha="center", fontsize=12)
        figure.suptitle(
            f"Spring Stiffness={self.spring_stiffness_in_n_per_m} N/m", fontsize=16
        )
        plt.show()

    @staticmethod
    def _check_local_maxima(data_series: np.ndarray, i: int) -> bool:
        """
        Check if the given index is a local maximum.

        Parameters
        ----------
        data_series : np.ndarray
            Data series to check.
        i : int
            Index to check.

        Returns
        -------
        bool
            True if the index is a local maximum, False otherwise.
        """
        return (
            data_series[i] > data_series[i -
                                         1] and data_series[i] > data_series[i + 1]
        )

    @staticmethod
    def _check_local_minima(data_series: np.ndarray, i: int) -> bool:
        """
        Check if the given index is a local minimum.

        Parameters
        ----------
        data_series : np.ndarray
            Data series to check.
        i : int
            Index to check.

        Returns
        -------
        bool
            True if the index is a local minimum, False otherwise.
        """
        return (
            data_series[i] < data_series[i -
                                         1] and data_series[i] < data_series[i + 1]
        )

    @staticmethod
    def get_local_extremes(
        data_series: np.ndarray
    ) -> Tuple[List[float], List[float]]:
        """
        Find the local maxima and minima in a data series.

        Parameters
        ----------
        data_series : np.ndarray
            Data series to analyze.

        Returns
        -------
        Tuple[List[float], List[float]]
            A tuple containing two lists:
            - local_maxima: List of local maxima.
            - local_minima: List of local minima.
        """
        local_maxima: List[float] = []
        local_minima: List[float] = []

        for i in range(1, len(data_series) - 1):
            if CarouselSimulation._check_local_maxima(data_series, i):
                local_maxima.append(data_series[i])
            if CarouselSimulation._check_local_minima(data_series, i):
                local_minima.append(data_series[i])

        return local_minima, local_maxima

    @staticmethod
    def get_global_extremes(
        local_minima: List[float], local_maxima: List[float]
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Calculate the global extremes (minima and maxima) from the local
        extremes.

        Parameters
        ----------
        local_minima : List[float]
            List of local minima.
        local_maxima : List[float]
            List of local maxima.

        Returns
        -------
        Tuple[Optional[float], Optional[float], Optional[float]]
            A tuple containing:
            - global_minima: Global minimum value (rounded to 5 decimal
              places).
            - global_maxima: Global maximum value (rounded to 5 decimal
              places).
            - difference: Difference between global maxima and minima (rounded
              to 5 decimal places).
            Returns (None, None, None) if either local_minima or local_maxima
            is empty.
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
        Analyze the displacement data to find global extremes and check if the
        criteria are met.

        Parameters
        ----------
        position_data : np.ndarray
            Radial position data from the simulation.

        Returns
        -------
        DisplacementAnalysisResult
            Result of the displacement analysis.
        """
        local_minima, local_maxima = self.get_local_extremes(position_data)
        extremes: Tuple[
            Optional[float], Optional[float], Optional[float]
        ] = self.get_global_extremes(local_minima, local_maxima)
        global_minima, global_maxima, displacement_in_m = extremes

        criteria_met: bool = False
        if (
            displacement_in_m is not None
            and displacement_in_m <= self.max_radial_displacement_in_m
        ):
            criteria_met = True

        return DisplacementAnalysisResult(
            global_minima=global_minima,
            global_maxima=global_maxima,
            displacement_in_m=displacement_in_m,
            criteria_met=criteria_met,
        )

    def print_simulation_result(
        self, analysis_result: DisplacementAnalysisResult
    ) -> None:
        """
        Print the simulation results with colored logging.

        Parameters
        ----------
        analysis_result : DisplacementAnalysisResult
            Result of the displacement analysis.
        """
        global_minima: Optional[float] = analysis_result.global_minima
        global_maxima: Optional[float] = analysis_result.global_maxima
        displacement_in_m: Optional[float] = analysis_result.displacement_in_m
        criteria_met: bool = analysis_result.criteria_met

        if displacement_in_m is not None:
            message: str = (
                f"[Spring Stiffness: {self.spring_stiffness_in_n_per_m:<8} N/m] "
                f"Global maxima: {global_maxima:<8} m | "
                f"Global minima: {global_minima:<8} m | "
                f"Displacement: {displacement_in_m:<8} m"
            )
            log_color: str = "green" if criteria_met else "red"
            logging.info(colored(message, log_color))

    def process_stiffness_result(
        self,
        simulation_data: Tuple,
        analysis_result: DisplacementAnalysisResult,
        show_results: bool,
    ) -> Optional[SimulationResult]:
        """
        Process the results of a single simulation run for a given spring
        stiffness.

        Parameters
        ----------
        simulation_data : Tuple
            Raw data from the simulation.
        analysis_result : DisplacementAnalysisResult
            Result of the displacement analysis.
        show_results : bool
            Whether to display the simulation plots.

        Returns
        -------
        Optional[SimulationResult]
            SimulationResult if the displacement criteria are met, None
            otherwise.
        """
        if not analysis_result.criteria_met:
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
    ) -> Optional[SimulationResult]:
        """
        Run a single simulation with the given spring stiffness.

        Parameters
        ----------
        stiffness_value_in_n_per_m : float
            Spring stiffness value to use for the simulation.
        show_results : bool
            Whether to display the simulation plots.

        Returns
        -------
        Optional[SimulationResult]
            SimulationResult if the simulation was successful, None otherwise.
        """
        self.spring_stiffness_in_n_per_m = stiffness_value_in_n_per_m

        try:
            simulation_data: Tuple = self.run_simulation()
            _, position_data, _, _, _ = simulation_data

            analysis_result: DisplacementAnalysisResult = self.analyze_displacement(
                position_data
            )

            self.print_simulation_result(analysis_result)

            return self.process_stiffness_result(
                simulation_data,
                analysis_result,
                show_results,
            )

        except (ValueError, RuntimeError) as error:
            error_message: str = (
                f"[Spring Stiffness: {self.spring_stiffness_in_n_per_m:<8} N/m] "
                f"{type(error).__name__}: {error}"
            )
            logging.error(colored(error_message, "red"))
            return None

    def find_optimal_spring_stiffness(
        self, search_config: Dict[str, Any]
    ) -> List[SimulationResult]:
        """
        Find the optimal spring stiffness based on the given search
        configuration.

        Parameters
        ----------
        search_config : Dict[str, Any]
            Configuration for the search:
            - start_n_per_m: Starting value for the spring stiffness search
              (default is 0).
            - step_size_n_per_m: Step size for incrementing the spring
              stiffness (default is 100,000).
            - num_results: Number of consecutive valid results to find
              (default is 1).
            - show_results: Whether to display the simulation plots (default is
              True).

        Returns
        -------
        List[SimulationResult]
            List of SimulationResult objects representing the valid results
            found.
        """
        start_in_n_per_m: float = search_config.get("start_n_per_m", 0)
        step_size_in_n_per_m: float = search_config.get(
            "step_size_n_per_m", 100_000)
        num_results: int = search_config.get("num_results", 1)
        show_results: bool = search_config.get("show_results", True)

        consecutive_valid_results_count: int = 0
        results: List[SimulationResult] = []
        max_stiffness: float = 999_999_999_999

        for stiffness in np.arange(
            start_in_n_per_m, max_stiffness, step_size_in_n_per_m
        ):
            result: Optional[SimulationResult] = self._run_single_simulation(
                stiffness, show_results
            )

            if result:
                results.append(result)
                consecutive_valid_results_count += 1
                if consecutive_valid_results_count >= num_results:
                    break
            else:
                consecutive_valid_results_count = 0

        return results


def _run_simulations(
    simulator: CarouselSimulation, search_config: Dict
) -> List[SimulationResult]:
    """
    Run the carousel simulations to find optimal spring stiffness.

    Parameters
    ----------
    simulator : CarouselSimulation
        The CarouselSimulation object to use.
    search_config : Dict
        Configuration for the search.

    Returns
    -------
    List[SimulationResult]
        List of SimulationResult objects representing the valid results found.
    """
    return simulator.find_optimal_spring_stiffness(search_config)


def main() -> Optional[List[SimulationResult]]:
    """
    Execute the main functionality of the script.

    This function creates a CarouselSimulation object, runs the simulations to
    find the optimal spring stiffness, and returns the results.

    Returns
    -------
    Optional[List[SimulationResult]]
        List of SimulationResult objects if simulations are successful, None
        otherwise.
    """
    simulator: CarouselSimulation = CarouselSimulation()

    results: List[SimulationResult] = _run_simulations(
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
