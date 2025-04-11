"""
Carousel Simulation Module
===========================

This module provides a simulation of a carousel, focusing on the
dynamics of the gondola.

It includes functionalities to simulate the motion of a carousel
gondola, analyze its displacement, and find the optimal spring
stiffness for the gondola suspension to meet specific displacement
criteria. The simulation considers various parameters such as
gravity, tilt angle, carousel radius, gondola mass, rotation speed,
and gondola dimensions.

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
...     print(result.spring_stiffness_in_n_per_m)
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp  # type: ignore

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
    Dataclass to hold the parameters for the carousel simulation.

    Attributes
    ----------
    gravity: float
        The gravitational acceleration (default is 9.81 m/s^2).
    tilt_angle_in_degrees: float
        The tilt angle of the carousel in degrees (default is 30
        degrees).
    carousel_radius_in_m: float
        The radius of the carousel in meters (default is 6 m).
    gondola_mass_in_kg: float
        The mass of the gondola in kilograms (default is 300 kg).
    carousel_rotation_speed_in_rps: float
        The rotation speed of the carousel in revolutions per second
        (default is 0.2333 rps).
    gondola_width_in_m: float
        The width of the gondola in meters (default is 1.5 m).
    gondola_height_in_m: float
        The height of the gondola in meters (default is 1.5 m).
    """

    gravity: float = 9.81
    tilt_angle_in_degrees: float = 30
    carousel_radius_in_m: float = 6
    gondola_mass_in_kg: float = 300
    carousel_rotation_speed_in_rps: float = 0.2333
    gondola_width_in_m: float = 1.5
    gondola_height_in_m: float = 1.5


class DisplacementAnalysisResult(NamedTuple):
    """
    A NamedTuple to store the results of the displacement analysis.

    Attributes
    ----------
    global_minima: Union[float, None]
        The global minimum displacement of the gondola, if available.
    global_maxima: Union[float, None]
        The global maximum displacement of the gondola, if available.
    displacement_in_m: Union[float, None]
        The total displacement (difference between global maxima and
        minima) of the gondola, if available.
    criteria_was_met: bool
        A boolean indicating whether the displacement criteria were met.
    """

    global_minima: Union[float, None]
    global_maxima: Union[float, None]
    displacement_in_m: Union[float, None]
    criteria_was_met: bool


class SimulationResult(NamedTuple):
    """
    A NamedTuple to store the results of a single simulation run.

    Attributes
    ----------
    spring_stiffness_in_n_per_m: float
        The spring stiffness used in the simulation.
    global_maxima_in_m: Union[float, None]
        The global maximum displacement observed in the simulation, if
        available.
    global_minima_in_m: Union[float, None]
        The global minimum displacement observed in the simulation, if
        available.
    displacement_in_m: Union[float, None]
        The total displacement (difference between global maxima and
        minima) observed, if available.
    simulation_data: Tuple
        The raw data from the simulation.
    """

    spring_stiffness_in_n_per_m: float
    global_maxima_in_m: Union[float, None]
    global_minima_in_m: Union[float, None]
    displacement_in_m: Union[float, None]
    simulation_data: Tuple


class CarouselSimulation:
    """
    A class to simulate the dynamics of a carousel gondola.

    This class provides methods to simulate the motion of a carousel
    gondola, analyze its displacement, and find the optimal spring
    stiffness for the gondola suspension to meet specific
    displacement criteria.

    Attributes
    ----------
    params: CarouselParameters
        The parameters for the carousel simulation.
    simulation_time_in_s: float
        The total time for the simulation in seconds.
    max_radial_displacement_in_m: float
        The maximum allowed radial displacement for the gondola.
    tilt_angle_in_rad: float
        The tilt angle of the carousel in radians.
    spring_stiffness_in_n_per_m: float
        The spring stiffness of the gondola suspension.
    initial_conditions: List[float]
        The initial conditions for the simulation.

    Methods
    -------
    run_simulation() -> Tuple[np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray]:
        Run the simulation and return the time and state arrays.
    plot_results(simulation_data: Tuple) -> None:
        Plot the simulation results.
    find_optimal_spring_stiffness(search_config: Dict[str, Any]) ->
    List[SimulationResult]:
        Find and return the optimal spring stiffness based on the given
        search configuration.
    """

    def __init__(
        self,
        params: Union[CarouselParameters, None] = None,
        simulation_time_in_s: float = 10,
        max_radial_displacement_in_m: float = 0.005,
    ) -> None:
        """
        Initialize the CarouselSimulation object.

        Parameters
        ----------
        params: Union[CarouselParameters, None]
            The parameters for the carousel simulation. If None, default
            parameters are used.
        simulation_time_in_s: float
            The total time for the simulation in seconds.
        max_radial_displacement_in_m: float
            The maximum allowed radial displacement for the gondola.
        """
        self.params = params if params else CarouselParameters()
        self.simulation_time_in_s = simulation_time_in_s
        self.max_radial_displacement_in_m = max_radial_displacement_in_m
        self.tilt_angle_in_rad = math.radians(
            self.params.tilt_angle_in_degrees
        )
        self.spring_stiffness_in_n_per_m: float = 0

        self.initial_conditions: List[float] = [
            self.params.carousel_radius_in_m,
            0,
            0,
            2 * np.pi * self.params.carousel_rotation_speed_in_rps,
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
            The current position of the gondola.
        angular_velocity_in_rads: float
            The current angular velocity of the gondola.
        angle_in_rad: float
            The current angle of the carousel.

        Returns
        -------
        float
            The calculated rate of change of velocity.
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
            The current position of the gondola.
        velocity_in_ms: float
            The current velocity of the gondola.
        angle_in_rad: float
            The current angle of the carousel.
        angular_velocity_in_rads: float
            The current angular velocity of the gondola.

        Returns
        -------
        float
            The calculated rate of change of angular velocity.
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
            * (
                self.params.gondola_width_in_m**2
                + self.params.gondola_height_in_m**2
            )
            + 20 * self.params.carousel_radius_in_m**2
        )
        return numerator / denominator

    def system_dynamics(self, _: float, state: List[float]) -> List[float]:
        """
        Define the system dynamics for the carousel gondola.

        Parameters
        ----------
        _: float
            The current time (not used in the calculation but required
            by solve_ivp).
        state: List[float]
            The current state of the system, containing position,
            velocity, angle, and angular velocity.

        Returns
        -------
        List[float]
            The derivatives of the state variables [dposition_dt,
            dvelocity_dt, dangle_dt, dangular_velocity_dt].
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
            A tuple containing the time array and the state variable
            arrays (position, velocity, angle, and angular velocity).
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
        axes: Any,
        time_array_in_s: np.ndarray,
        data: np.ndarray,
        ylabel: str,
        label: str,
    ) -> None:
        """
        Plot a single set of results on the given axes.

        Parameters
        ----------
        axes: Any
            The matplotlib axes on which to plot.
        time_array_in_s: np.ndarray
            The array of time values.
        data: np.ndarray
            The data to plot.
        ylabel: str
            The label for the y-axis.
        label: str
            The label for the data series.
        """
        axes.plot(time_array_in_s, data, label=label)
        axes.set_ylabel(ylabel)
        axes.legend(loc="upper right")
        axes.grid()
        axes.set_xlim(min(time_array_in_s), max(time_array_in_s))

    def _plot_angle_results(
        self, axes: Any, time_array_in_s: np.ndarray, angle_in_degrees: np.ndarray
    ) -> None:
        """
        Plot the angle results, highlighting full rotations.

        Parameters
        ----------
        axes: Any
            The matplotlib axes on which to plot.
        time_array_in_s: np.ndarray
            The array of time values.
        angle_in_degrees: np.ndarray
            The array of angle values in degrees.
        """
        self._plot_single_result(
            axes,
            time_array_in_s,
            angle_in_degrees,
            r"$\alpha$ [deg]",
            r"$\alpha(t)$",
        )

        for i in range(1, int(max(angle_in_degrees) / 360) + 1):
            indices = np.where(np.isclose(
                angle_in_degrees, 360 * i, atol=1))[0]
            if len(indices) > 0:
                axes.plot(
                    time_array_in_s[indices[0]
                                    ], angle_in_degrees[indices[0]], "ro"
                )
                for axis in axes:
                    axis.axvline(
                        x=time_array_in_s[indices[0]],
                        color="r",
                        linestyle="--",
                        linewidth=1,
                    )

        y_ticks: np.ndarray = np.arange(0, max(angle_in_degrees) + 360, 360)
        axes.set_yticks(y_ticks)
        axes.set_yticklabels([f"{int(y)}" for y in y_ticks])

    def plot_results(self, simulation_data: Tuple) -> None:
        """
        Plot the simulation results for position, velocity, angle, and
        angular velocity.

        Parameters
        ----------
        simulation_data: Tuple
            The simulation data returned by `run_simulation`.
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
            axes[0], time_array_in_s, position_in_m, r"$x$ [m]", r"$x(t)$"
        )
        self._plot_single_result(
            axes[1],
            time_array_in_s,
            velocity_in_ms,
            r"$\dot{x}$ [m/s]",
            r"$\dot{x}(t)$",
        )

        angle_in_degrees: np.ndarray = np.degrees(angle_in_rad)
        self._plot_angle_results(axes[2], time_array_in_s, angle_in_degrees)

        angular_velocity_in_degs: np.ndarray = np.degrees(
            angular_velocity_in_rads)
        self._plot_single_result(
            axes[3],
            time_array_in_s,
            angular_velocity_in_degs,
            r"$\dot{\alpha}$ [deg/s]",
            r"$\dot{\alpha}(t)$",
        )

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
    ) -> Tuple[List[float], List[float]]:
        """
        Identify and return the local minima and maxima in a data series.

        Parameters
        ----------
        data_series: np.ndarray
            The data series to analyze.

        Returns
        -------
        Tuple[List[float], List[float]]
            A tuple containing lists of local minima and local maxima.
        """
        local_maxima: List[float] = []
        local_minima: List[float] = []

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
        local_minima: List[float], local_maxima: List[float]
    ) -> Tuple[Union[float, None], Union[float, None], Union[float, None]]:
        """
        Determine the global minima, maxima, and their difference from
        lists of local extremes.

        Parameters
        ----------
        local_minima: List[float]
            The list of local minima.
        local_maxima: List[float]
            The list of local maxima.

        Returns
        -------
        Tuple[Union[float, None], Union[float, None], Union[float, None]]
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
        Analyze the displacement of the gondola from the position data.

        Parameters
        ----------
        position_data: np.ndarray
            The array of position data from the simulation.

        Returns
        -------
        DisplacementAnalysisResult
            The result of the displacement analysis, including global
            extremes and whether the criteria were met.
        """
        local_minima, local_maxima = self.get_local_extremes(position_data)
        extremes: Tuple[
            Union[float, None], Union[float, None], Union[float, None]
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
        Print the simulation results, highlighting whether the
        displacement criteria were met.

        Parameters
        ----------
        analysis_result: DisplacementAnalysisResult
            The result of the displacement analysis.
        """
        global_minima: Union[float, None] = analysis_result.global_minima
        global_maxima: Union[float, None] = analysis_result.global_maxima
        displacement_in_m: Union[float,
                                 None] = analysis_result.displacement_in_m
        criteria_was_met: bool = analysis_result.criteria_was_met

        if displacement_in_m is not None:
            message: str = (
                f"[Spring Stiffness: {self.spring_stiffness_in_n_per_m:<8} N/m] "
                f"Global maxima: {global_maxima:<8} m | "
                f"Global minima: {global_minima:<8} m | "
                f"Displacement: {displacement_in_m:<8} m"
            )
            color: str = "green" if criteria_was_met else "red"
            print(
                "\033[32m" + message + "\033[0m"
                if color == "green"
                else "\033[31m" + message + "\033[0m"
            )

    def process_stiffness_result(
        self,
        simulation_data: Tuple,
        analysis_result: DisplacementAnalysisResult,
        show_results: bool,
    ) -> Union[SimulationResult, None]:
        """
        Process the results of a simulation run for a given spring
        stiffness.

        Parameters
        ----------
        simulation_data: Tuple
            The raw data from the simulation.
        analysis_result: DisplacementAnalysisResult
            The result of analyzing the displacement from the simulation
            data.
        show_results: bool
            Whether to display the results graphically.

        Returns
        -------
        Union[SimulationResult, None]
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
    ) -> Union[SimulationResult, None]:
        """
        Run a single simulation with a specified spring stiffness.

        Parameters
        ----------
        stiffness_value_in_n_per_m: float
            The spring stiffness to use for the simulation.
        show_results: bool
            Whether to display the results graphically.

        Returns
        -------
        Union[SimulationResult, None]
            The simulation result if the displacement criteria were met,
            otherwise None.

        Raises
        ------
        ValueError
            If the numerical integration fails due to issues like
            stiffness.
        RuntimeError
            If any other runtime error occurs during the simulation.
        """
        self.spring_stiffness_in_n_per_m = stiffness_value_in_n_per_m

        try:
            simulation_data: Tuple = self.run_simulation()
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
            print("\033[31m" + error_message + "\033[0m")
            return None

        except RuntimeError as error:
            error_message: str = (
                f"[Spring Stiffness: {self.spring_stiffness_in_n_per_m:<8} N/m] "
                f"Runtime Error: {error}"
            )
            print("\033[31m" + error_message + "\033[0m")
            return None

    def find_optimal_spring_stiffness(
        self, search_config: Dict[str, Any]
    ) -> List[SimulationResult]:
        """
        Iteratively search for the optimal spring stiffness that meets
        the displacement criteria.

        Parameters
        ----------
        search_config: Dict[str, Any]
            Configuration for the search, including start value, step
            size, number of results, and whether to show results.

        Returns
        -------
        List[SimulationResult]
            A list of SimulationResult objects that meet the
            displacement criteria, with the length of the list
            determined by 'num_results' in search_config.
        """
        start_in_n_per_m: float = search_config.get("start_n_per_m", 0)
        step_size_in_n_per_m: float = search_config.get(
            "step_size_n_per_m", 100_000
        )
        number_of_results: int = search_config.get("num_results", 1)
        show_results: bool = search_config.get("show_results", True)

        consecutive_valid_results_count: int = 0
        results: List[SimulationResult] = []
        max_stiffness: float = 999_999_999_999

        for stiffness in np.arange(
            start_in_n_per_m, max_stiffness, step_size_in_n_per_m
        ):
            result: Union[SimulationResult, None] = self._run_single_simulation(
                stiffness, show_results
            )

            if result:
                results.append(result)
                consecutive_valid_results_count += 1
                if consecutive_valid_results_count >= number_of_results:
                    break
            else:
                consecutive_valid_results_count = 0

        return results


def run_simulation_and_find_optimal_stiffness(
    simulator: CarouselSimulation, search_config: Dict[str, Any]
) -> List[SimulationResult]:
    """
    Run the simulation and find the optimal spring stiffness.

    Parameters
    ----------
    simulator : CarouselSimulation
        The CarouselSimulation instance to use.
    search_config : Dict[str, Any]
        Configuration for the search, including start value, step size,
        number of results, and whether to show results.

    Returns
    -------
    List[SimulationResult]
        A list of SimulationResult objects that meet the displacement
        criteria.
    """
    return simulator.find_optimal_spring_stiffness(search_config)


def main() -> Union[List[SimulationResult], None]:
    """
    Execute the main functionality of the script.

    This function initializes a CarouselSimulation, runs the simulation
    to find the optimal spring stiffness based on the provided search
    configuration, and returns the results.

    Returns
    -------
    Union[List[SimulationResult], None]
        A list of SimulationResult objects if successful, None
        otherwise.
    """
    simulator: CarouselSimulation = CarouselSimulation()

    results: List[SimulationResult] = run_simulation_and_find_optimal_stiffness(
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
