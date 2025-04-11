import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Tuple, Union

import colorama
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

colorama.init()


@dataclass
class CarouselParameters:
    """
    Represents the physical parameters of the carousel.

    Attributes
    ----------
    gravity : float
        Acceleration due to gravity (default is 9.81 m/s^2).
    tilt_angle_in_degrees : float
        Tilt angle of the carousel in degrees (default is 30 degrees).
    carousel_radius_in_m : float
        Radius of the carousel in meters (default is 6 m).
    gondola_mass_in_kg : float
        Mass of a single gondola in kilograms (default is 300 kg).
    carousel_rotation_speed_in_rps : float
        Rotation speed of the carousel in revolutions per second (default
        is 0.2333 rps).
    gondola_width_in_m : float
        Width of a gondola in meters (default is 1.5 m).
    gondola_height_in_m : float
        Height of a gondola in meters (default is 1.5 m).
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
    Represents the result of the displacement analysis.

    Attributes
    ----------
    global_minima : float | None
        Global minimum displacement in meters.
    global_maxima : float | None
        Global maximum displacement in meters.
    displacement_in_m : float | None
        Total displacement (difference between global maxima and minima) in
        meters.
    criteria_was_met : bool
        Indicates whether the displacement criteria was met.
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
    spring_stiffness_in_n_per_m : float
        The spring stiffness value used in the simulation.
    global_maxima_in_m : float | None
        Global maximum displacement observed in the simulation.
    global_minima_in_m : float | None
        Global minimum displacement observed in the simulation.
    displacement_in_m : float | None
        Total displacement (difference between global maxima and minima) in
        meters.
    simulation_data : Tuple
        Raw data from the simulation (time, position, velocity, angle,
        angular velocity).
    """

    spring_stiffness_in_n_per_m: float
    global_maxima_in_m: float | None
    global_minima_in_m: float | None
    displacement_in_m: float | None
    simulation_data: Tuple


class CarouselSimulation:
    """
    Simulates the motion of a gondola on a tilted carousel.

    This class models the dynamics of a gondola attached to a rotating
    carousel, considering factors such as gravity, tilt angle, carousel and
    gondola dimensions, and spring stiffness. It provides methods to run the
    simulation, analyze the displacement of the gondola, and find the optimal
    spring stiffness to minimize this displacement.

    Attributes
    ----------
    params : CarouselParameters
        Physical parameters of the carousel.
    simulation_time_in_s : float
        Total simulation time in seconds.
    max_radial_displacement_in_m : float
        Maximum allowed radial displacement for the gondola.
    tilt_angle_in_rad : float
        Tilt angle of the carousel in radians.
    spring_stiffness_in_n_per_m : float
        Spring stiffness value, adjusted during simulations.
    initial_conditions : List[float]
        Initial conditions for the simulation (position, velocity, angle,
        angular velocity).

    Methods
    -------
    run_simulation() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray]:
        Runs the simulation and returns the time, position, velocity, angle,
        and angular velocity data.
    plot_results(simulation_data: Tuple) -> None:
        Plots the simulation results.
    analyze_displacement(position_data: np.ndarray) ->
    DisplacementAnalysisResult:
        Analyzes the displacement of the gondola from the simulation data.
    find_optimal_spring_stiffness(search_config: Dict[str, Any]) ->
    List[SimulationResult]:
        Searches for the optimal spring stiffness based on the provided
        search configuration.
    """

    def __init__(
        self,
        params: CarouselParameters | None = None,
        simulation_time_in_s: float = 10,
        max_radial_displacement_in_m: float = 0.005,
    ) -> None:
        """
        Constructs all the necessary attributes for the CarouselSimulation
        object.

        Parameters
        ----------
        params : CarouselParameters | None
            Physical parameters of the carousel. If None, default parameters
            are used.
        simulation_time_in_s : float
            Total simulation time in seconds (default is 10 seconds).
        max_radial_displacement_in_m : float
            Maximum allowed radial displacement for the gondola (default is
            0.005 m).
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
        position_in_m : float
            Current position of the gondola in meters.
        angular_velocity_in_rads : float
            Current angular velocity of the carousel in radians per second.
        angle_in_rad : float
            Current angle of the carousel in radians.

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
        position_in_m : float
            Current position of the gondola in meters.
        velocity_in_ms : float
            Current velocity of the gondola in meters per second.
        angle_in_rad : float
            Current angle of the carousel in radians.
        angular_velocity_in_rads : float
            Current angular velocity of the carousel in radians per second.

        Returns
        -------
        float
            The rate of change of angular velocity.
        """
        numerator = -(
            2
            * angular_velocity_in_rads
            * velocity_in_ms
            * position_in_m
            + self.params.gravity
            * np.sin(self.tilt_angle_in_rad)
            * np.sin(angle_in_rad)
            * position_in_m
        )
        denominator = (
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
        Define the system dynamics for the carousel simulation.

        Parameters
        ----------
        _ : float
            Time (not used in the calculation but required by solve_ivp).
        state : List[float]
            Current state of the system (position, velocity, angle, angular
            velocity).

        Returns
        -------
        List[float]
            The derivatives of the state variables (dposition_dt,
            dvelocity_dt, dangle_dt, dangular_velocity_dt).
        """
        (
            position_in_m,
            velocity_in_ms,
            angle_in_rad,
            angular_velocity_in_rads,
        ) = state

        dposition_dt = velocity_in_ms
        dvelocity_dt = self._calculate_dvelocity_dt(
            position_in_m, angular_velocity_in_rads, angle_in_rad
        )

        dangle_dt = angular_velocity_in_rads
        dangular_velocity_dt = self._calculate_dangular_velocity_dt(
            position_in_m, velocity_in_ms, angle_in_rad, angular_velocity_in_rads
        )

        return [dposition_dt, dvelocity_dt, dangle_dt, dangular_velocity_dt]

    def run_simulation(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the carousel simulation.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            A tuple containing the simulation results:
            - time_array_in_s: Time points of the simulation.
            - position_in_m: Position of the gondola over time.
            - velocity_in_ms: Velocity of the gondola over time.
            - angle_in_rad: Angle of the carousel over time.
            - angular_velocity_in_rads: Angular velocity of the carousel over
              time.
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
        axis,
        time_array_in_s: np.ndarray,
        data: np.ndarray,
        ylabel: str,
        label: str,
    ) -> None:
        """
        Plot a single data series on a given axis.

        Parameters
        ----------
        axis
            The matplotlib axis object to plot on.
        time_array_in_s : np.ndarray
            Array of time values.
        data : np.ndarray
            Array of data values to plot.
        ylabel : str
            Label for the y-axis.
        label : str
            Label for the data series.
        """
        axis.plot(time_array_in_s, data, label=label)
        axis.set_ylabel(ylabel)
        axis.legend(loc="upper right")
        axis.grid()
        axis.set_xlim(min(time_array_in_s), max(time_array_in_s))

    def _plot_angle_results(
        self,
        axis,
        time_array_in_s: np.ndarray,
        angle_in_degrees: np.ndarray,
        angular_velocity_in_degs: np.ndarray,
    ) -> None:
        """
        Plot angle and angular velocity on a given axis.

        Parameters
        ----------
        axis
            The matplotlib axis object to plot on.
        time_array_in_s : np.ndarray
            Array of time values.
        angle_in_degrees : np.ndarray
            Array of angle values in degrees.
        angular_velocity_in_degs : np.ndarray
            Array of angular velocity values in degrees per second.
        """
        self._plot_single_result(
            axis,
            time_array_in_s,
            angle_in_degrees,
            r"$\alpha$ [deg]",
            r"$\alpha(t)$",
        )
        self._plot_single_result(
            axis.twinx(),
            time_array_in_s,
            angular_velocity_in_degs,
            r"$\dot{\alpha}$ [deg/s]",
            r"$\dot{\alpha}(t)$",
        )

        for i in range(1, int(max(angle_in_degrees) / 360) + 1):
            idx = np.where(np.isclose(angle_in_degrees, 360 * i, atol=1))[0]
            if len(idx) > 0:
                axis.plot(time_array_in_s[idx[0]],
                          angle_in_degrees[idx[0]], "ro")
                axis.axvline(
                    x=time_array_in_s[idx[0]],
                    color="r",
                    linestyle="--",
                    linewidth=1,
                )

        y_ticks = np.arange(0, max(angle_in_degrees) + 360, 360)
        axis.set_yticks(y_ticks)
        axis.set_yticklabels([f"{int(y)}" for y in y_ticks])

    def plot_results(self, simulation_data: Tuple) -> None:
        """
        Plot the simulation results.

        Parameters
        ----------
        simulation_data : Tuple
            The simulation data returned by `run_simulation`.
        """
        (
            time_array_in_s,
            position_in_m,
            velocity_in_ms,
            angle_in_rad,
            angular_velocity_in_rads,
        ) = simulation_data
        figure, axes = plt.subplots(3, 1, figsize=(10, 9))

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
        angular_velocity_in_degs: np.ndarray = np.degrees(
            angular_velocity_in_rads
        )
        self._plot_angle_results(
            axes[2], time_array_in_s, angle_in_degrees, angular_velocity_in_degs
        )

        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        figure.text(0.5, 0.01, "Time (t)", ha="center", fontsize=12)
        figure.suptitle(
            (
                "Spring Stiffness="
                f"{self.spring_stiffness_in_n_per_m} N/m"
            ),
            fontsize=16,
        )
        plt.show()

    @staticmethod
    def get_local_extremes(
        data_series: np.ndarray
    ) -> Tuple[List[float], List[float]]:
        """
        Find the local maxima and minima in a data series.

        Parameters
        ----------
        data_series : np.ndarray
            The data series to analyze.

        Returns
        -------
        Tuple[List[float], List[float]]
            A tuple containing two lists: local minima and local maxima.
        """
        local_maxima: List[float] = []
        local_minima: List[float] = []

        for i in range(1, len(data_series) - 1):
            is_maxima: bool = (
                data_series[i] > data_series[i - 1]
                and data_series[i] > data_series[i + 1]
            )
            is_minima: bool = (
                data_series[i] < data_series[i - 1]
                and data_series[i] < data_series[i + 1]
            )

            if is_maxima:
                local_maxima.append(data_series[i])
            if is_minima:
                local_minima.append(data_series[i])

        return local_minima, local_maxima

    @staticmethod
    def get_global_extremes(
        local_minima: List[float], local_maxima: List[float]
    ) -> Tuple[float | None, float | None, float | None]:
        """
        Calculate the global extremes (minima, maxima, and difference) from
        local extremes.

        Parameters
        ----------
        local_minima : List[float]
            List of local minima.
        local_maxima : List[float]
            List of local maxima.

        Returns
        -------
        Tuple[float | None, float | None, float | None]
            A tuple containing the global minima, global maxima, and their
            difference.
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
        Analyze the displacement of the gondola.

        Parameters
        ----------
        position_data : np.ndarray
            The position data from the simulation.

        Returns
        -------
        DisplacementAnalysisResult
            An object containing the global minima, global maxima,
            displacement, and whether the criteria was met.
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
        Print the results of the simulation.

        Parameters
        ----------
        analysis_result : DisplacementAnalysisResult
            The result of the displacement analysis.
        """
        global_minima: float | None = analysis_result.global_minima
        global_maxima: float | None = analysis_result.global_maxima
        displacement_in_m: float | None = analysis_result.displacement_in_m
        criteria_was_met: bool = analysis_result.criteria_was_met

        if displacement_in_m is not None:
            message: str = (
                f"[Spring Stiffness: {self.spring_stiffness_in_n_per_m:<8} "
                "N/m] Global maxima: "
                f"{global_maxima:<8} m | Global minima: {global_minima:<8} "
                f"m | Displacement: {displacement_in_m:<8} m"
            )
            color: str = "green" if criteria_was_met else "red"
            print(colored(message, color))

    def process_stiffness_result(
        self,
        simulation_data: Tuple,
        analysis_result: DisplacementAnalysisResult,
        show_results: bool,
    ) -> SimulationResult | None:
        """
        Process the results for a given spring stiffness.

        Parameters
        ----------
        simulation_data : Tuple
            The simulation data.
        analysis_result : DisplacementAnalysisResult
            The result of the displacement analysis.
        show_results : bool
            Whether to plot the results.

        Returns
        -------
        SimulationResult | None
            The simulation result if the displacement criteria was met,
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
        Run a single simulation with a specific spring stiffness.

        Parameters
        ----------
        stiffness_value_in_n_per_m : float
            The spring stiffness value to use.
        show_results : bool
            Whether to plot the results.

        Returns
        -------
        SimulationResult | None
            The simulation result if the displacement criteria was met,
            otherwise None.
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

        except (ValueError, RuntimeError) as error:
            error_message: str = (
                f"[Spring Stiffness: {self.spring_stiffness_in_n_per_m:<8} "
                f"N/m] Error: {str(error)}"
            )
            print(colored(error_message, "red"))
            return None

    def find_optimal_spring_stiffness(
        self, search_config: Dict[str, Any]
    ) -> List[SimulationResult]:
        """
        Find the optimal spring stiffness by running multiple simulations.

        Parameters
        ----------
        search_config : Dict[str, Any]
            Configuration for the search, including:
            - start_n_per_m: Starting value for spring stiffness (default is
              0).
            - step_size_n_per_m: Step size for incrementing spring stiffness
              (default is 100,000).
            - num_results: Number of consecutive valid results to find
              (default is 1).
            - show_results: Whether to plot the results (default is True).

        Returns
        -------
        List[SimulationResult]
            A list of SimulationResult objects that meet the displacement
            criteria.
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
            result: SimulationResult | None = self._run_single_simulation(
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


def run_simulations(
    simulator: CarouselSimulation, search_config: Dict
) -> List:
    """
    Run the carousel simulations to find optimal spring stiffness.

    Parameters
    ----------
    simulator : CarouselSimulation
        The CarouselSimulation object to use.
    search_config : Dict
        Configuration for the search (see `find_optimal_spring_stiffness` for
        details).

    Returns
    -------
    List
        A list of SimulationResult objects that meet the displacement
        criteria.
    """
    return simulator.find_optimal_spring_stiffness(search_config)


def main() -> None:
    """
    Execute the main functionality of the script.

    This function creates a CarouselSimulation object, runs simulations with
    specified search configurations, and displays the results.
    """
    simulator: CarouselSimulation = CarouselSimulation()

    results = run_simulations(
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
