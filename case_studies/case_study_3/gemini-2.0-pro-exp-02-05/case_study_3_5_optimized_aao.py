import math
from dataclasses import dataclass
from typing import Any, NamedTuple

import colorama
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp  # type: ignore
from termcolor import colored

colorama.init()


@dataclass
class CarouselParameters:
    """
    A class to represent the parameters for a carousel simulation.

    Attributes
    ----------
    gravity: float
        The gravitational acceleration (default is 9.81 m/s^2).
    tilt_angle_deg: float
        The tilt angle of the carousel in degrees (default is 30 degrees).
    carousel_radius_m: float
        The radius of the carousel in meters (default is 6 meters).
    gondola_mass_kg: float
        The mass of the gondola in kilograms (default is 300 kg).
    carousel_rot_speed_rps: float
        The rotational speed of the carousel in revolutions per second
        (default is 0.2333 rps).
    gondola_width_m: float
        The width of the gondola in meters (default is 1.5 meters).
    gondola_height_m: float
        The height of the gondola in meters (default is 1.5 meters).
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
    Represents the result of the displacement analysis.

    Attributes
    ----------
    global_minima: float | None
        The global minimum displacement in meters, if available.
    global_maxima: float | None
        The global maximum displacement in meters, if available.
    displacement_m: float | None
        The total displacement (difference between global maxima and minima)
        in meters, if available.
    criteria_met: bool
        Indicates whether the displacement criteria were met.
    """

    global_minima: float | None
    global_maxima: float | None
    displacement_m: float | None
    criteria_met: bool


class SimulationResult(NamedTuple):
    """
    Represents the result of a single carousel simulation.

    Attributes
    ----------
    spring_stiffness_n_per_m: float
        The spring stiffness used in the simulation (N/m).
    global_maxima_m: float | None
        The global maximum displacement observed in the simulation (m).
    global_minima_m: float | None
        The global minimum displacement observed in the simulation (m).
    displacement_m: float | None
        The total displacement (difference between global maxima and minima)
        observed in the simulation (m).
    simulation_data: tuple
        The raw data from the simulation (time, position, velocity, angle,
        angular velocity).
    """

    spring_stiffness_n_per_m: float
    global_maxima_m: float | None
    global_minima_m: float | None
    displacement_m: float | None
    simulation_data: tuple


class CarouselSimulation:
    """
    A class to simulate the dynamics of a carousel with a suspended gondola.

    Attributes
    ----------
    params: CarouselParameters
        The physical parameters of the carousel.
    simulation_time_s: float
        The total simulation time in seconds.
    max_radial_displacement_m: float
        The maximum allowable radial displacement of the gondola.
    tilt_angle_rad: float
        The tilt angle of the carousel in radians.
    spring_stiffness_n_per_m: float
        The stiffness of the spring connecting the gondola to the carousel.
    initial_conditions: list[float]
        The initial conditions for the simulation
        [position, velocity, angle, angular velocity].

    Methods
    -------
    system_dynamics(_: float, state: list[float]) -> list[float]:
        Calculate the system dynamics (derivatives) for the given state.
    run_simulation() -> tuple[np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray]:
        Run the simulation and return the results.
    plot_results(simulation_data: tuple) -> None:
        Plot the simulation results.
    get_local_extremes(data_series: np.ndarray) ->
    tuple[list[float], list[float]]:
        Find the local minima and maxima in a data series.
    get_global_extremes(local_minima: list[float], local_maxima: list[float])
    -> tuple[float | None, float | None, float | None]:
        Calculate the global extremes (min, max, and difference) from local
        extremes.
    analyze_displacement(position_data: np.ndarray) ->
    DisplacementAnalysisResult:
        Analyze the displacement of the gondola and determine if criteria are met.
    print_simulation_result(analysis_result: DisplacementAnalysisResult) -> None:
        Print the simulation results, highlighting whether criteria were met.
    process_stiffness_result(simulation_data: tuple, analysis_result:
    DisplacementAnalysisResult, show_results: bool) -> SimulationResult | None:
        Process the results for a given spring stiffness.
    _run_single_simulation(stiffness_value_n_per_m: float, show_results: bool)
    -> SimulationResult | None:
        Run a single simulation with a specified spring stiffness.
    find_optimal_spring_stiffness(search_config: dict[str, Any]) ->
    list[SimulationResult]:
        Find the optimal spring stiffness based on the provided search
        configuration.
    """

    def __init__(
        self,
        params: CarouselParameters | None = None,
        simulation_time_s: float = 10.0,
        max_radial_displacement_m: float = 0.005,
    ) -> None:
        """
        Construct all the necessary attributes for the CarouselSimulation object.

        Parameters
        ----------
        params: CarouselParameters | None, optional
            The physical parameters of the carousel (default is None, which
            uses default parameters).
        simulation_time_s: float, optional
            The total simulation time in seconds (default is 10 seconds).
        max_radial_displacement_m: float, optional
            The maximum allowable radial displacement of the gondola in meters
            (default is 0.005 meters).
        """
        if params is None:
            params = CarouselParameters()

        self.params: CarouselParameters = params
        self.simulation_time_s: float = simulation_time_s
        self.max_radial_displacement_m: float = max_radial_displacement_m
        self.tilt_angle_rad: float = math.radians(params.tilt_angle_deg)
        self.spring_stiffness_n_per_m: float = 0.0

        self.initial_conditions: list[float] = [
            self.params.carousel_radius_m,
            0.0,
            0.0,
            2 * np.pi * self.params.carousel_rot_speed_rps,
        ]

    def system_dynamics(self, _: float, state: list[float]) -> list[float]:
        """
        Calculate the system dynamics (derivatives) for the given state.

        Parameters
        ----------
        _: float
            The current time (not used in the calculation, but required by
            solve_ivp).
        state: list[float]
            The current state of the system
            [position, velocity, angle, angular velocity].

        Returns
        -------
        list[float]
            The derivatives of the state variables
            [dposition_dt, dvelocity_dt, dangle_dt, dangular_velocity_dt].
        """
        (
            position_m,
            velocity_ms,
            angle_rad,
            angular_velocity_rads,
        ) = state

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
            (
                2 * angular_velocity_rads * velocity_ms * position_m
                + self.params.gravity
                * np.sin(self.tilt_angle_rad)
                * np.sin(angle_rad)
                * position_m
            )
            / (
                position_m**2
                + (5 / 3)
                * (self.params.gondola_width_m**2 + self.params.gondola_height_m**2)
                + 20 * self.params.carousel_radius_m**2
            )
        )

        return [dposition_dt, dvelocity_dt, dangle_dt, dangular_velocity_dt]

    def run_simulation(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the simulation and return the results.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            A tuple containing the simulation results:
            - time_array_s: The time points at which the solution is evaluated.
            - position_m: The position of the gondola over time.
            - velocity_ms: The velocity of the gondola over time.
            - angle_rad: The angle of the gondola over time.
            - angular_velocity_rads: The angular velocity of the gondola over time.
        """
        time_span_s: list[float] = [0.0, self.simulation_time_s]
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
        Plot the simulation results.

        Parameters
        ----------
        simulation_data: tuple
            The raw data from the simulation (time, position, velocity, angle,
            angular velocity).
        """
        (
            time_array_s,
            position_m,
            velocity_ms,
            angle_rad,
            angular_velocity_rads,
        ) = simulation_data
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
            index: np.ndarray = np.where(
                np.isclose(angle_deg, 360 * i, atol=1))[0]
            if len(index) > 0:
                axes[2].plot(time_array_s[index[0]], angle_deg[index[0]], "ro")
                for axis in axes:
                    axis.axvline(
                        x=time_array_s[index[0]],
                        color="r",
                        linestyle="--",
                        linewidth=1,
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
        Find the local minima and maxima in a data series.

        Parameters
        ----------
        data_series: np.ndarray
            The data series to analyze.

        Returns
        -------
        tuple[list[float], list[float]]
            A tuple containing two lists:
            - local_minima: The local minima found in the data series.
            - local_maxima: The local maxima found in the data series.
        """
        local_maxima: list[float] = []
        local_minima: list[float] = []

        for i in range(1, len(data_series) - 1):
            data_is_local_maximum: bool = (
                data_series[i] > data_series[i - 1]
                and data_series[i] > data_series[i + 1]
            )
            if data_is_local_maximum:
                local_maxima.append(data_series[i])

            data_is_local_minimum: bool = (
                data_series[i] < data_series[i - 1]
                and data_series[i] < data_series[i + 1]
            )
            if data_is_local_minimum:
                local_minima.append(data_series[i])

        return local_minima, local_maxima

    @staticmethod
    def get_global_extremes(
        local_minima: list[float], local_maxima: list[float]
    ) -> tuple[float | None, float | None, float | None]:
        """
        Calculate the global extremes (min, max, and difference) from local extremes.

        Parameters
        ----------
        local_minima: list[float]
            The local minima found in the data series.
        local_maxima: list[float]
            The local maxima found in the data series.

        Returns
        -------
        tuple[float | None, float | None, float | None]
            A tuple containing:
            - global_minima: The global minimum value, or None if local_minima
            is empty.
            - global_maxima: The global maximum value, or None if local_maxima
            is empty.
            - difference: The difference between the global maximum and minimum,
            or None if either is None.
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
        Analyze the displacement of the gondola and determine if criteria are met.

        Parameters
        ----------
        position_data: np.ndarray
            The position data of the gondola over time.

        Returns
        -------
        DisplacementAnalysisResult
            The result of the displacement analysis.
        """
        local_minima, local_maxima = self.get_local_extremes(position_data)
        extremes = self.get_global_extremes(local_minima, local_maxima)
        global_minima, global_maxima, displacement_m = extremes

        criteria_met: bool = False
        displacement_is_valid: bool = (
            displacement_m is not None
            and displacement_m <= self.max_radial_displacement_m
        )
        if displacement_is_valid:
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
        Print the simulation results, highlighting whether criteria were met.

        Parameters
        ----------
        analysis_result: DisplacementAnalysisResult
            The result of the displacement analysis.
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
        Process the results for a given spring stiffness and optionally plot them.

        Parameters
        ----------
        simulation_data: tuple
            The raw data from the simulation.
        analysis_result: DisplacementAnalysisResult
            The result of analyzing the displacement from the simulation data.
        show_results: bool
            Whether to display the results graphically.

        Returns
        -------
        SimulationResult | None
            The simulation result if the displacement criteria are met,
            otherwise None.
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
        Run a single simulation with a specified spring stiffness.

        Parameters
        ----------
        stiffness_value_n_per_m: float
            The spring stiffness to use for the simulation (N/m).
        show_results: bool
            Whether to display the results graphically.

        Returns
        -------
        SimulationResult | None
            The simulation result if the displacement criteria are met,
            otherwise None.
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
        Find the optimal spring stiffness based on the provided search configuration.

        Parameters
        ----------
        search_config: dict[str, Any]
            A dictionary containing the search configuration:
            - start_n_per_m: The starting value for the spring stiffness search
            (default is 0).
            - step_size_n_per_m: The step size for incrementing the spring
            stiffness (default is 100,000).
            - num_results: The number of consecutive valid results required to
            terminate the search (default is 1).
            - show_results: Whether to display the results graphically for each
            simulation (default is True).

        Returns
        -------
        list[SimulationResult]
            A list of SimulationResult objects representing the valid results
            found during the search.
        """
        start_n_per_m: float = search_config.get("start_n_per_m", 0.0)
        step_size_n_per_m: float = search_config.get(
            "step_size_n_per_m", 100_000.0)
        num_results: int = search_config.get("num_results", 1)
        show_results: bool = search_config.get("show_results", True)

        consecutive_valid_results_count: int = 0
        results_list: list[SimulationResult] = []
        max_stiffness: float = 999_999_999_999.0

        for stiffness in np.arange(start_n_per_m, max_stiffness, step_size_n_per_m):
            result = self._run_single_simulation(stiffness, show_results)

            if result:
                results_list.append(result)
                consecutive_valid_results_count += 1
                results_found: bool = consecutive_valid_results_count >= num_results
                if results_found:
                    break
            else:
                consecutive_valid_results_count = 0

        return results_list


def main() -> list[SimulationResult]:
    """
    Execute the main functionality of the script:
    - Initialize a CarouselSimulation.
    - Find and return the optimal spring stiffness based on the given
    configuration.
    """
    simulator = CarouselSimulation()

    results = simulator.find_optimal_spring_stiffness(
        {
            "start_n_per_m": 0.0,
            "step_size_n_per_m": 100_000.0,
            "num_results": 1,
            "show_results": True,
        }
    )
    return results


if __name__ == "__main__":
    main()
