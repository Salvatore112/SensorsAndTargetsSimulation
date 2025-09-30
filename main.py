import numpy as np
import matplotlib.colors as mcolors
import os
import random
import math
import matplotlib.pyplot as plt
import pickle

from matplotlib.animation import FuncAnimation, PillowWriter
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class Target:
    id: int
    initial_position: Tuple[float, float]
    velocity: Tuple[float, float]


@dataclass
class Sensor:
    id: int
    position: Tuple[float, float]


class Simulation:
    def __init__(
        self,
        duration: float,
        time_step: float = 1.0,
        output_dir: str = "simulation_results",
    ):
        self.duration = duration
        self.time_step = time_step
        self.sensors: List[Sensor] = []
        self.targets: List[Target] = []
        self.simulation_data: Dict = {}
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def add_sensor(self, sensor_id: int, position: Tuple[float, float]):
        self.sensors.append(Sensor(sensor_id, position))

    def add_target(
        self,
        obj_id: int,
        initial_position: Tuple[float, float],
        velocity: Tuple[float, float],
    ):
        self.targets.append(Target(obj_id, initial_position, velocity))

    def add_uniform_target(self, obj_id: int, area_size: float = 50):
        initial_x = random.uniform(-area_size, area_size)
        initial_y = random.uniform(-area_size, area_size)

        speed = random.uniform(0.5, 3.0)
        angle = random.uniform(0, 2 * math.pi)
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)

        self.targets.append(Target(obj_id, (initial_x, initial_y), (vx, vy)))
        return (initial_x, initial_y), (vx, vy)

    def add_uniform_sensor(self, sensor_id: int, area_size: float = 50):
        pos_x = random.uniform(-area_size, area_size)
        pos_y = random.uniform(-area_size, area_size)

        self.sensors.append(Sensor(sensor_id, (pos_x, pos_y)))
        return (pos_x, pos_y)

    def calculate_distance(
        self, pos1: Tuple[float, float], pos2: Tuple[float, float]
    ) -> float:
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def get_target_position(self, obj: Target, time: float) -> Tuple[float, float]:
        x = obj.initial_position[0] + obj.velocity[0] * time
        y = obj.initial_position[1] + obj.velocity[1] * time
        return (x, y)

    def run_simulation(self):
        time_points = np.arange(0, self.duration + self.time_step, self.time_step)

        self.simulation_data = {
            "time_points": time_points,
            "sensors": {sensor.id: sensor for sensor in self.sensors},
            "targets": {target.id: target for target in self.targets},
        }

    def get_distance(self, sensor_id: int, target_id: int, time: float) -> float:
        if time < 0 or time > self.duration:
            raise ValueError(
                f"Time {time} is outside the simulation range [0, {self.duration}]"
            )

        sensor = self.simulation_data["sensors"][sensor_id]
        target_obj = self.simulation_data["targets"][target_id]

        target_pos = self.get_target_position(target_obj, time)
        distance = self.calculate_distance(sensor.position, target_pos)

        return distance

    def get_target_position_at_time(
        self, target_id: int, time: float
    ) -> Tuple[float, float]:
        if time < 0 or time > self.duration:
            raise ValueError(
                f"Time {time} is outside the simulation range [0, {self.duration}]"
            )

        target_obj = self.simulation_data["targets"][target_id]
        return self.get_target_position(target_obj, time)

    def print_distances_at_time(self, time: float):
        print(f"\n=== Distances at time t={time} seconds ===")
        for sensor in self.sensors:
            for obj in self.targets:
                distance = self.get_distance(sensor.id, obj.id, time)
                obj_pos = self.get_target_position_at_time(obj.id, time)
                print(f"Sensor {sensor.id} -> Target {obj.id}: {distance:.2f} units")
                print(
                    f"  Target {obj.id} position: ({obj_pos[0]:.1f}, {obj_pos[1]:.1f})"
                )

    def print_multiple_times(self, times: List[float]):
        for time in times:
            self.print_distances_at_time(time)

    def plot_trajectories(self, save_file: bool = True):
        plt.figure(figsize=(12, 10))

        obj_colors = list(mcolors.TABLEAU_COLORS.keys())
        sensor_colors = ["red", "green", "blue", "purple", "orange", "brown"]

        for i, obj in enumerate(self.targets):
            color = obj_colors[i % len(obj_colors)]

            positions = [
                self.get_target_position(obj, t)
                for t in self.simulation_data["time_points"]
            ]
            x_vals = [p[0] for p in positions]
            y_vals = [p[1] for p in positions]

            plt.plot(
                x_vals,
                y_vals,
                color=color,
                linewidth=2,
                label=f"Target {obj.id}",
                alpha=0.7,
            )

            plt.scatter(
                x_vals[0],
                y_vals[0],
                color=color,
                s=100,
                marker="o",
                edgecolors="black",
                zorder=5,
                label=f"Target {obj.id} start",
            )
            plt.scatter(
                x_vals[-1],
                y_vals[-1],
                color=color,
                s=100,
                marker="s",
                edgecolors="black",
                zorder=5,
                label=f"Target {obj.id} finish",
            )

            if len(positions) > 1:
                mid_idx = len(positions) // 2
                plt.annotate(
                    "",
                    xy=positions[mid_idx + 1],
                    xytext=positions[mid_idx],
                    arrowprops=dict(arrowstyle="->", color=color, lw=2),
                )

        for i, sensor in enumerate(self.sensors):
            color = sensor_colors[i % len(sensor_colors)]
            plt.scatter(
                sensor.position[0],
                sensor.position[1],
                color=color,
                s=200,
                marker="^",
                label=f"Sensor {sensor.id}",
                edgecolors="black",
                zorder=5,
            )

            plt.annotate(
                f"S{sensor.id}",
                (sensor.position[0], sensor.position[1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=12,
                fontweight="bold",
            )

        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.title("Target trajectories and sensor positions")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis("equal")
        plt.tight_layout()

        if save_file:
            filename = os.path.join(self.output_dir, "trajectories.png")
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"Trajectories saved to: {filename}")

        plt.show()

    def create_animation(self, interval: int = 100, save_gif: bool = True):
        fig, ax = plt.subplots(figsize=(12, 10))
        time_points = self.simulation_data["time_points"]

        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        ax.set_title("Target movement animation")
        ax.grid(True, alpha=0.3)

        all_x = []
        all_y = []
        for obj in self.targets:
            positions = [self.get_target_position(obj, t) for t in time_points]
            all_x.extend([p[0] for p in positions])
            all_y.extend([p[1] for p in positions])
        for sensor in self.sensors:
            all_x.append(sensor.position[0])
            all_y.append(sensor.position[1])

        margin = 5
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

        obj_points = []
        obj_trails = []
        obj_colors = list(mcolors.TABLEAU_COLORS.keys())

        for i, obj in enumerate(self.targets):
            color = obj_colors[i % len(obj_colors)]
            (trail,) = ax.plot([], [], color=color, linewidth=2, alpha=0.5)
            (point,) = ax.plot(
                [], [], color=color, marker="o", markersize=10, markeredgecolor="black"
            )
            obj_trails.append(trail)
            obj_points.append(point)

            ax.text(
                obj.initial_position[0],
                obj.initial_position[1],
                f"O{obj.id}",
                fontsize=10,
                fontweight="bold",
            )

        sensor_colors = ["red", "green", "blue", "purple", "orange", "brown"]
        for i, sensor in enumerate(self.sensors):
            color = sensor_colors[i % len(sensor_colors)]
            ax.scatter(
                sensor.position[0],
                sensor.position[1],
                color=color,
                s=150,
                marker="^",
                label=f"Sensor {sensor.id}",
                edgecolors="black",
                zorder=5,
            )
            ax.annotate(
                f"S{sensor.id}",
                (sensor.position[0], sensor.position[1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=12,
                fontweight="bold",
            )

        time_text = ax.text(
            0.02,
            0.95,
            "",
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        def animate(frame):
            current_time = time_points[frame]
            time_text.set_text(f"Time: {current_time:.1f} sec")

            for i, obj in enumerate(self.targets):
                trail_x = []
                trail_y = []
                for t in time_points[: frame + 1]:
                    pos = self.get_target_position(obj, t)
                    trail_x.append(pos[0])
                    trail_y.append(pos[1])

                obj_trails[i].set_data(trail_x, trail_y)

                current_pos = self.get_target_position(obj, current_time)
                obj_points[i].set_data([current_pos[0]], [current_pos[1]])

            return obj_trails + obj_points + [time_text]

        anim = FuncAnimation(
            fig,
            animate,
            frames=len(time_points),
            interval=interval,
            blit=True,
            repeat=True,
        )

        plt.legend()
        plt.tight_layout()

        if save_gif:
            try:
                gif_filename = os.path.join(self.output_dir, "animation.gif")
                anim.save(
                    gif_filename, writer=PillowWriter(fps=1000 // interval), dpi=100
                )
                print(f"Animation saved to: {gif_filename}")
            except Exception as e:
                print(f"Error saving GIF: {e}")

        plt.show()

        return anim

    def save_simulation(self, filename: str = "simulation_data.pkl"):
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "duration": self.duration,
                    "time_step": self.time_step,
                    "sensors": self.sensors,
                    "targets": self.targets,
                    "simulation_data": self.simulation_data,
                },
                f,
            )
        print(f"Simulation saved to: {filepath}")

    def load_simulation(self, filename: str = "simulation_data.pkl"):
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.duration = data["duration"]
            self.time_step = data["time_step"]
            self.sensors = data["sensors"]
            self.targets = data["targets"]
            self.simulation_data = data["simulation_data"]
        print(f"Simulation loaded from: {filepath}")

    def print_simulation_info(self):
        print(f"Simulation duration: {self.duration} seconds")
        print(f"Time step: {self.time_step} seconds")
        print(f"Number of sensors: {len(self.sensors)}")
        print(f"Number of targets: {len(self.targets)}")
        print("\nSensors:")
        for sensor in self.sensors:
            print(
                f"  Sensor {sensor.id}: position ({sensor.position[0]:.1f}, {sensor.position[1]:.1f})"
            )
        print("\nTargets:")
        for obj in self.targets:
            print(
                f"  Target {obj.id}: initial position ({obj.initial_position[0]:.1f}, {obj.initial_position[1]:.1f}), "
                f"velocity ({obj.velocity[0]:.2f}, {obj.velocity[1]:.2f})"
            )


if __name__ == "__main__":
    sim = Simulation(duration=50, time_step=1.0, output_dir="simulation_results")

    print("Adding random sensors...")
    for i in range(1, 4):
        pos = sim.add_uniform_sensor(i, area_size=30)
        print(f"Sensor {i}: position ({pos[0]:.1f}, {pos[1]:.1f})")

    print("\nAdding random targets...")
    for i in range(1, 6):
        init_pos, velocity = sim.add_uniform_target(i, area_size=30)
        print(
            f"Target {i}: initial position ({init_pos[0]:.1f}, {init_pos[1]:.1f}), "
            f"velocity ({velocity[0]:.2f}, {velocity[1]:.2f})"
        )

    sim.run_simulation()

    print("\n" + "=" * 50)
    sim.print_simulation_info()

    print("\nSaving simulation to file...")
    sim.save_simulation()

    print("\nCreating trajectory visualization...")
    sim.plot_trajectories()

    print("\nCreating animation...")
    sim.create_animation(interval=100, save_gif=True)

    print("\n" + "=" * 50)
    print("SPECIFIC DISTANCES AT SPECIFIC TIMES")
    print("=" * 50)

    sim.print_multiple_times([10, 25, 40])

    print("\n" + "=" * 50)
    print("INDIVIDUAL QUERIES")
    print("=" * 50)

    random_times = [random.randint(0, 50) for _ in range(3)]
    for time in random_times:
        distance1 = sim.get_distance(1, 1, time)
        distance2 = sim.get_distance(2, 3, time)
        print(
            f"Time t={time}: Sensor1->Target1: {distance1:.2f}, Sensor2->Target3: {distance2:.2f}"
        )

    print("\n" + "=" * 50)
    print("DEMONSTRATING LOADING SIMULATION")
    print("=" * 50)

    new_sim = Simulation(duration=10, time_step=1.0)
    new_sim.load_simulation()
    new_sim.print_simulation_info()
