import numpy as np
import matplotlib.colors as mcolors
import os
import random
import math
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, PillowWriter
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class target:
    id: int
    initial_position: Tuple[float, float]
    velocity: Tuple[float, float]  # (vx, vy)


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
        self.targets: List[target] = []
        self.simulation_data: Dict = {}
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def add_sensor(self, sensor_id: int, position: Tuple[float, float]):
        """Добавить сенсор в симуляцию"""
        self.sensors.append(Sensor(sensor_id, position))

    def add_target(
        self,
        obj_id: int,
        initial_position: Tuple[float, float],
        velocity: Tuple[float, float],
    ):
        """Добавить объект в симуляцию"""
        self.targets.append(target(obj_id, initial_position, velocity))

    def add_uniform_target(self, obj_id: int, area_size: float = 50):
        """Добавить объект со случайными параметрами"""
        # Случайная начальная позиция
        initial_x = random.uniform(-area_size, area_size)
        initial_y = random.uniform(-area_size, area_size)

        # Случайная скорость
        speed = random.uniform(0.5, 3.0)
        angle = random.uniform(0, 2 * math.pi)
        vx = speed * math.cos(angle)
        vy = speed * math.sin(angle)

        self.targets.append(target(obj_id, (initial_x, initial_y), (vx, vy)))
        return (initial_x, initial_y), (vx, vy)

    def add_uniform_sensor(self, sensor_id: int, area_size: float = 50):
        """Добавить сенсор со случайной позицией"""
        pos_x = random.uniform(-area_size, area_size)
        pos_y = random.uniform(-area_size, area_size)

        self.sensors.append(Sensor(sensor_id, (pos_x, pos_y)))
        return (pos_x, pos_y)

    def calculate_distance(
        self, pos1: Tuple[float, float], pos2: Tuple[float, float]
    ) -> float:
        """Вычислить евклидово расстояние между двумя точками"""
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def get_target_position(self, obj: target, time: float) -> Tuple[float, float]:
        """Получить позицию объекта в заданный момент времени"""
        x = obj.initial_position[0] + obj.velocity[0] * time
        y = obj.initial_position[1] + obj.velocity[1] * time
        return (x, y)

    def run_simulation(self):
        """Запустить симуляцию - теперь только сохраняем базовую информацию"""
        time_points = np.arange(0, self.duration + self.time_step, self.time_step)

        # Сохраняем только временные точки и базовую информацию
        self.simulation_data = {
            "time_points": time_points,
            "sensors": {sensor.id: sensor for sensor in self.sensors},
            "targets": {target.id: target for target in self.targets},
        }

    def get_distance(self, sensor_id: int, target_id: int, time: float) -> float:
        """Получить расстояние от сенсора до объекта в заданный момент времени (вычисляется на лету)"""
        if time < 0 or time > self.duration:
            raise ValueError(
                f"Время {time} вне диапазона симуляции [0, {self.duration}]"
            )

        sensor = self.simulation_data["sensors"][sensor_id]
        target_obj = self.simulation_data["targets"][target_id]

        target_pos = self.get_target_position(target_obj, time)
        distance = self.calculate_distance(sensor.position, target_pos)

        return distance

    def get_target_position_at_time(
        self, target_id: int, time: float
    ) -> Tuple[float, float]:
        """Получить координаты объекта в заданный момент времени (вычисляется на лету)"""
        if time < 0 or time > self.duration:
            raise ValueError(
                f"Время {time} вне диапазона симуляции [0, {self.duration}]"
            )

        target_obj = self.simulation_data["targets"][target_id]
        return self.get_target_position(target_obj, time)

    def print_distances_at_time(self, time: float):
        """Вывести все расстояния в конкретный момент времени"""
        print(f"\n=== Расстояния в момент времени t={time} секунд ===")
        for sensor in self.sensors:
            for obj in self.targets:
                distance = self.get_distance(sensor.id, obj.id, time)
                obj_pos = self.get_target_position_at_time(obj.id, time)
                print(f"Сенсор {sensor.id} -> Объект {obj.id}: {distance:.2f} единиц")
                print(
                    f"  Позиция объекта {obj.id}: ({obj_pos[0]:.1f}, {obj_pos[1]:.1f})"
                )

    def print_multiple_times(self, times: List[float]):
        """Вывести расстояния для нескольких моментов времени"""
        for time in times:
            self.print_distances_at_time(time)

    def plot_trajectories(self, save_file: bool = True):
        """Визуализировать траектории движения объектов и позиции сенсоров"""
        plt.figure(figsize=(12, 10))

        # Цвета для объектов и сенсоров
        obj_colors = list(mcolors.TABLEAU_COLORS.keys())
        sensor_colors = ["red", "green", "blue", "purple", "orange", "brown"]

        # Рисуем траектории объектов
        for i, obj in enumerate(self.targets):
            color = obj_colors[i % len(obj_colors)]

            # Вычисляем позиции на лету для построения траекторий
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
                label=f"Объект {obj.id}",
                alpha=0.7,
            )

            # Начальная и конечная точки
            plt.scatter(
                x_vals[0],
                y_vals[0],
                color=color,
                s=100,
                marker="o",
                edgecolors="black",
                zorder=5,
                label=f"Объект {obj.id} старт",
            )
            plt.scatter(
                x_vals[-1],
                y_vals[-1],
                color=color,
                s=100,
                marker="s",
                edgecolors="black",
                zorder=5,
                label=f"Объект {obj.id} финиш",
            )

            # Стрелка направления
            if len(positions) > 1:
                mid_idx = len(positions) // 2
                plt.annotate(
                    "",
                    xy=positions[mid_idx + 1],
                    xytext=positions[mid_idx],
                    arrowprops=dict(arrowstyle="->", color=color, lw=2),
                )

        # Рисуем сенсоры
        for i, sensor in enumerate(self.sensors):
            color = sensor_colors[i % len(sensor_colors)]
            plt.scatter(
                sensor.position[0],
                sensor.position[1],
                color=color,
                s=200,
                marker="^",
                label=f"Сенсор {sensor.id}",
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

        plt.xlabel("X координата")
        plt.ylabel("Y координата")
        plt.title("Траектории движения объектов и позиции сенсоров")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis("equal")
        plt.tight_layout()

        if save_file:
            filename = os.path.join(self.output_dir, "trajectories.png")
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"Траектории сохранены в: {filename}")

        plt.show()

    def create_animation(self, interval: int = 100, save_gif: bool = True):
        """Создать анимацию движения объектов"""
        fig, ax = plt.subplots(figsize=(12, 10))
        time_points = self.simulation_data["time_points"]

        # Настройка графика
        ax.set_xlabel("X координата")
        ax.set_ylabel("Y координата")
        ax.set_title("Анимация движения объектов")
        ax.grid(True, alpha=0.3)

        # Определяем границы для стабильного отображения
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

        # Подготавливаем элементы анимации
        obj_points = []
        obj_trails = []
        obj_colors = list(mcolors.TABLEAU_COLORS.keys())

        # Создаем следы и точки для объектов
        for i, obj in enumerate(self.targets):
            color = obj_colors[i % len(obj_colors)]
            (trail,) = ax.plot([], [], color=color, linewidth=2, alpha=0.5)
            (point,) = ax.plot(
                [], [], color=color, marker="o", markersize=10, markeredgecolor="black"
            )
            obj_trails.append(trail)
            obj_points.append(point)

            # Подписываем объекты
            ax.text(
                obj.initial_position[0],
                obj.initial_position[1],
                f"O{obj.id}",
                fontsize=10,
                fontweight="bold",
            )

        # Рисуем сенсоры (статические)
        sensor_colors = ["red", "green", "blue", "purple", "orange", "brown"]
        for i, sensor in enumerate(self.sensors):
            color = sensor_colors[i % len(sensor_colors)]
            ax.scatter(
                sensor.position[0],
                sensor.position[1],
                color=color,
                s=150,
                marker="^",
                label=f"Сенсор {sensor.id}",
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

        # Текст времени
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
            time_text.set_text(f"Время: {current_time:.1f} сек")

            for i, obj in enumerate(self.targets):
                # Получаем позиции до текущего момента (вычисляем на лету)
                trail_x = []
                trail_y = []
                for t in time_points[: frame + 1]:
                    pos = self.get_target_position(obj, t)
                    trail_x.append(pos[0])
                    trail_y.append(pos[1])

                # Обновляем след
                obj_trails[i].set_data(trail_x, trail_y)

                # Обновляем текущую позицию
                current_pos = self.get_target_position(obj, current_time)
                obj_points[i].set_data([current_pos[0]], [current_pos[1]])

            return obj_trails + obj_points + [time_text]

        # Создаем анимацию
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

        # Сохраняем GIF
        if save_gif:
            try:
                gif_filename = os.path.join(self.output_dir, "animation.gif")
                anim.save(
                    gif_filename, writer=PillowWriter(fps=1000 // interval), dpi=100
                )
                print(f"Анимация сохранена в: {gif_filename}")
            except Exception as e:
                print(f"Ошибка при сохранении GIF: {e}")

        plt.show()

        return anim

    def print_simulation_info(self):
        """Вывести информацию о симуляции"""
        print(f"Длительность симуляции: {self.duration} секунд")
        print(f"Шаг времени: {self.time_step} секунд")
        print(f"Количество сенсоров: {len(self.sensors)}")
        print(f"Количество объектов: {len(self.targets)}")
        print("\nСенсоры:")
        for sensor in self.sensors:
            print(
                f"  Сенсор {sensor.id}: позиция ({sensor.position[0]:.1f}, {sensor.position[1]:.1f})"
            )
        print("\nОбъекты:")
        for obj in self.targets:
            print(
                f"  Объект {obj.id}: начальная позиция ({obj.initial_position[0]:.1f}, {obj.initial_position[1]:.1f}), "
                f"скорость ({obj.velocity[0]:.2f}, {obj.velocity[1]:.2f})"
            )


# Пример использования со случайными параметрами
if __name__ == "__main__":
    # Создаем симуляцию
    sim = Simulation(duration=50, time_step=1.0, output_dir="simulation_results")

    # Добавляем случайные сенсоры
    print("Добавляем случайные сенсоры...")
    for i in range(1, 4):
        pos = sim.add_uniform_sensor(i, area_size=30)
        print(f"Сенсор {i}: позиция ({pos[0]:.1f}, {pos[1]:.1f})")

    # Добавляем случайные объекты
    print("\nДобавляем случайные объекты...")
    for i in range(1, 6):
        init_pos, velocity = sim.add_uniform_target(i, area_size=30)
        print(
            f"Объект {i}: начальная позиция ({init_pos[0]:.1f}, {init_pos[1]:.1f}), "
            f"скорость ({velocity[0]:.2f}, {velocity[1]:.2f})"
        )

    # Запускаем симуляцию (теперь это легковесная операция)
    sim.run_simulation()

    # Выводим информацию о симуляции
    print("\n" + "=" * 50)
    sim.print_simulation_info()

    # Визуализируем траектории
    print("\nСоздаем визуализацию траекторий...")
    sim.plot_trajectories()

    # Создаем анимацию
    print("\nСоздаем анимацию...")
    sim.create_animation(interval=100, save_gif=True)

    # Примеры запросов конкретных расстояний
    print("\n" + "=" * 50)
    print("КОНКРЕТНЫЕ РАССТОЯНИЯ В КОНКРЕТНЫЕ МОМЕНТЫ ВРЕМЕНИ")
    print("=" * 50)

    # Конкретные моменты времени
    sim.print_multiple_times([10, 25, 40])

    # Отдельные запросы
    print("\n" + "=" * 50)
    print("ОТДЕЛЬНЫЕ ЗАПРОСЫ")
    print("=" * 50)

    # Случайные моменты времени для демонстрации
    random_times = [random.randint(0, 50) for _ in range(3)]
    for time in random_times:
        distance1 = sim.get_distance(1, 1, time)
        distance2 = sim.get_distance(2, 3, time)
        print(
            f"Время t={time}: Сенсор1->Объект1: {distance1:.2f}, Сенсор2->Объект3: {distance2:.2f}"
        )
