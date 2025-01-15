import numpy as np
import matplotlib.pyplot as plt
import copy
import math
from QuinticPolynomialsPlanner.quintic_polynomials_planner import QuinticPolynomial
from CubicSpline import cubic_spline_planner

# Simulation Parameters
DT = 0.2  # time tick [s]
ROBOT_RADIUS = 2.0  # robot radius [m]

class FrenetPath:
    def __init__(self):
        self.t = []
        self.d = []
        self.s = []
        self.x = []
        self.y = []
        self.yaw = []


class ReverseParking:
    def __init__(self, parking_slot, stop_point):
        self.parking_slot = parking_slot  # [x, y, width, length]
        self.stop_point = stop_point  # [x, y]

    def generate_reverse_path(self):
        # Generate a simple reverse trajectory into the parking slot
        parking_x, parking_y = self.parking_slot[:2]
        stop_x, stop_y = self.stop_point

        # Create waypoints for reverse trajectory
        waypoints_x = [stop_x, parking_x]
        waypoints_y = [stop_y, parking_y]

        return waypoints_x, waypoints_y

    def visualize_parking(self, path, parking_slot):
        plt.cla()
        plt.plot(path.x, path.y, "-r", label="Planned Path")
        plt.scatter([self.stop_point[0]], [self.stop_point[1]], color="blue", label="Stop Point")
        plt.scatter([parking_slot[0]], [parking_slot[1]], color="green", label="Parking Slot")

        plt.legend()
        plt.axis("equal")
        plt.grid(True)
        plt.show()


class TrajectoryPlanner:
    def __init__(self, waypoints_x, waypoints_y):
        self.waypoints_x = waypoints_x
        self.waypoints_y = waypoints_y
        self.csp = cubic_spline_planner.CubicSpline2D(waypoints_x, waypoints_y)

    def generate_frenet_path(self):
        frenet_path = FrenetPath()

        s_values = np.arange(0, self.csp.s[-1], 0.1)
        for s in s_values:
            x, y = self.csp.calc_position(s)
            yaw = self.csp.calc_yaw(s)
            frenet_path.x.append(x)
            frenet_path.y.append(y)
            frenet_path.yaw.append(yaw)

        return frenet_path


def main():
    # Define parking slot and stop point
    parking_slot = [10.0, -5.0, 4.0, 6.0]  # [x, y, width, length]
    stop_point = [10.0, 5.0]  # [x, y]

    # Initialize reverse parking planner
    reverse_parking = ReverseParking(parking_slot, stop_point)

    # Generate waypoints for reverse parking
    waypoints_x, waypoints_y = reverse_parking.generate_reverse_path()

    # Plan trajectory
    trajectory_planner = TrajectoryPlanner(waypoints_x, waypoints_y)
    frenet_path = trajectory_planner.generate_frenet_path()

    # Visualize result
    reverse_parking.visualize_parking(frenet_path, parking_slot)


if __name__ == "__main__":
    main()
