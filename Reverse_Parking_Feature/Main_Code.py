import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import copy
import math
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from QuinticPolynomialsPlanner.quintic_polynomials_planner import \
    QuinticPolynomial
from QuinticPolynomialsPlanner.quintic_polynomials_planner import quintic_polynomials_planner
from CubicSpline import cubic_spline_planner
from matplotlib.patches import Rectangle
from Functions_Classes import*

from Functions_Classes import (generate_target_course, check_for_transition, Target_speed, parkx, parky, pspot_length, psport_breadth)


show_animation = True

def main():
    print(__file__ + " start!!")

    area = 35.0
    wx = [0.0, 10.0, 20.0, 40.0, 50.0, 65.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0]  # Reference Path
    wy = [0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]

    tx, ty, tyaw, tc, csp = generate_target_course(wx, wy)

    c_speed = Target_speed  # Current speed [m/s]
    c_accel = 0.0  # Current acceleration [m/s^2]
    c_d = 0.0  # Current lateral position [m]
    c_d_d = 0.0  # Current lateral speed [m/s]
    c_d_dd = 0.0  # Current lateral acceleration [m/s^2]
    s0 = 0.0  # Current course position

    current_state = 'forward'
    r_speed = Target_speed  # Reverse speed [m/s]
    r_accel = -2.0  # Reverse acceleration [m/s^2]
    r_d = 0.0  # Reverse lateral position [m]
    r_d_d = 0.0  # Reverse lateral speed [m/s]
    r_d_dd = 0.0  # Reverse lateral acceleration [m/s^2]
    rs0 = s0  # Current reverse course position

    csp.lx = [x - 2.5*math.cos(i_yaw + math.pi / 2.0) for x,i_yaw in zip(tx,tyaw) ]
    csp.ly = [y - 2.5*math.sin(i_yaw + math.pi / 2.0) for y,i_yaw in zip(ty,tyaw) ]
    csp.mx = [x + 2.5*math.cos(i_yaw + math.pi / 2.0) for x,i_yaw in zip(tx,tyaw) ]
    csp.my = [y + 2.5*math.sin(i_yaw + math.pi / 2.0) for y,i_yaw in zip(ty,tyaw) ]
    initial_lx = csp.lx.copy()
    initial_ly = csp.ly.copy()
    initial_mx = csp.mx.copy()
    initial_my = csp.my.copy()

    is_parallel_parking = pspot_length > psport_breadth

    for i in range(SIM_LOOP):

        if current_state == 'forward':
            s0, c_d, c_d_d, c_d_dd, c_speed, c_accel, path, ego_x, ego_y,fplist = forward_parking(
                csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, parkx, parky, Target_speed
            )

            if s0 >= parkx:
                current_state = 'interme'
                plt.pause(0.02)
                print("Parking Spot Detected!!!! Initiating Parking Maneuver")
                continue

        elif current_state == 'interme':
            s0, c_d, c_d_d, c_d_dd, c_speed, c_accel, path, ego_x, ego_y,fplist = forward_parking(
                csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, parkx, parky, Target_speed
            )

            if c_speed < 0:
                plt.pause(0.04)
                current_state = 'reverse'
                xvalue, yvalue = csp.calc_position(path.s[1])

                if is_parallel_parking:
                    x1, y1 = generate_s_trajectory(xvalue, yvalue, parkx, parky, pspot_length, psport_breadth)
                    s1, s2 = straight_into_reverse(parkx, parky, pspot_length, psport_breadth)
                    new_wx = list(x1) + s1
                    new_wy = list(y1) + s2
                    tx, ty, tyaw, tc, csp = generate_target_course(new_wx, new_wy)

                elif parky < 0:
                    newx1 = [ego_x, ego_x - 1.0, parkx, parkx, parkx, parkx]
                    newy1 = [ego_y, ego_y, parky + 4, parky + 2, parky - 1.0, parky - 2.0]
                    tx, ty, tyaw, tc, csp = generate_target_course(newx1, newy1)

                elif parky > 0:
                    newwx1 = [ego_x, ego_x - 1.0, parkx, parkx, parkx, parkx, parkx]
                    newwy1 = [ego_y, ego_y, parky - 4, parky - 2, parky, parky + 1, parky + 1.5]
                    tx, ty, tyaw, tc, csp = generate_target_course(newwx1, newwy1)

                continue

        elif current_state == 'reverse':
            rs0, r_d, r_d_d, r_d_dd, r_speed, r_accel, path, ego_x, ego_y,fplist = reverse_parking(
                csp, rs0, r_speed, r_accel, r_d, r_d_d, r_d_dd, parkx, parky, Target_speed
            )

            if np.hypot(path.x[1] - tx[-1], path.y[1] - ty[-1]) <= 2.5:
                break

        #Visualizing the ego_vehicle and the parking simulation
        if show_animation:
            plt.clf()
            plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
    
    # Plotting road boundaries & parking trajectory at a later time
            plt.cla()
            plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(tx, ty)    
            csp.lx = [x - 2.5*math.cos(i_yaw + math.pi / 2.0) for x,i_yaw in zip(tx,tyaw) ]
            csp.ly = [y - 2.5*math.sin(i_yaw + math.pi / 2.0) for y,i_yaw in zip(ty,tyaw) ]
            csp.mx = [x + 2.5*math.cos(i_yaw + math.pi / 2.0) for x,i_yaw in zip(tx,tyaw) ]
            csp.my = [y + 2.5*math.sin(i_yaw + math.pi / 2.0) for y,i_yaw in zip(ty,tyaw) ]  
            csp.ax = [x - 2.5*math.cos(i_yaw + math.pi / 2.0) for x,i_yaw in zip(tx,tyaw) ]
            csp.ay = [y - 2.5*math.sin(i_yaw + math.pi / 2.0) for y,i_yaw in zip(ty,tyaw) ]
            csp.bx = [x + 2.5*math.cos(i_yaw + math.pi / 2.0) for x,i_yaw in zip(tx,tyaw) ]
            csp.by = [y + 2.5*math.sin(i_yaw + math.pi / 2.0) for y,i_yaw in zip(ty,tyaw) ]   
            plt.plot(initial_lx, initial_ly, label='Road Boundaries', color='black')
            plt.plot(initial_mx, initial_my, label='Road Boundaries', color='black')       
            plt.plot(csp.mx, csp.my, '--')
            plt.plot(csp.lx, csp.ly, '--')
            plt.plot(csp.ax, csp.ay, '--')
            plt.plot(csp.bx, csp.by, '--')
    
    # Plot parking spot location
            plt.gca().add_patch(Rectangle((parkx - pspot_length / 2, parky - psport_breadth / 2),
            pspot_length, psport_breadth,linewidth=2, edgecolor='blue', facecolor='blue', label="Parking Spot"))
            plt.text(parkx, parky,'Parking Spot',color='blue', fontsize=10, ha='center', va='center', bbox=dict(facecolor='none', alpha=0.7, edgecolor='none', pad = 0.02))
    
   
    # Plot ego vehicle
            px, py, yaw = path.x[1], path.y[1], path.yaw[1]
            ego_yaw = path.yaw[1] * 180 /np.pi
            ego_vehicle = Rectangle((px - ego_length / 2.0, py - ego_width / 2.0),ego_length, ego_width, facecolor='red', edgecolor='black', label="Ego Vehicle")
            t = transforms.Affine2D().rotate_deg_around(path.x[1], path.y[1], ego_yaw) + plt.gca().transData
            ego_vehicle.set_transform(t)
            plt.gca().add_patch(ego_vehicle)
            plt.text(px, py + 9,f"Speed: {c_speed * 3.6:.2f} km/h \nState: {current_state}",color='black', fontsize=10, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
     # Plot candidate paths
           
            for i, fp in enumerate(fplist[:150]):
                plt.plot(fp.x, fp.y, '--', alpha=0.5, label=f"Frenet Paths" if i == 0 else "")
    
    # Configure plot appearance
            plt.xlim(path.x[1] - area, path.x[1] + area)
            plt.ylim(path.y[1] - area, path.y[1] + area)
            plt.title("Reverse Parking Simulation", fontsize=16)
            plt.xlabel("X-axis (m)", fontsize=12)
            plt.ylabel("Y-axis (m)", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(loc='upper right', fontsize=10)
    
            plt.pause(0.02)

    
    print("Finish")

    if show_animation:  # pragma: no cover
        plt.grid(True)
        plt.pause(0.02)
        plt.show()


if __name__ == '__main__':
    main()
