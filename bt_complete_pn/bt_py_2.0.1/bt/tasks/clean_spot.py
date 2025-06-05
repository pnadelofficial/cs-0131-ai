import bt_library as btl
from ..globals import BATTERY_LEVEL, CURRENT_COORDINATES
import random
import turtle
import time

class CleanSpot(btl.Task):
    """
    Implementation of the Task "Clean Spot".
    """
    def run(self, blackboard: btl.Blackboard) -> btl.ResultEnum:
        self.print_message("Cleaning spot")
        
        spot_coords = (random.randint(-250, 250), random.randint(-250, 250))
        self.print_message(f"Dirty spot at {spot_coords}")

        curr_coords = blackboard.get_in_environment(CURRENT_COORDINATES, (0,0))
        print(f"Current coords: {curr_coords}")
        dist_x = spot_coords[0] - curr_coords[0]
        direction_x = "Left" if dist_x > 0 else "Right"
        dist_y = spot_coords[1] - curr_coords[1]
        direction_y = "Up" if dist_y > 0 else "Down"
        path_string = f"{dist_x} units {direction_x}, {abs(dist_y)} units {direction_y}"
        self.print_message(path_string)
        
        turtle.forward(dist_x)
        time.sleep(1)
        if dist_y > 0:
            turtle.left(-90)
        else:
            turtle.left(90)
        turtle.forward(dist_y)
        blackboard.set_in_environment(CURRENT_COORDINATES, spot_coords)

        for _ in range(5): 
            self.print_message("Cleaning...")
            blackboard.set_in_environment(BATTERY_LEVEL, blackboard.get_in_environment(BATTERY_LEVEL, 0) - 1)
        return self.report_succeeded(blackboard)
