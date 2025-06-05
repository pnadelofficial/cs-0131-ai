#
# Behavior Tree framework for A1 Behavior trees assignment.
# CS 131 - Artificial Intelligence
#
# Version 2.0.1 - copyright (c) 2023-2024 Santini Fabrizio. All rights reserved.
#

import bt_library as btl
from ..globals import HOME_PATH, CURRENT_COORDINATES, HOME_COORDINATES

import turtle
import time

class FindHome(btl.Task):
    """
    Implementation of the Task "Find Home".
    """
    def run(self, blackboard: btl.Blackboard) -> btl.ResultEnum:
        self.print_message("Looking for a home")

        # blackboard.set_in_environment(HOME_PATH, "Up Left Left Up Right")
        curr_coords = blackboard.get_in_environment(CURRENT_COORDINATES, (0, 0))
        home_coords = blackboard.get_in_environment(HOME_COORDINATES, (150, -150))
        dist_x = home_coords[0] - curr_coords[0]
        direction_x = "Left" if dist_x > 0 else "Right"
        dist_y = home_coords[1] - curr_coords[1]
        direction_y = "Up" if dist_y > 0 else "Down"
        path_string = f"{dist_x} units {direction_x}, {abs(dist_y)} units {direction_y}"
        blackboard.set_in_environment(HOME_PATH, path_string)

        turtle.forward(dist_x)
        time.sleep(1)
        if dist_y > 0:
            turtle.left(-90)
        else:
            turtle.left(90)
        turtle.forward(dist_y)
        blackboard.set_in_environment(CURRENT_COORDINATES, home_coords)

        return self.report_succeeded(blackboard)
