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
        curr_coords = blackboard.get_in_environment(CURRENT_COORDINATES, (0, 0))
        self.print_message(f"Looking for home. Current coordinates: {curr_coords}")
        
        home_coords = blackboard.get_in_environment(HOME_COORDINATES, (150, -150))
        dist_x = home_coords[0] - curr_coords[0]
        dist_y = home_coords[1] - curr_coords[1]
        direction_x = "right" if dist_x > 0 else "left"
        direction_y = "up" if dist_y > 0 else "down"
        
        self.print_message(f"Found a way home: X={dist_x} ({direction_x}), Y={dist_y} ({direction_y})")
        path_home = f"{abs(dist_x)} {direction_x},{abs(dist_y)} {direction_y}"
        blackboard.set_in_environment(HOME_PATH, path_home)
        
        # Change color
        turtle.color("red")
        # Move in X direction
        self.print_message("Moving in X direction")
        turtle.setx(home_coords[0])
        time.sleep(1)

        # Move in Y direction
        self.print_message("Moving in Y direction")
        turtle.sety(home_coords[1])
        time.sleep(1)

        # Update current coordinates in the blackboard
        blackboard.set_in_environment(CURRENT_COORDINATES, home_coords)

        return self.report_succeeded(blackboard)
