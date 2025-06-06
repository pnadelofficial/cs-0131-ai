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
        
        spot_coords = (random.randint(-100, 100), random.randint(-100, 100)) # creating a random spot
        self.print_message(f"Dirty spot at {spot_coords}")

        curr_coords = blackboard.get_in_environment(CURRENT_COORDINATES, (0,0))
        print(f"Current coords: {curr_coords}")

        # Change color
        turtle.color("blue")
        # Move in X direction
        self.print_message("Moving in X direction")
        turtle.setx(spot_coords[0])
        time.sleep(1)

        # Move in Y direction
        self.print_message("Moving in Y direction")
        turtle.sety(spot_coords[1])
        time.sleep(1)

        # Update current coordinates in the blackboard
        blackboard.set_in_environment(CURRENT_COORDINATES, spot_coords)

        # Cleaing spot
        for _ in range(5): 
            self.print_message("Cleaning...")
            blackboard.set_in_environment(BATTERY_LEVEL, blackboard.get_in_environment(BATTERY_LEVEL, 0) - 1)
        return self.report_succeeded(blackboard)
