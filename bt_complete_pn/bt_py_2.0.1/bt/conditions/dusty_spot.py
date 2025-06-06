import bt_library as btl
from ..globals import DUSTY_SPOT_SENSOR, CURRENT_COORDINATES
import random
import turtle

class DustySpot(btl.Condition):
    """
    Implementation of the condition "Dusty Spot".
    """
    def run(self, blackboard: btl.Blackboard) -> btl.ResultEnum:
        self.print_message("Checking a dusty spot")

        if blackboard.get_in_environment(DUSTY_SPOT_SENSOR, False):
            self.print_message("Dusty spot detected!")
            # Create random dusty spot
            dusty_spot = (random.randint(-100, 100), random.randint(-100, 100))

            # Change color
            turtle.color("blue")
            # Move to dusty spot
            self.print_message(f"Moving to dusty spot at X={dusty_spot[0]} and Y={dusty_spot[1]}")
            turtle.setx(dusty_spot[0])
            turtle.sety(dusty_spot[1])
            
            # Update current coordinates in the blackboard
            blackboard.set_in_environment(CURRENT_COORDINATES, dusty_spot)

            return self.report_succeeded(blackboard)
        else:
            return self.report_failed(blackboard)