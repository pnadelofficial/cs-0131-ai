import bt_library as btl
from ..globals import BATTERY_LEVEL, CURRENT_COORDINATES
import turtle

class CleanFloor(btl.Task):
    """
    Implementation of the Task "Clean Floor".
    """
    def run(self, blackboard: btl.Blackboard) -> btl.ResultEnum:
        self.print_message("Cleaning floor")
        for _ in range(3):
            self.print_message("Cleaning...")

            # Move to the center
            turtle.setx(0)
            turtle.sety(0)

            # Update current coordinates in the blackboard
            blackboard.set_in_environment(CURRENT_COORDINATES, (0, 0))

            # Change color 
            turtle.color("green")
            # Clean floor in spiral
            for i in range(120):
                turtle.circle(i, 10) # spiral pattern
            
            # Reset to middle for next cycle
            turtle.setx(0)
            turtle.sety(0)
            blackboard.set_in_environment(CURRENT_COORDINATES, (0, 0))
            
            blackboard.set_in_environment(BATTERY_LEVEL, blackboard.get_in_environment(BATTERY_LEVEL, 0) - 15) # costs 15 battery per general cleaning cycle
        self.print_message("Floor cleaned")
        return self.report_failed(blackboard)