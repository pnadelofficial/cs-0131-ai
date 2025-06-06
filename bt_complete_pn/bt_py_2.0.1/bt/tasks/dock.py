import bt_library as btl
from ..globals import BATTERY_LEVEL
import random
import time

class Dock(btl.Task):
    """
    Implementation of the Task "Go Home".
    """
    def run(self, blackboard: btl.Blackboard) -> btl.ResultEnum:
        self.print_message("Docking and charging")
        current_battery_level = blackboard.get_in_environment(BATTERY_LEVEL, 0)
        self.print_message(f"Current battery level: {current_battery_level}%")
        for _ in range(100-current_battery_level, 100, 10):
            if random.random() < 0.2:  # simulate a charging
                time.sleep(.5)
            self.print_message("Charging...")
        self.print_message("Battery fully charged!")
        blackboard.set_in_environment(BATTERY_LEVEL, 100)  # assuming one time step to recharge -- doesn't always reach 100%

        return self.report_succeeded(blackboard)