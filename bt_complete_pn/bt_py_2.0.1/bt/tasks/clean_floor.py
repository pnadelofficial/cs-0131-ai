import bt_library as btl
from ..globals import BATTERY_LEVEL

class CleanFloor(btl.Task):
    """
    Implementation of the Task "Clean Floor".
    """
    def run(self, blackboard: btl.Blackboard) -> btl.ResultEnum:
        self.print_message("Cleaning floor")
        for _ in range(10):
            self.print_message("Cleaning...")
            blackboard.set_in_environment(BATTERY_LEVEL, blackboard.get_in_environment(BATTERY_LEVEL, 0) - 1)
        self.print_message("Floor cleaned")
        return self.report_failed(blackboard)