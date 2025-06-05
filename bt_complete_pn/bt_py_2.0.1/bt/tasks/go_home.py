import bt_library as btl
from ..globals import HOME_PATH, BATTERY_LEVEL

class GoHome(btl.Task):
    """
    Implementation of the Task "Go Home".
    """
    def run(self, blackboard: btl.Blackboard) -> btl.ResultEnum:
        self.print_message("Going for home")

        for direction in blackboard.get_in_environment(HOME_PATH, "").split(","):
            self.print_message(f"Moving {direction.strip()}")
        self.print_message("Reached home!")
        blackboard.set_in_environment(HOME_PATH, "")
        blackboard.set_in_environment(BATTERY_LEVEL, blackboard.get_in_environment(BATTERY_LEVEL, 0) - 1) # takes batter to go home too

        return self.report_succeeded(blackboard)
