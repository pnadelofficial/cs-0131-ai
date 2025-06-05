import bt_library as btl
from ..globals import DUSTY_SPOT_SENSOR

class DustySpot(btl.Condition):
    """
    Implementation of the condition "Spot Cleaning".
    """
    def run(self, blackboard: btl.Blackboard) -> btl.ResultEnum:
        self.print_message("Checking a dusty spot")

        if blackboard.get_in_environment(DUSTY_SPOT_SENSOR, False):
            return self.report_succeeded(blackboard)
        else:
            return self.report_failed(blackboard)