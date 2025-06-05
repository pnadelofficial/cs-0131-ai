import bt_library as btl
from ..globals import GENERAL_CLEANING

class DoneGeneral(btl.Task):
    """
    Implementation of the Task "Clean Spot".
    """
    def run(self, blackboard: btl.Blackboard) -> btl.ResultEnum:
        self.print_message("Done with general cleaning")
        blackboard.set_in_environment(GENERAL_CLEANING, False)
        return self.report_succeeded(blackboard) # assuming the task is completed successfully