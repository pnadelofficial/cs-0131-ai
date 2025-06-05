import bt_library as btl
from ..globals import GENERAL_CLEANING

class GeneralCleaning(btl.Condition):
    """
    Implementation of the condition "General Cleaning".
    """
    def run(self, blackboard: btl.Blackboard) -> btl.ResultEnum:
        self.print_message("Checking user input for General Cleaning")

        if blackboard.get_in_environment(GENERAL_CLEANING, False):
            return self.report_succeeded(blackboard)
        else:
            return self.report_failed(blackboard)