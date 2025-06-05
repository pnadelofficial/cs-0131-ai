import bt_library as btl

class DoneSpot(btl.Task):
    """
    Implementation of the Task "Done Spot".
    """
    def run(self, blackboard: btl.Blackboard) -> btl.ResultEnum:
        self.print_message("Spot cleaning done")
        return self.report_succeeded(blackboard)  # assuming the task is always successful