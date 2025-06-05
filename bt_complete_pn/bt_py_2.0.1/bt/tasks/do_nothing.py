import bt_library as btl

class DoNothing(btl.Task):
    """
    Implementation of the Task "Do Nothing".
    This will always return a running status.
    """
    def run(self, blackboard: btl.Blackboard) -> btl.ResultEnum:
        self.print_message("Doin' nuffin...")
        return self.report_running(blackboard, 0)