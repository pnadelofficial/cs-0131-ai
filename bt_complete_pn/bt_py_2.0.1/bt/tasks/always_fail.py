import bt_library as btl

class AlwaysFail(btl.Task):
    """
    Implementation of the Task "Always Fail".
    This will always return a failure status.
    """
    def run(self, blackboard: btl.Blackboard) -> btl.ResultEnum:
        return self.report_failed(blackboard)