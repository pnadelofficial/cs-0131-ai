import bt_library as btl

class UntilFail(btl.Decorator):
    """
    Implementation of the UntilFail decorator.
    This decorator will keep executing its child until it fails.
    """
    
    def run(self, blackboard: btl.Blackboard) -> btl.ResultEnum:
        self.print_message("Executing UntilFail decorator")
       
        while True:
            result = self.child.run(blackboard)
            if result == btl.ResultEnum.FAILED:
                return self.report_failed(blackboard)
            elif result == btl.ResultEnum.SUCCEEDED:
                continue  # keep executing until it fails
            else:
                return self.report_running(blackboard)