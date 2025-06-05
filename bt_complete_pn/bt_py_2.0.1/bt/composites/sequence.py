import bt_library as btl


class Sequence(btl.Composite):
    """
    Specific implementation of the sequence composite.
    """

    def __init__(self, children: btl.NodeListType):
        """
        Default constructor.

        :param children: List of children for this node
        """
        super().__init__(children)

    def run(self, blackboard: btl.Blackboard) -> btl.ResultEnum:
        """
        Execute the behavior of the node.

        :param blackboard: Blackboard with the current state of the problem
        :return: The result of the execution
        """
        for child in self.children:
            result_child = child.run(blackboard)
            if result_child != btl.ResultEnum.SUCCEEDED:
                return result_child

        return self.report_succeeded(blackboard)