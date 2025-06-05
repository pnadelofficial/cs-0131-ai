import bt_library as btl

class Priority(btl.Composite):
    """
    Specific implementation of the priority composite.
    """

    def __init__(self, children: btl.NodeListType, priorities: list):
        """
        Default constructor.

        :param children: List of children for this node
        """
        super().__init__(children)
        self.priorities = priorities

    def run(self, blackboard: btl.Blackboard) -> btl.ResultEnum:
        """
        Execute the behavior of the node.

        :param blackboard: Blackboard with the current state of the problem
        :param priorities: List of priorities for the children
        :return: The result of the execution
        """
        child_priorities = sorted(zip(self.children, self.priorities), key=lambda x: x[1])
        for child_position, (child, prio) in enumerate(child_priorities):
            self.print_message(f"Running child {child.pretty_id} with priority: {prio}")
            result_child = child.run(blackboard)
            # if result_child == btl.ResultEnum.SUCCEEDED:
            #     return result_child

            # if result_child == btl.ResultEnum.RUNNING:
            #     return result_child

        return result_child# self.report_failed(blackboard, 0)