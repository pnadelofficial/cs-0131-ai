#
# Behavior Tree framework for A1 Behavior trees assignment.
# CS 131 - Artificial Intelligence
#
# Version 2.0.1 - copyright (c) 2023-2024 Santini Fabrizio. All rights reserved.
#

from .tree_node import TreeNode
import time

class Decorator(TreeNode):
    """
    The generic definition of the decorator node class.
    """
    __child: TreeNode  # Child associated with this decorator

    def __init__(self, child: TreeNode):
        """
        Default constructor.

        :param child: Child for this node
        """
        super().__init__()

        self.__child = child

    @property
    def child(self) -> TreeNode:
        """
        :return: Return the child associated with this decorator
        """
        return self.__child

class TimerDecorator(Decorator):
    """
    Specific implementation of the timer decorator.
    """
    def __init__(self, time_to_run: int, child: TreeNode):
        """
        Default constructor.

        :param time: Duration of the timer [counts]
        :param child: Child associated to the decorator
        """
        super().__init__(child)
        self.time_to_run = time_to_run
        self.start_time = time.time()
    
    def run(self):
        """
        Execute the behavior of the node.

        :return: The result of the execution
        """
        elapsed_time = time.time() - self.start_time
        
        try:
            if elapsed_time >= self.time_to_run:
                return True
            else:
                return self.child.run()
        except Exception as e:
            print(f"Error during execution: {e}")
            return False
