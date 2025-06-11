import argparse
import heapq
import random
from typing import Iterable # , Self

class TreeNode:
    """
    TreeNode Class
    ~~~~~~~~~~~~~~
    This class represents a single stack configuation, along with the series of actions taken to arrive at this state. 
    """
    def __init__(self, state:Iterable, parent, position:int):
        """
        Creates a new TreeNode object

        Args 
        ----
        state: Any iterable that can represent a configuration of the pancake stack
        parent: The TreeNode preceding the current TreeNode
        position: The amount of flips needed to get to have gotten to this TreeNode
        """
        self.state = state
        self.parent = parent
        self.position = position

        self.backward_cost = 0 # Setting this to be 0 by default, but will be filled in later
    
    def __lt__(self, other):
        my_total_cost = self.total_cost()
        others_total_cost = other.total_cost()
        if my_total_cost == others_total_cost:
            return self.position < other.position
        return my_total_cost < others_total_cost


    # less than
    # greater than
    # equals operators

    def total_cost(self):
        """
        Calculates the total cost for a flip -- the sum of the heuristic (forward cost) and the backward cost
        
        Returns
        -------
        The total cost of the flip
        """
        return self.heuristic() + self.backward_cost

    def heuristic(self):
        """
        Gap heuristic, taken from "Landmark Heuristics for the Pancake Problem" by Malte Helmert, in which we count the number of non-adjecent stack positions

        Returns
        -------
        The gap heuristic for the state
        """
        gaps = 0
        for prev, curr in zip(self.state, self.state[1:]):
            if abs(curr - prev) != 1:
                gaps +=1
        return gaps
        # for i, s in enumerate(self.state):
        #     if i+1 != len(self.state): # Avoids IndexError for last pancake
        #         this_pancake = s
        #         # Check gaps below
        #         next_pancake = self.state[i+1]
        #         if abs(this_pancake - next_pancake) != 1: # Gap! Not adjacent
        #             gaps += 1
        # return gaps

    def flip(self, k):
        """
        Flips the first k elements of the stack.

        Returns
        -------
        A new state of the stack with the top k elements flipped
        """
        flipped = self.state[:k][::-1] # Cutting the first k elements and reversing them
        rest = self.state[k:] # Keeping the rest of the stack unchanged
        full = flipped + rest # Concatenating the flipped part with the rest of the stack
        return TreeNode(full, None, None)

class PriorityQueue:
    """
    PriorityQueue Class
    -------------------
    This class implements a priority queue for the A* algorithm, using a min-heap.
    """
    def __init__(self):
        self.pq = []

    def push(self, item):
        heapq.heappush(self.pq, item)

    def pop(self):
        return heapq.heappop(self.pq)

    def is_empty(self):
        return len(self.pq) == 0
    
    def replace(self, item):
        """
        Replaces the top item in the priority queue with a new item if the new item has a lower total cost.
        
        Args
        ----
        item: The new item to be added to the priority queue
        """
        if not self.is_empty() and item < self.pq[0]:
            heapq.heapreplace(self.pq, item)
        else:
            self.push(item)
        
    def has(self, state):
        """
        Checks if the priority queue contains a state.
        
        Args
        ----
        state: The state to check for in the priority queue
        
        Returns
        -------
        True if the state is in the priority queue, False otherwise
        """
        return any(node.state == state for node in self.pq)

class AStar:
    """
    A* Class
    --------
    This call implements the A* algorithm for a stack of pancakes.
    """
    def __init__(self, initial_state:Iterable):
        """
        Creates a new A* instance with the initial state of the pancake stack

        Args
        ----
        initial_state: the starting, randomized state of the pancake stack
        """
        self.initial_state = initial_state
        self.length = len(initial_state)
        self.visited = set() # Will keep track of the visited configurations
        self.position = 0 # Will keep track of the order of the configurations
        self.pq = PriorityQueue()

        self.root = TreeNode(initial_state, None, self.position)
        self.pq.push(self.root)
        self.position += 1
    
    def search(self):
        while True:
            if self.pq.is_empty():
                self.solution = False
                return 
            
            current_node = self.pq.pop()
            self.visited.add(tuple(current_node.state))

            if current_node.heuristic() == 0:
                self.solution = current_node
                return
            
            self.generate_successors(current_node)
    
    def generate_successors(self, current_node:TreeNode):
        for i in range(1, self.length+1):
            flipped = current_node.flip(i)
            flipped.parent = current_node
            flipped.position = self.position

            if (tuple(flipped.state) not in self.visited) and (not self.pq.has(flipped.state)):
                self.pq.push(flipped)
            
            if self.pq.has(flipped.state):
                self.pq.replace(flipped)
            
            self.position += 1

    def print_solution(self):
        if not self.solution:
            print("No solution found")
        elif isinstance(self.solution, TreeNode):
            steps = []
            current_node = self.solution

            try:
                while current_node.state is not None:
                    steps.append(current_node)
                    current_node = current_node.parent
            except AttributeError:
                for step in steps[::-1]:
                    print(step.state)

def main():
    test_stack = [i for i in range(1, 11)]
    random.shuffle(test_stack)
    print(f"Initial stack: {test_stack}")

    astar = AStar(test_stack)
    astar.search()
    astar.print_solution()

if __name__ == '__main__':
    main()