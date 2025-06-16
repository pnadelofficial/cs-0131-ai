import argparse
import heapq
import random
from typing import Iterable, Self
import time
import copy

GOLD = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

class TreeNode:
    """
    TreeNode Class
    --------------
    This class represents a single stack configuation, along with the series of actions taken to arrive at this state. 
    """
    def __init__(self, state:Iterable, parent:Self, position:int, ucs:bool):
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
        self.ucs = ucs
    
    def __lt__(self, other):
        my_total_cost = self.total_cost()
        others_total_cost = other.total_cost()
        if my_total_cost == others_total_cost:
            return self.position < other.position
        return my_total_cost < others_total_cost

    def __gt__(self, other):
        my_total_cost = self.total_cost()
        others_total_cost = other.total_cost()
        if my_total_cost == others_total_cost:
            return self.position > other.position
        return my_total_cost > others_total_cost

    def __eq__(self, other):
        my_total_cost = self.total_cost()
        others_total_cost = other.total_cost()
        if my_total_cost == others_total_cost:
            return self.position == other.position
        return my_total_cost == others_total_cost

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
        if self.ucs:
            return 0
        gaps = 0
        for prev, curr in zip(self.state, self.state[1:]):
            if abs(curr - prev) != 1:
                gaps +=1
        return gaps

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
        
        self.state = full
        self.backward_cost = k
        # flipped_node = TreeNode(full, None, None, self.ucs)
        # flipped_node.backward_cost = k
        # return flipped_node

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
        for node in self.pq:
            if node.state == state:
                return True
        return False

class AStar:
    """
    A* Class
    --------
    This call implements the A* algorithm for a stack of pancakes.
    """
    def __init__(self, initial_state:Iterable, ucs:bool=False):
        """
        Creates a new A* instance with the initial state of the pancake stack

        Args
        ----
        initial_state: the starting, randomized state of the pancake stack
        """
        self.initial_state = initial_state
        self.length = len(initial_state)
        self.ucs = ucs

        self.visited = set() # Will keep track of the visited configurations
        self.position = 0 # Will keep track of the order of the configurations
        self.pq = PriorityQueue()

        self.root = TreeNode(initial_state, None, self.position, self.ucs)
        self.pq.push(self.root)
        self.position += 1
    
    def search(self):
        while True:
            if self.pq.is_empty():
                self.solution = False
                return 
            
            current_node = self.pq.pop()
            self.visited.add(tuple(current_node.state))

            if current_node.state == GOLD:
                self.solution = current_node
                return
            
            self.generate_successors(current_node)
    
    def generate_successors(self, current_node:TreeNode):
        for i in range(2, self.length):
            flipped = copy.deepcopy(current_node)
            flipped.flip(i)
            flipped.parent = current_node
            flipped.position = self.position

            if (tuple(flipped.state) not in self.visited) and (not self.pq.has(flipped.state)):
                self.pq.push(flipped)
            
            elif self.pq.has(flipped.state):
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
                    print(f"Current stack: {step.state}")
                print()
                print("Pancake problem solved!")

def main():
    test_stack = [i for i in range(1, 11)]
    random.shuffle(test_stack)
    print(f"Initial stack: {test_stack}")
    print()

    curr_time = time.time()

    astar = AStar(test_stack, ucs=False)
    astar.search()
    astar.print_solution()

    elapsed_time = time.time() - curr_time
    print(f"Pancake problem completed in {round(elapsed_time, 4)} seconds")

if __name__ == '__main__':
    main()