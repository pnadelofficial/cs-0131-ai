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
        for i, s in enumerate(self.state):
            if i+1 != len(self.state): # Avoids IndexError for last pancake
                this_pancake = s
                # Check gaps below
                next_pancake = self.state[i+1]
                if abs(this_pancake - next_pancake) > 1: # Gap! Not adjacent
                    gaps += 1
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
        return TreeNode(full, None, None)

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
        self.pq = []

        self.root = TreeNode(initial_state, None, self.position)
        heapq.heappush(self.pq, self.root)
        self.position += 1
    
    def search(self):
        done = True
        while done:
            if len(self.pq) == 0:
                print("All avenues checked. No solution possible.")
                return
            current_node = heapq.heappop(self.pq)
            self.visited.add(tuple(current_node.state))
            if current_node.heuristic() == 0:
                print("Solution found.")
                self.print_solution()
                return
            self.generate_successors(current_node)
    
    def generate_successors(self, current_node:TreeNode):
        for i in range(1, self.length):
            flipped = current_node.flip(i)
            flipped.parent = current_node
            flipped.position = self.position

            if (tuple(flipped.state) not in self.visited) and (flipped.state not in [n.state for n in self.pq]):
                heapq.heappush(self.pq, flipped)
            elif flipped.state in [n.state for n in self.pq]:
                for node in self.pq:
                    if (node.state == flipped.state) and (node.total_cost() < flipped.total_cost()):
                        self.pq[self.pq.index(node)] = flipped
            
            self.position += 1


    def print_solution():
        pass

def main():
    test_stack = [i for i in range(1, 11)]
    random.shuffle(test_stack)

    astar = AStar(test_stack)
    astar.search()

if __name__ == '__main__':
    main()