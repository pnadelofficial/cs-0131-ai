import argparse
import random 
import heapq
import time
from typing import Tuple, Self, Union

GOLD = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

def gap_heuristic(state:Tuple) -> int:
    """
    Heuristic taken from Landmark Heuristics for the Pancake Problem by Malte Helmert
    Args:
        state (tuple): Order of pancakes
    Returns:
        int: The gap heuristic, i.e. the number of gaps between consecutive pancakes
    """
    gaps = 0
    for prev, curr in zip(state, state[1:]): # pythonic way to compare pancake placements in a stack
        if abs(curr - prev) != 1: # gap found
            gaps +=1 # increment number of gaps found
    return gaps

def no_heuristic(state:Tuple) -> int:
    """
    Uniform Cost Search heuristic -- a heuristic that always returns 0
    Args:
        state (tuple): Order of pancakes
    Returns:
        int: Always returns zero as the UCS algorithm ignores heuristics
    """
    return 0

class PancakeStack:
    """
    Abstraction for representing a possible stack of pancakes. This will be a single node in a tree which will then be traveresed using the A* algorithm. 
    """
    def __init__(self, state:Tuple, parent:Self, added:int, b_cost:int, heuristic:callable=gap_heuristic) -> None:
        """
        Initializes the PancakeStack
        Args: 
            state (tuple): Order of pancakes
            parent (PancakeStack): The parent node, the flipping of which spawned this stack
            added (int): The order that the pancake was added in -- useful for breaking ties
            b_cost (int): The backward cost of flipping this stack
            heuristic (callable): The heuristic function to be used to navigate the search tree -- in the literature as h(x)
        """
        self.state = tuple(state)
        self.parent = parent
        self.added = added
        self.b_cost = b_cost
        self.heuristic = heuristic

        self.obsolete = False # an obsolete stack is one whose cost is greater but whose state is the same as another stack found elsewhere in the search tree

    def __lt__(self, other) -> bool:
        """
        Less than function, used with the < operator and for the PriorityQueue object
        Args: 
            other (PancakeStack): A stack to compare with
        Returns:
            bool: Whether self is less than other
        """
        my_total_cost = self.total_cost()
        others_total_cost = other.total_cost()
        if my_total_cost == others_total_cost:
            return self.added > other.added
        return my_total_cost < others_total_cost

    def total_cost(self) -> int:
        """
        Calculates total cost of a stack
        Returns:
            int: Total cost of a stack, the sum of the heuristic given the state and the backward cost
        """
        return self.heuristic(self.state) + self.b_cost

    def flip(self, k) -> Self:
        """
        Flips the stack at a given place, k
        Args:
            k (int): The place the cook inserts the spatula to make a flip
        Returns:
            PancakeStack: New pancake stack made up for the flipped state with a new added and backward cost
        """
        new_state = self.state[:k+1][::-1] + self.state[k+1:]
        new_b_cost = self.b_cost + 1 # increment backward cost
        return PancakeStack(
            state=new_state,
            parent=self,
            added=self.added + 1, # increment added
            b_cost=new_b_cost,
            heuristic=self.heuristic # must use the same heuristic and must specify because it defaults to gap_heuristic
        )

class PriorityQueue:
    """
    Abstraction for representing the priority queue -- object which will keep track of the frontier as we explore the tree of PancakeStacks
    """
    def __init__(self) -> None:
        """
        Initializes a PriorityQueue
        """
        self.pq = [] # will hold all of the nodes
        self.state2node = {} # mapping between states to nodes for faster look up
    
    def push(self, item) -> None:
        """
        Pushes a node to the queue
        Args:
            item (PancakeStack): item to be put on the queue
        """
        heapq.heappush(self.pq, item)
        self.state2node[item.state] = item
    
    def pop(self) -> Union[None, PancakeStack]:
        """
        Pops a node from the queue and returns that node
        Returns:
            PancakeStack: the item that was popped
        """
        # Because we have added state2node we need to make sure that the states of state2node and pq are aligned. 
        # We have to introduce an obsolete trigger, which tracks nodes whose cost is larger than the a node with the same state. 
        # If we try to pop an obsolete node, then we just continue on the next node and ignore it. Otherwise, we actually pop the node and delete its key in state2node
        while self.pq:
            item = heapq.heappop(self.pq)
            if item.obsolete:
                continue
            if self.state2node.get(item.state) == item:
                del self.state2node[item.state]
            return item
        return None # return none if we only have obsolete nodes -- this shouldn't happen except in degenerate cases of the problem as a stack is only marked obsolete if there is another node that costs less than it with the same state.

    def is_empty(self):
        """
        Checks if the pq is empty which would trigger an end to the A* algorithm
        Returns:
            bool: Whether the pq is empty
        """
        return len(self.pq) == 0

    def get_node_with_state(self, state):
        """
        Retrieves a node given a state
        Args:
            state (tuple): The state the node of which we want to retrieve
        Return:
            PancakeStack: The retrieved node with the given state -- by definition the node with the lowest cost for that given state (see discussion of obsolete above)
        """
        return self.state2node.get(state, None)

    def has(self, state):
        """
        Checks whether a state is already in the pq
        Args:
            state (tuple): The state the node of which we want to check
        Return:
            bool: Whether this state is in the pq
        """
        return state in self.state2node

class AStar:
    def __init__(self, initial_state, heuristic=gap_heuristic):
        self.initial_state = initial_state
        self.length = len(initial_state)
        self.heuristic = heuristic

        self.solution = False
        self.visited = set()
        self.pq = PriorityQueue()

        self.root = PancakeStack(initial_state, None, 0, 0, heuristic)
        self.pq.push(self.root)
        self.added = 1
    
    def search(self):
        while True:
            if self.pq.is_empty():
                return
        
            current = self.pq.pop()
            if current is None: 
                return
            
            self.visited.add(current.state) 

            if current.state == GOLD:
                self.solution = current
                return
        
            self.generate_successors(current)
    
    def generate_successors(self, current):
        for i in range(2, self.length+1):
            successor = current.flip(i)
            
            if successor.state in self.visited:
                continue
        
            if not self.pq.has(successor.state):
                self.pq.push(successor)
            else:
                possible_existing_stack = self.pq.get_node_with_state(successor.state)
                if possible_existing_stack and (successor < possible_existing_stack):
                    possible_existing_stack.obsolete = True
                    self.pq.push(successor)

    def print_results(self):
        results = []
        
        current = self.solution
        while current.parent:
            results.append(current.state)
            current = current.parent
        
        for i, result in enumerate(results[::-1]):
            print(f"Stack after flip #{i+1}: {result}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(dest="stack", nargs="*", type=int, default=[])
    parser.add_argument("-u", "--ucs", action="store_true")
    args = parser.parse_args()

    stack = args.stack
    if stack == []:
        stack = [i for i in range(1, 11)]
        random.shuffle(stack)
        print(f"Initial stack: {stack}")

    if args.ucs:
        astar = AStar(stack, heuristic=no_heuristic)
    else:
        astar = AStar(stack, heuristic=gap_heuristic)
    
    cur = time.time()
    astar.search()
    astar.print_results()
    elapsed = time.time() - cur

    print(f"Pancake Problem sovled in {round(elapsed, 4)} seconds")

if __name__ == '__main__':
    main()