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
    Uniform Cost Search heuristic -- a heuristic that always returns None
    Args:
        state (tuple): Order of pancakes
    Returns:
        int: Always returns None as the UCS algorithm ignores heuristics
    """
    return None

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
            return self.added > other.added # using the added to break ties: stacks added later will be given preference
        return my_total_cost < others_total_cost

    def total_cost(self) -> int:
        """
        Calculates total cost of a stack
        Returns:
            int: Total cost of a stack, the sum of the heuristic given the state and the backward cost
        """
        h = self.heuristic(self.state) 
        if h: # check if h returns None
            return self.heuristic(self.state) + self.b_cost
        else:
            return self.b_cost

    def flip(self, k) -> Self:
        """
        Flips the stack at a given place, k
        Args:
            k (int): The place the cook inserts the spatula to make a flip
        Returns:
            PancakeStack: New pancake stack made up for the flipped state with a new added and backward cost
        """
        new_state = self.state[:k+1][::-1] + self.state[k+1:] # pythonic way to do flip: cuts the stack at k+1, reverses everything before and concatenates it to the rest of the original stack
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

    def is_empty(self) -> bool:
        """
        Checks if the pq is empty which would trigger an end to the A* algorithm
        Returns:
            bool: Whether the pq is empty
        """
        return len(self.pq) == 0

    def get_node_with_state(self, state:Tuple) -> PancakeStack:
        """
        Retrieves a node given a state
        Args:
            state (tuple): The state the node of which we want to retrieve
        Return:
            PancakeStack: The retrieved node with the given state -- by definition the node with the lowest cost for that given state (see discussion of obsolete above)
        """
        return self.state2node.get(state, None)

    def has(self, state) -> bool:
        """
        Checks whether a state is already in the pq
        Args:
            state (tuple): The state the node of which we want to check
        Return:
            bool: Whether this state is in the pq
        """
        return state in self.state2node

class AStar:
    """
    Implementation of the A* algorithm to solve the Pancake Problem
    """
    def __init__(self, initial_state:Tuple, heuristic:callable=gap_heuristic) -> None:
        """
        Initializes the A* object for searching the given initial stack
        Args:
            initial_state (tuple): the initial state of the stack, the starting point of the pancake problem
            heuristic (callable): the heuristic to use when solving the problem 
        """
        self.initial_state = initial_state
        self.length = len(initial_state)
        self.heuristic = heuristic

        self.solution = False
        self.visited = set() # set of all visited nodes to avoid visiting the same node twice
        self.pq = PriorityQueue()

        self.root = PancakeStack(initial_state, None, 0, 0, heuristic) # initial PancakeStack
        self.pq.push(self.root)
        self.added = 1 # set added to 1 
    
    def search(self) -> None:
        while True:
            # print(self.pq.pq)
            if self.pq.is_empty(): # if pq is empty then we end
                return
        
            current = self.pq.pop()
            if current is None: # if pq pops a None then we end (see pop for more explanation)
                return
            
            self.visited.add(current.state) # add current state to visited

            if current.heuristic(current.state) is None: # in ucs, need to check if we have any gaps wit hthe gap heuristic, used now as a check
                gaps = gap_heuristic(current.state)
                if gaps == 0:
                    self.solution = current
                    return    
            
            if current.heuristic(current.state) == 0:                
                if current.state[0] < current.state[-1]: # if it is lower first
                    current = current.flip(len(current.state)+1) # flip it at the last position
                self.solution = current # set solution to the current stack
                return
        
            self.generate_successors(current) # expand frontier
    
    def generate_successors(self, current:PancakeStack) -> None:
        for i in range(2, self.length+1): # iterate across all of the flips which would change the stack
            successor = current.flip(i)
            
            if successor.state in self.visited: # pass if visited the flipped stack
                continue
        
            if not self.pq.has(successor.state): # push if we haven't visited the stack and the stack is not in the pq
                self.pq.push(successor)
            else:
                # if it is in the pq then we have some options:
                # the existing stack either costs more or less than the exisiting stack 
                # if it cost more we can move on
                possible_existing_stack = self.pq.get_node_with_state(successor.state)
                if possible_existing_stack and (successor < possible_existing_stack): # if it costs less
                    possible_existing_stack.obsolete = True # then we mark the existing stack as obsolete
                    self.pq.push(successor) # and push the newly popped stack

    def print_results(self) -> None:
        results = []
        
        current = self.solution
        if not current:
            print("No solution found.")
            return

        while current.parent: # backtrack to find parents
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
        random.shuffle(stack) # if no stack is given we randomly create our own
    stack = tuple(stack)

    if args.ucs:
        astar = AStar(stack, heuristic=no_heuristic)
    else:
        astar = AStar(stack, heuristic=gap_heuristic)

    print(f"Initial stack: {stack}")

    cur = time.time()
    astar.search()
    elapsed = time.time() - cur

    astar.print_results()
    
    print(f"Ran for {round(elapsed, 4)} seconds | Visited {len(astar.visited)} nodes")
    return elapsed, len(astar.visited)

if __name__ == '__main__':
    main()