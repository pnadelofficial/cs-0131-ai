import argparse
import random 
import heapq
import time

GOLD = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

def gap_heuristic(state):
    gaps = 0
    for prev, curr in zip(state, state[1:]):
        if abs(curr - prev) != 1:
            gaps +=1
    return gaps

def no_heuristic(state):
    return 0

class PancakeStack:
    def __init__(self, state, parent, added, b_cost, heuristic=gap_heuristic):
        self.state = tuple(state)
        self.parent = parent
        self.added = added
        self.b_cost = b_cost
        self.heuristic = heuristic

        self.obsolete = False

    def __lt__(self, other):
        my_total_cost = self.total_cost()
        others_total_cost = other.total_cost()
        if my_total_cost == others_total_cost:
            return self.added > other.added
        return my_total_cost < others_total_cost

    def total_cost(self):
        return self.heuristic(self.state) + self.b_cost

    def flip(self, k):
        new_state = self.state[:k+1][::-1] + self.state[k+1:]
        new_b_cost = self.b_cost + 1
        return PancakeStack(
            state=new_state,
            parent=self,
            added=self.added + 1,
            b_cost=new_b_cost,
            heuristic=self.heuristic
        )

class PriorityQueue:
    def __init__(self):
        self.pq = []
        self.state2node = {}
    
    def push(self, item):
        heapq.heappush(self.pq, item)
        self.state2node[item.state] = item
    
    def pop(self):
        while self.pq:
            item = heapq.heappop(self.pq)
            if item.obsolete:
                continue
            if self.state2node.get(item.state) == item:
                del self.state2node[item.state]
        # item = heapq.heappop(self.pq)
        # del self.state2node[item.state]
            return item
        return None

    def is_empty(self):
        return len(self.pq) == 0

    def get_node_with_state(self, state):
        return self.state2node.get(state, None)

    def has(self, state):
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