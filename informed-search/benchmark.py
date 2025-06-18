from pancake_problem import gap_heuristic, top_heuristic, no_heuristic, top_prime_heuristic, l_top_prime_heuristic, AStar
import time
import random
from functools import partial
import collections
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

HEURISTICS = {
    "UCS": no_heuristic,
    "Gap": gap_heuristic,
    "Top": top_heuristic,
    "Top'": top_prime_heuristic,
    "3-Top'": partial(l_top_prime_heuristic, l=3),
    "4-Top'": partial(l_top_prime_heuristic, l=4),
    "5-Top'": partial(l_top_prime_heuristic, l=5),
}

def benchmark():
    # generate 100 random stacks and test all results
    stacks = []
    for _ in range(100):
        stack = [i for i in range(1, 6)] # lowering the number of pancakes for testing
        random.shuffle(stack)
        stacks.append(tuple(stack))
    
    # test them on all of the heuristics and record results
    results = collections.defaultdict(list)
    for heuristic in HEURISTICS:
        for stack in stacks:
            astar = AStar(stack, heuristic=HEURISTICS[heuristic])

            cur = time.time()
            astar.search()
            elapsed = time.time() - cur
            num_visited = len(astar.visited)

            results[heuristic].append((elapsed, num_visited))

    return results, stacks

results, stacks = benchmark()
# two plots for each metric:
# line chart of elapsed and of visited nodes
# box plot of elasped and of visited nodes

## elapsed line chart
plt.figure(figsize=(7, 5))
for heuristic in HEURISTICS:
    plt.plot(range(1,101), [e[0] for e in results[heuristic]], label=heuristic)

plt.title("Time to sort vs. Stacks")
plt.xlabel("Stack")
plt.ylabel("Elasped time to sort")
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend()
plt.savefig("elapsed_by_heuristic.pdf", dpi=300, bbox_inches='tight')
plt.clf()

## nodes line chart
plt.figure(figsize=(7, 5))
for heuristic in HEURISTICS:
    plt.plot(range(1,101), [e[1] for e in results[heuristic]], label=heuristic)

plt.title("Visited nodes vs. Stacks")
plt.xlabel("Stack")
plt.ylabel("Visited ndoes")
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend()
plt.savefig("nodes_by_heuristic.pdf", dpi=300, bbox_inches='tight')
plt.clf()

## elapsed box plot
plt.figure(figsize=(7, 5))
data = np.array([[e[0] for e in results[heuristic]] for heuristic in HEURISTICS]).T
plt.boxplot(data, tick_labels=HEURISTICS.keys(), patch_artist=True)
plt.title("Elasped time to sort")
plt.xlabel("Heuristic")
plt.ylabel("Elasped time to sort")
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend()
plt.savefig("elapsed_box.pdf", dpi=300, bbox_inches='tight')
plt.clf()

## nodes box plot
plt.figure(figsize=(7, 5))
data = np.array([[e[1] for e in results[heuristic]] for heuristic in HEURISTICS]).T
plt.boxplot(data, tick_labels=HEURISTICS.keys(), patch_artist=True)
plt.title("Visited nodes")
plt.xlabel("Heuristic")
plt.ylabel("Visited nodes")
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend()
plt.savefig("nodes_box.pdf", dpi=300, bbox_inches='tight')
plt.clf()
