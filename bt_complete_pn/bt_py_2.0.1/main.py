#
# Behavior Tree framework for A1 Behavior trees assignment.
# CS 131 - Artificial Intelligence
#
# version 2.0.1 - copyright (c) 2023-2024 Santini Fabrizio. All rights reserved.
#

import bt_library as btl

from bt.behavior_tree import tree_root
from bt.globals import BATTERY_LEVEL, GENERAL_CLEANING, SPOT_CLEANING, DUSTY_SPOT_SENSOR, HOME_PATH, HOME_COORDINATES, CURRENT_COORDINATES

import random

# Main body of the assignment
current_blackboard = btl.Blackboard()
current_blackboard.set_in_environment(BATTERY_LEVEL, 29)
current_blackboard.set_in_environment(SPOT_CLEANING, False)
current_blackboard.set_in_environment(GENERAL_CLEANING, True)
current_blackboard.set_in_environment(DUSTY_SPOT_SENSOR, False)
current_blackboard.set_in_environment(HOME_PATH, "")
current_blackboard.set_in_environment(HOME_COORDINATES, (150, -150))
current_blackboard.set_in_environment(CURRENT_COORDINATES, (0, 0))

times_general = 0
times_spot = 0
done = False
while not done:
    # Each cycle in this while-loop is equivalent to 1 second time

    # Step 1: Change the environment
    #   - Change the battery level
    #   - Simulate the response of the dusty spot sensor
    #   - Simulate user input commands

    spot_or_general_cleaning = input("Enter 'spot' for spot cleaning or 'general' for general cleaning: ").strip().lower()
    if spot_or_general_cleaning == 'spot':
        current_blackboard.set_in_environment(SPOT_CLEANING, True)
        current_blackboard.set_in_environment(GENERAL_CLEANING, False)
        times_spot += 1
    elif spot_or_general_cleaning == 'general':
        current_blackboard.set_in_environment(SPOT_CLEANING, False)
        current_blackboard.set_in_environment(GENERAL_CLEANING, True)
        
        rand_dusty_spot = random.choice([True, False])
        current_blackboard.set_in_environment(DUSTY_SPOT_SENSOR, rand_dusty_spot)
        times_general += 1
    else:
        print("Invalid input. Please enter 'spot' or 'general'. Please try again.")
        continue

    # Step 2: Evaluating the tree
    result = tree_root.run(current_blackboard)

    # Step 3: Determine if your solution must terminate
    current_battery = current_blackboard.get_in_environment(BATTERY_LEVEL, 0)
    print(f"Current battery level: {current_battery}%")
    if current_battery == 0: # Should never be met
        done = True
    
     # If the robot has done general cleaning 10 times or spot cleaning 10 times it will stop
    if times_general == 10:
        print("Cleaned the whole room! You're welcome...")
        done = True
    if times_spot == 10:
        print("Piles of dirt eliminated! Try to be neater next time...")
        done = True
