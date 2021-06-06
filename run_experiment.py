#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 17:46:18 2021

@author: frederic

***** Perfect Score!! *****
    Q = [25,10,2,9,8,7]
    target 449 , tree value 449
    
    -
    |+
    ||*
    |||*
    ||||2
    |||
    ||||9
    ||
    |||25
    |
    ||7
    
    |8
    
**** Perfect Score!! *****
    Q = [50,75,9,10,2,2]
    target 533 , tree value 533
    
    +
    |+
    ||75
    |
    ||-
    |||*
    ||||9
    |||
    ||||50
    ||
    |||2
    
    |10

"""

import numpy as np
import time
import random
from number_game import pick_numbers, eval_tree, display_tree
import multiprocessing
from threading import Thread, Event
from genetic_algorithm import  evolve_pop
import sys
import os
import statistics

def test():
    #Q = pick_numbers()
    #target = np.random.randint(1,1000)
    
    
    Q = [100, 50, 3, 3, 10, 75]
    #target = 322
    
    # Q = [25,10,2,9,8,7]
    target = 449
    
    #Q = [50,75,9,10,2,2]
    #target = 533
    
    #Q = [100,25,7,5,3,1]
    #target = 728
    
    Q.sort()
    
    print('List of drawn numbers is ',Q)
    
    v, T = evolve_pop(Q, target, 
                      max_num_iteration = 200,
                      population_size = 1000,
                      parents_portion = 0.3)
    
    
    
    
    print('----------------------------')
    if v==0:
        print("\n***** Perfect Score!! *****")
    print(f'\ntarget {target} , tree value {eval_tree(T)}\n')
    display_tree(T)
    print('the costt at the end is->>>',v)
stop_event = Event()

pop_list = [583, 18, 21, 957, 861, 355, 754, 609, 305, 468, 452, 888, 294, 285, 244, 363, 345, 250, 637, 374]
'''
for i in range(20):
    r = random.randint(5, 1000)
    pop_list.append(r)
'''
pop_list = sorted(pop_list);
print("POP:", pop_list)
max_gen_list = []
max_gen_list.append(statistics.median([1112, 1153, 1421, 1354, 1496])) #18
max_gen_list.append(statistics.median([932, 987, 1105, 793, 1106])) #21
max_gen_list.append(statistics.median([88, 122, 118, 97, 115])) #244
max_gen_list.append(statistics.median([130, 85, 118, 89, 85])) #250
max_gen_list.append(statistics.median([78, 98, 140, 101, 81])) #285
max_gen_list.append(statistics.median([76, 113, 75, 98, 82])) #294
max_gen_list.append(statistics.median([72, 84, 96, 87, 102])) #305
max_gen_list.append(statistics.median([65, 80, 85, 64, 82])) #345
max_gen_list.append(statistics.median([76, 79, 80, 64, 63])) #355
max_gen_list.append(statistics.median([74, 65, 80, 66, 72])) #363
max_gen_list.append(statistics.median([73, 51, 52, 55, 57])) #374
max_gen_list.append(statistics.median([68, 52, 53, 59, 50])) #452
max_gen_list.append(statistics.median([63, 56, 70, 65, 54])) #468
max_gen_list.append(statistics.median([45, 37, 42, 53, 44])) #583
max_gen_list.append(statistics.median([38, 54, 48, 42, 43])) #609
max_gen_list.append(statistics.median([33, 22, 32, 25, 32])) #637
max_gen_list.append(statistics.median([27, 28, 33, 32, 32])) #754
max_gen_list.append(statistics.median([5, 25, 5, 30, 4])) #861
max_gen_list.append(statistics.median([6, 5, 5, 5, 5])) #888
max_gen_list.append(statistics.median([4, 3, 4, 23, 3])) #957
print("MAX GEN: ", max_gen_list)

Q = [100, 50, 3, 3, 10, 75]
target = 533
all_result = [[5, 1, 2, 2, 8, 1, 1, 1, 2, 1, 2, 8, 1, 8, 7, 358, 1, 3, 2, 1, 1, 1, 1, 1, 1, 2, 233, 33, 2, 1], [33, 1, 2, 1, 2, 1, 2, 1, 33, 1, 2, 33, 1, 3, 1, 1, 2, 5, 1, 1, 2, 2, 2, 1, 1, 2, 1, 2, 33, 1], [2, 0, 0, 1, 2, 0, 1, 3, 1, 2, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 2, 1, 0, 1, 0], [0, 0, 2, 0, 1, 0, 2, 0, 1, 1, 0, 1, 2, 1, 0, 2, 0, 0, 1, 0, 1, 1, 1, 0, 1, 2, 1, 1, 0, 0], [1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 2, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 2], [1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0], [1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0], [0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1], [1, 0, 0, 0, 0, 2, 0, 1, 1, 0, 1, 1, 2, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1], [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 2, 0, 0], [0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1], [0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1], [1, 1, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0], [1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 2, 0, 1, 0, 1, 0, 0, 1, 1, 0, 2, 1, 0, 0, 1, 2, 0, 1, 0]]
'''
for i in range(len(pop_list)):
    this_result = []
    for index in range(30):
        v, T = evolve_pop(Q, target, 
                     max_num_iteration = max_gen_list[i],
                    population_size = pop_list[i] ,
                parents_portion = 0.3)
        this_result.append(v)
    all_result.append(this_result)
    '''
    
    #18 0 
#print(all_result)
print('---------------------------------------------------------------------------------')
for index in range(len(pop_list)):
        print("Population Size: ", pop_list[index])
        print("Maximum Generation: ", max_gen_list[index])
        print("Results: ", all_result[index])
        print()
print('----------------------------------------------------------------------------------')
'''
maxCostList = []
for listing in all_result:
    maxCost = max(listing)
    maxCostList.append(maxCost)
    
print(maxCostList)
'''

numberlist = []
for listing in all_result:
    num_zero = listing.count(0)
    numberlist.append(num_zero)
    
print(numberlist)

best_pop = pop_list[numberlist.index(max(numberlist))]
best_max_gen = max_gen_list[numberlist.index(max(numberlist))]
best_value = [best_pop ,best_max_gen]

print('The best pair of population and max generation is: ',best_value )
print('The maximum value of population was: ', max(pop_list))
print('The maximum value in generation was: ',max(max_gen_list))



def task2():
    
      
    
        
    start=time.time()
        
    PERIOD_OF_TIME = 2
    v, T = evolve_pop(Q, target, 
                     max_num_iteration = 150,
                    population_size = 363 ,
                parents_portion = 0.3)
        
    if stop_event.is_set():
        cost = v
        print("cost: ", cost)
        
    
        
    print(time.time() - start)
    print(pop_list)
    
   
    


'''
if __name__ == '__main__':
    # We create another Thread
    action_thread = Thread(target=task2)
 
    # Here we start the thread and we wait 5 seconds before the code continues to execute.
    action_thread.start()
    action_thread.join(timeout=2)
 
    # We send a signal that the other thread should stop.
    stop_event.set()
 
    print("Hey there! I timed out! You can do things after me!")
'''

#test()
#task2()
