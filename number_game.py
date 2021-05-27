'''

In the Letters and Numbers (L&N) game,
One contestant chooses how many "small" and "large" numbers they would like 
to make up six randomly chosen numbers. Small numbers are between 
1 and 10 inclusive, and large numbers are 25, 50, 75, or 100. 
All large numbers will be different, 
so at most four large numbers may be chosen. 


How to represent a computation?

Let Q = [q0, q1, q2, q3, q4, q5] be the list of drawn numbers

The building blocks of the expression trees are
 the arithmetic operators  +,-,*
 the numbers  q0, q1, q2, q3, q4, q5

We can encode arithmetic expressions with Polish notation
    op arg1 arg2
where op is one of the operators  +,-,*

or with expression trees:
    (op, left_tree, right_tree)
    
Recursive definition of an Expression Tree:
 an expression tree is either a 
 - a scalar   or
 - a binary tree (op, left_tree, right_tree)
   where op is in  {+,-,*}  and  
   the two subtrees left_tree, right_tree are expressions trees.

When an expression tree is reduced to a scalar, we call it trivial.


Author: f.maire@qut.edu.au

Created on April 1 , 2021
    

This module contains functions to manipulate expression trees occuring in the
L&N game.

'''



import numpy as np
import random

import copy # for deepcopy

import collections


SMALL_NUMBERS = tuple(range(1,11))
LARGE_NUMBERS = (25, 50, 75, 100)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
    return [ (10355243, 'Nihar', 'Rupareliya'), (10467114, 'Cory', 'Bullen'), (10330895, 'Ricky', 'Lau') ]


# ----------------------------------------------------------------------------

def pick_numbers():
    '''    
    Create a random list of numbers according to the L&N game rules.
    
    Returns
    -------
    Q : int list
        list of numbers drawn randomly for one round of the game
    '''
    LN = set(LARGE_NUMBERS)
    Q = []
    for i in range(6):
        x = random.choice(list(SMALL_NUMBERS)+list(LN))
        Q.append(x)
        if x in LN:
            LN.remove(x)
    return Q


# ----------------------------------------------------------------------------

def bottom_up_creator(Q):
    '''
    Create a random algebraic expression tree
    that respects the L&N rules.
    
    Warning: Q is shuffled during the process

    Parameters
    ----------
    Q : non empty list of available numbers
        

    Returns  T, U
    -------
    T : expression tree 
    U : values used in the tree

    '''
    n = random.randint(1,6) # number of values we are going to use
    
    random.shuffle(Q)
    # Q[:n]  # list of the numbers we should use
    U = Q[:n].copy()
    
    if n==1:
        # return [U[0], None, None], [U[0]] # T, U
        return U[0], [U[0]] # T, U
        
    F = [u for u in U]  # F is initially a forest of values
    # we start with at least two trees in the forest
    while len(F)>1:
        # pick two trees and connect then with an arithmetic operator
        random.shuffle(F)
        op = random.choice(['-','+','*'])
        T = [op,F[-2],F[-1]]  # combine the last two trees
        F[-2:] = [] # remove the last two trees from the forest
        # insert the new tree in the forest
        F.append(T)
    # assert len(F)==1
    return F[0], U
  
# ---------------------------------------------------------------------------- 

def display_tree(T, indent=0):
    '''
    
    Eval the algebraic expression represented by T
    
    Parameters
    ----------
    T : Expression Tree
    indent: indentation for the recursive call

    Returns None

    '''
    # if T is a scalar, then we return it directly
    if isinstance(T, int):
        print('|'*indent,T, sep='')
        return
    # T is non trivial
    root_item = T[0]
    print('|'*indent, root_item, sep='')
    display_tree(T[1], indent+1)
    print('|'*indent)
    display_tree(T[2], indent+1)
   
# ---------------------------------------------------------------------------- 

def eval_tree(T):
    '''
    
    Eval the algebraic expression represented by T
    
    Parameters
    ----------
    T : Expression Tree

    Returns
    -------
    value of the algebraic expression represented by the T

    '''
    # if T is a scalar, then we return it directly
    if isinstance(T, int):
        return T
    # T is non trivial
    root_item = T[0]
    # assert root_item in ('-','+','*')
    left_value = eval_tree(T[1])
    right_value = eval_tree(T[2])
    return eval( str(left_value) +root_item + str(right_value) )
    # return eval(root_item.join([str(left_value), str(right_value)]))
   
     
# ---------------------------------------------------------------------------- 

def expr_tree_2_polish_str(T):
    '''
    Convert the Expression Tree into Polish notation

    Parameters
    ----------
    T : expression tree

    Returns
    -------
    string in Polish notation represention the expression tree T

    '''
    if isinstance(T, int):
        return str(T)
    root_item = T[0]
    # assert root_item in ('-','+','*')
    left_str = expr_tree_2_polish_str(T[1])
    right_str = expr_tree_2_polish_str(T[2])
    return '[' + ','.join([root_item,left_str,right_str]) + ']'
    

# ----------------------------------------------------------------------------

def polish_str_2_expr_tree(pn_str):
    '''

    Convert a polish notation string of an expression tree
    into an expression tree T.

    Parameters
    ----------
    pn_str : string representing an L&N algebraic expression

    Returns
    -------
    T

    '''


    result = []
    result1 = []
    T1 = 0
    T2 = 0

    if len(pn_str) < 10:
        return pn_str

    for k in range(len(pn_str)):
        if pn_str[k+1] == '[':
            result.append(k+1)
        if pn_str[k] == ']':
            result1.append(k)
        if len(result) == len(result1) & len(result) > 0:  # if amount of '[' placed equals the amount of ']'
            T1 = pn_str[result[0]:result1[-1]+1]  # left side tree
            T2 = pn_str[result1[-1]+2:-1]  # right side tree
            break

    if pn_str[5].isnumeric():
        T2 = T1
        T1 = pn_str[3]+pn_str[4]+pn_str[5]
    elif pn_str[4].isnumeric():
        T2 = T1
        T1 = pn_str[3]+pn_str[4]
    elif pn_str[3].isnumeric():
        T2 = T1
        T1 = pn_str[3]

    op = pn_str[1]
    print(op)
    print(T1)
    print(T2)
 # unfinished
   
# ----------------------------------------------------------------------------

def op_address_list(T, prefix = None):
    '''
    Return the address list L of the internal nodes of the expresssion tree T
    
    If T is a scalar, then L = []

    Note that the function 'decompose' is more general.

    Parameters
    ----------
    T : expression tree
    prefix: prefix to prepend to the addresses returned in L

    Returns
    -------
    L
    '''
    if isinstance(T, int):
        return []
    
    if prefix is None:
        prefix = []
        
    L = [prefix.copy()+[0]] # first adddress is the op of the root of T
    left_al = op_address_list(T[1], prefix.copy()+[1])
    L.extend(left_al)
    right_al = op_address_list(T[2], prefix.copy()+[2])
    L.extend(right_al)
    
    return L


# ----------------------------------------------------------------------------

#Example tree: T1 = ['-', ['+', ['-', 75, ['-', 10, 3]], ['-', 100, 50]], 3]
#Example tree: T2 = T = ['+', ['*', 5, 4] ,['-', 100, ['/', 20, 2] ]]
def get_ops(arg): # returns (list of operators, list of their indices)
    '''
    Compute
        ops: list of Operators inside the tree
        idx: list of index of the operators inside the tree
    
        For example, if 
    
        T =  ['-', ['+', ['-', 75, ['-', 10, 3]], ['-', 100, 50]], 3]
    
        then, 
    
         idx is  [[0], [1, 0], [1, 1, 0], [1, 1, 2, 0], [1, 2, 0]] 
     
         ops is ['-', '+', '-', '-', '-'] 
      
        Parameters
        ----------
        arg : expression tree 


        Returns
        -------
        ops,idx
    
    '''
    ops = [arg[0]] # first elem is an operator, so add it
    idx = [[0]] # add this index 0
    for i in (1, 2): # for each position 1 and 2,
        if type(arg[i]) is list: # if NOT scalar
            ops_sub, idx_sub = get_ops(arg[i]) # recurse!
            ops += ops_sub # add the list of ops from the sublist
            for x in idx_sub: # add the indices from the sublist, while prepending the position of this sublist for each
                idx.append([i] + x) #Appending the current index to idx
    return ops, idx 

def get_nums(arg): # basically the same goes here

    '''
    Compute
        nums: list of numbers inside the tree
        idx: list of index of the numbers inside the tree
    
        For example, if 
    
        T =  ['-', ['+', ['-', 75, ['-', 10, 3]], ['-', 100, 50]], 3]
    
        then, 
    
         idx is [[1, 1, 1], [1, 1, 2, 1], [1, 1, 2, 2], [1, 2, 1], [1, 2, 2], [2]] 
    
         nums is [75, 10, 3, 100, 50, 3]   
      
        Parameters
        ----------
        arg : expression tree 


        Returns
        -------
        nums,idx
    
    '''
    nums = []
    idx = []
    for i in (1, 2):
        if type(arg[i]) is list:
            nums_sub, idx_sub = get_nums(arg[i])
            nums += nums_sub
            for x in idx_sub:
                idx.append([i] + x)
        else:
            nums.append(arg[i])
            idx.append([i])
    return nums, idx

def decompose(T, prefix = None):
    '''
    Compute
        Aop : address list of the operators
        Lop : list of the operators
        Anum : address of the numbers
        Lnum : list of the numbers
    
    For example, if 
    
    T =  ['-', ['+', ['-', 75, ['-', 10, 3]], ['-', 100, 50]], 3]
    
    then, 
    
     Aop is  [[0], [1, 0], [1, 1, 0], [1, 1, 2, 0], [1, 2, 0]] 
    
     Lop is ['-', '+', '-', '-', '-'] 
    
     Anum is [[1, 1, 1], [1, 1, 2, 1], [1, 1, 2, 2], [1, 2, 1], [1, 2, 2], [2]] 
    
     Lnum is [75, 10, 3, 100, 50, 3]    
        
    
    Parameters
    ----------
    T : expression tree 
    
    prefix : address to preprend 

    Returns
    -------
    Aop, Lop, Anum, Lnum

    '''
    if prefix is None: #If prefix is none
        prefix = [] #Create an empty list

    if isinstance(T, int): #If the Tree is just an integer
        Aop = []  
        Lop = [] 
        Anum = [prefix]
        Lnum = [T]
        return Aop, Lop, Anum, Lnum
    
    assert isinstance(T, list) #Checking if the tree is an list
     
    Lop,Aop = get_ops(T) #Getting the information about the operators inside tree
    Lnum, Anum = get_nums(T) #Getting the information about the numbers inside the tres
    
    return Aop, Lop, Anum, Lnum
    
    


# ----------------------------------------------------------------------------

def get_item(T, a):
    '''
    Get the item at address a in the expression tree T

    Parameters
    ----------
    T : expression tree
    a : valid address of an item in the tree

    Returns
    -------
    the item at address a

    '''
    if len(a)==0:
        return T
    # else
    return get_item(T[a[0]], a[1:])
        
# ----------------------------------------------------------------------------

def replace_subtree(T, a, S):
    '''
    Replace the subtree at address a
    with the subtree S in the expression tree T
    
    The address a is a sequence of integers in {0,1,2}.
    
    If a == [] , then we return S
    If a == [1], we replace the left subtree of T with S
    If a == [2], we replace the right subtree of T with S

    Returns
    ------- 
    The modified tree

    Warning: the original tree T is modified. 
             Use copy.deepcopy()  if you want to preserve the original tree.
    '''    
    
    # base case, address empty
    if len(a)==0:
        return S
    
    # recursive case
    T[a[0]] = replace_subtree(T[a[0]], a[1:], S)
    return T


# ----------------------------------------------------------------------------

def mutate_num(T, Q):
    '''
    Mutate one of the numbers of the expression tree T

    Parameters
    ----------
    T : expression tree
    Q : list of numbers initially available in the game

    Returns
    -------
    A mutated copy of T

    '''

    if isinstance(T, int):
        return T

    Aop, Lop, Anum, Lnum = decompose(T)
    mutant_T = copy.deepcopy(T)
    counter_Q = collections.Counter(Q)  # some small numbers can be repeated

    valid_num = set(Q) - set(Lnum)
    valid_num = list(valid_num)

    if len(valid_num) == 0:
        return T

    rand_num = random.choice(Anum)
    i = rand_num[0]

    if len(rand_num) < 2:
        mutant_T[i] = random.choice(valid_num)

    elif len(rand_num) == 2:
        j = rand_num[1]
        mutant_T[i][j] = random.choice(valid_num)

    elif len(rand_num) == 3:
        j = rand_num[1]
        k = rand_num[2]
        mutant_T[i][j][k] = random.choice(valid_num)

    elif len(rand_num) == 4:
        j = rand_num[1]
        k = rand_num[2]
        l = rand_num[3]
        mutant_T[i][j][k][l] = random.choice(valid_num)

    elif len(rand_num) > 4:
        j = rand_num[1]
        k = rand_num[2]
        l = rand_num[3]
        m = rand_num[4]
        mutant_T[i][j][k][l][m] = random.choice(valid_num)

    return mutant_T
    

# ----------------------------------------------------------------------------

def mutate_op(T):
    '''
    Mutate an operator of the expression tree T
    If T is a scalar, return T

    Parameters
    ----------
    T : non trivial expression tree

    Returns
    -------
    A mutated copy of T

    '''
    if isinstance(T, int):
        return T

    La = op_address_list(T)
    a = random.choice(La)  # random address of an op in T
    op_c = get_item(T, a)       # the char of the op
    op_list = (['-', '+', '*'])

    testT = T.copy()
    i = a[0]

    if len(a) < 2:
        old_op = testT[i]
        op_list.remove(old_op)
        testT[i] = random.choice(op_list)

    elif len(a) == 2:
        j = a[1]
        old_op = testT[i][j]
        op_list.remove(old_op)
        testT[i][j] = random.choice(op_list)

    elif len(a) == 3:
        j = a[1]
        k = a[2]
        old_op = testT[i][j][k]
        op_list.remove(old_op)
        testT[i][j][k] = random.choice(op_list)

    elif len(a) == 4:
        j = a[1]
        k = a[2]
        l = a[3]
        old_op = testT[i][j][k][l]
        op_list.remove(old_op)
        testT[i][j][k][l] = random.choice(op_list)

    elif len(a) > 4:
        j = a[1]
        k = a[2]
        l = a[3]
        m = a[4]
        old_op = testT[i][j][k][l][m]
        op_list.remove(old_op)
        testT[i][j][k][l][m] = random.choice(op_list)

    return testT
    

# ----------------------------------------------------------------------------

def cross_over(P1, P2, Q):
    '''
    Perform crossover on two non trivial parents

    Parameters
    ----------
    P1 : parent 1, non trivial expression tree  (root is an op)
    P2 : parent 2, non trivial expression tree  (root is an op)
        DESCRIPTION

    Q : list of the available numbers
        Q may contain repeated small numbers


    Returns
    -------
    C1, C2 : two children obtained by crossover
    '''

    def get_num_ind(aop, Anum):
        '''
        Return the indices [a,b) of the range of numbers
        in Anum and Lum that are in the sub-tree
        rooted at address aop

        Parameters
        ----------
        aop : address of an operator (considered as the root of a subtree).
              The address aop is an element of Aop
        Anum : the list of addresses of the numbers

        Returns
        -------
        a, b : endpoints of the semi-open interval

        '''
        d = len(aop) - 1  # depth of the operator.
        # Root of the expression tree is a depth 0
        # K: list of the indices of the numbers in the subtrees
        # These numbers must have the same address prefix as aop
        p = aop[:d]  # prefix common to the elements of the subtrees
        K = [k for k in range(len(Anum)) if Anum[k][:d] == p]
        return K[0], K[-1] + 1
        # .........................................................

    Aop_1, Lop_1, Anum_1, Lnum_1 = decompose(P1)
    Aop_2, Lop_2, Anum_2, Lnum_2 = decompose(P2)

    C1 = copy.deepcopy(P1)
    C2 = copy.deepcopy(P2)

    i1 = np.random.randint(0, len(Lop_1))  # pick a subtree in C1 by selecting the index
    # of an op
    i2 = np.random.randint(0, len(Lop_2))  # Select a subtree in C2 in a similar way

    # i1, i2 = 4, 0 # DEBUG

    # Try to swap in C1 and C2 the sub-trees S1 and S2
    # at addresses Lop_1[i1] and Lop_2[i2].
    # That's our crossover operation!

    # Compute some auxiliary number lists

    # Endpoints of the intervals of the subtrees
    a1, b1 = get_num_ind(Aop_1[i1], Anum_1)  # indices of the numbers in S1
    # wrt C1 number list Lnum_1
    a2, b2 = get_num_ind(Aop_2[i2], Anum_2)  # same for S2 wrt C2

    # Lnum_1[a1:b1] is the list of numbers in S1
    # Lnum_2[a2:b2] is the list of numbers in S2

    # numbers is C1 not used in S1
    nums_C1mS1 = Lnum_1[:a1] + Lnum_1[b1:]
    # numbers is C2-S2
    nums_C2mS2 = Lnum_2[:a2] + Lnum_2[b2:]

    # S2 is a fine replacement of S1 in C1
    # if nums_S2 + nums_C1mS1 is contained in Q
    # if not we can bottom up a subtree with  Q-nums_C1mS1

    counter_Q = collections.Counter(Q)  # some small numbers can be repeated

    d1 = len(Aop_1[i1]) - 1
    aS1 = Aop_1[i1][:d1]  # address of the subtree S1
    S1 = get_item(C1, aS1)

    d2 = len(Aop_2[i2]) - 1
    aS2 = Aop_2[i2][:d2]
    S2 = get_item(C2, aS2)


    # count the numbers (their occurences) in the candidate child C1
    counter_1 = collections.Counter(Lnum_2[a2:b2] + nums_C1mS1)

    # Test whether child C1 is ok
    if all(counter_Q[v] >= counter_1[v] for v in counter_Q):
        # candidate is fine!  :-)
        C1 = replace_subtree(C1, aS1, S2)
    else:
        available_nums = counter_Q.copy()
        available_nums.subtract(
            collections.Counter(nums_C1mS1)
        )
        R1, _ = bottom_up_creator(list(available_nums.elements()))
        C1 = replace_subtree(C1, aS1, R1)

    # count the numbers (their occurences) in the candidate child C2
    counter_2 = collections.Counter(Lnum_1[a1:b1] + nums_C2mS2)

    # Test whether child C2 is ok
    if all(counter_Q[v] >= counter_2[v] for v in counter_Q):
        # candidate is fine!  :-)
        C2 = replace_subtree(C2, aS2, S1)
    else:
        available_nums1 = counter_Q.copy()
        available_nums1.subtract(
            collections.Counter(nums_C2mS2)
        )
        R2, _ = bottom_up_creator(list(available_nums1.elements()))
        C2 = replace_subtree(C2, aS2, R2)

    return C1, C2

