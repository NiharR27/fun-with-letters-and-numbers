# -*- coding: utf-8 -*-
"""
Created on Mon May 24 11:14:36 2021

@author: Nihar
"""

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
    if prefix is None:
        prefix = []

    if isinstance(T, int):
        Aop = []
        Lop = [] 
        Anum = [prefix]
        Lnum = [T]
        return Aop, Lop, Anum, Lnum
    
    assert isinstance(T, list)
    
    raise NotImplementedError()
    
