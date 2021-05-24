T1 = ['-', ['+', ['-', 75, ['-', 10, 3]], ['-', 100, 50]], 3]
T2 = T = ['+', ['*', 5, 4] ,['-', 100, ['/', 20, 2] ]]

def evaluate(arg):
    if type(arg) is list:
        return eval(f'{evaluate(arg[1])} {arg[0]} {evaluate(arg[2])}')
    else:
        return arg

def get_ops(arg):
    ops = [arg[0]]
    idx = [[0]]
    for i in (1, 2):
        if type(arg[i]) is list:
            ops_sub, idx_sub = get_ops(arg[i])
            ops += ops_sub
            for x in idx_sub:
                idx.append([i] + x)
    return ops, idx

def get_nums(arg):
    nums = []
    idx = []
    for i in (1, 2):
        if type(arg[i]) is list:
            nums_sub, idx_sub = get_nums(arg[i])
            nums += nums_sub
            for x in idx_sub:
                idx.append([i] + x)
        else: # if scalar
            nums.append(arg[i])
            idx.append([i])
    return nums, idx

operatorlist,operatorindex = get_ops(T2)

numberlist, numberindex = get_nums(T2)

print('this is the tree: ',T2)
print()
print(operatorlist)
print()
print(operatorindex)
print()
print(numberlist)
print()
print(numberindex)
