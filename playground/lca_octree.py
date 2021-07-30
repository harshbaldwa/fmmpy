# checking for least common ancestor in an oct tree
def lca_octree(index1, index2, curr_level):
    xor = index1 ^ index2
    for i in range(curr_level, 0, -1):
        if xor > ((1 << (i - 1) * 3) - 1):
            return index1 >> 3 * i
    return index1


a = int("000101", 2)
b = int("000111", 2)
print(bin(lca_octree(a, b, 4))[2:])
