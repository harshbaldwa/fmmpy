# checking for least common ancestor in an oct tree
def lca_octree(index1, index2, curr_level):
    xor = index1 ^ index2
    if xor == 0:
        return index1, curr_level
    for i in range(curr_level + 1, 0, -1):
        if xor > ((1 << (i - 1) * 3) - 1):
            return index1 >> 3 * i, curr_level - i
    return 0, 0


a = int("101111001001", 2)
b = int("101111001110", 2)
lca, level = lca_octree(a, b, 4)
print(bin(lca)[2:])
print(level)
