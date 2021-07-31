# checking for least common ancestor in an oct tree
def lca_octree(index1, index2, curr_level_1, curr_level_2):
    level_diff = abs(curr_level_1 - curr_level_2)

    if curr_level_1 - curr_level_2 > 0:
        index2 = index2 << 3 * (curr_level_1 - curr_level_2)
        curr_level_2 = curr_level_1
    elif curr_level_1 - curr_level_2 < 0:
        index1 = index1 << 3 * (-curr_level_1 + curr_level_2)
        curr_level_1 = curr_level_2

    xor = index1 ^ index2

    if xor == 0:
        return index1 >> 3 * level_diff, curr_level_1 - level_diff
    for i in range(curr_level_1 + 1, 0, -1):
        if xor > ((1 << (i - 1) * 3) - 1):
            return index1 >> 3 * i, curr_level_1 - i
    return 0, 0


a = int("101111001", 2)
b = int("101111001000", 2)
lca, level = lca_octree(a, b, 3, 4)
print(bin(lca)[2:])
print(level)
