import numpy as np
import math


def counting_sort(A, digit, radix, index, level):
    B = [0] * len(A)
    C = [0] * int(radix)
    result_index = [0] * len(index)
    result_level = [0] * len(level)

    for i in range(len(A)):
        digit_Ai = int((A[i] / radix ** digit) % radix)
        C[digit_Ai] += 1

    for i in range(1, radix):
        C[i] += C[i - 1]

    for i in range(len(A) - 1, -1, -1):
        digit_Ai = int((A[i] / radix ** digit) % radix)
        B[C[digit_Ai] - 1] = A[i]
        result_index[C[digit_Ai] - 1] = index[i]
        result_level[C[digit_Ai] - 1] = level[i]
        C[digit_Ai] -= 1

    return B, result_index, result_level


def radix_sort(A, index, level):

    max_digit_level = int(math.floor(math.log(max(level), 10)) + 1)

    for digit in range(max_digit_level):
        level[::-1], index[::-1], A[::-1] = counting_sort(level, digit, 10, index, A)

    max_digit = int(math.floor(math.log(max(A), 10)) + 1)
    for digit in range(max_digit):
        A, index, level = counting_sort(A, digit, 10, index, level)

    return A, index, level


TYPE_TREE = 3
index = [17, 2, 23, 3, 1]
level = [3, 2, 3, 1, 2]
new_index_sort = [0] * len(index)
max_l = max(level)

for i in range(len(new_index_sort)):
    new_index_sort[i] = ((index[i] + 1) << TYPE_TREE * (max_l - level[i])) - 1

B, result_index, result_level = radix_sort(new_index_sort, index, level)

print(result_index)
print(result_level)
