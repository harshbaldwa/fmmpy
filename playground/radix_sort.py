import numpy as np
import math


def counting_sort(A, digit, radix, index):
    B = [0] * len(A)
    C = [0] * int(radix)
    result_index = [0] * len(index)

    for i in range(len(A)):
        digit_Ai = int((A[i] / radix ** digit) % radix)
        C[digit_Ai] += 1

    for i in range(1, radix):
        C[i] += C[i - 1]

    for i in range(len(A) - 1, -1, -1):
        digit_Ai = int((A[i] / radix ** digit) % radix)
        B[C[digit_Ai] - 1] = A[i]
        result_index[C[digit_Ai] - 1] = index[i]
        C[digit_Ai] -= 1

    return B, result_index


def radix_sort(A, index):
    max_digit = int(math.floor(math.log(max(A), 10)) + 1)
    for digit in range(max_digit):
        A, index = counting_sort(A, digit, 10, index)
    return A, index


# random array
A = [3, 7, 5, 43, 76, 34, 11, 10, 100, 2]
index = np.arange(len(A))
B, result_index = radix_sort(A, index)

print(B)
print(result_index)
