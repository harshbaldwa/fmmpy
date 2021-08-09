from tree_compyle import *


def test_copy_arr():
    arr = ary.Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    arr2 = arr.empty(10)

    backend = 'cython'

    copy_value = CopyValue('copy_value', ['a', 'b']).function
    e = Elementwise(copy_value, backend=backend)
    e(arr, arr2)
    print(arr2 == arr)
    assert arr2 == arr
