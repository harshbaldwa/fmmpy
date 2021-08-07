import numpy as np
from math import sin
from compyle.types import annotate
from compyle.template import Template
import compyle.array as ary
from compyle.parallel import Elementwise
from compyle.types import KnownType
from compyle.api import declare
from compyle.low_level import cast


class SetConstant(Template):
    def __init__(self, name, arrays):
        super(SetConstant, self).__init__(name=name)
        self.arrays = arrays
        self.number = len(arrays)

    def extra_args(self):
        return self.arrays, {"intp": ','.join(self.arrays)}

    @annotate(i='int')
    def template(self, i):
        '''
        % for t in range(obj.number//2):
        ${obj.arrays[t]}[i] = ${obj.arrays[obj.number//2+t]}[i]
        % endfor
        '''


backend = 'cython'
set_const = SetConstant('set_const', ['x', 'y', 'z', 'w']).function
# set_const = SetConstant('set_const', ['x']).function
x = ary.ones(10, dtype=np.int32, backend=backend)
y = ary.zeros(10, dtype=np.int32, backend=backend)
z = ary.ones(10, dtype=np.int32, backend=backend)
w = ary.zeros(10, dtype=np.int32, backend=backend)

print(set_const.source)

# e = Elementwise(set_const, backend=backend)
# e(x, y, z)
# # e(x)

# print(x, y, z)
