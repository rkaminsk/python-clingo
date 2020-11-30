from cffi import FFI
import re

ffi = FFI()

cnt = []
with open('/home/kaminski/.local/opt/potassco/release/include/clingo.h') as f:
    for line in f:
        if not re.match(r' *(#|//|extern *"C" *{|}$|$)', line):
            cnt.append(line.replace('CLINGO_VISIBILITY_DEFAULT ', ''))


cnt.append('extern "Python" bool _clingo_solve_event_callback(clingo_solve_event_type_t type, void *event, void *data, bool *goon);')

ffi.set_source(
    '_clingo',
    '#include <clingo.h>',
    include_dirs=['/home/kaminski/.local/opt/potassco/release/include'],
    library_dirs=['/home/kaminski/.local/opt/potassco/release/lib'],
    extra_link_args=['-Wl,-rpath=/home/kaminski/.local/opt/potassco/release/lib'],
    libraries=['clingo'])
ffi.cdef(''.join(cnt))
ffi.compile()

#ctl = ffi.new('clingo_control_t **')
#lib.clingo_control_new(ffi.NULL, 0, ffi.NULL, ffi.NULL, 20, ctl)

#print(ctl)
