from cffi import FFI
import re

ffi = FFI()

cnt = []
with open('/home/kaminski/.local/opt/potassco/release/include/clingo.h') as f:
    for line in f:
        if not re.match(r' *(#|//|extern *"C" *{|}$|$)', line):
            cnt.append(line.replace('CLINGO_VISIBILITY_DEFAULT ', ''))


cnt.append('extern "Python" bool _clingo_solve_event_callback(clingo_solve_event_type_t type, void *event, void *data, bool *goon);')
cnt.append('extern "Python" void _clingo_logger_callback(clingo_warning_t code, char const *message, void *data);')
cnt.append('extern "Python" bool _clingo_ground_callback(clingo_location_t const *location, char const *name, clingo_symbol_t const *arguments, size_t arguments_size, void *data, clingo_symbol_callback_t symbol_callback, void *symbol_callback_data);')
cnt.append('extern "Python" bool _clingo_propagator_init(clingo_propagate_init_t *init, void *data);')
cnt.append('extern "Python" bool _clingo_propagator_propagate(clingo_propagate_control_t *control, clingo_literal_t const *changes, size_t size, void *data);')
cnt.append('extern "Python" void _clingo_propagator_undo(clingo_propagate_control_t const *control, clingo_literal_t const *changes, size_t size, void *data);')
cnt.append('extern "Python" bool _clingo_propagator_check(clingo_propagate_control_t *control, void *data);')
cnt.append('extern "Python" bool _clingo_propagator_decide(clingo_id_t thread_id, clingo_assignment_t const *assignment, clingo_literal_t fallback, void *data, clingo_literal_t *decision);')

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
