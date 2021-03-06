import re
from cffi import FFI

embed = False
clingo_dir = '/home/kaminski/.local/opt/potassco/debug'
script_h = '''\
'''
clingo_h = '''\
#include <clingo.h>
'''

ffi = FFI()

cnt = []
with open(f'{clingo_dir}/include/clingo.h') as f:
    for line in f:
        if not re.match(r' *(#|//|extern *"C" *{|}$|$)', line):
            cnt.append(line.replace('CLINGO_VISIBILITY_DEFAULT ', ''))
    if embed:
        cnt.append(script_h.replace('CLINGO_VISIBILITY_DEFAULT ', ''))

# callbacks
cnt.append('extern "Python" bool pyclingo_solve_event_callback(clingo_solve_event_type_t type, void *event, void *data, bool *goon);')
cnt.append('extern "Python" void pyclingo_logger_callback(clingo_warning_t code, char const *message, void *data);')
cnt.append('extern "Python" bool pyclingo_ground_callback(clingo_location_t const *location, char const *name, clingo_symbol_t const *arguments, size_t arguments_size, void *data, clingo_symbol_callback_t symbol_callback, void *symbol_callback_data);')
# propagator callbacks
cnt.append('extern "Python" bool pyclingo_propagator_init(clingo_propagate_init_t *init, void *data);')
cnt.append('extern "Python" bool pyclingo_propagator_propagate(clingo_propagate_control_t *control, clingo_literal_t const *changes, size_t size, void *data);')
cnt.append('extern "Python" void pyclingo_propagator_undo(clingo_propagate_control_t const *control, clingo_literal_t const *changes, size_t size, void *data);')
cnt.append('extern "Python" bool pyclingo_propagator_check(clingo_propagate_control_t *control, void *data);')
cnt.append('extern "Python" bool pyclingo_propagator_decide(clingo_id_t thread_id, clingo_assignment_t const *assignment, clingo_literal_t fallback, void *data, clingo_literal_t *decision);')
# observer callbacks
cnt.append('extern "Python" bool pyclingo_observer_init_program(bool incremental, void *data);')
cnt.append('extern "Python" bool pyclingo_observer_begin_step(void *data);')
cnt.append('extern "Python" bool pyclingo_observer_end_step(void *data);')
cnt.append('extern "Python" bool pyclingo_observer_rule(bool choice, clingo_atom_t const *head, size_t head_size, clingo_literal_t const *body, size_t body_size, void *data);')
cnt.append('extern "Python" bool pyclingo_observer_weight_rule(bool choice, clingo_atom_t const *head, size_t head_size, clingo_weight_t lower_bound, clingo_weighted_literal_t const *body, size_t body_size, void *data);')
cnt.append('extern "Python" bool pyclingo_observer_minimize(clingo_weight_t priority, clingo_weighted_literal_t const* literals, size_t size, void *data);')
cnt.append('extern "Python" bool pyclingo_observer_project(clingo_atom_t const *atoms, size_t size, void *data);')
cnt.append('extern "Python" bool pyclingo_observer_output_atom(clingo_symbol_t symbol, clingo_atom_t atom, void *data);')
cnt.append('extern "Python" bool pyclingo_observer_output_term(clingo_symbol_t symbol, clingo_literal_t const *condition, size_t size, void *data);')
cnt.append('extern "Python" bool pyclingo_observer_output_csp(clingo_symbol_t symbol, int value, clingo_literal_t const *condition, size_t size, void *data);')
cnt.append('extern "Python" bool pyclingo_observer_external(clingo_atom_t atom, clingo_external_type_t type, void *data);')
cnt.append('extern "Python" bool pyclingo_observer_assume(clingo_literal_t const *literals, size_t size, void *data);')
cnt.append('extern "Python" bool pyclingo_observer_heuristic(clingo_atom_t atom, clingo_heuristic_type_t type, int bias, unsigned priority, clingo_literal_t const *condition, size_t size, void *data);')
cnt.append('extern "Python" bool pyclingo_observer_acyc_edge(int node_u, int node_v, clingo_literal_t const *condition, size_t size, void *data);')
cnt.append('extern "Python" bool pyclingo_observer_theory_term_number(clingo_id_t term_id, int number, void *data);')
cnt.append('extern "Python" bool pyclingo_observer_theory_term_string(clingo_id_t term_id, char const *name, void *data);')
cnt.append('extern "Python" bool pyclingo_observer_theory_term_compound(clingo_id_t term_id, int name_id_or_type, clingo_id_t const *arguments, size_t size, void *data);')
cnt.append('extern "Python" bool pyclingo_observer_theory_element(clingo_id_t element_id, clingo_id_t const *terms, size_t terms_size, clingo_literal_t const *condition, size_t condition_size, void *data);')
cnt.append('extern "Python" bool pyclingo_observer_theory_atom(clingo_id_t atom_id_or_zero, clingo_id_t term_id, clingo_id_t const *elements, size_t size, void *data);')
cnt.append('extern "Python" bool pyclingo_observer_theory_atom_with_guard(clingo_id_t atom_id_or_zero, clingo_id_t term_id, clingo_id_t const *elements, size_t size, clingo_id_t operator_id, clingo_id_t right_hand_side_id, void *data);')
# application callbacks
cnt.append('extern "Python" char const *pyclingo_application_program_name(void *data);')
cnt.append('extern "Python" char const *pyclingo_application_version(void *data);')
cnt.append('extern "Python" unsigned pyclingo_application_message_limit(void *data);')
cnt.append('extern "Python" bool pyclingo_application_main(clingo_control_t *control, char const *const * files, size_t size, void *data);')
cnt.append('extern "Python" void pyclingo_application_logger(clingo_warning_t code, char const *message, void *data);')
cnt.append('extern "Python" bool pyclingo_application_print_model(clingo_model_t const *model, clingo_default_model_printer_t printer, void *printer_data, void *data);')
cnt.append('extern "Python" bool pyclingo_application_register_options(clingo_options_t *options, void *data);')
cnt.append('extern "Python" bool pyclingo_application_validate_options(void *data);')
# application options callbacks
cnt.append('extern "Python" bool pyclingo_application_options_parse(char const *value, void *data);')
# ast callbacks
cnt.append('extern "Python" bool pyclingo_ast_callback(clingo_ast_t const *, void *);')

if embed:
    ffi.embedding_api('''\
bool pyclingo_execute_(void *loc, char const *code, void *data);
bool pyclingo_call_(void *loc, char const *name, void *arguments, size_t size, void *symbol_callback, void *symbol_callback_data, void *data);
bool pyclingo_callable_(char const * name, bool *ret, void *data);
bool pyclingo_main_(void *ctl, void *data);
''')

    ffi.embedding_init_code("""\
from collections.abc import Iterable
from traceback import format_exception
import sys
import __main__

sys.path.insert(0, '/home/kaminski/git/rkaminsk/python-clingo')
from clingo._internal import _ffi, _handle_error, _lib
from clingo.control import Control
from clingo.symbol import Symbol

def _cb_error_top_level(exception, exc_value, traceback):
    msg = format_exception(exception, exc_value, traceback)
    _lib.clingo_set_error(_lib.clingo_error_runtime, msg)
    return False

@_ffi.def_extern(onerror=_cb_error_top_level)
def pyclingo_execute_(loc, code, data):
    exec(_ffi.string(code).decode(), __main__.__dict__, __main__.__dict__)
    return True

@_ffi.def_extern(onerror=_cb_error_top_level)
def pyclingo_call_(loc, name, arguments, size, symbol_callback, symbol_callback_data, data):
    context = _ffi.from_handle(data).data
    py_name = _ffi.string(name).decode()
    fun = getattr(__main__ if context is None else context, py_name)

    args = []
    for i in range(arguments_size):
        args.append(Symbol(arguments[i]))

    ret = fun(*args)
    symbols = list(ret) if isinstance(ret, Iterable) else [ret]

    c_symbols = _ffi.new('clingo_symbol_t[]', len(symbols))
    for i, sym in enumerate(symbols):
        c_symbols[i] = sym._rep
    _handle_error(symbol_callback(c_symbols, len(symbols), symbol_callback_data))

@_ffi.def_extern(onerror=_cb_error_top_level)
def pyclingo_callable_(name, ret, data):
    py_name = _ffi.string(name).decode()
    ret[0] = py_name in __main__.__dict__ and callable(__main__.__dict__[py_name])
    return True

@_ffi.def_extern(onerror=_cb_error_top_level)
def pyclingo_main_(ctl, data):
    __main__.main(Control(_ffi.cast('clingo_control_t*', ctl)))
    return True
""")

if embed:
    clingo_h += script_h

ffi.set_source(
    '_clingo',
    clingo_h,
    include_dirs=[f'{clingo_dir}/include'],
    library_dirs=[f'{clingo_dir}/lib'],
    extra_link_args=[f'-Wl,-rpath={clingo_dir}/lib'],
    libraries=['clingo'])
ffi.cdef(''.join(cnt))

ffi.compile(verbose=1)
