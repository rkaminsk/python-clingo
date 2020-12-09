from cffi import FFI
import re

ffi = FFI()

cnt = []
with open('/home/kaminski/.local/opt/potassco/release/include/clingo.h') as f:
    for line in f:
        if not re.match(r' *(#|//|extern *"C" *{|}$|$)', line):
            cnt.append(line.replace('CLINGO_VISIBILITY_DEFAULT ', ''))


# callbacks
cnt.append('extern "Python" bool _clingo_solve_event_callback(clingo_solve_event_type_t type, void *event, void *data, bool *goon);')
cnt.append('extern "Python" void _clingo_logger_callback(clingo_warning_t code, char const *message, void *data);')
cnt.append('extern "Python" bool _clingo_ground_callback(clingo_location_t const *location, char const *name, clingo_symbol_t const *arguments, size_t arguments_size, void *data, clingo_symbol_callback_t symbol_callback, void *symbol_callback_data);')
# propagator callbacks
cnt.append('extern "Python" bool _clingo_propagator_init(clingo_propagate_init_t *init, void *data);')
cnt.append('extern "Python" bool _clingo_propagator_propagate(clingo_propagate_control_t *control, clingo_literal_t const *changes, size_t size, void *data);')
cnt.append('extern "Python" void _clingo_propagator_undo(clingo_propagate_control_t const *control, clingo_literal_t const *changes, size_t size, void *data);')
cnt.append('extern "Python" bool _clingo_propagator_check(clingo_propagate_control_t *control, void *data);')
cnt.append('extern "Python" bool _clingo_propagator_decide(clingo_id_t thread_id, clingo_assignment_t const *assignment, clingo_literal_t fallback, void *data, clingo_literal_t *decision);')
# observer callbacks
cnt.append('extern "Python" bool _clingo_observer_init_program(bool incremental, void *data);')
cnt.append('extern "Python" bool _clingo_observer_begin_step(void *data);')
cnt.append('extern "Python" bool _clingo_observer_end_step(void *data);')
cnt.append('extern "Python" bool _clingo_observer_rule(bool choice, clingo_atom_t const *head, size_t head_size, clingo_literal_t const *body, size_t body_size, void *data);')
cnt.append('extern "Python" bool _clingo_observer_weight_rule(bool choice, clingo_atom_t const *head, size_t head_size, clingo_weight_t lower_bound, clingo_weighted_literal_t const *body, size_t body_size, void *data);')
cnt.append('extern "Python" bool _clingo_observer_minimize(clingo_weight_t priority, clingo_weighted_literal_t const* literals, size_t size, void *data);')
cnt.append('extern "Python" bool _clingo_observer_project(clingo_atom_t const *atoms, size_t size, void *data);')
cnt.append('extern "Python" bool _clingo_observer_output_atom(clingo_symbol_t symbol, clingo_atom_t atom, void *data);')
cnt.append('extern "Python" bool _clingo_observer_output_term(clingo_symbol_t symbol, clingo_literal_t const *condition, size_t size, void *data);')
cnt.append('extern "Python" bool _clingo_observer_output_csp(clingo_symbol_t symbol, int value, clingo_literal_t const *condition, size_t size, void *data);')
cnt.append('extern "Python" bool _clingo_observer_external(clingo_atom_t atom, clingo_external_type_t type, void *data);')
cnt.append('extern "Python" bool _clingo_observer_assume(clingo_literal_t const *literals, size_t size, void *data);')
cnt.append('extern "Python" bool _clingo_observer_heuristic(clingo_atom_t atom, clingo_heuristic_type_t type, int bias, unsigned priority, clingo_literal_t const *condition, size_t size, void *data);')
cnt.append('extern "Python" bool _clingo_observer_acyc_edge(int node_u, int node_v, clingo_literal_t const *condition, size_t size, void *data);')
cnt.append('extern "Python" bool _clingo_observer_theory_term_number(clingo_id_t term_id, int number, void *data);')
cnt.append('extern "Python" bool _clingo_observer_theory_term_string(clingo_id_t term_id, char const *name, void *data);')
cnt.append('extern "Python" bool _clingo_observer_theory_term_compound(clingo_id_t term_id, int name_id_or_type, clingo_id_t const *arguments, size_t size, void *data);')
cnt.append('extern "Python" bool _clingo_observer_theory_element(clingo_id_t element_id, clingo_id_t const *terms, size_t terms_size, clingo_literal_t const *condition, size_t condition_size, void *data);')
cnt.append('extern "Python" bool _clingo_observer_theory_atom(clingo_id_t atom_id_or_zero, clingo_id_t term_id, clingo_id_t const *elements, size_t size, void *data);')
cnt.append('extern "Python" bool _clingo_observer_theory_atom_with_guard(clingo_id_t atom_id_or_zero, clingo_id_t term_id, clingo_id_t const *elements, size_t size, clingo_id_t operator_id, clingo_id_t right_hand_side_id, void *data);')
# application callbacks
cnt.append('extern "Python" char const *_clingo_application_program_name(void *data);')
cnt.append('extern "Python" char const *_clingo_application_version(void *data);')
cnt.append('extern "Python" unsigned _clingo_application_message_limit(void *data);')
cnt.append('extern "Python" bool _clingo_application_main(clingo_control_t *control, char const *const * files, size_t size, void *data);')
cnt.append('extern "Python" void _clingo_application_logger(clingo_warning_t code, char const *message, void *data);')
cnt.append('extern "Python" bool _clingo_application_print_model(clingo_model_t const *model, clingo_default_model_printer_t printer, void *printer_data, void *data);')
cnt.append('extern "Python" bool _clingo_application_register_options(clingo_options_t *options, void *data);')
cnt.append('extern "Python" bool _clingo_application_validate_options(void *data);')
# application options callbacks
cnt.append('extern "Python" bool _clingo_application_options_parse(char const *value, void *data);')
# ast callbacks
cnt.append('extern "Python" bool _clingo_ast_callback(clingo_ast_t const *, void *);')

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
