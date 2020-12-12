"""
Helper to generate constructor function for clingo's python API.

Attention: Because this code exists just for code generation and won't run in
production, the functions in this module have been written with the least
amount of effort possible. Always check the generated code!
"""

from _clingo import ffi as _ffi, lib as _lib
import sys
from itertools import chain

_an = _lib.g_clingo_ast_attribute_names
_cs = _lib.g_clingo_ast_constructors

def to_camel_case(s):
    components = s.split('_')
    return ''.join(x.title() for x in components)

def argument_type_str(idx):
    if idx == _lib.clingo_ast_attribute_type_number:
        return 'int'
    if idx == _lib.clingo_ast_attribute_type_string:
        return 'str'
    if idx == _lib.clingo_ast_attribute_type_symbol:
        return 'Symbol'
    if idx == _lib.clingo_ast_attribute_type_location:
        return 'Location'
    if idx == _lib.clingo_ast_attribute_type_ast:
        return 'AST'
    if idx == _lib.clingo_ast_attribute_type_optional_ast:
        return 'Optional[AST]'
    if idx == _lib.clingo_ast_attribute_type_ast_array:
        return 'Sequence[AST]'
    assert idx == _lib.clingo_ast_attribute_type_string_array
    return 'Sequence[str]'

def generate_arguments(constructor):
    args = []
    for i in range(constructor.size):
        argument = constructor.arguments[i]
        argument_type = argument_type_str(argument.type)
        name = _ffi.string(_an.names[argument.attribute]).decode()
        args.append((name, argument_type))
    return args

def generate_parameter(name, idx):
    if idx == _lib.clingo_ast_attribute_type_number:
        return [f"_ffi.cast('int', {name})"]
    if idx == _lib.clingo_ast_attribute_type_string:
        return [f"_ffi.new('char const[]', {name}.encode())"]
    if idx == _lib.clingo_ast_attribute_type_symbol:
        return [f"_ffi.cast('clingo_symbol_t', {name}._rep)"]
    if idx == _lib.clingo_ast_attribute_type_location:
        return [f'c_{name}[0]']
    if idx == _lib.clingo_ast_attribute_type_ast:
        return [f'{name}._rep']
    if idx == _lib.clingo_ast_attribute_type_optional_ast:
        return [f'_ffi.NULL if {name} is None else {name}._rep']
    if idx == _lib.clingo_ast_attribute_type_ast_array:
        return [f"_ffi.new('clingo_ast_t*[]', [ x._rep for x in {name} ])", f"_ffi.cast('size_t', len({name}))"]
    assert idx == _lib.clingo_ast_attribute_type_string_array
    return [f"_ffi.new('char*[]', c_{name})", f"_ffi.cast('size_t', len({name}))"]

def generate_aux(name, idx):
    if idx == _lib.clingo_ast_attribute_type_string_array:
        return [f"c_{name} = [ _ffi.new('char[]', x.encode()) for x in {name} ]"]
    if idx == _lib.clingo_ast_attribute_type_location:
        return [f"c_{name} = _c_location({name})"]
    return []

def generate_parameters(constructor):
    args, aux = [], []
    for i in range(constructor.size):
        argument = constructor.arguments[i]
        name = _ffi.string(_an.names[argument.attribute]).decode()
        args.extend(generate_parameter(name, argument.type))
        aux.extend(generate_aux(name, argument.type))
    return args, aux

def generate_sig(name, arguments):
    comma = False
    ret = f'def {name}('
    m = len(ret)
    n = m
    for a, t in arguments:
        if comma:
            ret += ", "
            n += 2
        else:
            comma = True
        arg = f'{a}: {t}'
        if n + len(arg) <= 111:
            n += len(arg)
        else:
            ret = ret[:-1] + "\n" + m * " "
            n = m + len(arg)
        ret += arg
    ret += ")"
    arg = ' -> AST:'
    assert n + len(arg) <= 120
    ret += arg
    return ret

def generate():
    for i in range(_cs.size):
        constructor = _cs.constructors[i]
        c_name = _ffi.string(constructor.name).decode()
        name = to_camel_case(c_name)
        sys.stdout.write(f'\n{generate_sig(name, generate_arguments(constructor))}\n')
        sys.stdout.write("    '''\n")
        sys.stdout.write(f'    Construct an AST node of type `ASTType.{name}`.\n')
        sys.stdout.write("    '''\n")
        parameters, aux = generate_parameters(constructor)
        sys.stdout.write(f"    p_ast = _ffi.new('clingo_ast_t**')\n")
        for x in aux:
            sys.stdout.write(f"    {x}\n")
        sys.stdout.write(f"    _handle_error(_lib.clingo_ast_build(\n")
        sys.stdout.write(f"        _lib.clingo_ast_type_{c_name}, p_ast")
        for param in parameters:
            sys.stdout.write(f",\n        {param}")
        sys.stdout.write(f"))\n")

        sys.stdout.write(f"    return AST(p_ast[0])\n")

generate()
