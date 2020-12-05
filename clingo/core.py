'''
This modules defines core functionality used throught the clingo module.
'''
from enum import Enum

from ._internal import _cb_error_panic, _ffi, _lib

def _clingo_version():
    p_major = _ffi.new('int*')
    p_minor = _ffi.new('int*')
    p_revision = _ffi.new('int*')
    _lib.clingo_version(p_major, p_minor, p_revision)
    return f"{p_major[0]}.{p_minor[0]}.{p_revision[0]}"

__version__: str = _clingo_version()

class MessageCode(Enum):
    '''
    Enumeration of the different types of messages.

    Attributes
    ----------
    OperationUndefined : MessageCode
        Inform about an undefined arithmetic operation or unsupported weight of an
        aggregate.
    RuntimeError : MessageCode
        To report multiple errors; a corresponding runtime error is raised later.
    AtomUndefined : MessageCode
        Informs about an undefined atom in program.
    FileIncluded : MessageCode
        Indicates that the same file was included multiple times.
    VariableUnbounded : MessageCode
        Informs about a CSP variable with an unbounded domain.
    GlobalVariable : MessageCode
        Informs about a global variable in a tuple of an aggregate element.
    Other : MessageCode
        Reports other kinds of messages.
    '''
    AtomUndefined = _lib.clingo_warning_atom_undefined
    FileIncluded = _lib.clingo_warning_file_included
    GlobalVariable = _lib.clingo_warning_global_variable
    OperationUndefined = _lib.clingo_warning_operation_undefined
    Other = _lib.clingo_warning_atom_undefined
    RuntimeError = _lib.clingo_warning_runtime_error
    VariableUnbounded = _lib.clingo_warning_variable_unbounded

@_ffi.def_extern(onerror=_cb_error_panic)
def _clingo_logger_callback(code, message, data):
    '''
    Low-level logger callback.
    '''
    handler = _ffi.from_handle(data)
    handler(MessageCode(code), message.encode())

class TruthValue(Enum):
    '''
    Enumeration of the different truth values.

    Attributes
    ----------
    True_ : TruthValue
        Represents truth value true.
    False_ : TruthValue
        Represents truth value true.
    Free : TruthValue
        Represents absence of a truth value.
    Release : TruthValue
        Indicates that an atom is to be released.
    '''
    False_ = _lib.clingo_truth_value_false
    Free = _lib.clingo_truth_value_free
    True_ = _lib.clingo_truth_value_true
    Release = 3
