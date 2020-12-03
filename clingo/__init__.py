'''
The clingo module.

This module provides functions and classes to control the grounding and solving
process.

If the clingo application is build with Python support, clingo will also be
able to execute Python code embedded in logic programs.  Functions defined in a
Python script block are callable during the instantiation process using
`@`-syntax.  The default grounding/solving process can be customized if a main
function is provided.

Note that gringo's precomputed terms (terms without variables and interpreted
functions), called symbols in the following, are wrapped in the Symbol class.
Furthermore, strings, numbers, and tuples can be passed wherever a symbol is
expected - they are automatically converted into a Symbol object.  Functions
called during the grounding process from the logic program must either return a
symbol or a sequence of symbols.  If a sequence is returned, the corresponding
`@`-term is successively substituted by the values in the sequence.

## Examples

The first example shows how to use the clingo module from Python.

    >>> import clingo
    >>> class Context:
    ...     def id(self, x):
    ...         return x
    ...     def seq(self, x, y):
    ...         return [x, y]
    ...
    >>> def on_model(m):
    ...     print (m)
    ...
    >>> ctl = clingo.Control()
    >>> ctl.add("base", [], """\
    ... p(@id(10)).
    ... q(@seq(1,2)).
    ... """)
    >>> ctl.ground([("base", [])], context=Context())
    >>> ctl.solve(on_model=on_model)
    p(10) q(1) q(2)
    SAT

The second example shows how to use Python code from clingo.

    #script (python)

    import clingo

    class Context:
        def id(x):
            return x

        def seq(x, y):
            return [x, y]

    def main(prg):
        prg.ground([("base", [])], context=Context())
        prg.solve()

    #end.

    p(@id(10)).
    q(@seq(1,2)).
'''
# pylint: disable=too-many-lines

from typing import (
    AbstractSet, Any, Callable, Collection, ContextManager, Generic, Hashable, Iterable, Iterator, KeysView, List,
    Mapping, MutableMapping, MutableSequence, Optional, Sequence, Set, Tuple, TypeVar, Union, ValuesView,
    cast)
from numbers import Real
from abc import ABCMeta, abstractmethod
from enum import Enum
from functools import total_ordering
from os import _exit
from traceback import print_exception
from collections import abc
import sys

from _clingo import ffi as _ffi, lib as _lib # type: ignore # pylint: disable=no-name-in-module

# {{{1 auxiliary functions

def _clingo_version():
    p_major = _ffi.new('int*')
    p_minor = _ffi.new('int*')
    p_revision = _ffi.new('int*')
    _lib.clingo_version(p_major, p_minor, p_revision)
    return f"{p_major[0]}.{p_minor[0]}.{p_revision[0]}"

def _str(f_size, f_str, *args):
    p_size = _ffi.new('size_t*')
    f_size(*args, p_size)
    p_str = _ffi.new('char[]', p_size[0])
    f_str(*args, p_str, p_size[0])
    return _ffi.string(p_str).decode()

def _c_call(c_type, c_fun, *args):
    '''
    Helper to simplify calling C functions where the last parameter is a
    reference to the return value.
    '''
    if isinstance(c_type, str):
        p_ret = _ffi.new(f'{c_type}*')
    else:
        p_ret = c_type
    _handle_error(c_fun(*args, p_ret))
    return p_ret[0]

def _c_call2(c_type1, c_type2, c_fun, *args):
    '''
    Helper to simplify calling C functions where the last two parameters are a
    reference to the return value.
    '''
    p_ret1 = _ffi.new(f'{c_type1}*')
    p_ret2 = _ffi.new(f'{c_type2}*')
    _handle_error(c_fun(*args, p_ret1, p_ret2))
    return p_ret1[0], p_ret2[0]

def _to_str(c_str) -> str:
    return _ffi.string(c_str).decode()

def _handle_error(ret: bool, handler=None):
    if not ret:
        code = _lib.clingo_error_code()
        if code == _lib.clingo_error_unknown and handler is not None and handler.error is not None:
            raise handler.error[0](handler.error[1]).with_traceback(handler.error[2])
        msg = _ffi.string(_lib.clingo_error_message()).decode()
        if code == _lib.clingo_error_bad_alloc:
            raise MemoryError(msg)
        raise RuntimeError(msg)

def _cb_error_handler(param: str):
    def handler(exception, exc_value, traceback) -> bool:
        if traceback is not None:
            handler = _ffi.from_handle(traceback.tb_frame.f_locals[param])
            handler.error = (exception, exc_value, traceback)
            _lib.clingo_set_error(_lib.clingo_error_unknown, "python exception".encode())
        else:
            _lib.clingo_set_error(_lib.clingo_error_runtime, "error in callback".encode())
        return False
    return handler

def _cb_error_panic(exception, exc_value, traceback):
    print_exception(exception, exc_value, traceback)
    sys.stderr.write('PANIC: exception in nothrow scope')
    _exit(1)

Key = TypeVar('Key')
Value = TypeVar('Value')
class Lookup(Generic[Key, Value], Collection[Value]):
    '''
    A collection of values with additional lookup by key.
    '''
    @abstractmethod
    def __getitem__(self, key: Key) -> Optional[Value]:
        pass

# {{{1 basics [100%]

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

# {{{1 symbols [100%]

class SymbolType(Enum):
    '''
    Enumeration of the different types of symbols.

    Attributes
    ----------
    Number : SymbolType
        A numeric symbol, e.g., `1`.
    String : SymbolType
        A string symbol, e.g., `"a"`.
    Function : SymbolType
        A function symbol, e.g., `c`, `(1, "a")`, or `f(1,"a")`.
    Infimum : SymbolType
        The `#inf` symbol.
    Supremum : SymbolType
        The `#sup` symbol
    '''
    Function = _lib.clingo_symbol_type_function
    Infimum  = _lib.clingo_symbol_type_infimum
    Number  = _lib.clingo_symbol_type_number
    String  = _lib.clingo_symbol_type_string
    Supremum  = _lib.clingo_symbol_type_supremum

@total_ordering
class Symbol:
    '''
    Represents a gringo symbol.

    This includes numbers, strings, functions (including constants with
    `len(arguments) == 0` and tuples with `len(name) == 0`), `#inf` and `#sup`.

    Symbol objects implement Python's rich comparison operators and are ordered
    like in gringo. They can also be used as keys in dictionaries. Their string
    representation corresponds to their gringo representation.

    Notes
    -----
    Note that this class does not have a constructor. Instead there are the
    functions `Number`, `String`, and `Function` to construct symbol objects or the
    preconstructed symbols `Infimum` and `Supremum`.
    '''
    def __init__(self, rep):
        self._rep = rep

    def __str__(self) -> str:
        return _str(_lib.clingo_symbol_to_string_size, _lib.clingo_symbol_to_string, self._rep)

    def __hash__(self) -> int:
        return _lib.clingo_symbol_hash(self._rep)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Symbol):
            return NotImplemented
        return _lib.clingo_symbol_is_equal_to(self._rep, other._rep)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Symbol):
            return NotImplemented
        return _lib.clingo_symbol_is_less_than(self._rep, other._rep)

    def match(self, name: str, arity: int, positive: bool = True) -> bool:
        '''
        Check if this is a function symbol with the given signature.

        Parameters
        ----------
        name : str
            The name of the function.

        arity : int
            The arity of the function.

        positive : bool
            Whether to match positive or negative signatures.

        Returns
        -------
        bool
            Whether the function matches.
        '''
        return (self.type == SymbolType.Function and
                self.positive == positive and
                self.name == name and
                len(self.arguments) == arity)

    @property
    def arguments(self) -> List['Symbol']:
        '''
        The arguments of a function.
        '''
        p_args = _ffi.new('clingo_symbol_t**')
        size = _c_call('size_t', _lib.clingo_symbol_arguments, self._rep, p_args)
        ret = []
        for i in range(size):
            ret.append(Symbol(p_args[0][i]))
        return ret

    @property
    def name(self) -> str:
        '''
        The name of a function.
        '''
        return _to_str(_c_call('char*', _lib.clingo_symbol_name, self._rep))

    @property
    def negative(self) -> bool:
        '''
        The inverted sign of a function.
        '''
        return _c_call('bool', _lib.clingo_symbol_is_negative, self._rep)

    @property
    def number(self) -> int:
        '''
        The value of a number.
        '''
        return _c_call('int', _lib.clingo_symbol_number, self._rep)

    @property
    def positive(self) -> bool:
        '''
        The sign of a function.
        '''
        return _c_call('bool', _lib.clingo_symbol_is_positive, self._rep)

    @property
    def string(self) -> str:
        '''
        The value of a string.
        '''
        return _to_str(_c_call('char*', _lib.clingo_symbol_string, self._rep))

    @property
    def type(self) -> SymbolType:
        '''
        The type of the symbol.
        '''
        return SymbolType(_lib.clingo_symbol_type(self._rep))

def Function(name: str, arguments: Sequence[Symbol]=[], positive: bool=True) -> Symbol:
    '''
    Construct a function symbol.

    This includes constants and tuples. Constants have an empty argument list and
    tuples have an empty name. Functions can represent classically negated atoms.
    Argument `positive` has to be set to false to represent such atoms.

    Parameters
    ----------
    name : str
        The name of the function (empty for tuples).
    arguments : Sequence[Symbol]=[]
        The arguments in form of a list of symbols.
    positive : bool=True
        The sign of the function (tuples must not have signs).

    Returns
    -------
    Symbol
    '''
    # pylint: disable=protected-access,invalid-name,dangerous-default-value
    p_sym = _ffi.new('clingo_symbol_t*')
    c_args = _ffi.new('clingo_symbol_t[]', len(arguments))
    for i, arg in enumerate(arguments):
        c_args[i] = arg._rep
    _handle_error(_lib.clingo_symbol_create_function(name.encode(), c_args, len(arguments), positive, p_sym))
    return Symbol(p_sym[0])

def Number(number: int) -> Symbol:
    '''
    Construct a numeric symbol given a number.

    Parameters
    ----------
    number : int
        The given number.

    Returns
    -------
    Symbol
    '''
    # pylint: disable=invalid-name
    p_rep = _ffi.new('clingo_symbol_t*')
    _lib.clingo_symbol_create_number(number, p_rep)
    return Symbol(p_rep[0])

def String(string: str) -> Symbol:
    '''
    Construct a string symbol given a string.

    Parameters
    ----------
    string : str
        The given string.

    Returns
    -------
    Symbol
    '''
    # pylint: disable=invalid-name
    return Symbol(_c_call('clingo_symbol_t', _lib.clingo_symbol_create_string, string.encode()))

def Tuple_(arguments: Sequence[Symbol]) -> Symbol:
    '''
    A shortcut for `Function("", arguments)`.

    Parameters
    ----------
    arguments : Sequence[Symbol]
        The arguments in form of a list of symbols.

    Returns
    -------
    Symbol

    See Also
    --------
    clingo.Function
    '''
    # pylint: disable=invalid-name
    return Function("", arguments)

_p_infimum = _ffi.new('clingo_symbol_t*')
_p_supremum = _ffi.new('clingo_symbol_t*')
_lib.clingo_symbol_create_infimum(_p_infimum)
_lib.clingo_symbol_create_supremum(_p_supremum)

Infimum: Symbol = Symbol(_p_infimum[0])
Supremum: Symbol = Symbol(_p_supremum[0])

@_ffi.def_extern(onerror=_cb_error_panic)
def _clingo_logger_callback(code, message, data):
    '''
    Low-level logger callback.
    '''
    handler = _ffi.from_handle(data)
    handler(MessageCode(code), message.encode())

def parse_term(string: str, logger: Callable[[MessageCode,str],None]=None, message_limit: int=20) -> Symbol:
    '''
    Parse the given string using gringo's term parser for ground terms.

    The function also evaluates arithmetic functions.

    Parameters
    ----------
    string : str
        The string to be parsed.
    logger : Callable[[MessageCode,str],None]=None
        Function to intercept messages normally printed to standard error.
    message_limit : int=20
        Maximum number of messages passed to the logger.

    Returns
    -------
    Symbol

    Examples
    --------
        >>> import clingo
        >>> clingo.parse_term('p(1+2)')
        p(3)
    '''
    if logger is not None:
        # pylint: disable=protected-access
        c_handle = _ffi.new_handle(logger)
        c_cb = _lib._clingo_logger_callback
    else:
        c_handle = _ffi.NULL
        c_cb = _ffi.NULL
    return Symbol(_c_call('clingo_symbol_t', _lib.clingo_parse_term, string.encode(), c_cb, c_handle, message_limit))

# {{{1 symbolic atoms [100%]

class SymbolicAtom:
    '''
    Captures a symbolic atom and provides properties to inspect its state.
    '''
    def __init__(self, rep, it):
        self._rep = rep
        self._it = it

    def match(self, name: str, arity: int, positive: bool=True) -> bool:
        '''
        Check if the atom matches the given signature.

        Parameters
        ----------
        name : str
            The name of the function.

        arity : int
            The arity of the function.

        positive : bool
            Whether to match positive or negative signatures.

        Returns
        -------
        bool
            Whether the function matches.

        See Also
        --------
        Symbol.match
        '''
        return self.symbol.match(name, arity, positive)

    @property
    def is_external(self) -> bool:
        '''
        Whether the atom is an external atom.
        '''
        return _c_call('bool', _lib.clingo_symbolic_atoms_is_external, self._rep, self._it)

    @property
    def is_fact(self) -> bool:
        '''
        Whether the atom is a fact.
        '''
        return _c_call('bool', _lib.clingo_symbolic_atoms_is_fact, self._rep, self._it)

    @property
    def literal(self) -> int:
        '''
        The program literal associated with the atom.
        '''
        return _c_call('clingo_literal_t', _lib.clingo_symbolic_atoms_literal, self._rep, self._it)

    @property
    def symbol(self) -> Symbol:
        '''
        The representation of the atom in form of a symbol.
        '''
        return Symbol(_c_call('clingo_symbol_t', _lib.clingo_symbolic_atoms_symbol, self._rep, self._it))

class SymbolicAtoms(Lookup[Symbol,SymbolicAtom]):
    '''
    This class provides read-only access to the atom base of the grounder.

    Implements: `Lookup[Symbol,SymbolicAtom]`.

    Examples
    --------

        >>> import clingo
        >>> prg = clingo.Control()
        >>> prg.add('base', [], """\
        ... p(1).
        ... { p(3) }.
        ... #external p(1..3).
        ...
        ... q(X) :- p(X).
        ... """)
        >>> prg.ground([("base", [])])
        >>> len(prg.symbolic_atoms)
        6
        >>> prg.symbolic_atoms[clingo.Function("p", [2])] is not None
        True
        >>> prg.symbolic_atoms[clingo.Function("p", [4])] is None
        True
        >>> prg.symbolic_atoms.signatures
        [('p', 1L, True), ('q', 1L, True)]
        >>> [(x.symbol, x.is_fact, x.is_external)
        ...  for x in prg.symbolic_atoms.by_signature("p", 1)]
        [(p(1), True, False), (p(3), False, False), (p(2), False, True)]
    '''
    def __init__(self, rep):
        self._rep = rep

    def _iter(self, p_sig) -> Iterator[SymbolicAtom]:
        p_it = _ffi.new('clingo_symbolic_atom_iterator_t*')
        p_valid = _ffi.new('bool*')
        _handle_error(_lib.clingo_symbolic_atoms_begin(self._rep, p_sig, p_it))
        while _c_call(p_valid, _lib.clingo_symbolic_atoms_is_valid, self._rep, p_it[0]):
            yield SymbolicAtom(self._rep, p_it[0])
            _handle_error(_lib.clingo_symbolic_atoms_next(self._rep, p_it[0], p_it))

    def __iter__(self) -> Iterator[SymbolicAtom]:
        yield from self._iter(_ffi.NULL)

    def __contains__(self, symbol) -> bool:
        if not isinstance(symbol, Symbol):
            return False

        it = _c_call('clingo_symbolic_atom_iterator_t', _lib.clingo_symbolic_atoms_find, self._rep, symbol._rep)

        return _c_call('bool', _lib.clingo_symbolic_atoms_is_valid, self._rep, it)

    def __getitem__(self, symbol: Symbol) -> Optional[SymbolicAtom]:
        it = _c_call('clingo_symbolic_atom_iterator_t', _lib.clingo_symbolic_atoms_find, self._rep, symbol._rep)

        if not _c_call('bool', _lib.clingo_symbolic_atoms_is_valid, self._rep, it):
            return None

        return SymbolicAtom(self._rep, it)

    def __len__(self) -> int:
        return _c_call('size_t', _lib.clingo_symbolic_atoms_size, self._rep)

    def by_signature(self, name: str, arity: int, positive: bool=True) -> Iterator[SymbolicAtom]:
        '''
        Return an iterator over the symbolic atoms with the given signature.

        Arguments
        ---------
        name : str
            The name of the signature.
        arity : int
            The arity of the signature.
        positive : bool=True
            The sign of the signature.

        Returns
        -------
        Iterator[SymbolicAtom]
        '''
        p_sig = _ffi.new('clingo_signature_t*')
        _handle_error(_lib.clingo_signature_create(name.encode(), arity, positive, p_sig))
        yield from self._iter(p_sig)

    @property
    def signatures(self) -> List[Tuple[str,int,bool]]:
        '''
        The list of predicate signatures occurring in the program.

        The Boolean indicates the sign of the signature.
        '''
        size = _c_call('size_t', _lib.clingo_symbolic_atoms_signatures_size, self._rep)

        p_sigs = _ffi.new('clingo_signature_t[]', size)
        _handle_error(_lib.clingo_symbolic_atoms_signatures(self._rep, p_sigs, size))

        return [ (_to_str(_lib.clingo_signature_name(c_sig)),
                  _lib.clingo_signature_arity(c_sig),
                  _lib.clingo_signature_is_positive(c_sig)) for c_sig in p_sigs ]

# {{{1 theory atoms [100%]

class TheoryTermType(Enum):
    '''
    Enumeration of the different types of theory terms.

    Attributes
    ----------
    Function : TheoryTermType
        For a function theory terms.
    Number : TheoryTermType
        For numeric theory terms.
    Symbol : TheoryTermType
        For symbolic theory terms (symbol here means the term is a string).
    List : TheoryTermType
        For list theory terms.
    Tuple : TheoryTermType
        For tuple theory terms.
    Set : TheoryTermType
        For set theory terms.
    '''
    Function = _lib.clingo_theory_term_type_function
    List = _lib.clingo_theory_term_type_list
    Number = _lib.clingo_theory_term_type_number
    Set = _lib.clingo_theory_term_type_set
    Symbol = _lib.clingo_theory_term_type_symbol
    Tuple = _lib.clingo_theory_term_type_tuple

@total_ordering
class TheoryTerm:
    '''
    `TheoryTerm` objects represent theory terms.

    Theory terms have a readable string representation, implement Python's rich
    comparison operators, and can be used as dictionary keys.
    '''
    def __init__(self, rep, idx):
        self._rep = rep
        self._idx = idx

    def __hash__(self):
        return self._idx

    def __eq__(self, other):
        return self._idx == other._idx

    def __lt__(self, other):
        return self._idx < other._idx

    def __str__(self):
        return _str(_lib.clingo_theory_atoms_term_to_string_size,
                    _lib.clingo_theory_atoms_term_to_string,
                    self._rep, self._idx)

    @property
    def arguments(self) -> List['TheoryTerm']:
        '''
        The arguments of the term (for functions, tuples, list, and sets).
        '''
        args, size = _c_call2('clingo_id_t*', 'size_t', _lib.clingo_theory_atoms_term_arguments, self._rep, self._idx)
        return [TheoryTerm(self._rep, args[i]) for i in range(size)]

    @property
    def name(self) -> str:
        '''
        The name of the term (for symbols and functions).
        '''
        return _to_str(_c_call('char*', _lib.clingo_theory_atoms_term_name, self._rep, self._idx))

    @property
    def number(self) -> int:
        '''
        The numeric representation of the term (for numbers).
        '''
        return _c_call('int', _lib.clingo_theory_atoms_term_number, self._rep, self._idx)

    @property
    def type(self) -> TheoryTermType:
        '''
        The type of the theory term.
        '''
        type_ = _c_call('clingo_theory_term_type_t', _lib.clingo_theory_atoms_term_type, self._rep, self._idx)
        return TheoryTermType(type_)

@total_ordering
class TheoryElement:
    '''
    Class to represent theory elements.

    Theory elements have a readable string representation, implement Python's rich
    comparison operators, and can be used as dictionary keys.
    '''
    def __init__(self, rep, idx):
        self._rep = rep
        self._idx = idx

    def __hash__(self):
        return self._idx

    def __eq__(self, other):
        return self._idx == other._idx

    def __lt__(self, other):
        return self._idx < other._idx

    def __str__(self):
        return _str(_lib.clingo_theory_atoms_element_to_string_size,
                    _lib.clingo_theory_atoms_element_to_string,
                    self._rep, self._idx)

    @property
    def condition(self) -> List[int]:
        '''
        The condition of the element.
        '''
        cond, size = _c_call2('clingo_literal_t*', 'size_t', _lib.clingo_theory_atoms_element_condition,
                              self._rep, self._idx)
        return [cond[i] for i in range(size)]

    @property
    def condition_id(self) -> int:
        '''
        Each condition has an id. This id can be passed to
        `PropagateInit.solver_literal` to obtain a solver literal equivalent to
        the condition.
        '''
        return _c_call('clingo_literal_t', _lib.clingo_theory_atoms_element_condition_id, self._rep, self._idx)

    @property
    def terms(self) -> List[TheoryTerm]:
        '''
        The tuple of the element.
        '''
        terms, size = _c_call2('clingo_id_t*', 'size_t', _lib.clingo_theory_atoms_element_tuple, self._rep, self._idx)
        return [TheoryTerm(self._rep, terms[i]) for i in range(size)]

@total_ordering
class TheoryAtom:
    '''
    Class to represent theory atoms.

    Theory atoms have a readable string representation, implement Python's rich
    comparison operators, and can be used as dictionary keys.
    '''
    def __init__(self, rep, idx):
        self._rep = rep
        self._idx = idx

    def __hash__(self):
        return self._idx

    def __eq__(self, other):
        return self._idx == other._idx

    def __lt__(self, other):
        return self._idx < other._idx

    def __str__(self):
        return _str(_lib.clingo_theory_atoms_atom_to_string_size,
                    _lib.clingo_theory_atoms_atom_to_string,
                    self._rep, self._idx)

    @property
    def elements(self) -> List[TheoryElement]:
        '''
        The elements of the atom.
        '''
        elems, size = _c_call2('clingo_id_t*', 'size_t', _lib.clingo_theory_atoms_atom_elements, self._rep, self._idx)
        return [TheoryElement(self._rep, elems[i]) for i in range(size)]

    @property
    def guard(self) -> Optional[Tuple[str, TheoryTerm]]:
        '''
        The guard of the atom or None if the atom has no guard.
        '''
        if not _c_call('bool', _lib.clingo_theory_atoms_atom_has_guard, self._rep, self._idx):
            return None

        conn, term = _c_call2('char*', 'clingo_id_t', _lib.clingo_theory_atoms_atom_guard, self._rep, self._idx)

        return (_to_str(conn), TheoryTerm(self._rep, term))

    @property
    def literal(self) -> int:
        '''
        The program literal associated with the atom.
        '''
        return _c_call('clingo_literal_t', _lib.clingo_theory_atoms_atom_literal, self._rep, self._idx)

    @property
    def term(self) -> TheoryTerm:
        '''
        The term of the atom.
        '''
        term = _c_call('clingo_id_t', _lib.clingo_theory_atoms_atom_term, self._rep, self._idx)
        return TheoryTerm(self._rep, term)

# {{{1 solving [100%]

class SolveResult:
    '''
    Captures the result of a solve call.

    `SolveResult` objects cannot be constructed from Python. Instead they are
    returned by the solve methods of the Control object.
    '''
    def __init__(self, rep):
        self._rep = rep

    @property
    def exhausted(self) -> bool:
        '''
        True if the search space was exhausted.
        '''
        return (_lib.clingo_solve_result_exhausted & self._rep) != 0

    @property
    def interrupted(self) -> bool:
        '''
        True if the search was interrupted.
        '''
        return (_lib.clingo_solve_result_interrupted & self._rep) != 0

    @property
    def satisfiable(self) -> Optional[bool]:
        '''
        True if the problem is satisfiable, False if the problem is unsatisfiable, or
        None if the satisfiablity is not known.
        '''
        if (_lib.clingo_solve_result_satisfiable & self._rep) != 0:
            return True
        if (_lib.clingo_solve_result_unsatisfiable & self._rep) != 0:
            return False
        return None

    @property
    def unknown(self) -> bool:
        '''
        True if the satisfiablity is not known.

        This is equivalent to satisfiable is None.
        '''
        return self.satisfiable is None

    @property
    def unsatisfiable(self) -> Optional[bool]:
        '''
        True if the problem is unsatisfiable, false if the problem is satisfiable, or
        `None` if the satisfiablity is not known.
        '''
        if (_lib.clingo_solve_result_unsatisfiable & self._rep) != 0:
            return True
        if (_lib.clingo_solve_result_satisfiable & self._rep) != 0:
            return False
        return None

class SolveControl:
    '''
    Object that allows for controlling a running search.

    `SolveControl` objects cannot be constructed from Python. Instead they are
    available via `Model.context`.
    '''
    def __init__(self, rep):
        self._rep = rep

    def add_clause(self, literals: Sequence[Union[Tuple[Symbol,bool],int]]) -> None:
        '''
        Add a clause that applies to the current solving step during the search.

        Parameters
        ----------
        literals : Sequence[Union[Tuple[Symbol,bool],int]]
            List of literals either represented as pairs of symbolic atoms and Booleans
            or as program literals.

        Notes
        -----
        This function can only be called in a model callback or while iterating when
        using a `SolveHandle`.
        '''
        atoms = self.symbolic_atoms
        p_lits = _ffi.new('clingo_literal_t[]', len(literals))
        for i, lit in enumerate(literals):
            if isinstance(lit, int):
                p_lits[i] = lit
            else:
                atom = atoms[lit[0]]
                if atom is not None:
                    slit = atom.literal
                else:
                    slit = -1
                p_lits[i] = slit if lit[1] else -slit

        _handle_error(_lib.clingo_solve_control_add_clause(self._rep, p_lits, len(literals)))

    def _invert(self, lit: Union[Tuple[Symbol,bool],int]) -> Union[Tuple[Symbol,bool],int]:
        if isinstance(lit, int):
            return -lit
        return lit[0], not lit[1]

    def add_nogood(self, literals: Sequence[Union[Tuple[Symbol,bool],int]]) -> None:
        '''
        Equivalent to `SolveControl.add_clause` with the literals inverted.
        '''
        self.add_clause([self._invert(lit) for lit in literals])

    @property
    def symbolic_atoms(self) -> SymbolicAtoms:
        '''
        `SymbolicAtoms` object to inspect the symbolic atoms.
        '''
        atoms = _c_call('clingo_symbolic_atoms_t*', _lib.clingo_solve_control_symbolic_atoms, self._rep)
        return SymbolicAtoms(atoms)

class ModelType(Enum):
    '''
    Enumeration of the different types of models.

    Attributes
    ----------
    StableModel : ModelType
        The model captures a stable model.
    BraveConsequences : ModelType
        The model stores the set of brave consequences.
    CautiousConsequences : ModelType
        The model stores the set of cautious consequences.
    '''
    BraveConsequences = _lib.clingo_model_type_brave_consequences
    CautiousConsequences = _lib.clingo_model_type_cautious_consequences
    StableModel = _lib.clingo_model_type_stable_model

class Model:
    '''
    Provides access to a model during a solve call and provides a `SolveContext`
    object to provided limited support to influence the running search.

    Notes
    -----
    The string representation of a model object is similar to the output of models
    by clingo using the default output.

    `Model` objects cannot be constructed from Python. Instead they are obained
    during solving (see `Control.solve`). Furthermore, the lifetime of a model
    object is limited to the scope of the callback it was passed to or until the
    search for the next model is started. They must not be stored for later use.

    Examples
    --------
    The following example shows how to store atoms in a model for usage after
    solving:

        >>> import clingo
        >>> ctl = clingo.Control()
        >>> ctl.add("base", [], "{a;b}.")
        >>> ctl.ground([("base", [])])
        >>> ctl.configuration.solve.models="0"
        >>> models = []
        >>> with ctl.solve(yield_=True) as handle:
        ...     for model in handle:
        ...         models.append(model.symbols(atoms=True))
        ...
        >>> sorted(models)
        [[], [a], [a, b], [b]]
    '''
    def __init__(self, rep):
        self._rep = rep

    def contains(self, atom: Symbol) -> bool:
        '''
        contains(self, atom: Symbol) -> bool

        Efficiently check if an atom is contained in the model.

        Parameters
        ----------
        atom : Symbol
            The atom to lookup.

        Returns
        -------
        bool
            Whether the given atom is contained in the model.

        Notes
        -----
        The atom must be represented using a function symbol.
        '''
        # pylint: disable=protected-access
        return _c_call('bool', _lib.clingo_model_contains, self._rep, atom._rep)

    def extend(self, symbols: Sequence[Symbol]) -> None:
        '''
        Extend a model with the given symbols.

        Parameters
        ----------
        symbols : Sequence[Symbol]
            The symbols to add to the model.

        Returns
        -------
        None

        Notes
        -----
        This only has an effect if there is an underlying clingo application, which
        will print the added symbols.
        '''
        # pylint: disable=protected-access
        c_symbols = _ffi.new('clingo_symbol_t[]', len(symbols))
        for i, sym in enumerate(symbols):
            c_symbols[i] = sym._rep
        _handle_error(_lib.clingo_model_extend(self._rep, c_symbols, len(symbols)))

    def is_true(self, literal: int) -> bool:
        '''
        is_true(self, literal: int) -> bool

        Check if the given program literal is true.

        Parameters
        ----------
        literal : int
            The given program literal.

        Returns
        -------
        bool
            Whether the given program literal is true.
        '''
        return _c_call('bool', _lib.clingo_model_is_true, self._rep, literal)

    def symbols(self, atoms: bool=False, terms: bool=False, shown: bool=False, csp: bool=False,
                theory: bool=False, complement: bool=False) -> List[Symbol]:
        '''
        Return the list of atoms, terms, or CSP assignments in the model.

        Parameters
        ----------
        atoms : bool=False
            Select all atoms in the model (independent of `#show` statements).
        terms : bool=False
            Select all terms displayed with `#show` statements in the model.
        shown : bool=False
            Select all atoms and terms as outputted by clingo.
        csp : bool=False
            Select all csp assignments (independent of `#show` statements).
        theory : bool=False
            Select atoms added with `Model.extend`.
        complement : bool=False
            Return the complement of the answer set w.r.t. to the atoms known to the
            grounder. (Does not affect csp assignments.)

        Returns
        -------
        List[Symbol]
            The selected symbols.

        Notes
        -----
        Atoms are represented using functions (`Symbol` objects), and CSP assignments
        are represented using functions with name `"$"` where the first argument is the
        name of the CSP variable and the second its value.
        '''
        show = 0
        if atoms:
            show |= _lib.clingo_show_type_atoms
        if terms:
            show |= _lib.clingo_show_type_terms
        if shown:
            show |= _lib.clingo_show_type_shown
        if csp:
            show |= _lib.clingo_show_type_csp
        if theory:
            show |= _lib.clingo_show_type_theory
        if complement:
            show |= _lib.clingo_show_type_complement

        size = _c_call('size_t', _lib.clingo_model_symbols_size, self._rep, show)

        p_symbols = _ffi.new('clingo_symbol_t[]', size)
        _handle_error(_lib.clingo_model_symbols(self._rep, show, p_symbols, size))

        symbols = []
        for c_symbol in p_symbols:
            symbols.append(Symbol(c_symbol))
        return symbols

    def __str__(self):
        return " ".join(map(str, self.symbols(shown=True)))

    @property
    def context(self) -> SolveControl:
        '''
        Object that allows for controlling the running search.
        '''
        ctl = _c_call('clingo_solve_control_t*', _lib.clingo_model_context, self._rep)
        return SolveControl(ctl)

    @property
    def cost(self) -> List[int]:
        '''
        Return the list of integer cost values of the model.

        The return values correspond to clasp's cost output.
        '''
        size = _c_call('size_t', _lib.clingo_model_cost_size, self._rep)

        p_costs = _ffi.new('int64_t[]', size)
        _handle_error(_lib.clingo_model_cost(self._rep, p_costs, size))

        return list(p_costs)

    @property
    def number(self) -> int:
        '''
        The running number of the model.
        '''
        return _c_call('uint64_t', _lib.clingo_model_number, self._rep)

    @property
    def optimality_proven(self) -> bool:
        '''
        Whether the optimality of the model has been proven.
        '''
        return _c_call('bool', _lib.clingo_model_optimality_proven, self._rep)

    @property
    def thread_id(self) -> int:
        '''
        The id of the thread which found the model.
        '''
        return _c_call('clingo_id_t', _lib.clingo_model_thread_id, self._rep)

    @property
    def type(self) -> ModelType:
        '''
        The type of the model.
        '''
        return ModelType(_c_call('clingo_model_type_t', _lib.clingo_model_type, self._rep))

class SolveHandle:
    '''
    Handle for solve calls.

    `SolveHandle` objects cannot be created from Python. Instead they are returned
    by `Control.solve`. They can be used to control solving, like, retrieving
    models or cancelling a search.

    Implements: `ContextManager[SolveHandle]`.

    See Also
    --------
    Control.solve

    Notes
    -----
    A `SolveHandle` is a context manager and must be used with Python's `with`
    statement.

    Blocking functions in this object release the GIL. They are not thread-safe
    though.
    '''
    def __init__(self, rep, handler):
        self._rep = rep
        self._handler = handler

    def __iter__(self) -> Iterator[Model]:
        while True:
            self.resume()
            m = self.model()
            if m is None:
                break
            yield m

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _handle_error(_lib.clingo_solve_handle_close(self._rep), self._handler)

    def cancel(self) -> None:
        '''
        Cancel the running search.

        Returns
        -------
        None

        See Also
        --------
        Control.interrupt
        '''
        _handle_error(_lib.clingo_solve_handle_cancel(self._rep), self._handler)

    def core(self) -> List[int]:
        '''
        The subset of assumptions that made the problem unsatisfiable.

        Returns
        -------
        List[int]
        '''

        p_core = _ffi.new('clingo_literal_t**')
        p_size = _ffi.new('size_t*')
        _handle_error(_lib.clingo_solve_handle_core(self._rep, p_core, p_size), self._handler)

        return [p_core[0][i] for i in range(p_size[0])]

    def get(self) -> SolveResult:
        '''
        Get the result of a solve call.

        If the search is not completed yet, the function blocks until the result is
        ready.

        Returns
        -------
        SolveResult
        '''
        p_res = _ffi.new('clingo_solve_result_bitset_t*')
        _handle_error(_lib.clingo_solve_handle_get(self._rep, p_res), self._handler)
        return SolveResult(p_res[0])

    def model(self) -> Optional[Model]:
        '''
        model(self) -> Optional[Model]

        Get the current model if there is any.

        Examples
        --------
        The following example shows how to implement a custom solve loop. While more
        cumbersome than using a for loop, this kind of loop allows for fine grained
        timeout handling between models:

            >>> import clingo
            >>> ctl = clingo.Control()
            >>> ctl.configuration.solve.models = 0
            >>> ctl.add("base", [], "1 {a;b}.")
            >>> ctl.ground([("base", [])])
            >>> with prg.solve(yield_=True, async_=True) as hnd:
            ...     while True:
            ...         hnd.resume()
            ...         _ = hnd.wait()
            ...         m = hnd.model()
            ...         print(m)
            ...         if m is None:
            ...             break
            b
            a
            a b
            None
        '''
        p_model = _ffi.new('clingo_model_t**')
        _handle_error(
            _lib.clingo_solve_handle_model(self._rep, p_model),
            self._handler)
        if p_model[0] == _ffi.NULL:
            return None
        return Model(p_model[0])

    def resume(self) -> None:
        '''
        resume(self) -> None

        Discards the last model and starts searching for the next one.

        Notes
        -----
        If the search has been started asynchronously, this function starts the search
        in the background.
        '''
        _handle_error(_lib.clingo_solve_handle_resume(self._rep), self._handler)

    def wait(self, timeout: Optional[float]=None) -> bool:
        '''
        wait(self, timeout: Optional[float]=None) -> bool

        Wait for solve call to finish or the next result with an optional timeout.

        Parameters
        ----------
        timeout : Optional[float]=None
            If a timeout is given, the function blocks for at most timeout seconds.

        Returns
        -------
        bool
            Returns a Boolean indicating whether the solve call has finished or the
            next result is ready.
        '''
        p_res = _ffi.new('bool*')
        _lib.clingo_solve_handle_wait(self._rep, 0 if timeout is None else timeout, p_res)
        return p_res[0]

# {{{1 propagators [0%]

class Trail(Sequence[int], metaclass=ABCMeta):
    '''
    Object to access literals assigned by the solver in chronological order.

    Literals in the trail are ordered by decision levels, where the first literal
    with a larger level than the previous literals is a decision; the following
    literals with same level are implied by this decision literal. Each decision
    level up to and including the current decision level has a valid offset in the
    trail.

    Implements: `Sequence[int]`.
    '''
    def begin(self, level: int) -> int:
        '''
        begin(self, level: int) -> int

        Returns the offset of the decision literal with the given decision level in the
        trail.

        Parameters
        ----------
        level : int
            The decision level.

        Returns
        -------
        int
        '''

    def end(self, level: int) -> int:
        '''
        end(self, level: int) -> int

        Returns the offset following the last literal with the given decision literal
        in the trail.

        Parameters
        ----------
        level : int
            The decision level.

        Returns
        -------
        int
        '''


class Assignment(Sequence[int], metaclass=ABCMeta):
    '''
    Object to inspect the (parital) assignment of an associated solver.

    Assigns truth values to solver literals.  Each solver literal is either true,
    false, or undefined, represented by the Python constants `True`, `False`, or
    `None`, respectively.

    This class implements `Sequence[int]` to access the (positive)
    literals in the assignment.

    Implements: `Sequence[int]`.
    '''
    def decision(self, level: int) -> int:
        '''
        decision(self, level: int) -> int

        Return the decision literal of the given level.

        Parameters
        ----------
        literal : int
            The solver literal.

        Returns
        -------
        int
        '''

    def has_literal(self, literal : int) -> bool:
        '''
        has_literal(self, literal : int) -> bool

        Determine if the given literal is valid in this solver.

        Parameters
        ----------
        literal : int
            The solver literal.

        Returns
        -------
        bool
        '''

    def is_false(self, literal: int) -> bool:
        '''
        is_false(self, literal: int) -> bool

        Determine if the literal is false.

        Parameters
        ----------
        literal : int
            The solver literal.

        Returns
        -------
        bool
        '''

    def is_fixed(self, literal: int) -> bool:
        '''
        is_fixed(self, literal: int) -> bool

        Determine if the literal is assigned on the top level.

        Parameters
        ----------
        literal : int
            The solver literal.

        Returns
        -------
        bool
        '''

    def is_true(self, literal: int) -> bool:
        '''
        is_true(self, literal: int) -> bool

        Determine if the literal is true.

        Parameters
        ----------
        literal : int
            The solver literal.

        Returns
        -------
        bool
        '''

    def level(self, literal: int) -> int:
        '''
        level(self, literal: int) -> int

        The decision level of the given literal.

        Parameters
        ----------
        literal : int
            The solver literal.

        Returns
        -------
        int

        Notes
        -----
        Note that the returned value is only meaningful if the literal is assigned -
        i.e., `value(lit) is not None`.
        '''

    def value(self, literal) -> Optional[bool]:
        '''
        value(self, literal) -> Optional[bool]

        Get the truth value of the given literal or `None` if it has none.

        Parameters
        ----------
        literal : int
            The solver literal.

        Returns
        -------
        Optional[bool]
        '''

    decision_level: int
    '''
    decision_level: int

    The current decision level.
    '''
    has_conflict: bool
    '''
    has_conflict: bool

    True if the assignment is conflicting.
    '''
    is_total: bool
    '''
    is_total: bool

    Whether the assignment is total.
    '''
    root_level: int
    '''
    root_level: int

    The current root level.
    '''
    trail: Trail
    '''
    trail: Trail

    The trail of assigned literals.
    '''

class PropagatorCheckMode(Enum):
    '''
    Enumeration of supported check modes for propagators.

    Note that total checks are subject to the lock when a model is found. This
    means that information from previously found models can be used to discard
    assignments in check calls.

    Attributes
    ----------
    Off : PropagatorCheckMode
        Do not call `Propagator.check` at all.
    Total : PropagatorCheckMode
        Call `Propagator.check` on total assignments.
    Fixpoint : PropagatorCheckMode
        Call `Propagator.check` on propagation fixpoints.
    Both : PropagatorCheckMode
        Call `Propagator.check` on propagation fixpoints and total assignments.
    '''
    Both = _lib.clingo_propagator_check_mode_both
    Fixpoint = _lib.clingo_propagator_check_mode_fixpoint
    Off = _lib.clingo_propagator_check_mode_none
    Total = _lib.clingo_propagator_check_mode_total

class PropagateInit(metaclass=ABCMeta):
    '''
    Object that is used to initialize a propagator before each solving step.

    See Also
    --------
    Control.register_propagator
    '''
    def add_clause(self, clause: Iterable[int]) -> bool:
        '''
        add_clause(self, clause: Iterable[int]) -> bool

        Statically adds the given clause to the problem.

        Parameters
        ----------
        clause : Iterable[int]
            The clause over solver literals to add.

        Returns
        -------
        bool
            Returns false if the program becomes unsatisfiable.

        Notes
        -----
        If this function returns false, initialization should be stopped and no further
        functions of the `PropagateInit` and related objects should be called.
        '''

    def add_literal(self, freeze: bool=True) -> int:
        '''
        add_literal(self, freeze: bool=True) -> int

        Statically adds a literal to the solver.

        To be able to use the variable in clauses during propagation or add watches to
        it, it has to be frozen. Otherwise, it might be removed during preprocessing.

        Parameters
        ----------
        freeze : bool=True
            Whether to freeze the variable.

        Returns
        -------
        int
            Returns the added literal.

        Notes
        -----
        If literals are added to the solver, subsequent calls to `add_clause` and
        `propagate` are expensive. It is best to add literals in batches.
        '''

    def add_minimize(self, literal: int, weight: int, priority: int=0) -> None:
        '''
        add_minimize(self, literal: int, weight: int, priority: int=0) -> None

        Extends the solver's minimize constraint with the given weighted literal.

        Parameters
        ----------
        literal : int
            The literal to add.
        weight : int
            The weight of the literal.
        priority : int=0
            The priority of the literal.
        '''

    def add_watch(self, literal: int, thread_id: Optional[int]=None) -> None:
        '''
        add_watch(self, literal: int, thread_id: Optional[int]=None) -> None

        Add a watch for the solver literal in the given phase.

        Parameters
        ----------
        literal : int
            The solver literal to watch.
        thread_id : Optional[int]
            The id of the thread to watch the literal. If the is `None` then all active
            threads will watch the literal.

        Returns
        -------
        None
        '''

    def add_weight_constraint(self, literal: int, literals: Iterable[Tuple[int,int]],
                              bound: int, type_: int=0, compare_equal: bool=False) -> bool:
        '''
        Statically adds a constraint of form

            literal <=> { l=w | (l, w) in literals } >= bound

        to the solver.

        - If `type_ < 0`, then `<=>` is a left implication.
        - If `type_ > 0`, then `<=>` is a right implication.
        - Otherwise, `<=>` is an equivalence.

        Parameters
        ----------
        literal : int
            The literal associated with the constraint.
        literals : Iterable[Tuple[int,int]]
            The weighted literals of the constrain.
        bound : int
            The bound of the constraint.
        type_ : int
            Add a weight constraint of the given type_.
        compare_equal : bool=False
            A Boolean indicating whether to compare equal or less than equal.

        Returns
        -------
        bool
            Returns false if the program becomes unsatisfiable.

        Notes
        -----
        If this function returns false, initialization should be stopped and no further
        functions of the `PropagateInit` and related objects should be called.
        '''

    def propagate(self) -> bool:
        '''
        propagate(self) -> bool

        Propagates consequences of the underlying problem excluding registered propagators.

        Returns
        -------
        bool
            Returns false if the program becomes unsatisfiable.

        Notes
        -----
        This function has no effect if SAT-preprocessing is enabled.

        If this function returns false, initialization should be stopped and no further
        functions of the `PropagateInit` and related objects should be called.
        '''

    def solver_literal(self, literal: int) -> int:
        '''
        solver_literal(self, literal: int) -> int

        Maps the given program literal or condition id to its solver literal.

        Parameters
        ----------
        literal : int
            A program literal or condition id.

        Returns
        -------
        int
            A solver literal.
        '''

    assignment: Assignment
    '''
    `Assignment` object capturing the top level assignment.
    '''
    check_mode: PropagatorCheckMode
    '''
    `PropagatorCheckMode` controlling when to call `Propagator.check`.
    '''
    number_of_threads: int
    '''
    The number of solver threads used in the corresponding solve call.
    '''
    symbolic_atoms: SymbolicAtoms
    '''
    The symbolic atoms captured by a `SymbolicAtoms` object.
    '''
    theory_atoms: Iterator[TheoryAtom]
    '''
    An iterator over all theory atoms.
    '''

class PropagateControl(metaclass=ABCMeta):
    '''
    This object can be used to add clauses and to propagate them.

    See Also
    --------
    Control.register_propagator
    '''
    def add_clause(self, clause: Iterable[int], tag: bool=False, lock: bool=False) -> bool:
        '''
        Add the given clause to the solver.

        Parameters
        ----------
        clause : Iterable[int]
            List of solver literals forming the clause.
        tag : bool=False
            If true, the clause applies only in the current solving step.
        lock : bool=False
            If true, exclude clause from the solver's regular clause deletion policy.

        Returns
        -------
        bool
            This method returns false if the current propagation must be stopped.
        '''

    def add_literal(self) -> int:
        '''
        Adds a new positive volatile literal to the underlying solver thread.

        The literal is only valid within the current solving step and solver thread.
        All volatile literals and clauses involving a volatile literal are deleted
        after the current search.

        Returns
        -------
        int
            The added solver literal.
        '''

    def add_nogood(self, clause: Iterable[int], tag: bool=False, lock: bool=False) -> bool:
        '''
        Equivalent to `self.add_clause([-lit for lit in clause], tag, lock)`.
        '''

    def add_watch(self, literal: int) -> None:
        '''
        Add a watch for the solver literal in the given phase.

        Parameters
        ----------
        literal : int
            The target solver literal.

        Returns
        -------
        None

        Notes
        -----
        Unlike `PropagateInit.add_watch` this does not add a watch to all solver
        threads but just the current one.
        '''

    def has_watch(self, literal: int) -> bool:
        '''
        Check whether a literal is watched in the current solver thread.

        Parameters
        ----------
        literal : int
            The target solver literal.

        Returns
        -------
        bool
            Whether the literal is watched.
        '''

    def propagate(self) -> bool:
        '''
        Propagate literals implied by added clauses.

        Returns
        -------
        bool
            This method returns false if the current propagation must be stopped.
        '''

    def remove_watch(self, literal: int) -> None:
        '''
        Removes the watch (if any) for the given solver literal.

        Parameters
        ----------
        literal : int
            The target solver literal.

        Returns
        -------
        None
        '''

    assignment: Assignment
    '''
    `Assignment` object capturing the partial assignment of the current solver thread.
    '''
    thread_id: int
    '''
    The numeric id of the current solver thread.
    '''

class Propagator(metaclass=ABCMeta):
    '''
    Propagator interface for custom constraints.
    '''
    def init(self, init: PropagateInit) -> None:
        """
        This function is called once before each solving step.

        It is used to map relevant program literals to solver literals, add
        watches for solver literals, and initialize the data structures used
        during propagation.

        Parameters
        ----------
        init : PropagateInit
            Object to initialize the propagator.

        Returns
        -------
        None

        Notes
        -----
        This is the last point to access theory atoms.  Once the search has
        started, they are no longer accessible.
        """

    def propagate(self, control: PropagateControl, changes: Sequence[int]) -> None:
        """
        Can be used to propagate solver literals given a partial assignment.

        Parameters
        ----------
        control : PropagateControl
            Object to control propagation.
        changes : Sequence[int]
            List of watched solver literals assigned to true.

        Returns
        -------
        None

        Notes
        -----
        Called during propagation with a non-empty list of watched solver
        literals that have been assigned to true since the last call to either
        propagate, undo, (or the start of the search) - the change set. Only
        watched solver literals are contained in the change set. Each literal
        in the change set is true w.r.t. the current Assignment.
        `PropagateControl.add_clause` can be used to add clauses. If a clause
        is unit resulting, it can be propagated using
        `PropagateControl.propagate`. If either of the two methods returns
        False, the propagate function must return immediately.

            c = ...
            if not control.add_clause(c) or not control.propagate(c):
                return

        Note that this function can be called from different solving threads.
        Each thread has its own assignment and id, which can be obtained using
        `PropagateControl.thread_id`.
        """

    def undo(self, thread_id: int, assignment: Assignment,
             changes: Sequence[int]) -> None:
        """
        Called whenever a solver with the given id undos assignments to watched
        solver literals.

        Parameters
        ----------
        thread_id : int
            The solver thread id.
        assignment : Assignment
            Object for inspecting the partial assignment of the solver.
        changes : Sequence[int]
            The list of watched solver literals whose assignment is undone.

        Returns
        -------
        None

        Notes
        -----
        This function is meant to update assignment dependent state in a
        propagator but not to modify the current state of the solver.
        Furthermore, errors raised in the function lead to program termination.
        """

    def check(self, control: PropagateControl) -> None:
        """
        This function is similar to propagate but is called without a change
        set on propagation fixpoints.

        When exactly this function is called, can be configured using the @ref
        PropagateInit.check_mode property.

        Parameters
        ----------
        control : PropagateControl
            Object to control propagation.

        Returns
        -------
        None

        Notes
        -----
        This function is called even if no watches have been added.
        """

    def decide(self, thread_id: int, assignment: Assignment, fallback: int) -> int:
        """
        This function allows a propagator to implement domain-specific
        heuristics.

        It is called whenever propagation reaches a fixed point.

        Parameters
        ----------
        thread_id : int
            The solver thread id.
        assignment : Assignment
            Object for inspecting the partial assignment of the solver.
        fallback : int
            The literal choosen by the solver's heuristic.

        Returns
        -------
        int
            Тhe next solver literal to make true.

        Notes
        -----
        This function should return a free solver literal that is to be
        assigned true. In case multiple propagators are registered, this
        function can return 0 to let a propagator registered later make a
        decision. If all propagators return 0, then the fallback literal is
        used.
        """

# {{{1 ground program inspection/building [0%]

class HeuristicType(Enum):
    '''
    Enumeration of the different heuristic types.

    `HeuristicType` objects have a readable string representation, implement
    Python's rich comparison operators, and can be used as dictionary keys.

    Furthermore, they cannot be constructed from Python. Instead the following
    preconstructed class attributes  are available:

    Attributes
    ----------
    Level : HeuristicType
        Heuristic modification to set the level of an atom.
    Sign : HeuristicType
        Heuristic modification to set the sign of an atom.
    Factor : HeuristicType
        Heuristic modification to set the decaying factor of an atom.
    Init : HeuristicType
        Heuristic modification to set the inital score of an atom.
    True_ : HeuristicType
        Heuristic modification to make an atom true.
    False_ : HeuristicType
        Heuristic modification to make an atom false.
    '''
    # Factor: HeuristicType
    # False_: HeuristicType
    # Init: HeuristicType
    # Level: HeuristicType
    # Sign: HeuristicType
    # True_: HeuristicType

class Observer(metaclass=ABCMeta):
    """
    Interface that has to be implemented to inspect rules produced during
    grounding.
    """
    def init_program(self, incremental: bool) -> None:
        """
        Called once in the beginning.

        Parameters
        ----------
        incremental : bool
            Whether the program is incremental. If the incremental flag is
            true, there can be multiple calls to `Control.solve`.

        Returns
        -------
        None
        """

    def begin_step(self) -> None:
        """
        Marks the beginning of a block of directives passed to the solver.

        Returns
        -------
        None
        """

    def rule(self, choice: bool, head: Sequence[int], body: Sequence[int]) -> None:
        """
        Observe rules passed to the solver.

        Parameters
        ----------
        choice : bool
            Determines if the head is a choice or a disjunction.
        head : Sequence[int]
            List of program atoms forming the rule head.
        body : Sequence[int]
            List of program literals forming the rule body.

        Returns
        -------
        None
        """

    def weight_rule(self, choice: bool, head: Sequence[int], lower_bound: int,
                    body: Sequence[Tuple[int,int]]) -> None:
        """
        Observe rules with one weight constraint in the body passed to the
        solver.

        Parameters
        ----------
        choice : bool
            Determines if the head is a choice or a disjunction.
        head : Sequence[int]
            List of program atoms forming the head of the rule.
        lower_bound:
            The lower bound of the weight constraint in the rule body.
        body : Sequence[Tuple[int,int]]
            List of weighted literals (pairs of literal and weight) forming the
            elements of the weight constraint.

        Returns
        -------
        None
        """

    def minimize(self, priority: int, literals: Sequence[Tuple[int,int]]) -> None:
        """
        Observe minimize directives (or weak constraints) passed to the
        solver.

        Parameters
        ----------
        priority : int
            The priority of the directive.
        literals : Sequence[Tuple[int,int]]
            List of weighted literals whose sum to minimize (pairs of literal
            and weight).

        Returns
        -------
        None
        """

    def project(self, atoms: Sequence[int]) -> None:
        """
        Observe projection directives passed to the solver.

        Parameters
        ----------
        atoms : Sequence[int]
            The program atoms to project on.

        Returns
        -------
        None
        """

    def output_atom(self, symbol: Symbol, atom: int) -> None:
        """
        Observe shown atoms passed to the solver.  Facts do not have an
        associated program atom. The value of the atom is set to zero.

        Parameters
        ----------
        symbol : Symbolic
            The symbolic representation of the atom.
        atom : int
            The associated program atom (0 for facts).

        Returns
        -------
        None
        """

    def output_term(self, symbol: Symbol, condition: Sequence[int]) -> None:
        """
        Observe shown terms passed to the solver.

        Parameters
        ----------
        symbol : Symbol
            The symbolic representation of the term.
        condition : Sequence[int]
            List of program literals forming the condition when to show the
            term.

        Returns
        -------
        None
        """

    def output_csp(self, symbol: Symbol, value: int,
                   condition: Sequence[int]) -> None:
        """
        Observe shown csp variables passed to the solver.

        Parameters
        ----------
        symbol : Symbol
            The symbolic representation of the variable.
        value : int
            The integer value of the variable.
        condition : Sequence[int]
            List of program literals forming the condition when to show the
            variable with its value.

        Returns
        -------
        None
        """

    def external(self, atom: int, value: TruthValue) -> None:
        """
        Observe external statements passed to the solver.

        Parameters
        ----------
        atom : int
            The external atom in form of a program literal.
        value : TruthValue
            The truth value of the external statement.

        Returns
        -------
        None
        """

    def assume(self, literals: Sequence[int]) -> None:
        """
        Observe assumption directives passed to the solver.

        Parameters
        ----------
        literals : Sequence[int]
            The program literals to assume (positive literals are true and
            negative literals false for the next solve call).

        Returns
        -------
        None
        """

    def heuristic(self, atom: int, type_: HeuristicType, bias: int,
                  priority: int, condition: Sequence[int]) -> None:
        """
        Observe heuristic directives passed to the solver.

        Parameters
        ----------
        atom : int
            The program atom heuristically modified.
        type_ : HeuristicType
            The type of the modification.
        bias : int
            A signed integer.
        priority : int
            An unsigned integer.
        condition : Sequence[int]
            List of program literals.

        Returns
        -------
        None
        """

    def acyc_edge(self, node_u: int, node_v: int,
                  condition: Sequence[int]) -> None:
        """
        Observe edge directives passed to the solver.

        Parameters
        ----------
        node_u : int
            The start vertex of the edge (in form of an integer).
        node_v : int
            Тhe end vertex of the edge (in form of an integer).
        condition : Sequence[int]
            The list of program literals forming th condition under which to
            add the edge.

        Returns
        -------
        None
        """

    def theory_term_number(self, term_id: int, number: int) -> None:
        """
        Observe numeric theory terms.

        Parameters
        ----------
        term_id : int
            The id of the term.
        number : int
            The value of the term.

        Returns
        -------
        None
        """

    def theory_term_string(self, term_id : int, name : str) -> None:
        """
        Observe string theory terms.

        Parameters
        ----------
        term_id : int
            The id of the term.
        name : str
            The string value of the term.

        Returns
        -------
        None
        """

    def theory_term_compound(self, term_id: int, name_id_or_type: int,
                             arguments: Sequence[int]) -> None:
        """
        Observe compound theory terms.

        Parameters
        ----------
        term_id : int
            The id of the term.
        name_id_or_type : int
            The name or type of the term where
            - if it is -1, then it is a tuple
            - if it is -2, then it is a set
            - if it is -3, then it is a list
            - otherwise, it is a function and name_id_or_type refers to the id
            of the name (in form of a string term)
        arguments : Sequence[int]
            The arguments of the term in form of a list of term ids.

        Returns
        -------
        None
        """

    def theory_element(self, element_id: int, terms: Sequence[int],
                       condition: Sequence[int]) -> None:
        """
        Observe theory elements.

        Parameters
        ----------
        element_id : int
            The id of the element.
        terms : Sequence[int]
            The term tuple of the element in form of a list of term ids.
        condition : Sequence[int]
            The list of program literals forming the condition.

        Returns
        -------
        None
        """

    def theory_atom(self, atom_id_or_zero: int, term_id: int,
                    elements: Sequence[int]) -> None:
        """
        Observe theory atoms without guard.

        Parameters
        ----------
        atom_id_or_zero : int
            The id of the atom or zero for directives.
        term_id : int
            The term associated with the atom.
        elements : Sequence[int]
            The elements of the atom in form of a list of element ids.

        Returns
        -------
        None
        """

    def theory_atom_with_guard(self, atom_id_or_zero: int, term_id: int,
                               elements: Sequence[int], operator_id: int,
                               right_hand_side_id: int) -> None:
        """
        Observe theory atoms with guard.

        Parameters
        ----------
        atom_id_or_zero : int
            The id of the atom or zero for directives.
        term_id : int
            The term associated with the atom.
        elements : Sequence[int]
            The elements of the atom in form of a list of element ids.
        operator_id : int
            The id of the operator (a string term).
        right_hand_side_id : int
            The id of the term on the right hand side of the atom.

        Returns
        -------
        None
        """

    def end_step(self) -> None:
        """
        Marks the end of a block of directives passed to the solver.

        This function is called right before solving starts.

        Returns
        -------
        None
        """

class Backend(ContextManager['Backend'], metaclass=ABCMeta):
    '''
    Backend object providing a low level interface to extend a logic program.

    This class allows for adding statements in ASPIF format.

    Implements: `ContextManager[Backend]`.

    See Also
    --------
    Control.backend

    Notes
    -----
    The `Backend` is a context manager and must be used with Python's `with`
    statement.

    Examples
    --------
    The following example shows how to add a fact to a program:

        >>> import clingo
        >>> ctl = clingo.Control()
        >>> sym_a = clingo.Function("a")
        >>> with ctl.backend() as backend:
        ...     atm_a = backend.add_atom(sym_a)
        ...     backend.add_rule([atm_a])
        ...
        >>> ctl.symbolic_atoms[sym_a].is_fact
        True
        >>> ctl.solve(on_model=lambda m: print("Answer: {}".format(m)))
        Answer: a
        SAT
    '''
    def __init__(self, rep):
        self._rep = rep

    def add_acyc_edge(self, node_u: int, node_v: int, condition: Iterable[int]) -> None:
        '''
        Add an edge directive to the program.

        Parameters
        ----------
        node_u : int
            The start node represented as an unsigned integer.
        node_v : int
            The end node represented as an unsigned integer.
        condition : Iterable[int]
            List of program literals.

        Returns
        -------
        None
        '''

    def add_assume(self, literals: Iterable[int]) -> None:
        '''
        Add assumptions to the program.

        Parameters
        ----------
        literals : Iterable[int]
            The list of literals to assume true.

        Returns
        -------
        None
        '''

    def add_atom(self, symbol : Optional[Symbol]=None) -> int:
        '''
        Return a fresh program atom or the atom associated with the given symbol.

        If the given symbol does not exist in the atom base, it is added first. Such
        atoms will be used in subequents calls to ground for instantiation.

        Parameters
        ----------
        symbol : Optional[Symbol]=None
            The symbol associated with the atom.

        Returns
        -------
        int
            The program atom representing the atom.
        '''

    def add_external(self, atom : int, value : TruthValue=TruthValue.False_) -> None:
        '''
        Mark a program atom as external optionally fixing its truth value.

        Parameters
        ----------
        atom : int
            The program atom to mark as external.
        value : TruthValue=TruthValue.False_
            Optional truth value.

        Returns
        -------
        None

        Notes
        -----
        Can also be used to release an external atom using `TruthValue.Release`.
        '''

    def add_heuristic(self, atom: int, type_: HeuristicType, bias: int, priority: int,
                      condition: Iterable[int]) -> None:
        '''
        Add a heuristic directive to the program.

        Parameters
        ----------
        atom : int
            Program atom to heuristically modify.
        type_ : HeuristicType
            The type of modification.
        bias : int
            A signed integer.
        priority : int
            An unsigned integer.
        condition : Iterable[int]
            List of program literals.

        Returns
        -------
        None
        '''

    def add_minimize(self, priority: int, literals: Iterable[Tuple[int,int]]) -> None:
        '''
        Add a minimize constraint to the program.

        Parameters
        ----------
        priority : int
            Integer for the priority.
        literals : Iterable[Tuple[int,int]]
            List of pairs of program literals and weights.

        Returns
        -------
        None
        '''

    def add_project(self, atoms: Iterable[int]) -> None:
        '''
        Add a project statement to the program.

        Parameters
        ----------
        atoms : Iterable[int]
            List of program atoms to project on.

        Returns
        -------
        None
        '''

    def add_rule(self, head: Iterable[int], body: Iterable[int]=[], choice: bool=False) -> None:
        '''
        Add a disjuntive or choice rule to the program.

        Parameters
        ----------
        head : Iterable[int]
            The program atoms forming the rule head.
        body : Iterable[int]=[]
            The program literals forming the rule body.
        choice : bool=False
            Whether to add a disjunctive or choice rule.

        Returns
        -------
        None

        Notes
        -----
        Integrity constraints and normal rules can be added by using an empty or
        singleton head list, respectively.
        '''
        # pylint: disable=dangerous-default-value,unnecessary-pass
        pass

    def add_weight_rule(self, head: Iterable[int], lower: int, body: Iterable[Tuple[int,int]],
                        choice: bool=False) -> None:
        '''
        Add a disjuntive or choice rule with one weight constraint with a lower bound
        in the body to the program.

        Parameters
        ----------
        head : Iterable[int]
            The program atoms forming the rule head.
        lower : int
            The lower bound.
        body : Iterable[Tuple[int,int]]
            The pairs of program literals and weights forming the elements of the
            weight constraint.
        choice : bool=False
            Whether to add a disjunctive or choice rule.

        Returns
        -------
        None
        '''

# {{{1 configuration [100%]

class Configuration:
    '''
    Allows for changing the configuration of the underlying solver.

    Options are organized hierarchically. To change and inspect an option use:

        config.group.subgroup.option = "value"
        value = config.group.subgroup.option

    There are also arrays of option groups that can be accessed using integer
    indices:

        config.group.subgroup[0].option = "value1"
        config.group.subgroup[1].option = "value2"

    To list the subgroups of an option group, use the `Configuration.keys` member.
    Array option groups, like solver, have a non-negative length and can be
    iterated. Furthermore, there are meta options having key `configuration`.
    Assigning a meta option sets a number of related options.  To get further
    information about an option or option group `<opt>`, call `description(<opt>)`.

    Notes
    -----
    When integers are assigned to options, they are automatically converted to
    strings. The value of an option is always a string.

    Examples
    --------
    The following example shows how to modify the configuration to enumerate all
    models:

        >>> import clingo
        >>> prg = clingo.Control()
        >>> prg.configuration.solve.description("models")
        'Compute at most %A models (0 for all)\n'
        >>> prg.configuration.solve.models = 0
        >>> prg.add("base", [], "{a;b}.")
        >>> prg.ground([("base", [])])
        >>> prg.solve(on_model=lambda m: print("Answer: {}".format(m)))
        Answer:
        Answer: a
        Answer: b
        Answer: a b
        SAT
    '''
    def __init__(self, rep, key):
        # Note: we have to bypass __setattr__ to avoid infinite recursion
        super().__setattr__("_rep", rep)
        super().__setattr__("_key", key)

    @property
    def _type(self) -> int:
        return _c_call('clingo_configuration_type_bitset_t', _lib.clingo_configuration_type, self._rep, self._key)

    def _get_subkey(self, name: str) -> Optional[int]:
        if self._type & _lib.clingo_configuration_type_map:
            if _c_call('bool', _lib.clingo_configuration_map_has_subkey, self._rep, self._key, name.encode()):
                return _c_call('clingo_id_t', _lib.clingo_configuration_map_at, self._rep, self._key, name.encode())
        return None

    def __len__(self):
        if self._type & _lib.clingo_configuration_type_array:
            return _c_call('size_t', _lib.clingo_configuration_array_size, self._rep, self._key)
        return 0

    def __getitem__(self, idx: int) -> 'Configuration':
        if idx < 0 or idx >= len(self):
            raise IndexError("invalid index")

        key = _c_call('clingo_id_t', _lib.clingo_configuration_array_at, self._rep, self._key, idx)
        return Configuration(self._rep, key)

    def __getattr__(self, name: str) -> Union[None,str,'Configuration']:
        key = self._get_subkey(name)
        if key is None:
            raise AttributeError(f'no attribute: {name}')

        type_ = _c_call('clingo_configuration_type_bitset_t', _lib.clingo_configuration_type, self._rep, key)

        if type_ & _lib.clingo_configuration_type_value:
            if not _c_call('bool', _lib.clingo_configuration_value_is_assigned, self._rep, key):
                return None

            size = _c_call('size_t', _lib.clingo_configuration_value_get_size, self._rep, key)

            c_val = _ffi.new('char[]', size)
            _handle_error(_lib.clingo_configuration_value_get(self._rep, key, c_val, size))
            return _to_str(c_val)

        return Configuration(self._rep, key)

    def __setattr__(self, name: str, val: str) -> None:
        key = self._get_subkey(name)
        if key is None:
            super().__setattr__(name, val)
        else:
            _handle_error(_lib.clingo_configuration_value_set(self._rep, key, val.encode()))

    def description(self, name: str) -> str:
        '''
        Get a description for a option or option group.

        Parameters
        ----------
        name : str
            The name of the option.

        Returns
        -------
        str
        '''
        key = self._get_subkey(name)
        if key is None:
            raise RuntimeError(f'unknown option {name}')
        return _to_str(_c_call('char*', _lib.clingo_configuration_description, self._rep, key))

    @property
    def keys(self) -> Optional[List[str]]:
        '''
        The list of names of sub-option groups or options.

        The list is `None` if the current object is not an option group.
        '''
        ret = None
        if self._type & _lib.clingo_configuration_type_map:
            ret = []
            for i in range(_c_call('size_t', _lib.clingo_configuration_map_size, self._rep, self._key)):
                name = _c_call('char*', _lib.clingo_configuration_map_subkey_name, self._rep, self._key, i)
                ret.append(_to_str(name))
        return ret

# {{{1 statistics [100%]

StatisticsValue = Union['StatisticsArray', 'StatisticsMap', float]

def _mutable_statistics_value_type(value):
    if isinstance(value, str):
        raise TypeError('unexpected string')
    if isinstance(value, abc.Sequence):
        return _lib.clingo_statistics_type_array
    if isinstance(value, abc.Mapping):
        return _lib.clingo_statistics_type_map
    return _lib.clingo_statistics_type_value

def _mutable_statistics_type(stats, key):
    return _c_call('clingo_statistics_type_t', _lib.clingo_statistics_type, stats, key)

def _mutable_statistics(stats) -> 'StatisticsMap':
    key = _c_call('uint64_t', _lib.clingo_statistics_root, stats)
    return cast(StatisticsMap, _mutable_statistics_get(stats, key))

def _mutable_statistics_get(stats, key) -> StatisticsValue:
    type_ = _mutable_statistics_type(stats, key)

    if type_ == _lib.clingo_statistics_type_array:
        return StatisticsArray(stats, key)

    if type_ == _lib.clingo_statistics_type_map:
        return StatisticsMap(stats, key)

    assert type_ == _lib.clingo_statistics_type_value
    return _c_call('double', _lib.clingo_statistics_value_get, stats, key)

def _mutable_statistics_set(stats, key, type_, value, update):
    if type_ == _lib.clingo_statistics_type_map:
        StatisticsMap(stats, key).update(value)
    elif type_ == _lib.clingo_statistics_type_array:
        StatisticsArray(stats, key).update(value)
    else:
        assert type_ == _lib.clingo_statistics_type_value
        if callable(value):
            if update:
                uval = value(_c_call('double', _lib.clingo_statistics_value_get, stats, key))
            else:
                uval = value(None)
        else:
            uval = value
        _handle_error(_lib.clingo_statistics_value_set(stats, key, uval))

class StatisticsArray(MutableSequence[StatisticsValue]):
    '''
    Object to modify statistics stored in an array.

    Note that only inplace concatenation and no deletion is supported.

    Implements: `MutableSequence[Union[StatisticsArray,StatisticsMap,float]]`.

    See Also
    --------
    Control.solve

    Notes
    -----
    The `StatisticsArray.update` function provides convenient means to initialize
    and modify a statistics array.
    '''
    def __init__(self, rep, key):
        self._rep = rep
        self._key = key

    def __len__(self):
        return _c_call('size_T', _lib.clingo_statistics_array_size, self._rep, self._key)

    def __getitem__(self, index):
        key = _c_call('uint64_t', _lib.clingo_statistics_array_at, self._rep, self._key, index)
        return _mutable_statistics_get(self._rep, key)

    def __setitem__(self, index, value):
        key = _c_call('uint64_t', _lib.clingo_statistics_array_at, self._rep, self._key, index)
        type_ = _mutable_statistics_type(self._rep, key)
        _mutable_statistics_set(self._rep, key, type_, value, True)

    def insert(self, index, value):
        '''
        This method is not supported.
        '''
        raise NotImplementedError('insertion is not supported')

    def __delitem__(self, index):
        raise NotImplementedError('deletion is not supported')

    def __iadd__(self, other):
        for x in other:
            self.append(x)
        return self

    def append(self, value: Any) -> None:
        '''
        Append a value.

        Parameters
        ----------
        value : Any
            A nested structure composed of floats, sequences, and mappings.

        Returns
        -------
        None
        '''
        type_ = _mutable_statistics_value_type(value)
        key = _c_call('uint64_t', _lib.clingo_statistics_array_push, self._rep, self._key, type_)
        _mutable_statistics_set(self._rep, key, type_, value, False)

    def extend(self, values: Iterable[Any]) -> None:
        '''
        Extend the statistics array with the given values.

        Paremeters
        ----------
        values : Sequence[Any]
            A sequence of nested structures composed of floats, sequences, and
            mappings.

        Returns
        -------
        None

        See Also
        -----
        append
        '''
        self += values

    def update(self, values: Sequence[Any]) -> None:
        '''
        Update a statistics array.

        Parameters
        ----------
        values : Sequence[Any]
            A sequence of nested structures composed of floats, callable, sequences,
            and mappings. A callable can be used to update an existing value, it
            receives the previous numeric value (or None if absent) as argument and
            must return an updated numeric value.

        Returns
        -------
        None
        '''
        n = len(self)
        for idx, val in enumerate(values):
            if idx < n:
                self[idx] = val
            else:
                self.append(val)

class StatisticsMap(MutableMapping[str, StatisticsValue]):
    '''
    Object to capture statistics stored in a map.

    This class does not support item deletion.

    Implements: `MutableMapping[str,Union[StatisticsArray,StatisticsMap,float]]`.

    See Also
    --------
    Control.solve

    Notes
    -----
    The `StatisticsMap.update` function provides convenient means to initialize
    and modify a statistics map.
    '''
    def __init__(self, rep, key):
        self._rep = rep
        self._key = key

    def __len__(self):
        return _c_call('size_t', _lib.clingo_statistics_map_size, self._rep, self._key)

    def __getitem__(self, name: str):
        key = _c_call('uint64_t', _lib.clingo_statistics_map_at, self._rep, self._key, name.encode())
        return _mutable_statistics_get(self._rep, key)

    def __setitem__(self, name: str, value: Any):
        has_key = name in self
        if has_key:
            key = _c_call('uint64_t', _lib.clingo_statistics_map_at, self._rep, self._key, name.encode())
            type_ = _mutable_statistics_type(self._rep, key)
        else:
            type_ = _mutable_statistics_value_type(value)
            key = _c_call('uint64_t', _lib.clingo_statistics_map_add_subkey, self._rep, self._key, name.encode(), type_)
        _mutable_statistics_set(self._rep, key, type_, value, has_key)

    def __delitem__(self, index):
        raise NotImplementedError('deletion is not supported')

    def __contains__(self, name):
        # note: has to be done like this because of python's container model
        if not isinstance(name, str):
            return False
        return _c_call('bool', _lib.clingo_statistics_map_has_subkey, self._rep, self._key, name.encode())

    def __iter__(self):
        return iter(self.keys())

    def items(self) -> AbstractSet[Tuple[str, StatisticsValue]]:
        '''
        Return the items of the map.

        Returns
        -------
        AbstractSet[Tuple[str,StatisticsValue]]
            The items of the map.
        '''
        ret = []
        for i in range(len(self)):
            name = _c_call('char*', _lib.clingo_statistics_map_subkey_name, self._rep, self._key, i)
            key = _c_call('uint64_t', _lib.clingo_statistics_map_at, self._rep, self._key, name)
            ret.append((_to_str(name), _mutable_statistics_get(self._rep, key)))
        # Note: to shut up the type checker; this should work fine in practice
        return cast(AbstractSet[Tuple[str, StatisticsValue]], ret)

    def keys(self) -> KeysView[str]:
        '''
        Return the keys of the map.

        Returns
        -------
        KeysView[str]
            The keys of the map.
        '''
        ret = []
        for i in range(len(self)):
            ret.append(_to_str(_c_call('char*', _lib.clingo_statistics_map_subkey_name, self._rep, self._key, i)))
        # Note: to shut up the type checker; this should work fine in practice
        return cast(KeysView[str], ret)

    def update(self, values):
        '''
        Update the map with the given values.

        Parameters
        ----------
        values : Mapping[Any]
            A mapping of nested structures composed of floats, callable, sequences,
            and mappings. A callable can be used to update an existing value, it
            receives the previous numeric value (or None if absent) as argument and
            must return an updated numeric value.

        Returns
        -------
        None
        '''
        for key, value in values.items():
            self[key] = value

    def values(self) -> ValuesView[StatisticsValue]:
        '''
        Return the values of the map.

        Returns
        -------
        ValuesView[Union[StatisticsArray,StatisticsMap,float]]
            The values of the map.
        '''
        ret = []
        for i in range(len(self)):
            name = _c_call('char*', _lib.clingo_statistics_map_subkey_name, self._rep, self._key, i)
            key = _c_call('uint64_t', _lib.clingo_statistics_map_at, self._rep, self._key, name)
            ret.append(_mutable_statistics_get(self._rep, key))
        # Note: to shut up the type checker; this should work fine in practice
        return cast(ValuesView[StatisticsValue], ret)

# {{{1 control [90%]

class _SolveEventHandler:
    # pylint: disable=missing-function-docstring
    def __init__(self, on_model, on_statistics, on_finish):
        self.error = None
        self._on_model = on_model
        self._on_statistics = on_statistics
        self._on_finish = on_finish

    def on_model(self, m):
        ret = None
        if self._on_model is not None:
            ret = self._on_model(Model(m))
        return bool(ret or ret is None)

    def on_finish(self, res):
        if self._on_finish is not None:
            self._on_finish(SolveResult(res))

    def on_statistics(self, step, accu):
        if self._on_statistics is not None:
            self.on_statistics(_mutable_statistics(step), _mutable_statistics(accu))

@_ffi.def_extern(onerror=_cb_error_handler('data'))
def _clingo_solve_event_callback(type_, event, data, goon):
    '''
    Low-level solve event handler.
    '''
    handler = _ffi.from_handle(data)
    if type_ == _lib.clingo_solve_event_type_finish:
        handler.on_finish(_ffi.cast('clingo_solve_result_bitset_t*', event)[0])

    if type_ == _lib.clingo_solve_event_type_model:
        goon[0] = handler.on_model(_ffi.cast('clingo_model_t*', event))

    if type_ == _lib.clingo_solve_event_type_statistics:
        p_stats = _ffi.cast('clingo_statistics_t**', event)
        handler.on_statistics(p_stats[0], p_stats[1])

    return True

class _Context:
    def __init__(self, context):
        self.error = None
        self.context = context

    def __call__(self, name, args):
        context = self.context if hasattr(self.context, name) else sys.modules['__main__']
        return getattr(context, name)(*args)

@_ffi.def_extern(onerror=_cb_error_handler('data'))
def _clingo_ground_callback(location, name, arguments, arguments_size, data, symbol_callback, symbol_callback_data):
    '''
    Low-level ground callback.
    '''
    # Note: location could be attached to error message
    # pylint: disable=protected-access,unused-argument
    context = _ffi.from_handle(data)

    args = []
    for i in range(arguments_size):
        args.append(Symbol(arguments[i]))
    symbols = list(context(_ffi.string(name).decode(), args))

    c_symbols = _ffi.new('clingo_symbol_t[]', len(symbols))
    for i, sym in enumerate(symbols):
        c_symbols[i] = sym._rep
    symbol_callback(c_symbols, len(symbols), symbol_callback_data)

    return True

def _statistics(stats, key):
    '''
    Transform clingo's statistics into python type.
    '''
    type_ = _c_call('clingo_statistics_type_t ', _lib.clingo_statistics_type, stats, key)

    if type_ == _lib.clingo_statistics_type_value:
        return _c_call('double', _lib.clingo_statistics_value_get, stats, key)

    if type_ == _lib.clingo_statistics_type_array:
        ret = []
        for i in range(_c_call('size_t', _lib.clingo_statistics_array_size, stats, key)):
            ret.append(_statistics(stats, _c_call('uint64_t', _lib.clingo_statistics_array_at, stats, key, i)))
        return ret

    assert type_ == _lib.clingo_statistics_type_map
    ret = {}
    for i in range(_c_call('size_t', _lib.clingo_statistics_map_size, stats, key)):
        name = _c_call('char*', _lib.clingo_statistics_map_subkey_name, stats, key, i)
        subkey = _c_call('uint64_t', _lib.clingo_statistics_map_at, stats, key, name)
        ret[_to_str(name)] = _statistics(stats, subkey)
    return ret

class Control:
    '''
    Control object for the grounding/solving process.

    Parameters
    ----------
    arguments : Sequence[str]
        Arguments to the grounder and solver.
    logger : Callable[[MessageCode,str],None]=None
        Function to intercept messages normally printed to standard error.
    message_limit : int=20
        The maximum number of messages passed to the logger.

    Notes
    -----
    Note that only gringo options (without `--text`) and clasp's search options are
    supported. Furthermore, a `Control` object is blocked while a search call is
    active; you must not call any member function during search.
    '''
    def __init__(self, arguments: Sequence[str]=[],
                 logger: Callable[[MessageCode,str],None]=None, message_limit: int=20):
        # pylint: disable=protected-access,dangerous-default-value
        if logger is not None:
            c_handle = _ffi.new_handle(logger)
            c_cb = _lib._clingo_logger_callback
        else:
            c_handle = _ffi.NULL
            c_cb = _ffi.NULL
        c_mem = []
        c_args = _ffi.new('char*[]', len(arguments))
        for i, arg in enumerate(arguments):
            c_mem.append(_ffi.new("char[]", arg.encode()))
            c_args[i] = c_mem[-1]
        self._rep = _c_call('clingo_control_t *', _lib.clingo_control_new,
                            c_args, len(arguments), c_cb, c_handle, message_limit)
        self._handler = None
        self._statistics = None
        self._statistics_call = -1.0

    def __del__(self):
        _lib.clingo_control_free(self._rep)

    def add(self, name: str, parameters: Sequence[str], program: str) -> None:
        '''
        Extend the logic program with the given non-ground logic program in string form.

        Parameters
        ----------
        name : str
            The name of program block to add.
        parameters : Sequence[str]
            The parameters of the program block to add.
        program : str
            The non-ground program in string form.

        Returns
        -------
        None

        See Also
        --------
        Control.ground
        '''
        c_mem = []
        c_params = _ffi.new('char*[]', len(parameters))
        for i, param in enumerate(parameters):
            c_mem.append(_ffi.new("char[]", param.encode()))
            c_params[i] = c_mem[-1]
        _handle_error(_lib.clingo_control_add(self._rep, name.encode(), c_params, len(parameters), program.encode()))

    def _program_atom(self, lit: Union[Symbol,int]) -> int:
        if isinstance(lit, int):
            return lit
        satom = self.symbolic_atoms[lit]
        return 0 if satom is None else satom.literal

    def assign_external(self, external: Union[Symbol,int], truth: Optional[bool]) -> None:
        '''
        Assign a truth value to an external atom.

        Parameters
        ----------
        external : Union[Symbol,int]
            A symbol or program literal representing the external atom.
        truth : Optional[bool]
            A Boolean fixes the external to the respective truth value; and None leaves
            its truth value open.

        Returns
        -------
        None

        See Also
        --------
        Control.release_external, SolveControl.symbolic_atoms, SymbolicAtom.is_external

        Notes
        -----
        The truth value of an external atom can be changed before each solve call. An
        atom is treated as external if it has been declared using an `#external`
        directive, and has not been released by calling release_external() or defined
        in a logic program with some rule. If the given atom is not external, then the
        function has no effect.

        For convenience, the truth assigned to atoms over negative program literals is
        inverted.
        '''

        if truth is None:
            val = _lib.clingo_external_type_free
        elif truth:
            val = _lib.clingo_external_type_true
        else:
            val = _lib.clingo_external_type_false
        _handle_error(_lib.clingo_control_assign_external(self._rep, self._program_atom(external), val))

    def backend(self) -> Backend:
        '''
        Returns a `Backend` object providing a low level interface to extend a logic
        program.

        Returns
        -------
        Backend
        '''
        return Backend(_c_call('clingo_backend_t*', _lib.clingo_control_backend, self._rep))

    def cleanup(self) -> None:
        '''
        Cleanup the domain used for grounding by incorporating information from the
        solver.

        This function cleans up the domain used for grounding.  This is done by first
        simplifying the current program representation (falsifying released external
        atoms).  Afterwards, the top-level implications are used to either remove atoms
        from the domain or mark them as facts.

        Returns
        -------
        None

        See Also
        --------
        Control.enable_cleanup

        Notes
        -----
        Any atoms falsified are completely removed from the logic program. Hence, a
        definition for such an atom in a successive step introduces a fresh atom.

        With the current implementation, the function only has an effect if called
        after solving and before any function is called that starts a new step.

        Typically, it is not necessary to call this function manually because automatic
        cleanups are enabled by default.
        '''
        _handle_error(_lib.clingo_control_cleanup(self._rep))

    def get_const(self, name: str) -> Optional[Symbol]:
        '''
        Return the symbol for a constant definition of form: `#const name = symbol.`

        Parameters
        ----------
        name : str
            The name of the constant to retrieve.

        Returns
        -------
        Optional[Symbol]
            The function returns `None` if no matching constant definition exists.
        '''
        if not _c_call('bool', _lib.clingo_control_has_const, self._rep, name.encode()):
            return None

        return Symbol(_c_call('clingo_symbol_t', _lib.clingo_control_get_const, self._rep, name.encode()))

    def ground(self, parts: Sequence[Tuple[str,Sequence[Symbol]]], context: Any=None) -> None:
        '''
        Ground the given list of program parts specified by tuples of names and arguments.

        Parameters
        ----------
        parts : Sequence[Tuple[str,Sequence[Symbol]]]
            List of tuples of program names and program arguments to ground.
        context : Any=None
            A context object whose methods are called during grounding using the
            `@`-syntax (if omitted methods, those from the main module are used).

        Notes
        -----
        Note that parts of a logic program without an explicit `#program` specification
        are by default put into a program called `base` without arguments.

        Examples
        --------

            >>> import clingo
            >>> ctl = clingo.Control()
            >>> ctl.add("p", ["t"], "q(t).")
            >>> parts = []
            >>> parts.append(("p", [1]))
            >>> parts.append(("p", [2]))
            >>> ctl.ground(parts)
            >>> ctl.solve(on_model=lambda m: print("Answer: {}".format(m)))
            Answer: q(1) q(2)
            SAT
        '''
        # pylint: disable=protected-access,dangerous-default-value
        handler = None
        c_handler = _ffi.NULL
        c_cb = _ffi.NULL
        if context is not None:
            handler = _Context(context)
            c_handler = _ffi.new_handle(handler)
            c_cb = _lib._clingo_ground_callback

        c_mem = []
        c_parts = _ffi.new("clingo_part_t[]", len(parts))
        for part, c_part in zip(parts, c_parts):
            c_mem.append(_ffi.new("char[]", part[0].encode()))
            c_part.name = c_mem[-1]
            c_mem.append(_ffi.new("clingo_symbol_t[]", len(part[1])))
            c_part.params = c_mem[-1]
            for i, sym in enumerate(part[1]):
                c_part.params[i] = sym._rep
            c_part.size = len(part[1])

        _handle_error(_lib.clingo_control_ground(
            self._rep, c_parts, len(parts), c_cb, c_handler), handler)

    def interrupt(self) -> None:
        '''
        Interrupt the active solve call.

        Returns
        -------
        None

        Notes
        -----
        This function is thread-safe and can be called from a signal handler. If no
        search is active, the subsequent call to `Control.solve` is interrupted. The
        result of the `Control.solve` method can be used to query if the search was
        interrupted.
        '''
        _lib.clingo_control_interrupt(self._rep)

    def load(self, path: str) -> None:
        '''
        Extend the logic program with a (non-ground) logic program in a file.

        Parameters
        ----------
        path : str
            The path of the file to load.

        Returns
        -------
        None
        '''
        _handle_error(_lib.clingo_control_load(self._rep, path.encode()))

    def register_observer(self, observer: Observer, replace: bool=False) -> None:
        '''
        Registers the given observer to inspect the produced grounding.

        Parameters
        ----------
        observer : Observer
            The observer to register. See below for a description of the requirede
            interface.
        replace : bool=False
            If set to true, the output is just passed to the observer and nolonger to
            the underlying solver (or any previously registered observers).

        Returns
        -------
        None

        Notes
        -----
        Not all functions the `Observer` interface have to be implemented and can be
        omitted if not needed.
        '''
        raise RuntimeError('implement me!!!')

    def register_propagator(self, propagator: Propagator) -> None:
        '''
        Registers the given propagator with all solvers.

        Parameters
        ----------
        propagator : Propagator
            The propagator to register.

        Returns
        -------
        None

        Notes
        -----
        Each symbolic or theory atom is uniquely associated with a positive program
        atom in form of a positive integer. Program literals additionally have a sign
        to represent default negation. Furthermore, there are non-zero integer solver
        literals. There is a surjective mapping from program atoms to solver literals.

        All methods called during propagation use solver literals whereas
        `SymbolicAtom.literal` and `TheoryAtom.literal` return program literals. The
        function `PropagateInit.solver_literal` can be used to map program literals or
        condition ids to solver literals.

        Not all functions of the `Propagator` interface have to be implemented and can
        be omitted if not needed.
        '''
        raise RuntimeError('implement me!!!')

    def release_external(self, external: Union[Symbol,int]) -> None:
        '''
        Release an external atom represented by the given symbol or program literal.

        This function causes the corresponding atom to become permanently false if
        there is no definition for the atom in the program. Otherwise, the function has
        no effect.

        Parameters
        ----------
        external : Union[Symbol,int]
            The symbolic atom or program atom to release.

        Returns
        -------
        None

        Notes
        -----
        If the program literal is negative, the corresponding atom is released.

        Examples
        --------
        The following example shows the effect of assigning and releasing and external
        atom.

            >>> import clingo
            >>> ctl = clingo.Control()
            >>> ctl.add("base", [], "a. #external b.")
            >>> ctl.ground([("base", [])])
            >>> ctl.assign_external(clingo.Function("b"), True)
            >>> ctl.solve(on_model=lambda m: print("Answer: {}".format(m)))
            Answer: b a
            SAT
            >>> ctl.release_external(clingo.Function("b"))
            >>> ctl.solve(on_model=lambda m: print("Answer: {}".format(m)))
            Answer: a
            SAT
        '''
        _handle_error(_lib.clingo_control_release_external(self._rep, self._program_atom(external)))

    def solve(self,
              assumptions: Sequence[Union[Tuple[Symbol,bool],int]]=[],
              on_model: Callable[[Model],Optional[bool]]=None,
              on_statistics : Callable[[StatisticsMap,StatisticsMap],None]=None,
              on_finish: Callable[[SolveResult],None]=None,
              on_core: Callable[[Sequence[int]],None]=None,
              yield_: bool=False,
              async_: bool=False) -> Union[SolveHandle,SolveResult]:
        '''
        Starts a search.

        Parameters
        ----------
        assumptions : Sequence[Union[Tuple[Symbol,bool],int]]=[]
            List of (atom, boolean) tuples or program literals that serve
            as assumptions for the solve call, e.g., solving under
            assumptions `[(Function("a"), True)]` only admits answer sets
            that contain atom `a`.
        on_model : Callable[[Model],Optional[bool]]=None
            Optional callback for intercepting models.
            A `Model` object is passed to the callback.
            The search can be interruped from the model callback by
            returning False.
        on_statistics : Callable[[StatisticsMap,StatisticsMap],None]=None
            Optional callback to update statistics.
            The step and accumulated statistics are passed as arguments.
        on_finish : Callable[[SolveResult],None]=None
            Optional callback called once search has finished.
            A `SolveResult` also indicating whether the solve call has been intrrupted
            is passed to the callback.
        on_core : Callable[[Sequence[int]],None]=None
            Optional callback called with the assumptions that made a problem
            unsatisfiable.
        yield_ : bool=False
            The resulting `SolveHandle` is iterable yielding `Model` objects.
        async_ : bool=False
            The solve call and the method `SolveHandle.resume` of the returned handle
            are non-blocking.

        Returns
        -------
        Union[SolveHandle,SolveResult]
            The return value depends on the parameters. If either `yield_` or `async_`
            is true, then a handle is returned. Otherwise, a `SolveResult` is returned.

        Notes
        -----
        If neither `yield_` nor `async_` is set, the function returns a SolveResult right
        away.

        Note that in gringo or in clingo with lparse or text output enabled this
        function just grounds and returns a SolveResult where `SolveResult.unknown`
        is true.

        If this function is used in embedded Python code, you might want to start
        clingo using the `--outf=3` option to disable all output from clingo.

        Note that asynchronous solving is only available in clingo with thread support
        enabled. Furthermore, the on_model and on_finish callbacks are called from
        another thread. To ensure that the methods can be called, make sure to not use
        any functions that block Python's GIL indefinitely.

        This function as well as blocking functions on the `SolveHandle` release the GIL
        but are not thread-safe.

        Examples
        --------

        The following example shows how to intercept models with a callback:

            >>> import clingo
            >>> ctl = clingo.Control("0")
            >>> ctl.add("p", [], "1 { a; b } 1.")
            >>> ctl.ground([("p", [])])
            >>> ctl.solve(on_model=lambda m: print("Answer: {}".format(m)))
            Answer: a
            Answer: b
            SAT

        The following example shows how to yield models:

            >>> import clingo
            >>> ctl = clingo.Control("0")
            >>> ctl.add("p", [], "1 { a; b } 1.")
            >>> ctl.ground([("p", [])])
            >>> with ctl.solve(yield_=True) as handle:
            ...     for m in handle: print("Answer: {}".format(m))
            ...     handle.get()
            ...
            Answer: a
            Answer: b
            SAT

        The following example shows how to solve asynchronously:

            >>> import clingo
            >>> ctl = clingo.Control("0")
            >>> ctl.add("p", [], "1 { a; b } 1.")
            >>> ctl.ground([("p", [])])
            >>> with ctl.solve(on_model=lambda m: print("Answer: {}".format(m)), async_=True) as handle:
            ...     while not handle.wait(0): pass
            ...     handle.get()
            ...
            Answer: a
            Answer: b
            SAT
        '''
        # pylint: disable=protected-access,dangerous-default-value
        handler = _SolveEventHandler(on_model, on_statistics, on_finish)
        self._handler = _ffi.new_handle(handler)

        p_ass = _ffi.NULL
        if assumptions:
            atoms = None
            p_ass = _ffi.new('clingo_literal_t[]', len(assumptions))
            for i, lit in enumerate(assumptions):
                if isinstance(lit, int):
                    p_ass[i] = lit
                else:
                    if atoms is None:
                        atoms = self.symbolic_atoms
                    atom = self.symbolic_atoms[lit[0]]
                    slit = -1 if atom is None else atom.literal
                    p_ass[i] = slit if lit[1] else -slit

        mode = 0
        if yield_:
            mode |= _lib.clingo_solve_mode_yield
        if async_:
            mode |= _lib.clingo_solve_mode_async

        p_handle = _ffi.new('clingo_solve_handle_t**')
        _handle_error(_lib.clingo_control_solve(
            self._rep, mode,
            p_ass, len(assumptions),
            _lib._clingo_solve_event_callback, self._handler,
            p_handle), handler)
        handle = SolveHandle(p_handle[0], handler)

        if not yield_ and not async_:
            with handle:
                ret = handle.get()
                if on_core is not None and ret.unsatisfiable:
                    on_core(handle.core())
                return ret
        return handle

    @property
    def configuration(self) -> Configuration:
        '''
        `Configuration` object to change the configuration.
        '''
        conf = _c_call('clingo_configuration_t*', _lib.clingo_control_configuration, self._rep)
        key = _c_call('clingo_id_t', _lib.clingo_configuration_root, conf)
        return Configuration(conf, key)

    @property
    def enable_cleanup(self) -> bool:
        '''
        Whether to enable automatic calls to `Control.cleanup`.
        '''
        return _lib.clingo_control_get_enable_cleanup(self._rep)

    @enable_cleanup.setter
    def enable_cleanup(self, value: bool) -> None:
        _handle_error(_lib.clingo_control_set_enable_cleanup(self._rep, value))

    @property
    def enable_enumeration_assumption(self) -> bool:
        '''
        Whether do discard or keep learnt information from enumeration modes.

        If the enumeration assumption is enabled, then all information learnt from
        clasp's various enumeration modes is removed after a solve call. This includes
        enumeration of cautious or brave consequences, enumeration of answer sets with
        or without projection, or finding optimal models; as well as clauses added with
        `SolveControl.add_clause`.

        Notes
        -----
        Initially the enumeration assumption is enabled.

        In general, the enumeration assumption should be enabled whenever there are
        multiple calls to solve. Otherwise, the behavior of the solver will be
        unpredictable because there are no guarantees which information exactly is
        kept. There might be small speed benefits when disabling the enumeration
        assumption for single shot solving.
        '''
        return _lib.clingo_control_get_enable_enumeration_assumption(self._rep)

    @enable_enumeration_assumption.setter
    def enable_enumeration_assumption(self, value: bool) -> None:
        _handle_error(_lib.clingo_control_set_enable_enumeration_assumption(self._rep, value))

    @property
    def is_conflicting(self) -> bool:
        '''
        Whether the internal program representation is conflicting.

        If this (read-only) property is true, solve calls return immediately with an
        unsatisfiable solve result.

        Notes
        -----
        Conflicts first have to be detected, e.g., initial unit propagation results in
        an empty clause, or later if an empty clause is resolved during solving. Hence,
        the property might be false even if the problem is unsatisfiable.
        '''
        return _lib.clingo_control_is_conflicting(self._rep)

    @property
    def statistics(self) -> dict:
        '''
        A `dict` containing solve statistics of the last solve call.

        Notes
        -----
        The statistics correspond to the `--stats` output of clingo. The detail of the
        statistics depends on what level is requested on the command line. Furthermore,
        there are some functions like `Control.release_external` that start a new
        solving step resetting the current step statistics. It is best to access the
        statistics right after solving.

        This property is only available in clingo.

        Examples
        --------
        The following example shows how to dump the solving statistics in json format:

            >>> import json
            >>> import clingo
            >>> ctl = clingo.Control()
            >>> ctl.add("base", [], "{a}.")
            >>> ctl.ground([("base", [])])
            >>> ctl.solve()
            SAT
            >>> print(json.dumps(ctl.statistics['solving'], sort_keys=True, indent=4,
            ... separators=(',', ': ')))
            {
                "solvers": {
                    "choices": 1.0,
                    "conflicts": 0.0,
                    "conflicts_analyzed": 0.0,
                    "restarts": 0.0,
                    "restarts_last": 0.0
                }
            }

        '''
        stats = _c_call('clingo_statistics_t*', _lib.clingo_control_statistics, self._rep)

        p_key = _ffi.new('uint64_t*')
        key_root = _c_call(p_key, _lib.clingo_statistics_root, stats)

        key_summary = _c_call(p_key, _lib.clingo_statistics_map_at, stats, key_root, "summary".encode())
        key_call = _c_call(p_key, _lib.clingo_statistics_map_at, stats, key_summary, "call".encode())
        call = _c_call('double', _lib.clingo_statistics_value_get, stats, key_call)
        if self._statistics is not None and call != self._statistics_call:
            self._statistics = None

        if self._statistics is None:
            self._statistics_call = call
            self._statistics = _statistics(stats, key_root)

        return cast(dict, self._statistics)

    @property
    def symbolic_atoms(self) -> SymbolicAtoms:
        '''
        `SymbolicAtoms` object to inspect the symbolic atoms.
        '''
        return SymbolicAtoms(_c_call('clingo_symbolic_atoms_t*', _lib.clingo_control_symbolic_atoms, self._rep))

    @property
    def theory_atoms(self) -> Iterator[TheoryAtom]:
        '''
        An iterator over the theory atoms.
        '''
        atoms = _c_call('clingo_theory_atoms_t*', _lib.clingo_control_theory_atoms, self._rep)
        size = _c_call('size_t', _lib.clingo_theory_atoms_size, atoms)

        for idx in range(size):
            yield TheoryAtom(atoms, idx)

# {{{1 application [0%]

class Flag:
    '''
    Helper object to parse command-line flags.

    Parameters
    ----------
    value : bool=False
        The initial value of the flag.
    '''
    def __init__(self, value: bool=False):
        pass

    flag: bool
    '''
    The value of the flag.
    '''

class ApplicationOptions(metaclass=ABCMeta):
    '''
    Object to add custom options to a clingo based application.
    '''
    def add(self, group: str, option: str, description: str, parser: Callable[[str], bool],
            multi: bool=False, argument: str=None) -> None:
        '''
        Add an option that is processed with a custom parser.

        Parameters
        ----------
        group : str
            Options are grouped into sections as given by this string.
        option : str
            Parameter option specifies the name(s) of the option. For example,
            `"ping,p"` adds the short option `-p` and its long form `--ping`. It is
            also possible to associate an option with a help level by adding `",@l"` to
            the option specification. Options with a level greater than zero are only
            shown if the argument to help is greater or equal to `l`.
        description : str
            The description of the option shown in the help output.
        parser : Callable[[str],bool]
            An option parser is a function that takes a string as input and returns
            true or false depending on whether the option was parsed successively.
        multi : bool=False
            Whether the option can appear multiple times on the command-line.
        argument : str=None
            Optional string to change the value name in the generated help.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            An error is raised if an option with the same name already exists.

        Notes
        -----
        The parser also has to take care of storing the semantic value of the option
        somewhere.
        '''

    def add_flag(self, group: str, option: str, description: str, target: Flag) -> None:
        '''
        add_flag(self, group: str, option: str, description: str, target: Flag) -> None

        Add an option that is a simple flag.

        This function is similar to `ApplicationOptions.add` but simpler because
        it only supports flags, which do not have values. Note that the target
        parameter must be of type Flag, which is set to true if the flag is passed on
        the command line.

        Parameters
        ----------
        group : str
            Options are grouped into sections as given by this string.
        option : str
            Same as for `ApplicationOptions.add`.
        description : str
            The description of the option shown in the help output.
        target : Flag
            The object that receives the value.

        Returns
        -------
        None
        '''

class Application(metaclass=ABCMeta):
    """
    Interface that has to be implemented to customize clingo.

    Attributes
    ----------
    program_name: str = 'clingo'
        Optional program name to be used in the help output.

    message_limit: int = 20
        Maximum number of messages passed to the logger.
    """

    @abstractmethod
    def main(self, control: Control, files: Sequence[str]) -> None:
        """
        Function to replace clingo's default main function.

        Parameters
        ----------
        control : Control
            The main control object.
        files : Sequence[str]
            The files passed to clingo_main.

        Returns
        -------
        None
        """

    def register_options(self, options: ApplicationOptions) -> None:
        """
        Function to register custom options.

        Parameters
        ----------
        options : ApplicationOptions
            Object to register additional options

        Returns
        -------
        None
        """

    def validate_options(self) -> bool:
        """
        Function to validate custom options.

        This function should return false or throw an exception if option
        validation fails.

        Returns
        -------
        bool
        """

    def print_model(self, model: Model, printer: Callable[[], None]) -> None:
        """
        Function to print additional information when the text output is used.

        Parameters
        ----------
        model : model
            The current model
        printer : Callable[[], None]
            The default printer as used in clingo.

        Returns
        -------
        None
        """

    def logger(self, code: MessageCode, message: str) -> None:
        """
        Function to intercept messages normally printed to standard error.

        By default, messages are printed to stdandard error.

        Parameters
        ----------
        code : MessageCode
            The message code.
        message : str
            The message string.

        Returns
        -------
        None

        Notes
        -----
        This function should not raise exceptions.
        """

def clingo_main(application: Application, files: Iterable[str]=[]) -> int:
    '''
    clingo_main(application: Application, files: Iterable[str]=[]) -> int

    Runs the given application using clingo's default output and signal handling.

    The application can overwrite clingo's default behaviour by registering
    additional options and overriding its default main function.

    Parameters
    ----------
    application : Application
        The Application object (see notes).
    files : Iterable[str]
        The files to pass to the main function of the application.

    Returns
    -------
    int
        The exit code of the application.

    Notes
    -----
    The main function of the `Application` interface has to be implemented. All
    other members are optional.

    Examples
    --------
    The following example reproduces the default clingo application:

        import sys
        import clingo

        class Application(clingo.Application):
            def __init__(self, name):
                self.program_name = name

            def main(self, ctl, files):
                if len(files) > 0:
                    for f in files:
                        ctl.load(f)
                else:
                    ctl.load("-")
                ctl.ground([("base", [])])
                ctl.solve()

        clingo.clingo_main(Application(sys.argv[0]), sys.argv[1:])
    '''
    # pylint: disable=dangerous-default-value,unused-argument,unnecessary-pass
    pass
