'''
The clingo.ast-5.5.0 module.

The grammar below defines valid ASTs. For each upper case identifier there is a
matching function in the module. Arguments follow in parenthesis: each having a
type given on the right-hand side of the colon. The symbols `?`, `*`, and `+`
are used to denote optional arguments (`None` encodes abscence), list
arguments, and non-empty list arguments.

```
# Terms

term = Symbol
        ( location : Location
        , symbol   : clingo.Symbol
        )
     | Variable
        ( location : Location
        , name     : str
        )
     | UnaryOperation
        ( location : Location
        , operator : UnaryOperator
        , argument : term
        )
     | BinaryOperation
        ( location : Location
        , operator : BinaryOperator
        , left     : term
        , right    : term
        )
     | Interval
        ( location : Location
        , left     : term
        , right    : term
        )
     | Function
        ( location  : Location
        , name      : str
        , arguments : term*
        , external  : bool
        )
     | Pool
        ( location  : Location
        , arguments : term*
        )

csp_term = CSPSum
            ( location : Location
            , terms    : CSPProduct
                          ( location    : Location
                          , coefficient : term
                          , variable    : term?
                          )*
            )

theory_term = Symbol
               ( location : Location
               , symbol   : clingo.Symbol
               )
            | Variable
               ( location : Location
               , name     : str
               )
            | TheorySequence
               ( location : Location
               , sequence_type : TheorySequenceType
               , terms         : theory_term*
               )
            | TheoryFunction
               ( location  : Location
               , name      : str
               , arguments : theory_term*
               )
            | TheoryUnparsedTerm
               ( location : Location
               , elements : TheoryUnparsedTermElement
                             ( operators : str*
                             , term      : theory_term
                             )+
               )

# Literals

symbolic_atom = SymbolicAtom
                 ( term : term
                 )

literal = Literal
           ( location : Location
           , sign     : Sign
           , atom     : Comparison
                         ( comparison : ComparisonOperator
                         , left       : term
                         , right      : term
                         )
                      | BooleanConstant
                         ( value : bool
                         )
                      | symbolic_atom
           )

        | CSPLiteral
           ( location : Location
           , term     : csp_term
           , guards   : CSPGuard
                         ( comparison : ComparisonOperator
                         , term       : csp_term
                         )+
           )

# Head and Body Literals

aggregate_guard = AggregateGuard
                   ( comparison : ComparisonOperator
                   , term       : term
                   )

conditional_literal = ConditionalLiteral
                       ( location  : Location
                       , literal   : Literal
                       , condition : Literal*
                       )

aggregate = Aggregate
             ( location    : Location
             , left_guard  : aggregate_guard?
             , elements    : conditional_literal*
             , right_guard : aggregate_guard?
             )

theory_atom = TheoryAtom
               ( location : Location
               , term     : term
               , elements : TheoryAtomElement
                             ( tuple     : theory_term*
                             , condition : literal*
                             )*
               , guard    : TheoryGuard
                             ( operator_name : str
                             , term          : theory_term
                             )?
               )

body_atom = aggregate
          | BodyAggregate
             ( location    : Location
             , left_guard  : aggregate_guard?
             , function    : AggregateFunction
             , elements    : BodyAggregateElement
                              ( tuple     : term*
                              , condition : literal*
                              )*
             , right_guard : aggregate_guard?
             )
          | Disjoint
             ( location : Location
             , elements : DisjointElement
                           ( location  : Location
                           , tuple     : term*
                           , term      : csp_term
                           , condition : literal*
                           )*
             )
          | theory_atom

body_literal = literal
             | conditional_literal
             | Literal
                ( location : Location
                , sign     : Sign
                , atom     : body_atom
                )

head = literal
     | aggregate
     | HeadAggregate
        ( location    : Location
        , left_guard  : aggregate_guard?
        , function    : AggregateFunction
        , elements    : HeadAggregateElement
                         ( tuple     : term*
                         , condition : conditional_literal
                         )*
        , right_guard : aggregate_guard?
        )
     | Disjunction
        ( location : Location
        , elements : conditional_literal*
        )
     | theory_atom

# Statements

statement = Rule
             ( location : Location
             , head     : head
             , body     : body_literal*
             )
          | Definition
             ( location   : Location
             , name       : str
             , value      : term
             , is_default : bool
             )
          | ShowSignature
             ( location   : Location
             , name       : str
             , arity      : int
             , sign       : bool
             , csp        : bool
             )
          | Defined
             ( location   : Location
             , name       : str
             , arity      : int
             , sign       : bool
             )
          | ShowTerm
             ( location : Location
             , term     : term
             , body     : body_literal*
             , csp      : bool
             )
          | Minimize
             ( location : Location
             , weight   : term
             , priority : term
             , tuple    : term*
             , body     : body_literal*
             )
          | Script
             ( location    : Location
             , script_type : ScriptType
             , code        : str
             )
          | Program
             ( location   : Location
             , name       : str
             , parameters : Id
                             ( location : Location
                             , id       : str
                             )*
             )
          | External
             ( location : Location
             , atom     : symbolic_atom
             , body     : body_literal*
             , type     : term
             )
          | Edge
             ( location : Location
             , u        : term
             , v        : term
             , body     : body_literal*
             )
          | Heuristic
             ( location : Location
             , atom     : symbolic_atom
             , body     : body_literal*
             , bias     : term
             , priority : term
             , modifier : term
             )
          | ProjectAtom
             ( location : Location
             , atom     : symbolic_atom
             , body     : body_literal*
             )
          | ProjectSignature
             ( location   : Location
             , name       : str
             , arity      : int
             , positive   : bool
             )
          | TheoryDefinition
             ( location : Location
             , name     : str
             , terms    : TheoryTermDefinition
                           ( location  : Location
                           , name      : str
                           , operators : TheoryOperatorDefinition
                                          ( location      : Location
                                          , name          : str
                                          , priority      : int
                                          , operator_type : TheoryOperatorType
                                          )*
                           )
             , atoms    : TheoryAtomDefinition
                           ( location  : Location
                           , atom_type : TheoryAtomType
                           , name      : str
                           , arity     : int
                           , elements  : str
                           , guard     : TheoryGuardDefinition
                                          ( operators : str*
                                          , term      : str
                                          )?
                           )
             )
```
'''

from enum import Enum
from typing import Any, Callable, ContextManager, Iterable, List, Tuple
from abc import ABCMeta

from ._internal import (_CBData, _Error,
                        _cb_error_handler, _c_call, _ffi, _handle_error, _lib, _to_str)
from .core import MessageCode
from .control import Control

# It might be a good idea to implement the AST in C with only a few functions and enums:
#
# - Value: Union[int,str,Symbol,AST,List[str],List[AST]]
# - AST.location() -> Location
# - AST.type() -> ASTType
# - AST.size() -> int
# - AST.get_key(int) -> str
# - AST.get_value(str) -> Value
# - AST.set_item(str, Value) -> None
# - AST.check() -> bool (to check if all required values are present)
#
# With this interface, the ast can be implemented easily in any dynamically
# typed language. Ownership could be handled with shared pointers (requiring
# users not to build cycles). Recursion depth should be no problem in typical
# ASP programs.

def Aggregate(*args: Any, **kwargs: Any) -> Any:
    pass

def AggregateGuard(*args: Any, **kwargs: Any) -> Any:
    pass

def BinaryOperation(*args: Any, **kwargs: Any) -> Any:
    pass

def BodyAggregate(*args: Any, **kwargs: Any) -> Any:
    pass

def BodyAggregateElement(*args: Any, **kwargs: Any) -> Any:
    pass

def BooleanConstant(*args: Any, **kwargs: Any) -> Any:
    pass

def CSPGuard(*args: Any, **kwargs: Any) -> Any:
    pass

def CSPLiteral(*args: Any, **kwargs: Any) -> Any:
    pass

def CSPProduct(*args: Any, **kwargs: Any) -> Any:
    pass

def CSPSum(*args: Any, **kwargs: Any) -> Any:
    pass

def Comparison(*args: Any, **kwargs: Any) -> Any:
    pass

def ConditionalLiteral(*args: Any, **kwargs: Any) -> Any:
    pass

def Defined(*args: Any, **kwargs: Any) -> Any:
    pass

def Definition(*args: Any, **kwargs: Any) -> Any:
    pass

def Disjoint(*args: Any, **kwargs: Any) -> Any:
    pass

def DisjointElement(*args: Any, **kwargs: Any) -> Any:
    pass

def Disjunction(*args: Any, **kwargs: Any) -> Any:
    pass

def Edge(*args: Any, **kwargs: Any) -> Any:
    pass

def External(*args: Any, **kwargs: Any) -> Any:
    pass

def Function(*args: Any, **kwargs: Any) -> Any:
    pass

def HeadAggregate(*args: Any, **kwargs: Any) -> Any:
    pass

def HeadAggregateElement(*args: Any, **kwargs: Any) -> Any:
    pass

def Heuristic(*args: Any, **kwargs: Any) -> Any:
    pass

def Id(*args: Any, **kwargs: Any) -> Any:
    pass

def Interval(*args: Any, **kwargs: Any) -> Any:
    pass

def Literal(*args: Any, **kwargs: Any) -> Any:
    pass

def Minimize(*args: Any, **kwargs: Any) -> Any:
    pass

def Pool(*args: Any, **kwargs: Any) -> Any:
    pass

def Program(*args: Any, **kwargs: Any) -> Any:
    pass

def ProjectAtom(*args: Any, **kwargs: Any) -> Any:
    pass

def ProjectSignature(*args: Any, **kwargs: Any) -> Any:
    pass

def Rule(*args: Any, **kwargs: Any) -> Any:
    pass

def Script(*args: Any, **kwargs: Any) -> Any:
    pass

def ShowSignature(*args: Any, **kwargs: Any) -> Any:
    pass

def ShowTerm(*args: Any, **kwargs: Any) -> Any:
    pass

def Symbol(*args: Any, **kwargs: Any) -> Any:
    pass

def SymbolicAtom(*args: Any, **kwargs: Any) -> Any:
    pass

def TheoryAtom(*args: Any, **kwargs: Any) -> Any:
    pass

def TheoryAtomDefinition(*args: Any, **kwargs: Any) -> Any:
    pass

def TheoryAtomElement(*args: Any, **kwargs: Any) -> Any:
    pass

def TheoryDefinition(*args: Any, **kwargs: Any) -> Any:
    pass

def TheoryFunction(*args: Any, **kwargs: Any) -> Any:
    pass

def TheoryGuard(*args: Any, **kwargs: Any) -> Any:
    pass

def TheoryGuardDefinition(*args: Any, **kwargs: Any) -> Any:
    pass

def TheoryOperatorDefinition(*args: Any, **kwargs: Any) -> Any:
    pass

def TheorySequence(*args: Any, **kwargs: Any) -> Any:
    pass

def TheoryTermDefinition(*args: Any, **kwargs: Any) -> Any:
    pass

def TheoryUnparsedTerm(*args: Any, **kwargs: Any) -> Any:
    pass

def TheoryUnparsedTermElement(*args: Any, **kwargs: Any) -> Any:
    pass

def UnaryOperation(*args: Any, **kwargs: Any) -> Any:
    pass

def Variable(*args: Any, **kwargs: Any) -> Any:
    pass


class ASTType(Enum):
    '''
    Enumeration of ast node types.
    '''
    Id = _lib.clingo_ast_type_id
    Variable = _lib.clingo_ast_type_variable
    Symbol = _lib.clingo_ast_type_symbol
    UnaryOperation = _lib.clingo_ast_type_unary_operation
    BinaryOperation = _lib.clingo_ast_type_binary_operation
    Interval = _lib.clingo_ast_type_interval
    Function = _lib.clingo_ast_type_function
    Pool = _lib.clingo_ast_type_pool
    CspProduct = _lib.clingo_ast_type_csp_product
    CspSum = _lib.clingo_ast_type_csp_sum
    CspGuard = _lib.clingo_ast_type_csp_guard
    BooleanConstant = _lib.clingo_ast_type_boolean_constant
    SymbolicAtom = _lib.clingo_ast_type_symbolic_atom
    Comparison = _lib.clingo_ast_type_comparison
    CspLiteral = _lib.clingo_ast_type_csp_literal
    AggregateGuard = _lib.clingo_ast_type_aggregate_guard
    ConditionalLiteral = _lib.clingo_ast_type_conditional_literal
    Aggregate = _lib.clingo_ast_type_aggregate
    BodyAggregateElement = _lib.clingo_ast_type_body_aggregate_element
    BodyAggregate = _lib.clingo_ast_type_body_aggregate
    HeadAggregateElement = _lib.clingo_ast_type_head_aggregate_element
    HeadAggregate = _lib.clingo_ast_type_head_aggregate
    Disjunction = _lib.clingo_ast_type_disjunction
    DisjointElement = _lib.clingo_ast_type_disjoint_element
    Disjoint = _lib.clingo_ast_type_disjoint
    TheorySequence = _lib.clingo_ast_type_theory_sequence
    TheoryFunction = _lib.clingo_ast_type_theory_function
    TheoryUnparsedTermElement = _lib.clingo_ast_type_theory_unparsed_term_element
    TheoryUnparsedTerm = _lib.clingo_ast_type_theory_unparsed_term
    TheoryGuard = _lib.clingo_ast_type_theory_guard
    TheoryAtomElement = _lib.clingo_ast_type_theory_atom_element
    TheoryAtom = _lib.clingo_ast_type_theory_atom
    Literal = _lib.clingo_ast_type_literal
    TheoryOperatorDefinition = _lib.clingo_ast_type_theory_operator_definition
    TheoryTermDefinition = _lib.clingo_ast_type_theory_term_definition
    TheoryGuardDefinition = _lib.clingo_ast_type_theory_guard_definition
    TheoryAtomDefinition = _lib.clingo_ast_type_theory_atom_definition
    Rule = _lib.clingo_ast_type_rule
    Definition = _lib.clingo_ast_type_definition
    ShowSignature = _lib.clingo_ast_type_show_signature
    ShowTerm = _lib.clingo_ast_type_show_term
    Minimize = _lib.clingo_ast_type_minimize
    Script = _lib.clingo_ast_type_script
    Program = _lib.clingo_ast_type_program
    External = _lib.clingo_ast_type_external
    Edge = _lib.clingo_ast_type_edge
    Heuristic = _lib.clingo_ast_type_heuristic
    ProjectAtom = _lib.clingo_ast_type_project_atom
    ProjectSignature = _lib.clingo_ast_type_project_signature
    Defined = _lib.clingo_ast_type_defined
    TheoryDefinition = _lib.clingo_ast_type_theory_definition

_attributes = {
    "argument": _lib.clingo_ast_attribute_argument,
    "arguments": _lib.clingo_ast_attribute_arguments,
    "arity": _lib.clingo_ast_attribute_arity,
    "atom": _lib.clingo_ast_attribute_atom,
    "atoms": _lib.clingo_ast_attribute_atoms,
    "atom_type": _lib.clingo_ast_attribute_atom_type,
    "bias": _lib.clingo_ast_attribute_bias,
    "body": _lib.clingo_ast_attribute_body,
    "code": _lib.clingo_ast_attribute_code,
    "coefficient": _lib.clingo_ast_attribute_coefficient,
    "comparison": _lib.clingo_ast_attribute_comparison,
    "condition": _lib.clingo_ast_attribute_condition,
    "csp": _lib.clingo_ast_attribute_csp,
    "elements": _lib.clingo_ast_attribute_elements,
    "external": _lib.clingo_ast_attribute_external,
    "external_type": _lib.clingo_ast_attribute_external_type,
    "function": _lib.clingo_ast_attribute_function,
    "guard": _lib.clingo_ast_attribute_guard,
    "guards": _lib.clingo_ast_attribute_guards,
    "head": _lib.clingo_ast_attribute_head,
    "id": _lib.clingo_ast_attribute_id,
    "is_default": _lib.clingo_ast_attribute_is_default,
    "left": _lib.clingo_ast_attribute_left,
    "left_guard": _lib.clingo_ast_attribute_left_guard,
    "literal": _lib.clingo_ast_attribute_literal,
    "location": _lib.clingo_ast_attribute_location,
    "modifier": _lib.clingo_ast_attribute_modifier,
    "name": _lib.clingo_ast_attribute_name,
    "node_u": _lib.clingo_ast_attribute_node_u,
    "node_v": _lib.clingo_ast_attribute_node_v,
    "operator": _lib.clingo_ast_attribute_operator,
    "operator_name": _lib.clingo_ast_attribute_operator_name,
    "operator_type": _lib.clingo_ast_attribute_operator_type,
    "operators": _lib.clingo_ast_attribute_operators,
    "parameters": _lib.clingo_ast_attribute_parameters,
    "priority": _lib.clingo_ast_attribute_priority,
    "right": _lib.clingo_ast_attribute_right,
    "right_guard": _lib.clingo_ast_attribute_right_guard,
    "script_type": _lib.clingo_ast_attribute_script_type,
    "sequence_type": _lib.clingo_ast_attribute_sequence_type,
    "sign": _lib.clingo_ast_attribute_sign,
    "symbol": _lib.clingo_ast_attribute_symbol,
    "term": _lib.clingo_ast_attribute_term,
    "terms": _lib.clingo_ast_attribute_terms,
    "tuple": _lib.clingo_ast_attribute_tuple,
    "value": _lib.clingo_ast_attribute_value,
    "var": _lib.clingo_ast_attribute_var,
    "variable": _lib.clingo_ast_attribute_variable,
    "weight": _lib.clingo_ast_attribute_weight}

class AST:
    '''
    Represents a node in the abstract syntax tree.

    AST nodes implement Python's rich comparison operators and are ordered
    structurally ignoring the location. They can also be used as dictionary keys.
    Their string representation corresponds to their gringo representation.

    Implements: `Any`.

    Parameters
    ----------
    type : ASTType
        The type of the onde.
    **arguments : Any
        Additionally, the functions takes an arbitrary number of keyword arguments.
        These should contain the required fields of the node but can also be set
        later.

    Notes
    -----
    It is also possible to create AST nodes using one of the functions provided in
    this module. The parameters of the functions correspond to the nonterminals as
    given in the [grammar](.) above.
    '''
    #def __init__(self, type_: ASTType, **arguments: Any):
    def __init__(self, rep):
        super().__setattr__("_rep", rep)
        # bool clingo_ast_attribute_clear(clingo_ast_t *ast, clingo_ast_attribute_t attribute);
        # bool clingo_ast_attribute_set_number(clingo_ast_t *ast, clingo_ast_attribute_t attribute, int value);
        # bool clingo_ast_attribute_set_location(clingo_ast_t *ast, clingo_ast_attribute_t attribute, clingo_location_t const *value);
        # bool clingo_ast_attribute_set_string(clingo_ast_t *ast, clingo_ast_attribute_t attribute, char const *value);
        # bool clingo_ast_attribute_set_ast(clingo_ast_t *ast, clingo_ast_attribute_t attribute, clingo_ast_t *value);
        # bool clingo_ast_attribute_get_string_at(clingo_ast_t *ast, clingo_ast_attribute_t attribute, size_t index, char const **value);
        # bool clingo_ast_attribute_set_string_at(clingo_ast_t *ast, clingo_ast_attribute_t attribute, size_t index, char const *value);
        # bool clingo_ast_attribute_delete_string_at(clingo_ast_t *ast, clingo_ast_attribute_t attribute, size_t index);
        # bool clingo_ast_attribute_size_string_array(clingo_ast_t *ast, clingo_ast_attribute_t attribute, size_t *size);
        # bool clingo_ast_attribute_insert_string_at(clingo_ast_t *ast, clingo_ast_attribute_t attribute, size_t index, char const *value);
        # bool clingo_ast_attribute_get_ast_at(clingo_ast_t *ast, clingo_ast_attribute_t attribute, size_t index, clingo_ast_t **value);
        # bool clingo_ast_attribute_set_ast_at(clingo_ast_t *ast, clingo_ast_attribute_t attribute, size_t index, clingo_ast_t *value);
        # bool clingo_ast_attribute_delete_ast_at(clingo_ast_t *ast, clingo_ast_attribute_t attribute, size_t index);
        # bool clingo_ast_attribute_size_ast_array(clingo_ast_t *ast, clingo_ast_attribute_t attribute, size_t *size);
        # bool clingo_ast_attribute_insert_ast_at(clingo_ast_t *ast, clingo_ast_attribute_t attribute, size_t index, clingo_ast_t *value);

    @property
    def ast_type(self):
        return ASTType(_c_call('clingo_ast_type_t', _lib.clingo_ast_get_type, self._rep))

    def __getattr__(self, name):
        if not name in _attributes:
            raise AttributeError(f'no attribute: {name}')
        attr_id = _attributes[name]
        if not _c_call('bool', _lib.clingo_ast_has_attribute, self._rep, attr_id):
            raise AttributeError(f'no attribute: {name}')
        attr_type = _c_call('clingo_ast_attribute_type_t', _lib.clingo_ast_attribute_type, self._rep, attr_id)
        if attr_type == _lib.clingo_ast_attribute_type_empty:
            return None
        if attr_type == _lib.clingo_ast_attribute_type_string:
            return _to_str(_c_call('char*', _lib.clingo_ast_attribute_get_string, self._rep, attr_id))
        if attr_type == _lib.clingo_ast_attribute_type_number:
            return _to_str(_c_call('int', _lib.clingo_ast_attribute_get_number, self._rep, attr_id))
        if attr_type == _lib.clingo_ast_attribute_type_symbol:
            return Symbol(_c_call('clingo_symbol_t', _lib.clingo_ast_attribute_get_symbol, self._rep, attr_id))
        if attr_type == _lib.clingo_ast_attribute_type_location:
            # bool clingo_ast_attribute_get_location(clingo_ast_t *ast, clingo_ast_attribute_t attribute, clingo_location_t *value);
            raise RuntimeError("implement me")
        if attr_type == _lib.clingo_ast_attribute_type_ast:
            return AST(_c_call('clingo_ast_t*', _lib.clingo_ast_attribute_get_ast, self._rep, attr_id))
        if attr_type == _lib.clingo_ast_attribute_type_string_array:
            raise RuntimeError("implement me")
        assert attr_type == _lib.clingo_ast_attribute_type_ast_array
        raise RuntimeError("implement me")

    def items(self) -> List[Tuple[str,"AST"]]:
        '''
        items(self) -> List[Tuple[str,AST]]

        The list of items of the AST node.

        Returns
        -------
        List[Tuple[str,AST]]
        '''

    def keys(self) -> List[str]:
        '''
        keys(self) -> List[str]

        The list of keys of the AST node.

        Returns
        -------
        List[str]
        '''

    def values(self) -> List["AST"]:
        '''
        values(self) -> List[AST]

        The list of values of the AST node.

        Returns
        -------
        List[AST]
        '''

    child_keys: List[str]
    '''
    child_keys: List[str]

    List of names of all AST child nodes.

    '''
    type: "ASTType"
    '''
    type: ASTType

    The type of the node.

    '''

class AggregateFunction(Enum):
    '''
    Enumeration of aggegate functions.

    Attributes
    ----------
    Count : AggregateFunction
        The `#count` function.
    Sum : AggregateFunction
        The `#sum` function.
    SumPlus : AggregateFunction
        The `#sum+` function.
    Min : AggregateFunction
        The `#min` function.
    Max : AggregateFunction
        The `#max` function.
    '''
    # Count: AggregateFunction
    # Max: AggregateFunction
    # Min: AggregateFunction
    # Sum: AggregateFunction
    # SumPlus: AggregateFunction

class BinaryOperator(metaclass=ABCMeta):
    '''
    Enumeration of binary operators.

    `BinaryOperator` objects have a readable string representation, implement
    Python's rich comparison operators, and can be used as dictionary keys.

    Furthermore, they cannot be constructed from Python. Instead the
    following preconstructed class attributes are available:

    Attributes
    ----------
    XOr : BinaryOperator
        For bitwise exclusive or.
    Or : BinaryOperator
        For bitwise or.
    And : BinaryOperator
        For bitwise and.
    Plus : BinaryOperator
        For arithmetic addition.
    Minus : BinaryOperator
        For arithmetic subtraction.
    Multiplication : BinaryOperator
        For arithmetic multipilcation.
    Division : BinaryOperator
        For arithmetic division.
    Modulo : BinaryOperator
        For arithmetic modulo.
    Power : BinaryOperator
        For arithmetic exponentiation.
    '''
    # And: BinaryOperator
    # Division: BinaryOperator
    # Minus: BinaryOperator
    # Modulo: BinaryOperator
    # Multiplication: BinaryOperator
    # Or: BinaryOperator
    # Plus: BinaryOperator
    # Power: BinaryOperator
    # XOr: BinaryOperator

class ComparisonOperator(Enum):
    '''
    Enumeration of comparison operators.

    Attributes
    ----------
    GreaterThan : ComparisonOperator
        The `>` operator.
    LessThan : ComparisonOperator
        The `<` operator.
    LessEqual : ComparisonOperator
        The `<=` operator.
    GreaterEqual : ComparisonOperator
        The `>=` operator.
    NotEqual : ComparisonOperator
        The `!=` operator.
    Equal : ComparisonOperator
        The `=` operator
    '''
    # Equal: ComparisonOperator
    # GreaterEqual: ComparisonOperator
    # GreaterThan: ComparisonOperator
    # LessEqual: ComparisonOperator
    # LessThan: ComparisonOperator
    # NotEqual: ComparisonOperator

class ScriptType(metaclass=ABCMeta):
    '''
    Enumeration of theory atom types.

    `ScriptType` objects have a readable string representation, implement Python's
    rich comparison operators, and can be used as dictionary keys.

    Furthermore, they cannot be constructed from Python. Instead the following
    preconstructed class attributes are available:

    Attributes
    ----------
    Python : ScriptType
        For Python code.
    Lua : ScriptType
        For Lua code.
    '''
    # Lua: ScriptType
    # Python: ScriptType

class Sign(metaclass=ABCMeta):
    '''
    Enumeration of signs for literals.

    `Sign` objects have a readable string representation, implement Python's rich
    comparison operators, and can be used as dictionary keys.

    Furthermore, they cannot be constructed from Python. Instead the following
    preconstructed class attributes are available:

    Attributes
    ----------
    NoSign : Sign
        For positive literals.
    Negation : Sign
        For negative literals (with prefix `not`).
    DoubleNegation : Sign
        For double negated literals (with prefix `not not`)
    '''
    # DoubleNegation: Sign
    # Negation: Sign
    # NoSign: Sign

class TheoryAtomType(metaclass=ABCMeta):
    '''
    Enumeration of theory atom types.

    `TheoryAtomType` objects have a readable string representation, implement
    Python's rich comparison operators, and can be used as dictionary keys.

    Furthermore, they cannot be constructed from Python. Instead the following
    preconstructed class attributes are available:

    Attributes
    ----------
    Any : TheoryAtomType
        For atoms that can occur anywhere.
    Body : TheoryAtomType
        For atoms that can only occur in rule bodies.
    Head : TheoryAtomType
        For atoms that can only occur in rule heads.
    Directive : TheoryAtomType
        For atoms that can only occur in facts.
    '''
    # Any: TheoryAtomType
    # Body: TheoryAtomType
    # Directive: TheoryAtomType
    # Head: TheoryAtomType

class TheoryOperatorType(metaclass=ABCMeta):
    '''
    Enumeration of operator types.

    `TheoryOperatorType` objects have a readable string representation, implement
    Python's rich comparison operators, and can be used as dictionary keys.

    Furthermore, they cannot be constructed from Python. Instead the following
    preconstructed class attributes are available:

    Attributes
    ----------
    Unary : TheoryOperatorType
        For unary operators.
    BinaryLeft : TheoryOperatorType
        For binary left associative operators.
    BinaryRight : TheoryOperatorType
        For binary right associative operator.
    '''
    # BinaryLeft: TheoryOperatorType
    # BinaryRight: TheoryOperatorType
    # Unary: TheoryOperatorType

class TheorySequenceType(metaclass=ABCMeta):
    '''
    Enumeration of theory term sequence types.

    `TheorySequenceType` objects have a readable string representation, implement
    Python's rich comparison operators, and can be used as dictionary keys.

    Furthermore, they cannot be constructed from Python. Instead the following
    preconstructed class attributes are available:

    Attributes
    ----------
    Tuple : TheorySequenceType
        For sequences enclosed in parenthesis.
    List : TheorySequenceType
        For sequences enclosed in brackets.
    Set : TheorySequenceType
        For sequences enclosed in braces.
    '''
    # List: TheorySequenceType
    # Set: TheorySequenceType
    # Tuple: TheorySequenceType
    left_hand_side: str
    '''
    left_hand_side: str

    Left-hand side representation of the sequence.
    '''
    right_hand_side: str
    '''
    right_hand_side: str

    Right-hand side representation of the sequence.
    '''

class UnaryOperator(metaclass=ABCMeta):
    '''
    Enumeration of signs for literals.

    `UnaryOperator` objects have a readable string representation, implement
    Python's rich comparison operators, and can be used as dictionary keys.

    Furthermore, they cannot be constructed from Python. Instead the following
    preconstructed class attributes are available:

    Attributes
    ----------
    Negation : UnaryOperator
        For bitwise negation.
    Minus : UnaryOperator
        For unary minus and classical negation.
    Absolute : UnaryOperator
        For taking the absolute value.
    '''
    # Absolute: UnaryOperator
    # Minus: UnaryOperator
    # Negation: UnaryOperator
    left_hand_side: str
    '''
    left_hand_side: str

    Left-hand side representation of the operator.
    '''
    right_hand_side: str
    '''
    right_hand_side: str

    Right-hand side representation of the operator.
    '''

@_ffi.def_extern(onerror=_cb_error_handler('data'))
def _clingo_ast_callback(ast, data):
    '''
    Low-level ast callback.
    '''
    callback = _ffi.from_handle(data).data
    callback(AST(ast))

    return True

def parse_files(files: Iterable[str], callback: Callable[[AST], None],
                logger: Callable[[MessageCode,str],None]=None, message_limit: int=20) -> None:
    '''
    Parse the programs in the given files and return an abstract syntax tree for
    each statement via a callback.

    The function follows clingo's handling of files on the command line. Filename
    "-" is treated as stdin and if an empty list is given, then the parser will
    read from stdin.

    Parameters
    ----------
    files : Iterable[str]
        List of file names.
    callback : Callable[[AST],None]
        Callable taking an ast as argument.
    logger : Callable[[MessageCode,str],None]=None
        Function to intercept messages normally printed to standard error.
    message_limit : int=20
        The maximum number of messages passed to the logger.

    Returns
    -------
    None

    See Also
    --------
    ProgramBuilder
    '''
    # pylint: disable=protected-access
    if logger is not None:
        c_logger_data = _ffi.new_handle(logger)
        c_logger = _lib._clingo_logger_callback
    else:
        c_logger_data = _ffi.NULL
        c_logger = _ffi.NULL

    error = _Error()
    cb_data = _CBData(callback, error)
    c_cb_data = _ffi.new_handle(cb_data)

    _handle_error(_lib.clingo_ast_parse_files([ _ffi.new("char[]", f.encode()) for f in files ],
                                              _lib._clingo_ast_callback, c_cb_data,
                                              c_logger, c_logger_data,
                                              message_limit))

def parse_string(program: str, callback: Callable[[AST], None],
                 logger: Callable[[MessageCode,str],None]=None, message_limit: int=20) -> None:
    '''
    Parse the given program and return an abstract syntax tree for each statement
    via a callback.

    Parameters
    ----------
    program : str
        String representation of the program.
    callback : Callable[[AST],None]
        Callable taking an ast as argument.
    logger : Callable[[MessageCode,str],None]=None
        Function to intercept messages normally printed to standard error.
    message_limit : int=20
        The maximum number of messages passed to the logger.

    Returns
    -------
    None

    See Also
    --------
    ProgramBuilder
    '''
    # pylint: disable=protected-access
    if logger is not None:
        c_logger_data = _ffi.new_handle(logger)
        c_logger = _lib._clingo_logger_callback
    else:
        c_logger_data = _ffi.NULL
        c_logger = _ffi.NULL

    error = _Error()
    cb_data = _CBData(callback, error)
    c_cb_data = _ffi.new_handle(cb_data)

    _handle_error(_lib.clingo_ast_parse_string(program.encode(),
                                               _lib._clingo_ast_callback, c_cb_data,
                                               c_logger, c_logger_data,
                                               message_limit))

class ProgramBuilder(ContextManager['ProgramBuilder']):
    '''
    Object to build non-ground programs.

    Implements: `ContextManager[ProgramBuilder]`.

    See Also
    --------
    Control.builder, parse_program

    Notes
    -----
    A `ProgramBuilder` is a context manager and must be used with Python's `with`
    statement.

    Examples
    --------
    The following example parses a program from a string and passes the resulting
    `AST` to the builder:

        >>> import clingo
        >>> ctl = clingo.Control()
        >>> prg = "a."
        >>> with ctl.builder() as bld:
        ...    clingo.parse_program(prg, lambda stm: bld.add(stm))
        ...
        >>> ctl.ground([("base", [])])
        >>> ctl.solve(on_model=lambda m: print("Answer: {}".format(m)))
        Answer: a
        SAT
    '''
    def __init__(self, control: Control):
        pass

    def add(self, statement: AST) -> None:
        '''
        add(self, statement: ast.AST) -> None

        Adds a statement in form of an `ast.AST` node to the program.

        Parameters
        ----------
        statement : ast.AST
            The statement to add.

        Returns
        -------
        None
        '''
