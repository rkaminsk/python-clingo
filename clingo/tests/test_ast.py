'''
Tests for the ast module.

TODO:
- test comparison and hashing
- test locations
- test ast class
- test ast and str sequences
- test program builder
'''
from unittest import TestCase
from textwrap import dedent
from collections.abc import Sequence
from copy import copy, deepcopy

from .. import ast
from ..ast import AST, parse_string

class TestAST(TestCase):
    '''
    Tests for the ast module.
    '''
    def _deepcopy(self, node: AST):
        '''
        This functions tests manual deep copying of ast nodes using all the
        constructor functions in the ast module.
        '''
        cons_name = str(node.ast_type).split(".")[1]
        cons = getattr(ast, cons_name)
        args = dict(node.items())

        for key in node.child_keys:
            if isinstance(args[key], Sequence):
                args[key] = [ self._deepcopy(child) for child in args[key] ]
            elif isinstance(args[key], AST):
                args[key] = self._deepcopy(args[key])

        cpy = cons(**args)
        self.assertEqual(cpy, node)
        return cpy

    def _str(self, s, alt=None):
        # TODO: This function should also pass asts back to the parser.
        prg = []
        parse_string(s, prg.append)
        cpy = copy(deepcopy(self._deepcopy(prg[-1])))
        self.assertEqual(str(cpy), s if alt is None else alt)

        prg = []
        parse_string(str(cpy), prg.append)
        self.assertEqual(str(prg[-1]), s if alt is None else alt)

    def test_terms(self):
        '''
        Test terms.
        '''
        self._str("a.")
        self._str("-a.")
        self._str("a(X).")
        self._str("a(-X).")
        self._str("a(|X|).")
        self._str("a(~X).")
        self._str("a((X^Y)).")
        self._str("a((X?Y)).")
        self._str("a((X&Y)).")
        self._str("a((X+Y)).")
        self._str("a((X-Y)).")
        self._str("a((X*Y)).")
        self._str("a((X/Y)).")
        self._str("a((X\\Y)).")
        self._str("a((X**Y)).")
        self._str("a((X..Y)).")
        self._str("-a(f).")
        self._str("-a(-f).")
        self._str("-a(f(X)).")
        self._str("-a(f(X,Y)).")
        self._str("-a(()).")
        self._str("-a((a,)).")
        self._str("-a((a,b)).")
        self._str("-a(@f(a,b)).")
        self._str("-a(@f).")
        self._str("-a(a;b;c).")
        self._str("-a((a;b;c)).")
        self._str("-a(f(a);f(b);f(c)).")

    def test_theory_terms(self):
        '''
        Test theory terms.
        '''
        self._str("&a { 1 }.")
        self._str("&a { (- 1) }.")
        self._str("&a { X }.")
        self._str("&a { () }.")
        self._str("&a { (1,) }.")
        self._str("&a { (1,2) }.")
        self._str("&a { [] }.")
        self._str("&a { [1] }.")
        self._str("&a { [1,2] }.")
        self._str("&a { {} }.")
        self._str("&a { {1} }.")
        self._str("&a { {1,2} }.")
        self._str("&a { f }.")
        self._str("&a { f(X) }.")
        self._str("&a { f(X,Y) }.")
        self._str("&a { (+ a + - * b + c) }.")

    def test_literals(self):
        '''
        Test literals.
        '''
        self._str("a.")
        self._str("not a.")
        self._str("not not a.")
        self._str("1 < 2.")
        self._str("1 <= 2.")
        self._str("1 > 2.")
        self._str("1 >= 2.")
        self._str("1 = 2.")
        self._str("1 != 2.")
        self._str("#false.")
        self._str("#true.")

    def test_head_literals(self):
        '''
        Test head literals.
        '''
        self._str("{ }.")
        self._str("{ } < 2.", "2 > { }.")
        self._str("1 < { }.")
        self._str("1 < { } < 2.")
        self._str("{ b }.")
        self._str("{ a; b }.")
        self._str("{ a; b: c, d }.")
        self._str("#count { }.")
        self._str("#count { } < 2.", "2 > #count { }.")
        self._str("1 < #count { }.")
        self._str("1 < #count { } < 2.")
        self._str("#count { b: a }.")
        self._str("#count { b,c: a }.")
        self._str("#count { a: a; b: c }.")
        self._str("#count { a: d; b: x: c, d }.")
        self._str("#min { }.")
        self._str("#max { }.")
        self._str("#sum { }.")
        self._str("#sum+ { }.")
        self._str("a; b.")
        self._str("a; b: c.")
        self._str("a; b: c, d.")
        self._str("&a { }.")
        self._str("&a { 1 }.")
        self._str("&a { 1; 2 }.")
        self._str("&a { 1,2 }.")
        self._str("&a { 1,2: a }.")
        self._str("&a { 1,2: a, b }.")
        self._str("&a { } != x.")
        self._str("&a(x) { }.")

    def test_body_literals(self):
        '''
        Test body literals.
        '''
        self._str("a :- { }.")
        self._str("a :- not { }.")
        self._str("a :- not not { }.")
        self._str("a :- { } < 2.", "a :- 2 > { }.")
        self._str("a :- 1 < { }.")
        self._str("a :- 1 < { } < 2.")
        self._str("a :- { b }.")
        self._str("a :- { a; b }.")
        self._str("a :- { a; b: c, d }.")
        self._str("a :- #count { }.")
        self._str("a :- not #count { }.")
        self._str("a :- not not #count { }.")
        self._str("a :- #count { } < 2.", "a :- 2 > #count { }.")
        self._str("a :- 1 < #count { }.")
        self._str("a :- 1 < #count { } < 2.")
        self._str("a :- #count { b }.")
        self._str("a :- #count { b,c }.")
        self._str("a :- #count { a; b }.")
        self._str("a :- #count { a; b: c, d }.")
        self._str("a :- #min { }.")
        self._str("a :- #max { }.")
        self._str("a :- #sum { }.")
        self._str("a :- #sum+ { }.")
        self._str("a :- a; b.")
        self._str("a :- a; b: c.")
        self._str("a :- a; b: c, d.")
        self._str("a :- &a { }.")
        self._str("a :- &a { 1 }.")
        self._str("a :- &a { 1; 2 }.")
        self._str("a :- &a { 1,2 }.")
        self._str("a :- &a { 1,2: a }.")
        self._str("a :- &a { 1,2: a, b }.")
        self._str("a :- &a { } != x.")
        self._str("a :- &a(x) { }.")
        self._str("a :- a.")
        self._str("a :- not a.")
        self._str("a :- not not a.")
        self._str("a :- 1 < 2.")
        self._str("a :- 1 <= 2.")
        self._str("a :- 1 > 2.")
        self._str("a :- 1 >= 2.")
        self._str("a :- 1 = 2.")
        self._str("a :- 1 != 2.")
        self._str("a :- #false.")
        self._str("a :- #true.")

    def test_statements(self):
        '''
        Test statements.
        '''
        self._str("a.")
        self._str("#false.")
        self._str("#false :- a.")
        self._str("a :- a; b.")
        self._str("#const x = 10.")
        self._str("#const x = 10. [override]")
        self._str("#show p/1.")
        self._str("#show -p/1.")
        self._str("#defined p/1.")
        self._str("#defined -p/1.")
        self._str("#show x.")
        self._str("#show x : y; z.")
        self._str(":~ . [1@0]")
        self._str(":~ b; c. [1@2,s,t]")
        self._str("#script(lua) code #end.")
        self._str("#script(python) code #end.")
        self._str("#program x(y, z).")
        self._str("#program x.")
        self._str("#external a. [X]")
        self._str("#external a : b; c. [false]")
        self._str("#edge (1,2).")
        self._str("#edge (1,2) : x; y.")
        self._str("#heuristic a. [b@p,m]")
        self._str("#heuristic a : b; c. [b@p,m]")
        self._str("#project a.")
        self._str("#project a : b; c.")
        self._str("#project -a/0.")
        self._str("#project a/0.")
        self._str("#theory x {\n}.")
        self._str(dedent("""\
                         #theory x {
                           t {
                             + : 0, unary;
                             - : 1, binary, left;
                             * : 2, binary, right
                           };
                           &a/0: t, head;
                           &b/0: t, body;
                           &c/0: t, directive;
                           &d/0: t, { }, t, any;
                           &e/0: t, { =, !=, + }, t, any
                         }."""))
