'''
Tests for the ast module.
'''
from unittest import TestCase
from typing import cast
from clingo.symbol import Function
from clingo.ast import AST, ASTType, Id, parse_string, Literal, SymbolicAtom, Symbol as Symbol_

class TestAST(TestCase):
    '''
    Tests for the ast module.
    '''
    def cb(self, ast: AST):
        print("before:", ast)
        if ast.ast_type == ASTType.Program:
            #print("program with name:", ast.name)
            ast.parameters.append(Id(ast.location, "blub"))
            #print("program with parameters:", list(x.name for x in ast.parameters))
            #print(ast.keys())
        if ast.ast_type == ASTType.Rule:
            sym = Symbol_(ast.location, Function("x"))
            ast.body.append(Literal(ast.location, 0, SymbolicAtom(sym)))
            #print("rule head of type:", ast.head.ast_type)
            #print("program with name:", ast.body)
            #print(ast.keys())
            #print(ast.child_keys)
            #print(ast.location)
        print("after:", ast)

    def test_ast(self):
        parse_string("#program x(y, z). a. a(X) :- b.", self.cb)

