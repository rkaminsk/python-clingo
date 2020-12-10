'''
Tests for the ast module.
'''
from unittest import TestCase
from typing import cast
from clingo.ast import AST, ASTType, Id, parse_string

class TestAST(TestCase):
    '''
    Tests for the ast module.
    '''
    def cb(self, ast: AST):
        if ast.ast_type == ASTType.Program:
            print("program with name:", ast.name)
            ast.parameters.append(Id(ast.location, "blub"))
            print("program with parameters:", list(x.name for x in ast.parameters))
            print(ast.keys())
        if ast.ast_type == ASTType.Rule:
            print("rule head of type:", ast.head.ast_type)
            print("program with name:", ast.body)
            print(ast.keys())
            print(ast.child_keys)
            print(ast.location)

    def test_ast(self):
        parse_string("#program x(y, z). a. a :- b.", self.cb)

