'''
Tests for the ast module.
'''
from unittest import TestCase
from typing import cast
from clingo.ast import AST, ASTType, parse_string

class TestAST(TestCase):
    '''
    Tests for the ast module.
    '''
    def cb(self, ast: AST):
        if ast.ast_type == ASTType.Program:
            print("program with name:", ast.name)
        if ast.ast_type == ASTType.Rule:
            print("rule head of type:", ast.head.ast_type)

    def test_ast(self):
        parse_string("a. a :- b.", self.cb)

