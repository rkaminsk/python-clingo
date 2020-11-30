'''
Test clingo's Symbol class.

TODO: complete!
'''
from unittest import TestCase

from clingo import Number

class TestSymbol(TestCase):
    '''
    Tests for the program observer.
    '''

    def test_str(self):
        '''
        Test string representation of symbols.
        '''
        self.assertEqual(str(Number(10)), "10")
