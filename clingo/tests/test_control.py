'''
Tests control.
'''
from unittest import TestCase
from clingo import Control, Function, Number

class TestSymbol(TestCase):
    '''
    Tests basic solving and related functions.
    '''
    def cb(self, c):
        return [Number(c.number + 1), Number(c.number - 1)]

    def test_ground(self):
        '''
        Test grounding with context and parameters.
        '''
        ctl = Control()
        ctl.add('part', ['c'], 'p(@cb(c)).')
        ctl.ground([('part', [Number(1)])], self)
        symbols = [atom.symbol for atom in ctl.symbolic_atoms]
        self.assertEqual(sorted(symbols), [Function('p', [Number(0)]), Function('p', [Number(2)])])
