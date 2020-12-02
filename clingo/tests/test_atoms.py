'''
Tests for theory and symbolic atoms.
'''

from unittest import TestCase
from clingo import Control, Number, Function

class TestAtoms(TestCase):
    '''
    Tests for theory and symbolic atoms.
    '''

    def setUp(self):
        self.ctl = Control()

    def test_symbolic_atom(self):
        '''
        Test symbolic atom.
        '''
        self.ctl.add('base', [], 'p(1). {p(2)}. #external p(3).')
        self.ctl.ground([('base', [])])

        atoms = self.ctl.symbolic_atoms

        p1 = atoms[Function('p', [Number(1)])]
        self.assertIsNotNone(p1)
        self.assertTrue(p1.is_fact)
        self.assertFalse(p1.is_external)
        self.assertTrue(p1.literal >= 1)
        self.assertEqual(p1.symbol, Function('p', [Number(1)]))
        self.assertTrue(p1.match('p', 1, True))
        self.assertFalse(p1.match('p', 2, True))

        p2 = atoms[Function('p', [Number(2)])]
        self.assertIsNotNone(p2)
        self.assertFalse(p2.is_fact)
        self.assertFalse(p2.is_external)
        self.assertTrue(p2.literal >= 2)
        self.assertEqual(p2.symbol, Function('p', [Number(2)]))
        self.assertTrue(p2.match('p', 1, True))
        self.assertFalse(p2.match('p', 2, True))

        p3 = atoms[Function('p', [Number(3)])]
        self.assertIsNotNone(p3)
        self.assertFalse(p3.is_fact)
        self.assertTrue(p3.is_external)
        self.assertTrue(p3.literal >= 2)
        self.assertEqual(p3.symbol, Function('p', [Number(3)]))
        self.assertTrue(p3.match('p', 1, True))
        self.assertFalse(p3.match('p', 2, True))

        p4 = atoms[Function('p', [Number(4)])]
        self.assertIsNone(p4)

    def test_symbolic_atoms(self):
        '''
        Test symbolic atoms.
        '''
        self.ctl.add('base', [], 'p(1). {p(2)}. #external p(3). q(1). -p(1). {q(2)}. #external q(3).')
        self.ctl.ground([('base', [])])

        atoms = self.ctl.symbolic_atoms
        self.assertEqual(sorted(atoms.signatures), [('p', 1, False), ('p', 1, True), ('q', 1, True)])

        ps = list(atoms.by_signature('p', 1, True))
        self.assertEqual(len(ps), 3)
        for p in ps:
            self.assertEqual(p.symbol.name, 'p')
            self.assertTrue(p.symbol.positive)
            self.assertEqual(len(p.symbol.arguments), 1)

        nps = list(atoms.by_signature('p', 1, False))
        self.assertEqual(len(nps), 1)
        for p in nps:
            self.assertEqual(p.symbol.name, 'p')
            self.assertTrue(p.symbol.negative)
            self.assertEqual(p.symbol.arguments, [Number(1)])

        self.assertEqual(len(atoms), 7)
        self.assertEqual(len(list(atoms)), 7)

        self.assertIn(Function('p', [Number(1)], True), atoms)
        self.assertIn(Function('p', [Number(1)], False), atoms)
        self.assertIn(Function('q', [Number(2)], True), atoms)
        self.assertNotIn(Function('q', [Number(2)], False), atoms)

