'''
Tests for solving.

TODO:
- control
  - add_clause/add_nogood
- model
  - contains
  - extend
  - is_true
  - symbols
  - cost
  - number
  - optimality_proven
  - thread_id
'''
from unittest import TestCase
from typing import cast
from clingo import Control, ModelType, SolveHandle, SolveResult, parse_term

def _p(*models):
    return [[parse_term(symbol) for symbol in model] for model in models]

class _MCB:
    # pylint: disable=missing-function-docstring
    def __init__(self):
        self._models = []
        self._core = None
        self.last = None

    def on_core(self, c):
        self._core = c

    def on_model(self, m):
        self.last = (m.type, sorted(m.symbols(shown=True)))
        self._models.append(self.last[1])

    @property
    def core(self):
        return sorted(self._core)

    @property
    def models(self):
        return sorted(self._models)

class TestSymbol(TestCase):
    '''
    Tests basic solving and related functions.
    '''

    def setUp(self):
        self.mcb = _MCB()
        self.mit = _MCB()
        self.ctl = Control(['0'])

    def tearDown(self):
        self.mcb = None
        self.mit = None
        self.ctl = None

    def _check_sat(self, ret: SolveResult) -> None:
        self.assertTrue(ret.satisfiable is True)
        self.assertTrue(ret.unsatisfiable is False)
        self.assertTrue(ret.unknown is False)
        self.assertTrue(ret.exhausted is True)

    def test_solve_cb(self):
        '''
        Test solving using callback.
        '''
        self.ctl.add("base", [], "1 {a; b} 1. c.")
        self.ctl.ground([("base", [])])
        self._check_sat(cast(SolveResult, self.ctl.solve(on_model=self.mcb.on_model, yield_=False, async_=False)))
        self.assertEqual(self.mcb.models, _p(['a', 'c'], ['b', 'c']))
        self.assertEqual(self.mcb.last[0], ModelType.StableModel)

    def test_solve_async(self):
        '''
        Test asynchonous solving.
        '''
        self.ctl.add("base", [], "1 {a; b} 1. c.")
        self.ctl.ground([("base", [])])
        with cast(SolveHandle, self.ctl.solve(on_model=self.mcb.on_model, yield_=False, async_=True)) as hnd:
            self._check_sat(hnd.get())
            self.assertEqual(self.mcb.models, _p(['a', 'c'], ['b', 'c']))

    def test_solve_yield(self):
        '''
        Test solving yielding models.
        '''
        self.ctl.add("base", [], "1 {a; b} 1. c.")
        self.ctl.ground([("base", [])])
        with cast(SolveHandle, self.ctl.solve(on_model=self.mcb.on_model, yield_=True, async_=False)) as hnd:
            for m in hnd:
                self.mit.on_model(m)
            self._check_sat(hnd.get())
            self.assertEqual(self.mcb.models, _p(['a', 'c'], ['b', 'c']))
            self.assertEqual(self.mit.models, _p(['a', 'c'], ['b', 'c']))

    def test_solve_async_yield(self):
        '''
        Test solving yielding models asynchronously.
        '''
        self.ctl.add("base", [], "1 {a; b} 1. c.")
        self.ctl.ground([("base", [])])
        with self.ctl.solve(on_model=self.mcb.on_model, yield_=True, async_=True) as hnd:
            while True:
                hnd.resume()
                _ = hnd.wait()
                m = hnd.model()
                if m is None:
                    break
                self.mit.on_model(m)
            self._check_sat(hnd.get())
            self.assertEqual(self.mcb.models, _p(['a', 'c'], ['b', 'c']))
            self.assertEqual(self.mit.models, _p(['a', 'c'], ['b', 'c']))

    def test_solve_interrupt(self):
        '''
        Test interrupting solving.
        '''
        self.ctl.add("base", [], "1 { p(P,H): H=1..99 } 1 :- P=1..100.\n1 { p(P,H): P=1..100 } 1 :- H=1..99.")
        self.ctl.ground([("base", [])])
        with self.ctl.solve(async_=True) as hnd:
            hnd.resume()
            hnd.cancel()
            ret = hnd.get()
            self.assertTrue(ret.interrupted)

        with self.ctl.solve(async_=True) as hnd:
            hnd.resume()
            self.ctl.interrupt()
            ret = hnd.get()
            self.assertTrue(ret.interrupted)

    def test_solve_core(self):
        '''
        Test core retrieval.
        '''
        self.ctl.add("base", [], "3 { p(1..10) } 3.")
        self.ctl.ground([("base", [])])
        ass = []
        for atom in self.ctl.symbolic_atoms.by_signature("p", 1):
            ass.append(-atom.literal)
        ret = cast(SolveResult, self.ctl.solve(on_core=self.mcb.on_core, assumptions=ass))
        self.assertTrue(ret.unsatisfiable)
        self.assertTrue(len(self.mcb.core) > 7)

    def test_enum(self):
        '''
        Test core retrieval.
        '''
        self.ctl = Control(['0', '-e', 'cautious'])
        self.ctl.add("base", [], "1 {a; b} 1. c.")
        self.ctl.ground([("base", [])])
        self.ctl.solve(on_model=self.mcb.on_model)
        self.assertEqual(self.mcb.last[0], ModelType.CautiousConsequences)
        self.assertEqual([self.mcb.last[1]], _p(['c']))

        self.ctl = Control(['0', '-e', 'brave'])
        self.ctl.add("base", [], "1 {a; b} 1. c.")
        self.ctl.ground([("base", [])])
        self.ctl.solve(on_model=self.mcb.on_model)
        self.assertEqual(self.mcb.last[0], ModelType.BraveConsequences)
        self.assertEqual([self.mcb.last[1]], _p(['a', 'b', 'c']))
