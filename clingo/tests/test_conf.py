'''
Tests for configuration and statistics.
'''

from unittest import TestCase
from clingo import Configuration, Control

class TestConfig(TestCase):
    '''
    Tests for configuration and statistics.
    '''
    def test_config(self):
        '''
        Test configuration.
        '''
        ctl = Control(['-t', '2'])
        self.assertIn('solver', ctl.configuration.keys)
        self.assertEqual(len(ctl.configuration.solver), 2)
        self.assertIsInstance(ctl.configuration.solver[0], Configuration)
        self.assertIsInstance(ctl.configuration.solver[0].heuristic, str)
        self.assertIsInstance(ctl.configuration.solver[0].description('heuristic'), str)
        ctl.configuration.solver[0].heuristic = 'berkmin'
        self.assertTrue(ctl.configuration.solver[0].heuristic.startswith('berkmin'))

    def test_simple_stats(self):
        '''
        Test simple statistics.
        '''
        ctl = Control(['-t', '2', '--stats=2'])
        ctl.add('base', [], '1 { a; b }.')
        ctl.ground([('base', [])])
        ctl.solve()
        stats = ctl.statistics
        self.assertGreaterEqual(stats['problem']['lp']['atoms'], 2)
        self.assertGreaterEqual(stats['solving']['solvers']['choices'], 1)
