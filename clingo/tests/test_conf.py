'''
Tests for configuration and statistics.
'''

from unittest import TestCase
from clingo import Configuration, Control

class TestConfig(TestCase):
    '''
    Tests for configuration and statistics.
    '''

    def setUp(self):
        self.ctl = Control()

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

    def test_stats(self):
        '''
        Test statistics.
        '''
