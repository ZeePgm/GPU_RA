import unittest
from p0a_qualification import freeze, qualify, adjudicate


class QualificationTest(unittest.TestCase):
    def setUp(self):
        self.c = dict(target_ns=100, relative_margin=.1, alpha=.05,
                      endpoints=1, warmup_runs=4, pairs=4, order_seed=4102,
                      protocol='fresh_process', binary_sha256='fixture')

    def test_wide_interval_is_inconclusive(self):
        self.assertEqual(adjudicate(self.c, freeze(self.c), [-100,100,-100,100])['status'], 'inconclusive')

    def test_frozen_contract_cannot_change(self):
        digest = freeze(self.c)
        self.c['target_ns'] = 1000
        with self.assertRaises(ValueError):
            adjudicate(self.c, digest, [0]*4)

    def test_missing_evidence_blocks_qualification(self):
        self.assertEqual(qualify(self.c, [0]*4, [0]*4, {})['status'], 'inconclusive')

    def test_fixed_budget(self):
        with self.assertRaises(ValueError):
            adjudicate(self.c, freeze(self.c), [0]*3)

    def test_reuse_needs_reset_evidence(self):
        self.c['protocol']='reuse_context'
        evidence=dict.fromkeys(['frozen_before_calibration','independent_calibration','math_equal','access_identity_equal','frequency_recorded','warmup_fixed','crossover_order_fixed'],True)
        result=qualify(self.c,[0]*4,[0]*4,evidence)
        self.assertEqual(result['status'],'inconclusive')
        self.assertIn('kv_reset_verified',result['missing_evidence'])

    def test_pass_and_fail(self):
        digest=freeze(self.c)
        self.assertEqual(adjudicate(self.c,digest,[1]*4)['status'],'pass')
        self.assertEqual(adjudicate(self.c,digest,[11]*4)['status'],'fail')

    def test_no_relaxed_margin(self):
        self.c['relative_margin'] = .2
        with self.assertRaises(ValueError):
            freeze(self.c)


if __name__ == '__main__':
    unittest.main()
