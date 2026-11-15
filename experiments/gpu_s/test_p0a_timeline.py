import unittest
from p0a_timeline import union_ns
class TimelineTests(unittest.TestCase):
 def test_overlaps_are_not_added(self):
  self.assertEqual(union_ns([(0,10),(2,8),(8,20),(30,40)]),30)
 def test_bad_interval_rejected(self):
  with self.assertRaises(ValueError): union_ns([(2,1)])
if __name__=='__main__': unittest.main()
