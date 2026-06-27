import unittest

from engine.modules.calibration_engine import CalibrationEngine, calibration_summary


class TestCalibrationThresholds(unittest.TestCase):
    def setUp(self):
        self.ce = CalibrationEngine()
        # Keep original imported function to restore later
        self._orig_summary = calibration_summary

    def tearDown(self):
        # restore
        try:
            from engine.modules import calibration_engine as ce_mod
            ce_mod.calibration_summary = self._orig_summary
        except Exception:
            pass

    def _mock_summary(self, ece_value):
        # Return a fake summary dict with given ece
        # Map ece -> severity using the platform standard
        if ece_value < 0.05:
            sev = 'NONE'
        elif ece_value <= 0.10:
            sev = 'MEDIUM'
        else:
            sev = 'HIGH'
        return {
            'label': 'model',
            'n': 100,
            'n_bins': 10,
            'ece': ece_value,
            'mce': 0.0,
            'brier': 0.0,
            'bss': 0.0,
            'severity': sev,
            'curve': [],
            'interpretation': ''
        }

    def test_ece_0_02_none(self):
        # ECE 0.02 -> NONE
        from engine.modules import calibration_engine as ce_mod
        ce_mod.calibration_summary = lambda y_true, y_prob, n_bins, model_name: self._mock_summary(0.02)
        res = self.ce.evaluate([], [])
        self.assertEqual(res.get('severity'), 'NONE')

    def test_ece_0_07_medium(self):
        from engine.modules import calibration_engine as ce_mod
        ce_mod.calibration_summary = lambda y_true, y_prob, n_bins, model_name: self._mock_summary(0.07)
        res = self.ce.evaluate([], [])
        self.assertEqual(res.get('severity'), 'MEDIUM')

    def test_ece_0_13_high(self):
        from engine.modules import calibration_engine as ce_mod
        ce_mod.calibration_summary = lambda y_true, y_prob, n_bins, model_name: self._mock_summary(0.13)
        res = self.ce.evaluate([], [])
        self.assertEqual(res.get('severity'), 'HIGH')

    def test_ece_0_25_high(self):
        from engine.modules import calibration_engine as ce_mod
        ce_mod.calibration_summary = lambda y_true, y_prob, n_bins, model_name: self._mock_summary(0.25)
        res = self.ce.evaluate([], [])
        self.assertEqual(res.get('severity'), 'HIGH')

    def test_ece_rounding_and_severity_use_same_raw_float(self):
        from engine.modules import calibration_engine as ce_mod
        ce_mod.calibration_summary = lambda y_true, y_prob, n_bins, model_name: {
            'label': 'model',
            'n': 100,
            'n_bins': 10,
            'ece': 0.1325,
            'mce': 0.0,
            'brier': 0.0,
            'bss': 0.0,
            'severity': 'EXCELLENT',
            'curve': [],
            'interpretation': ''
        }
        res = self.ce.evaluate([], [])
        self.assertEqual(res.get('ece'), 0.1325)
        self.assertEqual(res.get('ece_score'), 0.1325)
        self.assertEqual(res.get('severity'), 'HIGH')


if __name__ == '__main__':
    unittest.main()
