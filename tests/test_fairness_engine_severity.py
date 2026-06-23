import unittest

from engine.modules.fairness_engine import FairnessEngine


class TestFairnessEngineSeverityMapping(unittest.TestCase):

    def _build_group_data(self, di_ratio_target, n_per_group=100, priv_rate=0.5):
        """
        Build y_true, y_pred, groups such that disparate impact approx equals di_ratio_target.
        We fix privileged group's selection rate to priv_rate and compute unpriv positives.
        """
        priv_ones = int(round(priv_rate * n_per_group))
        unpriv_ones = int(round(di_ratio_target * priv_rate * n_per_group))

        y_pred = [1] * priv_ones + [0] * (n_per_group - priv_ones) + \
                 [1] * unpriv_ones + [0] * (n_per_group - unpriv_ones)
        # Provide non-zero y_true so per_group base_rate != 0 (avoid all-zero guard)
        # Alternate 1/0 to ensure some positives are present in each group.
        y_true = [1 if i % 2 == 0 else 0 for i in range(2 * n_per_group)]
        groups = ["A"] * n_per_group + ["B"] * n_per_group
        return y_true, y_pred, groups

    def test_di_to_severity(self):
        cases = [
            (1.00, "NONE"),
            (0.95, "NONE"),
            (0.85, "LOW"),
            (0.70, "MEDIUM"),
            (0.50, "HIGH"),
            (0.20, "CRITICAL"),
        ]
        fe = FairnessEngine()
        for di, expected in cases:
            with self.subTest(di=di, expected=expected):
                y_true, y_pred, groups = self._build_group_data(di_ratio_target=di)
                fe.clear_axes()
                fe.register_axis("demo", groups, privileged="A", unprivileged="B")
                report = fe.run(y_true, y_pred)
                per_axis = report.get("per_axis", {})
                axis_report = per_axis.get("demo", {})
                severity = axis_report.get("severity")
                self.assertEqual(severity, expected, msg=f"DI={di} produced {severity}, expected {expected}")


if __name__ == "__main__":
    unittest.main()
