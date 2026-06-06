# Feature Leakage Modules
from .target_leakage import (
    mutual_information_score,
    permutation_importance_spike,
    availability_check,
    target_correlation_scan,
    leakage_score
)
from .temporal_leakage import (
    check_split_ordering,
    detect_future_features,
    rolling_window_audit,
    target_encoding_leakage,
    temporal_leakage_report
)
from .leakage_scanner import (
    scan,
    rank_leakage_suspects,
    leakage_summary
)
