import numpy as np
import pytest

from osc_tools.pdr.temporal_metrics import TemporalConfig, temporal_metrics


def test_conditional_errors_are_not_hidden_by_majority():
    truth = np.r_[np.ones(990, dtype=int), -np.ones(10, dtype=int)]
    m = temporal_metrics(truth, np.ones(1000, dtype=int), 1000)
    e = m["regions"]["all"]
    assert e["all"]["fraction"] == .01
    assert e["missed_invalid"]["fraction"] == 1
    assert e["false_forward"]["fraction"] is None


def test_wrong_state_changes_do_not_break_full_error():
    m = temporal_metrics(np.ones(30, dtype=int), np.r_[np.zeros(10), -np.ones(10), np.zeros(10)].astype(int), 1000)
    assert m["regions"]["all"]["all"]["max_episode_ms"] == 30
    assert m["regions"]["all"]["false_reverse"]["max_episode_ms"] == 10


def test_transition_masks_partition_without_joining_intervals():
    truth = np.r_[np.zeros(20), np.ones(20)].astype(int)
    m = temporal_metrics(truth, 1-truth, 1000)
    r = m["regions"]
    assert r["transition"]["all"]["support_ms"] == 6  # [t,t+5 мс], обе границы.
    assert r["transition"]["all"]["error_ms"] + r["outside_transition"]["all"]["error_ms"] == 40
    assert r["outside_transition"]["all"]["max_episode_ms"] == 20


def _match(m, hold=5, a=0, b=1):
    return next(x for x in m["event_matching"] if x["hold_ms"] == hold and
                x["window_ms"] == 100 and x["from"] == a and x["to"] == b)


def test_delay_holding_and_short_pulse():
    truth = np.r_[np.zeros(100), np.ones(200)].astype(int)
    pred = np.r_[np.zeros(102), np.ones(1), np.zeros(7), np.ones(190)].astype(int)
    m = temporal_metrics(truth, pred, 1000)
    event = _match(m)["events"][0]
    assert event["delay_ms"] == 10
    assert event["confirmation_ms"] == 15
    assert _match(m)["unmatched_predicted_events"] == 1
    assert _match(m)["by_deadline"]["5.0"] == {"eligible": 1, "responded": 0}
    assert _match(m)["by_deadline"]["20.0"] == {"eligible": 1, "responded": 1}


def test_miss_censor_and_early_are_distinct():
    truth = np.r_[np.zeros(100), np.ones(200)].astype(int)
    assert _match(temporal_metrics(truth, np.zeros(300, dtype=int), 1000))["events"][0]["status"] == "missed"
    assert _match(temporal_metrics(truth[:150], np.zeros(150, dtype=int), 1000))["events"][0]["status"] == "censored"
    pred = np.r_[np.zeros(95), np.ones(205)].astype(int)
    event = _match(temporal_metrics(truth, pred, 1000))["events"][0]
    assert event["delay_ms"] == -5
    assert event["confirmation_ms"] == 5


def test_startup_is_not_validity_event_and_no_fake_zero_denominator():
    truth = np.ones(200, dtype=int)
    pred = np.r_[-np.ones(20), np.ones(180)].astype(int)
    m = temporal_metrics(truth, pred, 1000, startup_samples=20)
    assert m["event_matching"] == []
    assert m["regions"]["startup"]["all"]["error_ms"] == 20
    assert m["stability"]["stable_switches"] == 0


def test_sliding_error_resists_fragmentation():
    truth = np.ones(200, dtype=int)
    pred = np.zeros(200, dtype=int); pred[::10] = 1
    m = temporal_metrics(truth, pred, 1000)
    assert m["regions"]["all"]["all"]["max_episode_ms"] == 9
    assert m["max_error_in_sliding_window_ms"] == 90
    assert temporal_metrics(truth[:20], pred[:20], 1000)["max_error_in_sliding_window_ms"] is None


def test_duration_invariant_to_sampling_rate():
    truth = np.r_[np.zeros(100), np.ones(200)].astype(int)
    pred = np.r_[np.zeros(110), np.ones(190)].astype(int)
    a = temporal_metrics(truth, pred, 1000)
    b = temporal_metrics(np.repeat(truth, 2), np.repeat(pred, 2), 2000)
    assert a["regions"]["all"] == b["regions"]["all"]
    assert _match(a)["events"] == _match(b)["events"]


def test_invalid_inputs_fail_not_masked():
    with pytest.raises(ValueError):
        temporal_metrics(np.array([1]), np.array([-998]), 1000)
    with pytest.raises(ValueError):
        temporal_metrics(np.array([1]), np.array([1]), 0)


def test_summary_clusters_rtds_sections():
    from scripts.phase5_experiments.review_pdr_analysis_results import _summarize_engineering
    rows = [{"source": "rtds", "algorithm": "test", "cluster": "same_experiment", "section": s,
             "metrics": temporal_metrics(np.ones(200,dtype=int), np.full(200,v,dtype=int), 1000)} for s,v in ((1,0),(2,1))]
    result = _summarize_engineering(rows, 0)["by_source"]["rtds"]["test"]["errors"]["all"]
    assert result["supported_groups"] == 1
    assert result["group_mean"] == .5
    assert result["group_mean_ci95"] is None


def test_next_reference_event_is_not_record_end_censoring():
    truth = np.r_[np.zeros(100), np.ones(30), np.zeros(200)].astype(int)
    result = temporal_metrics(truth, np.zeros_like(truth), 1000)
    event = _match(result)["events"][0]
    assert event["status"] == "next_reference_event"
    assert _match(result)["by_deadline"]["20.0"] == {"eligible": 1, "responded": 0}
