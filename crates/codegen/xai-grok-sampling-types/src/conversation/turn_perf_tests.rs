// mlx-node fork: per-turn perf totals over every model call.
use super::*;

fn call(ttft: f64, prefill_tps: f64, decode_tps: f64, prefill: u32, cached: u32) -> TurnPerf {
    TurnPerf {
        ttft_ms: Some(ttft),
        prefill_tokens_per_second: Some(prefill_tps),
        decode_tokens_per_second: Some(decode_tps),
        prefill_input_tokens: Some(prefill),
        cached_prefix_tokens: Some(cached),
        prefill_ms: None,
    }
}

fn close(a: Option<f64>, b: f64) {
    let a = a.expect("value present");
    assert!((a - b).abs() < 1e-9, "{a} != {b}");
}

#[test]
fn two_calls_are_time_and_token_weighted() {
    let mut totals = TurnPerfTotals::default();
    // 2000 ms at 1000 tok/s = 2000 work tokens; 100 out at 50 tok/s = 2 s.
    totals.add_call(Some(&call(2000.0, 1000.0, 50.0, 2000, 100)), 100);
    // 500 ms at 200 tok/s = 100 work tokens; 300 out at 100 tok/s = 3 s.
    totals.add_call(Some(&call(500.0, 200.0, 100.0, 100, 4000)), 300);
    let perf = totals.finish().expect("perf");
    close(perf.ttft_ms, 2000.0);
    close(perf.prefill_ms, 2500.0);
    // 2100 work tokens / 2.5 s.
    close(perf.prefill_tokens_per_second, 840.0);
    close(perf.prefill_work_tokens(), 2100.0);
    // 400 out / 5 s.
    close(perf.decode_tokens_per_second, 80.0);
    assert_eq!(perf.prefill_input_tokens, Some(2100));
    assert_eq!(perf.cached_prefix_tokens, Some(4100));
}

#[test]
fn first_call_ttft_wins_even_when_unreported() {
    let mut totals = TurnPerfTotals::default();
    totals.add_call(
        Some(&TurnPerf {
            decode_tokens_per_second: Some(40.0),
            ..Default::default()
        }),
        10,
    );
    totals.add_call(Some(&call(300.0, 1000.0, 40.0, 300, 0)), 10);
    let perf = totals.finish().expect("perf");
    assert_eq!(perf.ttft_ms, None);
    close(perf.prefill_ms, 300.0);
    close(perf.prefill_tokens_per_second, 1000.0);
}

#[test]
fn calls_missing_fields_are_skipped_per_field() {
    let mut totals = TurnPerfTotals::default();
    totals.add_call(Some(&call(1000.0, 600.0, 50.0, 600, 0)), 50);
    // ttft without a prefill rate, no counts, no decode rate.
    totals.add_call(
        Some(&TurnPerf {
            ttft_ms: Some(9000.0),
            ..Default::default()
        }),
        500,
    );
    // Unreported perf entirely.
    totals.add_call(None, 70);
    // Decode rate with zero output tokens.
    totals.add_call(
        Some(&TurnPerf {
            decode_tokens_per_second: Some(1.0),
            ..Default::default()
        }),
        0,
    );
    let perf = totals.finish().expect("perf");
    close(perf.ttft_ms, 1000.0);
    close(perf.prefill_ms, 1000.0);
    close(perf.prefill_tokens_per_second, 600.0);
    close(perf.decode_tokens_per_second, 50.0);
    assert_eq!(perf.prefill_input_tokens, Some(600));
    assert_eq!(perf.cached_prefix_tokens, Some(0));
}

#[test]
fn no_reported_counters_stays_none() {
    let mut totals = TurnPerfTotals::default();
    assert_eq!(totals.finish(), None);
    totals.add_call(None, 12);
    totals.add_call(Some(&TurnPerf::default()), 3);
    assert_eq!(totals.finish(), None);
}

#[test]
fn merge_matches_one_accumulator_across_rounds() {
    let a = call(2000.0, 1000.0, 50.0, 2000, 100);
    let b = call(500.0, 200.0, 100.0, 100, 4000);
    let c = call(800.0, 700.0, 60.0, 560, 4100);
    let mut whole = TurnPerfTotals::default();
    whole.add_call(Some(&a), 100);
    whole.add_call(Some(&b), 300);
    whole.add_call(Some(&c), 30);

    let mut round1 = TurnPerfTotals::default();
    round1.add_call(Some(&a), 100);
    round1.add_call(Some(&b), 300);
    let mut round2 = TurnPerfTotals::default();
    round2.add_call(Some(&c), 30);
    let mut turn = TurnPerfTotals::default();
    turn.merge(&round1);
    turn.merge(&round2);
    assert_eq!(turn.finish(), whole.finish());
    close(turn.finish().and_then(|p| p.ttft_ms), 2000.0);
}

#[test]
fn old_turn_perf_without_prefill_ms_round_trips() {
    let old = r#"{"ttft_ms":450.2,"prefill_tokens_per_second":412.5,"decode_tokens_per_second":38.1,"prefill_input_tokens":200,"cached_prefix_tokens":2100}"#;
    let perf: TurnPerf = serde_json::from_str(old).expect("old perf parses");
    assert_eq!(perf.prefill_ms, None);
    assert_eq!(perf.ttft_ms, Some(450.2));
    // Old files fall back to ttft for the prefill time.
    close(perf.prefill_time_ms(), 450.2);
    close(perf.prefill_work_tokens(), 412.5 * 0.4502);
    let json = serde_json::to_string(&perf).expect("serializes");
    assert!(!json.contains("prefill_ms"), "{json}");
    assert_eq!(serde_json::from_str::<TurnPerf>(&json).unwrap(), perf);

    let new = TurnPerf {
        prefill_ms: Some(2500.0),
        ..perf
    };
    let json = serde_json::to_string(&new).expect("serializes");
    assert!(json.contains(r#""prefill_ms":2500.0"#), "{json}");
    assert_eq!(serde_json::from_str::<TurnPerf>(&json).unwrap(), new);
    assert_eq!(
        serde_json::from_str::<TurnPerf>("{}").unwrap(),
        TurnPerf::default()
    );
}

#[test]
fn small_prefill_uses_prefill_ms_then_ttft() {
    let at = |prefill_ms: Option<f64>, ttft: Option<f64>, tps: f64| TurnPerf {
        ttft_ms: ttft,
        prefill_ms,
        prefill_tokens_per_second: Some(tps),
        ..Default::default()
    };
    // 1000 tok/s × 0.5 s = 500 work tokens.
    assert_eq!(
        at(Some(500.0), Some(9000.0), 1000.0).small_prefill(),
        Some((500.0, 500.0))
    );
    // 1000 tok/s × 0.512 s = 512 work tokens: a throughput.
    assert_eq!(at(Some(512.0), None, 1000.0).small_prefill(), None);
    assert_eq!(
        at(None, Some(100.0), 1000.0).small_prefill(),
        Some((100.0, 100.0))
    );
    assert_eq!(at(None, None, 1000.0).small_prefill(), None);
}
