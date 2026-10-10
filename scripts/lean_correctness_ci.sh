#!/usr/bin/env bash
# Required CPU lean-correctness lane: build the pinned verifier, audit its acceptance theorems,
# check its identity against lean-plow/approved-verifiers.json, then run the explicit list of
# ignored CPU-verifier tests. Every listed suite must execute at least one test: a renamed or
# deleted test fails here instead of silently leaving the lane.
#
#   nix develop --command scripts/lean_correctness_ci.sh
#
# GPU and artifact tests are not run here (docs/bringup/lean-correctness-inventory.md §6).
set -euo pipefail
cd "$(dirname "$0")/.."

(cd lean-plow && lake build plow_verify proof_audit && lake exe proof_audit)
export PLOW_VERIFY_BIN="$PWD/lean-plow/.lake/build/bin/plow_verify"
echo "verifier: $(sha256sum "$PLOW_VERIFY_BIN" | cut -d' ' -f1)"

# One suite per line: <cargo args> -- <exact test names>. `--include-ignored` also runs the
# suite's ordinary tests; `--exact` keeps the selection explicit.
suites=(
  "-p lean_verify --test approved_verifier -- built_verifier_is_approved_for_current_sources"
  "-p lean_verify --test end_to_end -- checkpoint_d_accepts_safe_schedule batch_preserves_individual_verdicts_and_envelopes checkpoint_d_rejects_missing_counter checkpoint_d_accepts_disjoint_bytes checkpoint_d_rejects_unordered_dependency_without_address_entries checkpoint_d_rejects_malformed_access_sets checkpoint_d_checks_removed_edge_witnesses checkpoint_f_reuses_d_verifier checkpoint_d_checks_address_paths_without_graph_saturation all_checkpoints_are_wired"
  "-p lean_verify --test media_geometry -- production_media_contracts_hold_and_each_mutation_rejects"
  "-p lean_verify --test kv_ring -- production_ring_launches_hold_and_each_mutation_rejects"
  "-p lean_verify --test measured_policy -- measured_policy_proves_only_supplied_domain_coverage_and_minimum"
  "-p lean_verify --test logical_effects -- packet_derived_effects_reject_missing_raw_war_waw speech_effects_order_noise_war_bands_and_splitk_scratch"
  "-p lean_verify --test lower_bound_end_to_end -- rejects_malformed_and_cyclic_graphs validates_rates_and_preserves_certificate_envelope"
  "-p lean_verify --test mla_layout -- padded_wv_address_domain_preserves_bf16_boundary_without_expanding_coverage"
  "-p lean_verify --test knobs_end_to_end -- production_recipe_and_registry_are_accepted negative_fixtures_are_rejected stale_registry_defaults_are_rejected contradictory_constraints_are_rejected generated_evaluator_agrees_with_lean_on_random_configs"
  "-p lean_verify --test scope_end_to_end -- identical_packets_pass_without_a_delta any_difference_without_a_delta_is_rejected a_narrowed_collective_is_outside_the_small_cus_scope a_removed_instruction_is_one_difference a_program_no_allowance_selects_must_be_unchanged a_removed_program_is_a_program_set_difference object_facts_are_filtered_by_name a_route_move_needs_a_route_allowance_over_both_programs skipped_segments_are_a_route_difference empty_effect_and_scope_slack_are_warned"
  "-p lean_verify --test memory_effects -- generations_reuse_only_after_retirement_including_queued_work_after_cancel issue_order_partial_thresholds_and_missing_fences_cannot_certify_effects raw_war_waw_require_completion_order_but_read_read_does_not"
  "-p lean_verify --test perf_end_to_end -- improvement_beyond_the_floor_is_accepted change_inside_the_floor_is_rejected neutral_rung_needs_evidence_and_no_regression floor_that_cannot_be_computed_is_insufficient numeric_scope_needs_passing_facts tier4_and_untouched_digests history_replay empty_and_failed_gate_never_qualify repeated_treatment_and_both_spreads_are_required"
  "-p plowc --bin plowc -- cli_tests::actual_devblob_hook_derives_effects_and_rejects_missing_order cli_tests::actual_devblob_hook_checks_builder_reduction_witness cli_tests::actual_devblob_hook_checks_strided_layout_domain"
  "-p plowc --test lean_verify_rewrite -- accepts_known_sound_rules accepts_empty_rules_list rejects_unknown_rule"
  "-p plowc --test lean_verify -- every_example_bucket_is_accepted_by_lean"
  "-p plowc --test prefetch -- every_example_compiles_with_prefetch_and_verifier"
  "-p plowc --test counter_elim -- every_example_compiles_with_counter_elim_and_verifier"
  "-p plowc --test scope_narrow -- every_example_compiles_with_scope_narrow_and_verifier"
  "-p plowc --test lean_verify_tile_partition -- accepts_divisor_partition_within_bound accepts_empty_list rejects_zero_bm accepts_tile_larger_than_gemm_dim rejects_cost_bound_exceeded accepts_non_divisor_partition_with_slack_bound"
  "-p plowc --test lean_verify_growable_kv -- growable_class_survives_payload_round_trip growable_entries_pass_strict_verifier overlapping_growable_writers_fail_strict_but_pass_loose"
  "-p plowc --test lean_verify_schema -- rejects_edge_out_of_range rejects_wrong_length_resource_array rejects_wrong_length_waits_array rejects_unknown_cls_value rejects_out_of_range_reader_index rejects_malformed_edge_pair accepts_a_minimal_well_formed_payload"
  "-p plowc --test lean_verify_sram_fit -- accepts_temporally_disjoint_budget_ok rejects_overlapping_windows rejects_producer_over_budget rejects_consumer_over_budget accepts_empty_handoff_list"
  "-p plowc --test gemm_policy -- actual_gemm_policy_population_checks_exact_shape_and_rejects_wrong_choice"
  "-p plowc --test moe_policy -- real_moe_policy_producer_checks_handoff_adjusted_selection"
  "-p plowc --test rewrite_body -- actual_engine_rules_bind_bodies_and_retain_precision_relevant_attributes"
  "-p plowc --test lean_verify_lds_fit -- accepts_fitting_staged_ops rejects_the_task9_shape"
  "-p plowc --test lean_verify_negative -- verifier_rejects_unordered_byte_overlap stripping_waits_alone_is_not_a_corruption"
  "-p plowc --test lean_verify_wire -- accepts_valid_round_trip accepts_empty_program rejects_encode_frames_diverges_from_raw rejects_length_mismatch rejects_truncated_stream json_u8_boundary_round_trips_through_lean abstract_framing_shape_matches_packet_body_ordering"
  "-p plowrt --features cuda,hsa --lib -- certificate_checks::tests::selected_gemm_receipt_reconstructs_wire_after_packet_hash_changes certificate_checks::tests::logical_effect_receipt_is_reconstructed_from_loaded_packet certificate_checks::tests::packet_receipts_replay_at_load_and_reject_tampering certificate_checks::tests::layout_receipt_is_reconstructed_from_loaded_packet certificate_checks::tests::strict_policy_replays_complete_receipts_and_rejects_substituted_evidence memory::vmm::ring_tests::vmm_ring_driver_traces_satisfy_the_lean_lifecycle_model"
)

fail=0
for suite in "${suites[@]}"; do
  cargo_args=${suite%% -- *}
  tests=${suite#* -- }
  expected=$(wc -w <<<"$tests")
  # shellcheck disable=SC2086
  out=$(cargo test $cargo_args -- --ignored --exact $tests 2>&1) || { echo "$out"; echo "FAIL: $cargo_args"; fail=1; continue; }
  ran=$(grep -E '^test result:' <<<"$out" | sed -E 's/.* ([0-9]+) passed.*/\1/' | awk '{s+=$1} END {print s+0}')
  if [ "$ran" -ne "$expected" ]; then
    echo "$out"
    echo "FAIL: $cargo_args ran $ran of $expected listed tests"
    fail=1
  else
    echo "ok: $cargo_args ($ran tests)"
  fi
done
exit $fail
