"""
Comprehensive Test Suite for Rate Parser 2.0 (Approach 1: Unified Modern Billing Engine Model).
Tests LexicalSurfaceCleaner, SemanticClassifier, InvariantVerification, and Downstream Emitters.
"""

import unittest
from decimal import Decimal
from rate_parser import (
    FeeRateParser,
    LexicalSurfaceCleaner,
    SemanticClassifier,
    InvariantVerification,
    RawParsedSchedule,
    RawTierBracket,
    TierBracket,
    InvariantViolationError,
    SemanticClassificationError
)


class TestRateParserPipeline(unittest.TestCase):

    # -----------------------------------------------------------------------
    # 1. LexicalSurfaceCleaner Tests
    # -----------------------------------------------------------------------
    def test_lexical_surface_cleaner(self):
        text1 = "  12%   "
        clean1, cur1 = LexicalSurfaceCleaner.clean(text1)
        self.assertEqual(clean1, "12%")
        self.assertEqual(cur1, "USD")

        text2 = "8% for products with a TOTAL sales price of ≤ $1,000.00"
        clean2, cur2 = LexicalSurfaceCleaner.clean(text2)
        self.assertIn("<=", clean2)
        self.assertIn("1000.00", clean2)
        self.assertNotIn(",", clean2)
        self.assertEqual(cur2, "USD")

        text_eur = "15% for price up to €300"
        clean_eur, cur_eur = LexicalSurfaceCleaner.clean(text_eur)
        self.assertEqual(cur_eur, "EUR")

    # -----------------------------------------------------------------------
    # 2. Example A: Fixed Rate Tests
    # -----------------------------------------------------------------------
    def test_fixed_rate_example_a(self):
        rule = FeeRateParser.parse("12%", fee_id="RULE_FIXED")
        self.assertEqual(rule.calculation_method, "FIXED")
        self.assertEqual(len(rule.brackets), 1)
        self.assertEqual(rule.brackets[0].lower_bound, Decimal("0.00"))
        self.assertIsNone(rule.brackets[0].upper_bound)
        self.assertEqual(rule.brackets[0].rate_percent, Decimal("12.0"))

        # Fee calculations
        self.assertEqual(rule.calculate_fee(100.0), 12.00)
        self.assertEqual(rule.calculate_fee(50.0), 6.00)
        self.assertEqual(rule.calculate_fee(0.0), 0.00)

    # -----------------------------------------------------------------------
    # 3. Example B: Two Whole-Price Tiers Tests
    # -----------------------------------------------------------------------
    def test_whole_price_example_b(self):
        raw = "8% for products with a TOTAL sales price of <=$10, and 15% for products with a TOTAL sales price >$10.00"
        rule = FeeRateParser.parse(raw, fee_id="RULE_EXAMPLE_B")

        self.assertEqual(rule.calculation_method, "WHOLE_PRICE")
        self.assertEqual(len(rule.brackets), 2)

        # Bracket 1: [0.00, 10.00) -> 8%
        self.assertEqual(rule.brackets[0].lower_bound, Decimal("0.00"))
        self.assertEqual(rule.brackets[0].upper_bound, Decimal("10.00"))
        self.assertEqual(rule.brackets[0].rate_percent, Decimal("8.0"))

        # Bracket 2: [10.00, None) -> 15%
        self.assertEqual(rule.brackets[1].lower_bound, Decimal("10.00"))
        self.assertIsNone(rule.brackets[1].upper_bound)
        self.assertEqual(rule.brackets[1].rate_percent, Decimal("15.0"))

        # Invariant check: base_amount for whole price must be 0.00
        self.assertEqual(rule.brackets[0].base_amount, Decimal("0.00"))
        self.assertEqual(rule.brackets[1].base_amount, Decimal("0.00"))

        # Calculations:
        # $8.00 -> 8% * $8 = $0.64
        self.assertEqual(rule.calculate_fee(8.0), 0.64)
        # $10.00 -> 8% * $10 = $0.80 (boundary check: <= 10 falls in tier 1)
        self.assertEqual(rule.calculate_fee(10.0), 0.80)
        # $12.00 -> 15% * $12 = $1.80 (whole price cliff)
        self.assertEqual(rule.calculate_fee(12.0), 1.80)

    # -----------------------------------------------------------------------
    # 4. Example C: Three Whole-Price Tiers Tests
    # -----------------------------------------------------------------------
    def test_whole_price_example_c(self):
        raw = "5% for products with a total sales price of <=$15.00, 10% for products with a total sales price >$15.00 and <=$20.00, 17% for products with a total sales price >$20.00"
        rule = FeeRateParser.parse(raw, fee_id="RULE_EXAMPLE_C")

        self.assertEqual(rule.calculation_method, "WHOLE_PRICE")
        self.assertEqual(len(rule.brackets), 3)

        self.assertEqual(rule.brackets[0].format_interval(), "[0.00, 15.00)")
        self.assertEqual(rule.brackets[1].format_interval(), "[15.00, 20.00)")
        self.assertEqual(rule.brackets[2].format_interval(), "[20.00, inf)")

        self.assertEqual(rule.brackets[0].rate_percent, Decimal("5.0"))
        self.assertEqual(rule.brackets[1].rate_percent, Decimal("10.0"))
        self.assertEqual(rule.brackets[2].rate_percent, Decimal("17.0"))

        # Exact boundary calculations
        self.assertEqual(rule.calculate_fee(10.0), 0.50)
        self.assertEqual(rule.calculate_fee(15.0), 0.75)
        self.assertEqual(rule.calculate_fee(18.0), 1.80)
        self.assertEqual(rule.calculate_fee(20.0), 2.00)
        self.assertEqual(rule.calculate_fee(25.0), 4.25)

    # -----------------------------------------------------------------------
    # 5. Example D: Marginal Progressive Rates Tests
    # -----------------------------------------------------------------------
    def test_marginal_progressive_example_d(self):
        raw = "15% for the portion of the total sales price up to $300.00, and 8% for any portion of the total sales price greater than $300.00"
        rule = FeeRateParser.parse(raw, fee_id="RULE_EXAMPLE_D")

        self.assertEqual(rule.calculation_method, "MARGINAL")
        self.assertEqual(len(rule.brackets), 2)

        # Tier 1: [0.00, 300.00) -> 15%, base_amount = 0.00
        self.assertEqual(rule.brackets[0].lower_bound, Decimal("0.00"))
        self.assertEqual(rule.brackets[0].upper_bound, Decimal("300.00"))
        self.assertEqual(rule.brackets[0].rate_percent, Decimal("15.0"))
        self.assertEqual(rule.brackets[0].base_amount, Decimal("0.00"))

        # Tier 2: [300.00, inf) -> 8%, base_amount = 300 * 0.15 = 45.00
        self.assertEqual(rule.brackets[1].lower_bound, Decimal("300.00"))
        self.assertIsNone(rule.brackets[1].upper_bound)
        self.assertEqual(rule.brackets[1].rate_percent, Decimal("8.0"))
        self.assertEqual(rule.brackets[1].base_amount, Decimal("45.00"))

        # O(1) Instant Calculations:
        # $100.00 -> 15% * 100 = $15.00
        self.assertEqual(rule.calculate_fee(100.0), 15.00)
        # $300.00 -> 15% * 300 = $45.00
        self.assertEqual(rule.calculate_fee(300.0), 45.00)
        # $500.00 -> $45.00 + (500 - 300) * 8% = $45 + $16 = $61.00
        self.assertEqual(rule.calculate_fee(500.0), 61.00)
        # $1000.00 -> $45.00 + (1000 - 300) * 8% = $45 + $56 = $101.00
        self.assertEqual(rule.calculate_fee(1000.0), 101.00)

    # -----------------------------------------------------------------------
    # 6. Marketplace Hybrid Model Tests
    # -----------------------------------------------------------------------
    def test_marketplace_hybrid(self):
        raw = "15% for the portion of the total sales price up to $300.00, and 8% for any portion greater than $300.00 + $0.30 per item with a minimum fee of $0.50"
        rule = FeeRateParser.parse(raw, fee_id="RULE_HYBRID")

        self.assertEqual(rule.calculation_method, "HYBRID")
        self.assertEqual(rule.min_fee, 0.50)
        self.assertEqual(rule.brackets[0].flat_fee, Decimal("0.30"))

        # At $1.00: 15% * $1 = $0.15 + $0.30 flat = $0.45 -> Floor raises to $0.50!
        self.assertEqual(rule.calculate_fee(1.0), 0.50)
        # At $10.00: 15% * $10 = $1.50 + $0.30 flat = $1.80
        self.assertEqual(rule.calculate_fee(10.0), 1.80)
        # At $500.00: $61.00 + $0.30 flat = $61.30
        self.assertEqual(rule.calculate_fee(500.0), 61.30)

    # -----------------------------------------------------------------------
    # 7. Invariant Verification Mathematical Proof Tests
    # -----------------------------------------------------------------------
    def test_invariant_domain_anchor_failure(self):
        """Proof 1: Rejects schedules that do not anchor at $0.00."""
        invalid_schedule = RawParsedSchedule(
            fee_id="ERR_ANCHOR",
            source_text="10% for >$15",
            cleaned_text="10% for >$15",
            calculation_method="WHOLE_PRICE",
            currency="USD",
            price_basis="TOTAL_SALES_PRICE",
            min_fee=Decimal("0.00"),
            max_fee=None,
            raw_brackets=[
                RawTierBracket(lower_bound=Decimal("15.00"), upper_bound=None, rate_percent=Decimal("10.0"))
            ]
        )
        with self.assertRaises(InvariantViolationError) as ctx:
            InvariantVerification.verify_and_compile(invalid_schedule)
        self.assertEqual(ctx.exception.violation_type, "DOMAIN_ANCHOR_FAILURE")

    def test_invariant_coverage_gap(self):
        """Proof 3: Rejects discontinuous gaps between tiers."""
        gap_schedule = RawParsedSchedule(
            fee_id="ERR_GAP",
            source_text="5% <= $10, 10% > $15",
            cleaned_text="5% <= $10, 10% > $15",
            calculation_method="WHOLE_PRICE",
            currency="USD",
            price_basis="TOTAL_SALES_PRICE",
            min_fee=Decimal("0.00"),
            max_fee=None,
            raw_brackets=[
                RawTierBracket(lower_bound=Decimal("0.00"), upper_bound=Decimal("10.00"), rate_percent=Decimal("5.0")),
                RawTierBracket(lower_bound=Decimal("15.00"), upper_bound=None, rate_percent=Decimal("10.0"))
            ]
        )
        with self.assertRaises(InvariantViolationError) as ctx:
            InvariantVerification.verify_and_compile(gap_schedule)
        self.assertEqual(ctx.exception.violation_type, "COVERAGE_GAP")

    def test_invariant_interval_overlap(self):
        """Proof 3: Rejects overlapping tiers."""
        overlap_schedule = RawParsedSchedule(
            fee_id="ERR_OVERLAP",
            source_text="5% <= $15, 10% > $10",
            cleaned_text="5% <= $15, 10% > $10",
            calculation_method="WHOLE_PRICE",
            currency="USD",
            price_basis="TOTAL_SALES_PRICE",
            min_fee=Decimal("0.00"),
            max_fee=None,
            raw_brackets=[
                RawTierBracket(lower_bound=Decimal("0.00"), upper_bound=Decimal("15.00"), rate_percent=Decimal("5.0")),
                RawTierBracket(lower_bound=Decimal("10.00"), upper_bound=None, rate_percent=Decimal("10.0"))
            ]
        )
        with self.assertRaises(InvariantViolationError) as ctx:
            InvariantVerification.verify_and_compile(overlap_schedule)
        self.assertEqual(ctx.exception.violation_type, "INTERVAL_OVERLAP")

    def test_invariant_unbounded_ceiling_failure(self):
        """Proof 4: Rejects schedules missing unbounded (+inf) final tier."""
        bounded_schedule = RawParsedSchedule(
            fee_id="ERR_CEILING",
            source_text="5% <= $10, 10% > $10 and <= $50",
            cleaned_text="5% <= $10, 10% > $10 and <= $50",
            calculation_method="WHOLE_PRICE",
            currency="USD",
            price_basis="TOTAL_SALES_PRICE",
            min_fee=Decimal("0.00"),
            max_fee=None,
            raw_brackets=[
                RawTierBracket(lower_bound=Decimal("0.00"), upper_bound=Decimal("10.00"), rate_percent=Decimal("5.0")),
                RawTierBracket(lower_bound=Decimal("10.00"), upper_bound=Decimal("50.00"), rate_percent=Decimal("10.0"))
            ]
        )
        with self.assertRaises(InvariantViolationError) as ctx:
            InvariantVerification.verify_and_compile(bounded_schedule)
        self.assertEqual(ctx.exception.violation_type, "UNBOUNDED_CEILING_FAILURE")

    def test_invariant_inverted_interval(self):
        """Proof 2: Rejects inverted intervals where lower >= upper."""
        inverted_schedule = RawParsedSchedule(
            fee_id="ERR_INVERTED",
            source_text="Invalid inverted",
            cleaned_text="Invalid inverted",
            calculation_method="WHOLE_PRICE",
            currency="USD",
            price_basis="TOTAL_SALES_PRICE",
            min_fee=Decimal("0.00"),
            max_fee=None,
            raw_brackets=[
                RawTierBracket(lower_bound=Decimal("20.00"), upper_bound=Decimal("10.00"), rate_percent=Decimal("5.0"))
            ]
        )
        with self.assertRaises(InvariantViolationError) as ctx:
            InvariantVerification.verify_and_compile(inverted_schedule)
        self.assertEqual(ctx.exception.violation_type, "INVERTED_INTERVAL")

    # -----------------------------------------------------------------------
    # 8. Downstream Multi-Modal Emitters Tests
    # -----------------------------------------------------------------------
    def test_downstream_emitters(self):
        raw = "15% for the portion of the total sales price up to $300.00, and 8% for any portion of the total sales price greater than $300.00"
        rule = FeeRateParser.parse(raw, fee_id="EX_D")

        # Canonical JSON
        canonical = rule.to_canonical_json()
        self.assertIn('"fee_id": "EX_D"', canonical)
        self.assertIn('"calculation_method": "MARGINAL"', canonical)
        self.assertIn('"base_amount": 45.0', canonical)

        # Micro-DSL
        dsl = rule.to_micro_dsl()
        self.assertTrue(dsl.startswith("MARGINAL: [0.00, 300.00)->15.0% | [300.00, inf)->8.0%"))

        # Compact LLM JSON
        compact = rule.to_llm_compact_json()
        self.assertIn('"m":"M"', compact)

        # ML Feature Vector with Synthetic Probes
        vec = rule.to_ml_feature_vector()
        self.assertEqual(vec["feat_model_type"], 2)  # MARGINAL
        self.assertEqual(vec["feat_tier_count"], 2)
        self.assertEqual(vec["probe_fee_at_100"], 15.00)
        self.assertEqual(vec["probe_fee_at_500"], 61.00)
        self.assertEqual(vec["probe_eff_rate_at_100"], 0.1500)
        self.assertEqual(vec["probe_eff_rate_at_500"], 0.1220)

    # -----------------------------------------------------------------------
    # 9. Configurable Output & Rate Card Benchmarks
    # -----------------------------------------------------------------------
    def test_configurable_format_output(self):
        raw = "12%"
        rule = FeeRateParser.parse(raw, fee_id="EX_FMT")

        # Test format_output modes
        json_out = rule.format_output("json")
        self.assertIn('"fee_id": "EX_FMT"', json_out)

        compact_out = rule.format_output("compact_json")
        self.assertIn('"m":"F"', compact_out)

        dsl_out = rule.format_output("dsl")
        self.assertEqual(dsl_out, "FIXED: [0.00, inf)->12.0%")

        dict_out = rule.format_output("dict")
        self.assertIsInstance(dict_out, dict)
        self.assertEqual(dict_out["fee_id"], "EX_FMT")

        summary_out = rule.format_output("summary")
        self.assertIn("[EX_FMT] FIXED", summary_out)

        with self.assertRaises(ValueError):
            rule.format_output("unsupported_format")

    def test_additional_rate_card_cases(self):
        from rate_parser import ADDITIONAL_BENCHMARK_CASES
        self.assertEqual(len(ADDITIONAL_BENCHMARK_CASES), 9)

        for cid, text in ADDITIONAL_BENCHMARK_CASES:
            rule = FeeRateParser.parse(text, fee_id=f"CARD_{cid}")
            self.assertEqual(rule.parse_status, "VALID")
            self.assertGreaterEqual(len(rule.brackets), 2)
            self.assertEqual(rule.brackets[0].lower_bound, Decimal("0.00"))
            self.assertIsNone(rule.brackets[-1].upper_bound)
            # Verify fee can be computed at boundary points without error
            fee_low = rule.calculate_fee(10.0)
            fee_high = rule.calculate_fee(2000.0)
            self.assertGreater(fee_low, 0.0)
            self.assertGreater(fee_high, fee_low)


if __name__ == "__main__":
    unittest.main()
