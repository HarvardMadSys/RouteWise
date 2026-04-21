# Step 5 Ground-Truth Check

## Check 1: A-share monotonic non-increasing
- cheapest_fixed: PASS
- lp_mix: PASS
- lp_hedge: PASS
- lp_explorer: PASS
- v2_only: PASS
- v2_p50_hedge: FAIL (1 violations)
- v2_explorer: FAIL (2 violations)

## Check 2: V2 drops A after band boundary (P50 > 110ms)
- v2_only: FAIL (max A-share=42.3%)
- v2_p50_hedge: FAIL (max A-share=49.7%)
- v2_explorer: FAIL (max A-share=47.3%)

## Check 3: LP A-share has no discontinuity > 40pp
- lp_mix: PASS
- lp_hedge: PASS
- lp_explorer: PASS
