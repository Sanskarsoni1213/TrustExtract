from trustextract.evaluation.evaluate import DocumentEvaluator

evaluator = DocumentEvaluator()

print("=================================================================")
print("EVALUATOR FREE_TEXT / MerchantName MATCHER VALIDATION & NEGATIVE TESTS")
print("=================================================================")

test_cases = [
    # Positive / expected matches
    ("STARBUCKS COFFEE #1042", "Starbucks Coffee", "MerchantName", True, "Expected: GT is contiguous substring of Extracted"),
    ("Starbucks Coffee", "Starbucks Coffee", "MerchantName", True, "Expected: Exact match"),
    ("STARBUCKS COFFEE", "Starbucks Coffee", "MerchantName", True, "Expected: Case/whitespace normalized match"),
    
    # Negative / deliberate rejection tests
    ("DUNKIN DONUTS #1042", "Starbucks Coffee", "MerchantName", False, "Deliberately wrong vendor name"),
    ("PEETS COFFEE", "Starbucks Coffee", "MerchantName", False, "Different coffee chain"),
    ("Acme Logistics Corp", "Starbucks Coffee", "MerchantName", False, "Completely unrelated entity"),
    ("Starbucks", "Starbucks Coffee", "MerchantName", False, "Extracted is truncated/shorter than GT (rejected by directional rule)"),
    ("1042", "Starbucks Coffee", "MerchantName", False, "Only store number extracted, merchant missing"),
    ("", "Starbucks Coffee", "MerchantName", False, "Empty extracted string"),
]

all_passed = True
for ext, gt, fname, expected, desc in test_cases:
    actual = evaluator.values_match(ext, gt, fname)
    passed = (actual == expected)
    status_str = "PASS" if passed else "FAIL"
    print(f"[{status_str}] Extracted: {ext!r:<25} | GT: {gt!r:<18} | Match: {actual!s:<5} (Expected: {expected!s:<5}) -> {desc}")
    if not passed:
        all_passed = False

print("-----------------------------------------------------------------")
print(f"All negative & positive matcher tests passed: {all_passed}")
print("=================================================================")
