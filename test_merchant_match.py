"""Quick negative test for the FREE_TEXT substring matcher."""
from trustextract.evaluation.evaluate import DocumentEvaluator

e = DocumentEvaluator()

# Test 1: Real case — extracted has store number, GT doesn't
r1 = e.values_match("STARBUCKS COFFEE #1042", "Starbucks Coffee", "MerchantName")
print(f"Test 1 (GT-in-extracted, should be True):  result={r1}")

# Test 2: Negative — nonsense wrapping GT. This SHOULD be True because
# "starbuckscoffee" is a substring of "xyzstarbuckscoffeenonsense"
# This is the GT-in-extracted direction — it IS accepted by design.
r2 = e.values_match("xyzstarbuckscoffeenonsense", "starbuckscoffee", "MerchantName")
print(f"Test 2 (GT-in-extracted with junk, accepted by design): result={r2}")

# Test 3: Reversed direction — extracted is SHORTER than GT. Should be False.
r3 = e.values_match("Coffee", "Starbucks Coffee", "MerchantName")
print(f"Test 3 (extracted-in-GT, should be False): result={r3}")

# Test 4: Exact match still works
r4 = e.values_match("Apex Technologies LLC", "Apex Technologies LLC", "VendorName")
print(f"Test 4 (exact match, should be True):      result={r4}")

# Test 5: Completely wrong value
r5 = e.values_match("McDonalds", "Starbucks Coffee", "MerchantName")
print(f"Test 5 (wrong value, should be False):      result={r5}")

# Test 6: Tiny GT (< 3 chars) should NOT trigger substring
r6 = e.values_match("A big string", "AB", "MerchantName")
print(f"Test 6 (tiny GT < 3 chars, should be False): result={r6}")
