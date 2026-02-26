
def get_categories_from_filename(file_name, categories):
    if not file_name or not categories:
        return []
    
    file_name_lower = file_name.lower()
    found_categories = set()
    
    for category in categories:
        short_codes = category.get("short_code", [])
        for code in short_codes:
            if code and code.lower() in file_name_lower:
                category_code = category.get("code", "UNCATEGORIZED")
                if category_code != "UNCATEGORIZED":
                    found_categories.add(category_code)
                    print(f"Match found! Code: {code} in {file_name}")
    
    return list(found_categories)

# Mock Categories
categories = [
    {"code": "TAX_RETURN", "name": "Tax Return", "short_code": ["Tax Return", "TR"]},
    {"code": "INVOICE", "name": "Invoice", "short_code": ["INV", "Invoice"]},
    {"code": "BANK_STATEMENT", "name": "Bank Statement", "short_code": ["Bank Statement", "BS"]}
]

# Test Cases
print("--- Test 1: Original Filename (Backfill Scenario) ---")
filename_original = "Tax Return 2024.pdf"
matches = get_categories_from_filename(filename_original, categories)
print(f"Filename: {filename_original} -> Matches: {matches}")

print("\n--- Test 2: Secure Filename (Manual Upload Scenario) ---")
# secure_filename replaces spaces with underscores
filename_secure = "Tax_Return_2024.pdf" 
matches_secure = get_categories_from_filename(filename_secure, categories)
print(f"Filename: {filename_secure} -> Matches: {matches_secure}")

print("\n--- Test 3: Proposed Fix (Replace underscores) ---")
filename_fixed = filename_secure.replace("_", " ")
matches_fixed = get_categories_from_filename(filename_fixed, categories)
print(f"Filename: {filename_fixed} -> Matches: {matches_fixed}")
