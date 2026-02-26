# Categorization Logic Update Documentation

## Overview
This document details the updates made to the file categorization logic within the Georgia Digitization Platform backend. The primary objective was to support **multiple categories per file** when identifying categories via short codes in filenames, resolving a limitation where only the first matching category was assigned.

## Problem Statement
Previously, the categorization logic scanned the filename for short codes (e.g., "INV", "CHK") but stopped after finding the *first* match. This meant that if a filename contained multiple short codes (e.g., `INV_CHK_123.pdf`), the system would only assign the first identified category (e.g., "INVOICES") and ignore the rest.

## Solution Implemented
The logic has been refactored to:
1.  Iterate through **all** configured categories.
2.  Collect **all** unique category codes that have a matching short code in the filename.
3.  Assign the full list of identified categories to the document metadata.

## Technical Changes

### 1. `backend/app/services/category_api_service.py`

**Change:** Refactored `get_category_from_filename` to `get_categories_from_filename`.

-   **Before:**
    -   Function: `get_category_from_filename(file_name, categories) -> str`
    -   Behavior: Returned the first matching category code as a single string.
    -   Return Value: `"INVOICES"` or `"UNCATEGORIZED"`.

-   **After:**
    -   Function: `get_categories_from_filename(file_name, categories) -> List[str]`
    -   Behavior: Iterates through all categories and collects all matches in a set to ensure uniqueness.
    -   Return Value: `["INVOICES", "CHECKS"]` or `[]` (empty list).

### 2. `backend/app/services/categorization_service.py`

**Change:** Updated `categorize_document` to handle the list response.

-   **Before:**
    ```python
    category_name_from_short_code = category_api_service.get_category_from_filename(...)
    if category_name_from_short_code and category_name_from_short_code != "UNCATEGORIZED":
        doc_ref.update({"categories": [category_name_from_short_code]})
    ```

-   **After:**
    ```python
    category_names_from_short_code = category_api_service.get_categories_from_filename(...)
    if category_names_from_short_code:
        doc_ref.update({"categories": category_names_from_short_code})
    ```

## Impact
-   **Multi-Category Support:** Files can now automatically receive multiple tags based on their naming convention (e.g., a file named `2024_INV_TAX_Report.pdf` can be tagged as both `INVOICES` and `TAX_FORMS`).
-   **Backward Compatibility:** If only one short code is present, it works as before but returns a list of one element.
-   **No "Uncategorized" Noise:** The new function returns an empty list `[]` instead of the string `"UNCATEGORIZED"`, allowing for cleaner conditional checks in the calling service.
