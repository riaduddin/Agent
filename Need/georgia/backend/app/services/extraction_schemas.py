# backend/app/services/extraction_schemas.py
"""
Document-Type-Specific Entity Extraction Schemas

Each schema defines:
- description: What this document type represents
- fields: All fields to extract with descriptions for Gemini
- primary_search_fields: Most important fields for search
- vector_restrictions: Fields to use as vector namespace restrictions

These schemas are used by TypeSpecificExtractionService to generate
type-specific Gemini prompts for maximum extraction accuracy.
"""

EXTRACTION_SCHEMAS = {
    
    "1099": {
        "description": "IRS 1099 Tax Forms (1099-MISC, 1099-NEC, 1099-INT, 1099-DIV, 1099-R, 1099-G, etc.)",
        "fields": {
            "recipient_name": "Full name of person or business receiving the 1099",
            "recipient_tin": "Recipient's Tax Identification Number (SSN format XXX-XX-XXXX or EIN format XX-XXXXXXX)",
            "recipient_address": "Complete mailing address of the recipient",
            "payer_name": "Name of the organization or person issuing the 1099",
            "payer_tin": "Payer's Employer Identification Number (EIN)",
            "payer_address": "Complete address of the payer",
            "tax_year": "Tax year in YYYY format (e.g., 2024)",
            "form_type": "Specific 1099 form type (1099-MISC, 1099-NEC, 1099-INT, 1099-DIV, 1099-R, 1099-G, 1099-K, 1099-S, 1099-B)",
            "total_amount": "Primary or total reported amount in dollars",
            "box1_amount": "Box 1 amount if present",
            "box2_amount": "Box 2 amount if present",
            "box3_amount": "Box 3 amount if present",
            "box4_federal_tax": "Federal income tax withheld",
            "box7_amount": "Box 7 nonemployee compensation if present",
            "state": "State name or abbreviation if state tax reported",
            "state_tax_withheld": "State tax withheld amount",
            "account_number": "Account number if present on the form"
        },
        "primary_search_fields": ["recipient_name", "recipient_tin", "payer_name", "tax_year", "form_type"],
        "vector_restrictions": ["recipient_name", "recipient_tin", "payer_name", "tax_year", "form_type"]
    },

    "CHECKS": {
        "description": "Checks, Direct Deposits, Payment Vouchers, and Payment Instruments",
        "fields": {
            "check_number": "Check number - This is the MOST CRITICAL field for search. Look for 'Check No.', 'Check #', 'No.', or numbers in top right corner",
            "payee_name": "Name of person or entity the check is payable to (after 'Pay to the Order of')",
            "payer_name": "Name on the account / Check issuer (usually printed at top of check)",
            "amount_numeric": "Check amount in numeric format (e.g., 500.00)",
            "amount_written": "Check amount written in words (e.g., Five Hundred and 00/100)",
            "date": "Check date in YYYY-MM-DD format",
            "bank_name": "Name of the issuing bank",
            "bank_address": "Address of the bank",
            "routing_number": "Bank routing number (9 digits at bottom of check)",
            "account_number_partial": "Last 4 digits of account number only (for privacy)",
            "memo": "Memo line or 'For' line content",
            "signature_name": "Name of signer if legible",
            "endorsement": "Endorsement details on back if visible",
            "void_status": "VOID if check is voided, otherwise leave empty",
            "check_type": "Type: PERSONAL, BUSINESS, CASHIER, MONEY_ORDER, DIRECT_DEPOSIT, VOUCHER"
        },
        "primary_search_fields": ["check_number", "payee_name", "payer_name", "amount_numeric", "date"],
        "vector_restrictions": ["check_number", "payee_name", "payer_name", "date"]
    },

    "CHILD_WELFARE_REPORTS": {
        "description": "Child Welfare Reports, CPS Reports, Social Services Documents, Foster Care Records",
        "fields": {
            "case_number": "Case ID, Case Number, or Reference Number - PRIMARY identifier",
            "child_name": "Full name of the child (may be multiple children - list primary or all)",
            "child_dob": "Child's date of birth in YYYY-MM-DD format",
            "child_age": "Child's age if date of birth not available",
            "parent_names": "Parent or guardian names (comma separated if multiple)",
            "caseworker_name": "Assigned social worker, caseworker, or investigator name",
            "caseworker_id": "Caseworker ID, badge number, or employee ID",
            "supervisor_name": "Supervisor name if present",
            "report_date": "Date of the report in YYYY-MM-DD format",
            "incident_date": "Date of incident if different from report date",
            "report_type": "Type of report: INVESTIGATION, ASSESSMENT, REVIEW, HOME_STUDY, COURT_REPORT, CLOSURE, INTAKE, REFERRAL",
            "county": "County name",
            "district": "District or regional office",
            "state": "State",
            "status": "Case status: OPEN, CLOSED, PENDING, ACTIVE, SUBSTANTIATED, UNSUBSTANTIATED",
            "placement_type": "Placement type: FOSTER_HOME, KINSHIP, RESIDENTIAL, GROUP_HOME, ADOPTIVE, REUNIFIED, IN_HOME",
            "court_date": "Next court date if applicable in YYYY-MM-DD format",
            "court_name": "Name of court if mentioned",
            "allegations_summary": "Brief summary of allegations or concerns (keep under 100 words)",
            "services_provided": "Services being provided (comma separated)"
        },
        "primary_search_fields": ["case_number", "child_name", "caseworker_name", "parent_names", "report_date"],
        "vector_restrictions": ["case_number", "child_name", "caseworker_name", "county", "status"]
    },

    "LEAVE_DOCUMENTS": {
        "description": "Employee Leave Requests, Leave Approvals, FMLA Forms, Time Off Requests",
        "fields": {
            "employee_name": "Full name of the employee requesting leave",
            "employee_id": "Employee ID, badge number, or personnel number",
            "ssn_last4": "Last 4 digits of SSN only (if visible)",
            "department": "Department name",
            "division": "Division or unit name",
            "job_title": "Employee's job title or position",
            "supervisor_name": "Supervisor or manager name",
            "supervisor_title": "Supervisor's title",
            "leave_type": "Type of leave: ANNUAL, SICK, FMLA, BEREAVEMENT, MILITARY, UNPAID, PERSONAL, MATERNITY, PATERNITY, JURY_DUTY, ADMINISTRATIVE",
            "start_date": "Leave start date in YYYY-MM-DD format",
            "end_date": "Leave end date in YYYY-MM-DD format",
            "return_date": "Expected return to work date",
            "hours_requested": "Total hours requested",
            "days_requested": "Total days requested",
            "status": "Status: SUBMITTED, PENDING, APPROVED, DENIED, CANCELLED, MODIFIED",
            "approval_date": "Date of approval or denial in YYYY-MM-DD format",
            "approver_name": "Name of person who approved/denied",
            "reason": "Reason for leave (brief)",
            "intermittent": "YES if intermittent leave, NO otherwise",
            "fmla_tracking_number": "FMLA tracking or case number if applicable"
        },
        "primary_search_fields": ["employee_name", "employee_id", "leave_type", "start_date", "department"],
        "vector_restrictions": ["employee_name", "employee_id", "department", "leave_type", "status"]
    },

    "MONTH_END_REPORTS": {
        "description": "Financial Month-End Reports: General Journal Entries (GJE), General Ledger (GL), Purchase Orders (PO), Trial Balance, Budget Reports",
        "fields": {
            "report_type": "Type: GJE, GENERAL_JOURNAL, GL, GENERAL_LEDGER, PO, PURCHASE_ORDER, TRIAL_BALANCE, BUDGET, AP_AGING, AR_AGING, BANK_RECONCILIATION",
            "report_title": "Full title of the report as shown",
            "report_number": "Report number or ID if present",
            "reporting_period": "Month and Year (e.g., December 2024 or 12/2024)",
            "period_start_date": "Period start date in YYYY-MM-DD",
            "period_end_date": "Period end date in YYYY-MM-DD",
            "fiscal_year": "Fiscal year (e.g., FY2024 or 2024)",
            "department": "Department name",
            "department_code": "Department code or number",
            "fund_name": "Fund name",
            "fund_code": "Fund code or number",
            "cost_center": "Cost center number or name",
            "project_code": "Project code if applicable",
            "account_number": "GL Account number(s) - list primary ones",
            "account_name": "GL Account name(s)",
            "total_debits": "Total debit amount",
            "total_credits": "Total credit amount",
            "net_amount": "Net amount or balance",
            "beginning_balance": "Beginning balance for period",
            "ending_balance": "Ending balance for period",
            "prepared_by": "Name of report preparer",
            "reviewed_by": "Name of reviewer",
            "approved_by": "Name of approver",
            "run_date": "Report generation date in YYYY-MM-DD",
            "batch_number": "Batch number if applicable"
        },
        "primary_search_fields": ["report_type", "reporting_period", "department", "account_number", "fund_code"],
        "vector_restrictions": ["report_type", "reporting_period", "department", "fund_code", "account_number"]
    },

    "OTHER": {
        "description": "Uncategorized, Miscellaneous, or Unknown Document Types",
        "fields": {
            "document_title": "Title, subject line, or header of the document",
            "document_type_guess": "Best guess at document type based on content",
            "primary_entity_name": "Main person or organization name found",
            "secondary_entity_name": "Secondary person or organization name",
            "primary_id": "Any primary identifier found (case #, file #, reference #, ID)",
            "secondary_id": "Any secondary identifier",
            "primary_date": "Most prominent or important date in YYYY-MM-DD format",
            "secondary_date": "Another relevant date",
            "location": "Any address, city, state, or location mentioned",
            "phone_number": "Any phone number found",
            "email": "Any email address found",
            "amount": "Any monetary amount",
            "reference_number": "Any reference, tracking, or file number",
            "author_name": "Author or sender name",
            "recipient_name": "Recipient or addressee name",
            "summary": "Brief 1-2 sentence summary of document content"
        },
        "primary_search_fields": ["primary_entity_name", "primary_id", "primary_date", "document_type_guess"],
        "vector_restrictions": ["primary_entity_name", "primary_id", "document_type_guess", "primary_date"]
    },

    "PAYROLL_REPORTS_N_DOCUMENTS": {
        "description": "Payroll Records, Pay Stubs, Earnings Statements, W-2 Forms, Payroll Registers",
        "fields": {
            "employee_name": "Employee full name - CRITICAL field",
            "employee_id": "Employee ID or badge number - CRITICAL field",
            "ssn_last4": "Last 4 digits of SSN only",
            "department": "Department name",
            "department_code": "Department code",
            "division": "Division name",
            "job_title": "Job title or position",
            "job_code": "Job or position code",
            "pay_grade": "Pay grade or level",
            "pay_period_start": "Pay period start date in YYYY-MM-DD",
            "pay_period_end": "Pay period end date in YYYY-MM-DD",
            "pay_date": "Check date or payment date in YYYY-MM-DD",
            "check_number": "Payroll check number",
            "direct_deposit": "YES if direct deposit, NO if paper check",
            "hours_regular": "Regular hours worked",
            "hours_overtime": "Overtime hours worked",
            "hours_total": "Total hours worked",
            "rate_regular": "Regular hourly rate",
            "rate_overtime": "Overtime hourly rate",
            "gross_pay": "Gross pay amount",
            "net_pay": "Net pay amount",
            "federal_tax": "Federal tax withheld",
            "state_tax": "State tax withheld",
            "social_security": "Social Security tax withheld",
            "medicare": "Medicare tax withheld",
            "retirement_deduction": "Retirement/401k deduction",
            "health_insurance": "Health insurance deduction",
            "other_deductions": "Other deduction amount",
            "ytd_gross": "Year-to-date gross earnings",
            "ytd_federal_tax": "Year-to-date federal tax withheld",
            "ytd_net": "Year-to-date net pay"
        },
        "primary_search_fields": ["employee_name", "employee_id", "check_number", "pay_date", "department"],
        "vector_restrictions": ["employee_name", "employee_id", "department", "pay_date", "check_number"]
    },

    "PENDING_FILES": {
        "description": "Files Pending Processing, Classification, or Action",
        "fields": {
            "file_reference": "File reference number, name, or identifier",
            "file_name": "Original filename if visible",
            "subject": "Subject or title of the document",
            "entity_name": "Primary person or organization name",
            "entity_id": "Any identifier associated with the entity",
            "date_received": "Date received or created in YYYY-MM-DD",
            "date_due": "Due date or deadline if applicable",
            "source": "Source of document (department, person, external)",
            "source_department": "Originating department",
            "priority": "Priority level: HIGH, MEDIUM, LOW, URGENT",
            "assigned_to": "Person or department assigned to handle",
            "assigned_date": "Date assigned in YYYY-MM-DD",
            "status": "Status: PENDING, IN_REVIEW, AWAITING_INFO, READY, ON_HOLD",
            "action_required": "Required action described briefly",
            "notes": "Any notes or comments on the file",
            "related_case": "Related case number if applicable",
            "document_type_detected": "Best guess at actual document type"
        },
        "primary_search_fields": ["file_reference", "entity_name", "date_received", "assigned_to", "status"],
        "vector_restrictions": ["file_reference", "entity_name", "entity_id", "date_received", "status"]
    },

    "PERSONNEL_FILES": {
        "description": "Employee Personnel Records, HR Files, Employment Documents",
        "fields": {
            "employee_name": "Employee full name - CRITICAL field for all personnel searches",
            "employee_id": "Employee ID or badge number - CRITICAL field",
            "ssn_last4": "Last 4 digits of Social Security Number only",
            "date_of_birth": "Date of birth in YYYY-MM-DD format",
            "hire_date": "Original hire date in YYYY-MM-DD",
            "termination_date": "Termination date if applicable in YYYY-MM-DD",
            "rehire_date": "Rehire date if applicable",
            "department": "Current department name",
            "department_code": "Department code",
            "division": "Division name",
            "job_title": "Current job title or position",
            "job_code": "Job or position code",
            "pay_grade": "Pay grade or level",
            "supervisor_name": "Current supervisor name",
            "supervisor_id": "Supervisor's employee ID",
            "work_location": "Work location, office, or site",
            "work_address": "Work address",
            "home_address": "Home address (if visible)",
            "phone_work": "Work phone number",
            "phone_personal": "Personal phone (if visible)",
            "email_work": "Work email address",
            "salary": "Salary or hourly rate if visible",
            "employment_status": "Status: FULL_TIME, PART_TIME, TEMPORARY, CONTRACT, INTERN",
            "employment_type": "Type: REGULAR, SEASONAL, PROBATIONARY",
            "document_subtype": "Specific document type: APPLICATION, RESUME, OFFER_LETTER, I9, W4, PERFORMANCE_REVIEW, DISCIPLINARY, PROMOTION, TRANSFER, TRAINING, CERTIFICATION, TERMINATION, EXIT_INTERVIEW, BACKGROUND_CHECK, DRUG_TEST, MEDICAL",
            "document_date": "Date of this specific document in YYYY-MM-DD",
            "effective_date": "Effective date of action/change if applicable",
            "review_period": "Review period if performance document",
            "review_rating": "Performance rating if applicable"
        },
        "primary_search_fields": ["employee_name", "employee_id", "department", "document_subtype", "job_title"],
        "vector_restrictions": ["employee_name", "employee_id", "department", "job_title", "document_subtype"]
    },

    "TRAVEL_REPORTS": {
        "description": "Travel Requests, Travel Authorizations, Travel Expense Reports, Trip Reports",
        "fields": {
            "traveler_name": "Name of employee traveling - CRITICAL field",
            "employee_id": "Employee ID of traveler",
            "department": "Department name",
            "department_code": "Department code",
            "division": "Division name",
            "supervisor_name": "Supervisor or approving manager name",
            "trip_purpose": "Business purpose or reason for travel",
            "trip_description": "Brief description of trip",
            "conference_name": "Conference or event name if applicable",
            "destination_city": "Primary destination city",
            "destination_state": "Destination state",
            "destination_country": "Destination country (if international)",
            "multiple_destinations": "YES if multiple destinations, list cities",
            "departure_date": "Trip start/departure date in YYYY-MM-DD",
            "return_date": "Trip end/return date in YYYY-MM-DD",
            "total_days": "Total number of travel days",
            "transportation_mode": "Primary transportation: AIR, CAR, TRAIN, BUS, RENTAL, PERSONAL_VEHICLE",
            "airline": "Airline name if flying",
            "flight_numbers": "Flight numbers if available",
            "rental_company": "Rental car company if applicable",
            "hotel_name": "Hotel or lodging name",
            "airfare_cost": "Airfare or transportation cost",
            "lodging_cost": "Hotel/lodging total cost",
            "meals_cost": "Meals and per diem amount",
            "mileage_cost": "Mileage reimbursement amount",
            "registration_cost": "Conference/event registration cost",
            "parking_cost": "Parking costs",
            "other_expenses": "Other miscellaneous expenses",
            "total_expense": "Total expense amount - CRITICAL for amount searches",
            "advance_amount": "Travel advance amount if any",
            "reimbursement_due": "Net reimbursement due to employee",
            "status": "Status: DRAFT, SUBMITTED, PENDING_APPROVAL, APPROVED, DENIED, COMPLETED, CANCELLED",
            "approval_date": "Date of approval in YYYY-MM-DD",
            "approved_by": "Name of approver",
            "project_code": "Project or grant code being charged",
            "fund_code": "Fund code for expense",
            "cost_center": "Cost center",
            "receipt_count": "Number of receipts attached",
            "report_number": "Travel report or authorization number"
        },
        "primary_search_fields": ["traveler_name", "destination_city", "departure_date", "total_expense", "trip_purpose"],
        "vector_restrictions": ["traveler_name", "employee_id", "department", "destination_city", "departure_date"]
    }
}


# Mapping from category codes to schema keys (handles variations)
CATEGORY_TO_SCHEMA_MAP = {
    # Direct mappings
    "1099": "1099",
    "CHECKS": "CHECKS",
    "CHILD_WELFARE_REPORTS": "CHILD_WELFARE_REPORTS",
    "LEAVE_DOCUMENTS": "LEAVE_DOCUMENTS",
    "MONTH_END_REPORTS": "MONTH_END_REPORTS",
    "OTHER": "OTHER",
    "PAYROLL_REPORTS_N_DOCUMENTS": "PAYROLL_REPORTS_N_DOCUMENTS",
    "PENDING_FILES": "PENDING_FILES",
    "PERSONNEL_FILES": "PERSONNEL_FILES",
    "TRAVEL_REPORTS": "TRAVEL_REPORTS",
    
    # Short code mappings
    "DD": "CHECKS",
    "CK": "CHECKS",
    "CHECK": "CHECKS",
    "CW": "CHILD_WELFARE_REPORTS",
    "CWR": "CHILD_WELFARE_REPORTS",
    "GJE": "MONTH_END_REPORTS",
    "GL": "MONTH_END_REPORTS",
    "PO": "MONTH_END_REPORTS",
    "MONTH_END": "MONTH_END_REPORTS",
    "GENERALLEDGER": "MONTH_END_REPORTS",
    "GENERAL_LEDGER": "MONTH_END_REPORTS",
    "PAYROLL": "PAYROLL_REPORTS_N_DOCUMENTS",
    "TRAVEL": "TRAVEL_REPORTS",
    
    # Common variations
    "INVOICE": "OTHER",  # Invoices fall under OTHER unless specific category exists
    "TAX": "1099",
    "TAX_FORM": "1099",
    "LEAVE": "LEAVE_DOCUMENTS",
    "PERSONNEL": "PERSONNEL_FILES",
    "HR": "PERSONNEL_FILES",
    "EMPLOYEE": "PERSONNEL_FILES",
    
    # General/Unknown document types - map to OTHER for generic extraction
    "GENERAL_DOCUMENT": "OTHER",
    "GENERAL": "OTHER",
    "UNKNOWN": "OTHER",
    "UNCLASSIFIED": "OTHER",
    "DOCUMENT": "OTHER",
    "MISC": "OTHER",
    "MISCELLANEOUS": "OTHER",
    "DEFAULT": "OTHER",
    "UNCATEGORIZED": "OTHER",
}


def get_schema_for_doc_type(doc_type: str) -> dict:
    """
    Get the extraction schema for a document type.
    
    Args:
        doc_type: Document type string (can be category code, short code, or variation)
        
    Returns:
        Schema dictionary with fields, primary_search_fields, and vector_restrictions
    """
    if not doc_type:
        return EXTRACTION_SCHEMAS["OTHER"]
    
    # Normalize to uppercase
    doc_type_upper = doc_type.upper().strip()
    
    # Try direct match in schemas
    if doc_type_upper in EXTRACTION_SCHEMAS:
        return EXTRACTION_SCHEMAS[doc_type_upper]
    
    # Try mapping
    mapped_type = CATEGORY_TO_SCHEMA_MAP.get(doc_type_upper)
    if mapped_type and mapped_type in EXTRACTION_SCHEMAS:
        return EXTRACTION_SCHEMAS[mapped_type]
    
    # Default to OTHER
    return EXTRACTION_SCHEMAS["OTHER"]


def get_all_schema_names() -> list:
    """Get list of all available schema names."""
    return list(EXTRACTION_SCHEMAS.keys())


def get_vector_restriction_fields(doc_type: str) -> list:
    """Get the vector restriction fields for a document type."""
    schema = get_schema_for_doc_type(doc_type)
    return schema.get("vector_restrictions", [])


def get_primary_search_fields(doc_type: str) -> list:
    """Get the primary search fields for a document type."""
    schema = get_schema_for_doc_type(doc_type)
    return schema.get("primary_search_fields", [])
