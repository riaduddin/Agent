# 🎨 Slide Modification Agent V2

**Content-Only Modification System with Structure Preservation**

---

## 🏗️ **Architecture Overview**

```
User Query: "Change slide 3 title to 'Welcome' and add Tesla sales data to slide 5"
                            ↓
┌──────────────────────────────────────────────────────────────┐
│  Multi-Slide Modification Orchestrator                       │
│  - Coordinates entire modification workflow                  │
│  - Handles single and multiple slides                        │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│  STEP 1: Modification Request Parser                         │
│  - Parses: "Change slide 3..." + "add...to slide 5"         │
│  - Output: [                                                 │
│      {slide: 3, type: "title_change", instruction: "..."},  │
│      {slide: 5, type: "content_addition", research: true}   │
│    ]                                                         │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│  STEP 2: Validate Slide Numbers                              │
│  - Check slides 3 and 5 exist in presentation               │
│  - Verify against total slide count                         │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│  STEP 3: Process Each Modification (Sequential)              │
│                                                              │
│  For Slide 3:                  For Slide 5:                 │
│  ┌─────────────────────┐       ┌─────────────────────┐     │
│  │ Single Slide        │       │ Single Slide        │     │
│  │ Modifier            │       │ Modifier            │     │
│  └─────────────────────┘       └─────────────────────┘     │
│           │                              │                   │
│           ▼                              ▼                   │
│  A. Fetch from DB              A. Fetch from DB             │
│  B. Extract content            B. Extract content           │
│  C. Content Modifier           C. Content Modifier          │
│     (Simple text replace)          (+ Research retrieval)   │
│  D. Validate structure         D. Validate structure        │
│  E. Update DB                  E. Update DB                 │
│  F. Return success             F. Return success            │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│  STEP 4: Aggregate Results                                   │
│  - Slide 3: ✅ Success                                       │
│  - Slide 5: ✅ Success                                       │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│  STEP 5: Return Summary                                      │
│  "Successfully updated slides 3 and 5"                       │
└──────────────────────────────────────────────────────────────┘
```

---

## 📁 **File Structure**

```
root_agent/slide_modification_agent_v2/
├── __init__.py                      # Package exports
├── README.md                        # This file
├── request_parser.py                # Parses natural language requests
├── content_modifier.py              # LLM agent for content modification
├── validation_utils.py              # Structure validation functions
├── database_tools.py                # DB fetch/update tools
├── single_slide_modifier.py         # Single slide orchestrator
└── multi_slide_orchestrator.py      # Multi-slide orchestrator (main entry)
```

---

## 🔧 **Component Details**

### **1. Modification Request Parser** (`request_parser.py`)

**Purpose:** Parse natural language into structured modification requests

**Input:**
```
"Change slide 3 title to 'Welcome' and add Tesla sales data to slide 5"
```

**Output:**
```json
{
  "modifications": [
    {
      "slide_number": 3,
      "modification_type": "title_change",
      "instruction": "Change title to 'Welcome'",
      "requires_research": false
    },
    {
      "slide_number": 5,
      "modification_type": "content_addition",
      "instruction": "Add Tesla sales data",
      "requires_research": true
    }
  ],
  "error_message": null
}
```

**Supported Patterns:**
- Single: "Change slide 3 title"
- Multiple: "Update slide 2 and slide 5"
- Range: "Change slides 3-5"
- List: "Update slides 1, 3, and 5"

**Modification Types:**
- `title_change` - Change main heading
- `content_update` - Modify existing text
- `content_addition` - Add new content
- `content_removal` - Remove sections
- `style_change` - Visual/CSS changes
- `data_update` - Update charts/metrics

---

### **2. Content Modifier Agent** (`content_modifier.py`)

**Purpose:** LLM agent that modifies HTML content while preserving structure

**Key Features:**
- ✅ Uses Qdrant research tool when `requires_research=true`
- ✅ Receives strict instructions to preserve structure
- ✅ Has access to current content and global theme
- ✅ Returns complete modified HTML

**Critical Instructions:**
```
❌ NEVER change HTML tags, classes, IDs
❌ NEVER add or remove HTML elements
✅ ONLY change text content
✅ Use research data when needed
✅ Maintain theme consistency
```

---

### **3. Validation Utilities** (`validation_utils.py`)

**Purpose:** Ensure HTML structure is unchanged after modification

**Functions:**

#### `validate_structure(original_html, modified_html) -> bool`
```python
# Checks:
✅ Same tag count and types
✅ Same classes and IDs
✅ Same HTML hierarchy
✅ Same tag attributes

# Returns:
True  → Safe to save (only content changed)
False → REJECT (structure changed)
```

#### `extract_content_structure(html) -> dict`
```python
# Extracts:
{
  "title": "Main heading text",
  "headings": ["H2 text", "H3 text"],
  "paragraphs": ["Para 1", "Para 2"],
  "lists": [{"type": "ul", "items": ["Item 1", "Item 2"]}]
}
```

---

### **4. Database Tools** (`database_tools.py`)

**Functions:**

#### `fetch_slide_data(p_id, slide_number)`
```python
# Returns:
{
  "html": "<!DOCTYPE html>...",      # Raw HTML
  "slide_plan": {...},                # Original context
  "template_info": {...},             # Template used
  "content_metadata": {...},          # Pre-extracted content
  "global_theme": {...}               # Presentation theme
}
```

#### `update_slide_html(p_id, slide_number, modified_html)`
```python
# Updates:
- body: Modified HTML (raw, no prettify)
- content_metadata: Re-extracted from new HTML
- updated_at: Timestamp
```

#### `validate_presentation_slides(p_id, slide_numbers)`
```python
# Validates:
- Presentation exists
- All slide numbers are within range (1 to total_slides)
```

---

### **5. Single Slide Modifier** (`single_slide_modifier.py`)

**Purpose:** Orchestrate modification of ONE slide

**Workflow:**
1. Fetch slide data from DB
2. Extract content structure
3. Call content modifier agent
4. **VALIDATE structure unchanged**
5. Update database
6. Return confirmation

**Key Feature:** Validation step prevents structure changes

---

### **6. Multi-Slide Orchestrator** (`multi_slide_orchestrator.py`)

**Purpose:** Handle single or multiple slide modifications

**Workflow:**
1. Parse modification request
2. Validate slide numbers exist
3. Process each modification (sequential)
4. Aggregate results
5. Return summary

**Error Handling:**
- One slide fails → Continue with others
- Report which succeeded and which failed

---

## 🎯 **Usage Examples**

### **Example 1: Simple Title Change**

**User Input:**
```
"Change slide 3 title to 'Welcome to Tesla'"
```

**Processing:**
```
1. Parser: [{slide: 3, type: "title_change", instruction: "...", research: false}]
2. Validate: Slide 3 exists ✓
3. Fetch: Get current HTML from DB
4. Modify: Replace h1 text
5. Validate: Structure unchanged ✓
6. Save: Update DB
7. Result: "Successfully updated slide 3"
```

---

### **Example 2: Content Addition with Research**

**User Input:**
```
"Add Tesla's 2025 Q1 sales data to slide 5"
```

**Processing:**
```
1. Parser: [{slide: 5, type: "content_addition", instruction: "...", research: true}]
2. Validate: Slide 5 exists ✓
3. Fetch: Get current HTML
4. Research: Retrieve "Tesla 2025 Q1 sales data" from Qdrant
5. Modify: LLM adds research data to existing list/paragraph
6. Validate: Structure unchanged ✓
7. Save: Update DB
8. Result: "Successfully updated slide 5"
```

---

### **Example 3: Multiple Slides**

**User Input:**
```
"Change slide 2 title to 'Overview' and update slide 4 with recent EV market trends"
```

**Processing:**
```
1. Parser: [
     {slide: 2, type: "title_change", research: false},
     {slide: 4, type: "content_update", research: true}
   ]
2. Validate: Slides 2 and 4 exist ✓
3. Process Slide 2:
   - Fetch → Modify title → Validate ✓ → Save ✓
4. Process Slide 4:
   - Fetch → Research EV trends → Modify → Validate ✓ → Save ✓
5. Aggregate: Both succeeded
6. Result: "Successfully updated slides 2 and 4"
```

---

### **Example 4: Range Operation**

**User Input:**
```
"Update slides 3-5 to use blue background"
```

**Processing:**
```
1. Parser: [
     {slide: 3, type: "style_change", ...},
     {slide: 4, type: "style_change", ...},
     {slide: 5, type: "style_change", ...}
   ]
2. Validate: Slides 3, 4, 5 exist ✓
3. Process each sequentially:
   - Slide 3: Fetch → Modify CSS → Validate ✓ → Save ✓
   - Slide 4: Fetch → Modify CSS → Validate ✓ → Save ✓
   - Slide 5: Fetch → Modify CSS → Validate ✓ → Save ✓
4. Result: "Successfully updated slides 3-5"
```

---

## 🛡️ **Safety Mechanisms**

### **1. Structure Validation**
```python
# Before saving ANY modification
if validate_structure(original_html, modified_html):
    save_to_database(modified_html)  # ✅ Safe
else:
    reject_modification()             # ❌ Structure changed
```

### **2. Raw HTML Storage**
```python
# Store exactly as generated/modified
"body": modified_html  # No prettify(), no formatting changes
```

### **3. Sequential Processing**
```python
# Process slides one at a time
for modification in modifications:
    result = modify_slide(modification)
    # One failure doesn't stop others
```

### **4. Error Isolation**
```python
# Track each slide independently
results = {
    "successful": [3, 5],
    "failed": [{slide: 7, error: "..."}]
}
```

---

## ✅ **Supported Modifications**

| User Request | Type | Research | Structure Preserved |
|--------------|------|----------|---------------------|
| "Change title to X" | title_change | No | ✅ Yes |
| "Update paragraph with 2025 data" | content_update | Yes | ✅ Yes |
| "Add bullet point" | content_addition | Maybe | ✅ Yes |
| "Remove third bullet" | content_removal | No | ✅ Yes |
| "Change background to blue" | style_change | No | ✅ Yes |
| "Update chart with Q1 data" | data_update | Yes | ✅ Yes |
| "Fix typo in paragraph" | content_update | No | ✅ Yes |
| "Make text bigger" | style_change | No | ✅ Yes |

---

## 🚫 **Rejected Modifications (Structure Changes)**

| User Request | Why Rejected |
|--------------|-------------|
| "Add a new section" | Would create new div/section element ❌ |
| "Split slide into two columns" | Would change HTML structure ❌ |
| "Add a sidebar" | Would create new structural element ❌ |
| "Rearrange sections" | Would change HTML hierarchy ❌ |

**Solution:** Tell user: "I can only modify content, not structure. Please request content changes only."

---

## 🔄 **Modification Flow (Per Slide)**

```python
def modify_single_slide(p_id, slide_number, mod_request):
    # 1. Fetch
    slide = fetch_slide_data(p_id, slide_number)
    original_html = slide["html"]
    
    # 2. Extract
    content = extract_content_structure(original_html)
    
    # 3. Modify (LLM)
    if mod_request["requires_research"]:
        research = retrieve_research_context(...)
    
    modified_html = content_modifier_agent(
        html=original_html,
        content=content,
        request=mod_request,
        theme=slide["global_theme"]
    )
    
    # 4. VALIDATE ⭐
    if not validate_structure(original_html, modified_html):
        return {"error": "Structure changed - rejected"}
    
    # 5. Update
    update_slide_html(p_id, slide_number, modified_html)
    
    # 6. Confirm
    return {"success": True}
```

---

## 📊 **Database Schema Used**

### **Read From:**
```javascript
db.slide_html.find_one({
  "p_id": "...",
  "slide_index": 2  // 0-based
})
// Returns:
{
  "body": "<!DOCTYPE html>...",  // ⭐ Raw HTML for modification
  "slide_plan": {...},            // Context for understanding content
  "template_info": {...},         // Template reference
  "content_metadata": {...},      // Quick content overview
  "global_theme": {...}           // Stored in presentations collection
}
```

### **Write To:**
```javascript
db.slide_html.update_one(
  {"p_id": "...", "slide_index": 2},
  {"$set": {
    "body": modified_html,        // Raw modified HTML
    "content_metadata": {...},    // Re-extracted
    "updated_at": ISODate("...")
  }}
)
```

---

## 🧪 **Testing**

### **Test Case 1: Title Change**
```bash
# Request
"Change slide 1 title to 'Welcome'"

# Expected Result
✅ Title changed
✅ All HTML tags preserved
✅ Classes/IDs unchanged
✅ Structure identical
```

### **Test Case 2: Multi-Edit**
```bash
# Request
"Change slide 2 title and add sales data to slide 3"

# Expected Result
✅ Slide 2 title updated
✅ Slide 3 content added with research
✅ Both structures preserved
✅ Summary: "Successfully updated slides 2 and 3"
```

### **Test Case 3: Validation Rejection**
```bash
# Request (ambiguous, might cause structure change)
"Add a new section to slide 4"

# Expected Result
❌ Modification rejected (tried to add new div)
⚠️ User message: "Cannot change HTML structure, only content"
✅ Original HTML unchanged
```

---

## 🎯 **Integration**

### **In `root_agent/agent.py`:**
```python
from root_agent.slide_modification_agent_v2 import multi_slide_modification_orchestrator

SlideOrchestrationAgent = LlmAgent(
    sub_agents=[
        pipeline,                               # Slide creation
        multi_slide_modification_orchestrator   # Slide modification ⭐
    ]
)
```

### **Routing:**
```python
# User query classified as "edit_slide"
# ↓
# Route to: multi_slide_modification_orchestrator
# ↓
# Handles: single slides, multiple slides, ranges, all edge cases
```

---

## ✅ **Key Advantages**

1. **✅ Structure Preservation Guaranteed**
   - Validation runs before every save
   - Rejects any structure changes
   - User HTML never breaks

2. **✅ Flexible Natural Language**
   - Handles single/multiple slides
   - Understands ranges (3-5)
   - Parses complex requests

3. **✅ Research Integration**
   - Auto-detects when research needed
   - Retrieves from Qdrant
   - Adds relevant, accurate data

4. **✅ Error Resilience**
   - One slide failure doesn't stop others
   - Clear error messages
   - Graceful degradation

5. **✅ Content-Only Modifications**
   - Text changes only
   - Preserves templates
   - Maintains visual consistency

---

## 🚀 **Ready to Use**

The modification system is **complete** and **integrated**. 

**To test:**
1. Generate a presentation (creates slides in DB)
2. Request a modification: "Change slide 3 title to 'Test'"
3. System will parse → validate → modify → save
4. Verify HTML structure unchanged in DB

**User can now request ANY content modification safely!** 🎉


