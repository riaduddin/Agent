# Slide Insertion Orchestrator Implementation Plan

## Overview

Implement a new `slide_insertion_orchestrator` agent as a sub-agent of `SlideOrchestrationAgent` that handles inserting new slides between existing slides with automatic renumbering, template selection, and intelligent content generation.

## Architecture

The agent will follow this workflow:

1. Parse user request to identify insertion position and content/topic
2. Validate insertion position against existing slide count
3. Fetch global theme and presentation metadata from database
4. Generate content using research tools if needed
5. Select appropriate template based on content and neighboring slides
6. Generate new slide HTML
7. Renumber existing slides (shift slide_number and slide_index)
8. Insert new slide at specified position
9. Update presentation outline in new collection

## Implementation Steps

### Step 1: Create Slide Insertion Module Directory Structure

Create new directory: `root_agent/slide_insertion_agent/`

Files to create:

- `__init__.py` - Export main orchestrator
- `insertion_orchestrator.py` - Main orchestrator agent
- `insertion_request_parser.py` - Parse user requests for insertion details
- `slide_renumbering_tool.py` - Database tool to renumber slides
- `database_tools.py` - Fetch theme, validate position, save outline
- `content_generator.py` - Generate slide content from topic/type

### Step 2: Create Insertion Request Parser Agent

File: `root_agent/slide_insertion_agent/insertion_request_parser.py`

Purpose: Parse natural language requests like "add a slide about Tesla's challenges between slide 3 and 4"

Returns:

```python
{
  "insert_after_slide": int,  # Insert after this slide number
  "content_provided": str or None,  # Full content if provided
  "topic": str or None,  # Topic if provided
  "slide_type": str or None,  # e.g., "comparison", "timeline", "data"
  "user_instructions": str,  # Any specific instructions
  "error_message": str or None
}
```

### Step 3: Create Database Tools

File: `root_agent/slide_insertion_agent/database_tools.py`

Tools needed:

1. `validate_insertion_position_tool(p_id, insert_after_slide)` - Verify position is valid
2. `fetch_presentation_context_tool(p_id)` - Get global_theme, presentation_metadata, total_slides
3. `fetch_neighboring_slides_tool(p_id, slide_number)` - Get slides before/after insertion point for context
4. `save_presentation_outline_tool(p_id, outline)` - Save new outline after insertion

### Step 4: Create Slide Renumbering Tool

File: `root_agent/slide_insertion_agent/slide_renumbering_tool.py`

Function: `renumber_slides_after_insertion(p_id, insert_after_slide)`

Logic:

```python
# Update all slides where slide_number > insert_after_slide
# Increment both slide_number and slide_index by 1
db.slide_html.update_many(
    {"p_id": p_id, "slide_number": {"$gt": insert_after_slide}},
    {"$inc": {"slide_number": 1, "slide_index": 1}}
)
```

### Step 5: Create Content Generator Sub-Agent

File: `root_agent/slide_insertion_agent/content_generator.py`

Purpose: Generate slide content when user only provides topic or type

Integration with existing tools:

- Use `qdrant_retrieval_tool` to fetch relevant stored research from presentation context
- Use research/browser tools if needed for additional information
- Generate slide plan (title, purpose, content_guidance, required_elements)

### Step 6: Create Main Insertion Orchestrator

File: `root_agent/slide_insertion_agent/insertion_orchestrator.py`

Agent workflow:

```
1. Parse insertion request → insertion_request_parser
2. Validate position → validate_insertion_position_tool
3. Fetch context → fetch_presentation_context_tool + fetch_neighboring_slides_tool
4. Generate content (if needed) → content_generator
5. Select template → reuse template_selector_agent from slide_creation_agent
6. Generate slide HTML → reuse enhanced_slide_generator from slide_creation_agent
7. Renumber slides → renumber_slides_after_insertion
8. Insert new slide → save_slide_to_database with correct slide_index
9. Update presentation outline → save_presentation_outline_tool
10. Return success message with new slide details
```

Key parameters:

- `p_id` from session state
- `enhanced_query` from main orchestrator
- Access to global_theme from presentations collection

### Step 7: Integrate with Main Orchestrator

File: `root_agent/agent.py`

Changes needed:

1. Import the new agent:
```python
from root_agent.slide_insertion_agent import slide_insertion_orchestrator
```

2. Add to sub_agents list:
```python
sub_agents=[
    pipeline,                               # name="SlideCreationPipeline"
    multi_slide_modification_orchestrator,  # name="multi_slide_modification_orchestrator"
    slide_insertion_orchestrator            # name="slide_insertion_orchestrator"
]
```

3. Update instruction to include new routing logic:
```python
# Add to intent classification section
- Possible results:
  - `create_presentation`
  - `edit_slide`
  - `insert_slide`  # NEW
  - `other`

# Add routing logic
- If insert_slide:
  - Ensure ctx.session.state contains:
      - p_id = {p_id}
      - enhanced_query
  - Now delegate the work:
    Call: transfer_to_agent(agent_name="slide_insertion_orchestrator")
```


### Step 8: Update Query Classifier

File: `root_agent/sub_agents/query_classifier_agent.py` (if exists) or create if needed

Add detection for insertion requests:

- Keywords: "add", "insert", "between", "after slide X", "before slide Y"
- Return "insert_slide" intent

### Step 9: Create Presentation Outlines Collection

Database: MongoDB collection `presentation_outlines`

Schema:

```javascript
{
  p_id: "abc123",
  user_id: "user123",
  slide_outline: [
    {
      slide_number: 1,
      slide_title: "...",
      slide_purpose: "...",
      suggested_type: "...",
      // ... other plan details
    }
  ],
  total_slides: 10,
  created_at: ISODate(),
  updated_at: ISODate()
}
```

### Step 10: Reuse Existing Components

Components to import and reuse:

1. `template_selector_agent` from `root_agent/slide_creation_agent/sub_agents/template_selector_agent.py`
2. `enhanced_slide_generator` from `root_agent/slide_creation_agent/sub_agents/enhanced_slide_generator.py`
3. `save_slide_to_database` from `root_agent/slide_creation_agent/sub_agents/lightweight_slide_pipeline.py`
4. `qdrant_retrieval_tool` from `qdrant_retrieval_tool.py`

### Step 11: Testing Considerations

Test scenarios:

1. Insert slide with full content provided
2. Insert slide with only topic (triggers research)
3. Insert slide with only type (e.g., "timeline")
4. Insert at beginning (after slide 0)
5. Insert in middle
6. Insert at end
7. Verify renumbering works correctly
8. Verify global_theme is applied
9. Verify outline is saved

## Key Design Decisions

1. **No full plan storage needed**: Only store individual slide plans in `presentation_outlines` collection after insertion
2. **Renumber only slide_number and slide_index**: Other collections remain unchanged
3. **Reuse template selector**: Use existing `template_selector_agent` with global_theme from database
4. **Research integration**: Use `qdrant_retrieval_tool` to fetch context from stored presentation research
5. **Separate orchestrator**: New top-level sub-agent alongside pipeline and modification orchestrator

## Files to Create/Modify

### New Files (7):

1. `root_agent/slide_insertion_agent/__init__.py`
2. `root_agent/slide_insertion_agent/insertion_orchestrator.py`
3. `root_agent/slide_insertion_agent/insertion_request_parser.py`
4. `root_agent/slide_insertion_agent/slide_renumbering_tool.py`
5. `root_agent/slide_insertion_agent/database_tools.py`
6. `root_agent/slide_insertion_agent/content_generator.py`
7. `root_agent/slide_insertion_agent/README.md`

### Modified Files (2):

1. `root_agent/agent.py` - Add new sub-agent and routing logic
2. `root_agent/sub_agents/query_classifier_agent.py` (or create) - Add insert_slide detection

## Dependencies

Required imports:

- google.adk.agents (LlmAgent, AgentTool)
- google.adk.tools (FunctionTool)
- Existing template_selector_agent
- Existing enhanced_slide_generator
- Existing save_slide_to_database function
- qdrant_retrieval_tool for research context
- MongoDB client from db.py