# Slide Insertion Agent

A comprehensive agent system for inserting new slides between existing slides in presentations with automatic renumbering, template selection, and intelligent content generation.

## Overview

The Slide Insertion Agent allows users to add new slides at any position in an existing presentation. It handles:
- Parsing natural language insertion requests
- Validating insertion positions
- Generating content from topics or user-provided content
- Selecting appropriate templates based on context
- Automatically renumbering existing slides
- Updating presentation outlines

## Architecture

```
slide_insertion_orchestrator (Main Agent)
├── insertion_request_parser (Parse user requests)
├── database_tools (Validation & context fetching)
├── content_generator (Generate slide plans)
├── slide_renumbering_tool (Renumber existing slides)
└── save_slide_to_database (Save new slide)
```

## Usage Examples

### Basic Insertion
```
User: "Add a slide about Tesla's challenges between slide 3 and 4"
Result: New slide inserted at position 4, slides 4+ renumbered
```

### Topic-Only Insertion
```
User: "Insert a timeline slide after slide 2"
Result: Generates timeline content using research tools
```

### Content-Provided Insertion
```
User: "Add a slide with this content: [full content]"
Result: Uses provided content directly
```

### End Insertion
```
User: "Add a conclusion slide"
Result: Inserts at the end of presentation
```

## Agent Components

### 1. Insertion Request Parser
**File**: `insertion_request_parser.py`

Parses natural language requests and extracts:
- Insertion position (after which slide)
- Content/topic information
- Slide type (comparison, timeline, data, etc.)
- User instructions

**Output**:
```json
{
  "insert_after_slide": 3,
  "content_provided": null,
  "topic": "Tesla's challenges",
  "slide_type": "comparison",
  "user_instructions": "Add a slide about Tesla's challenges between slide 3 and 4",
  "error_message": null
}
```

### 2. Database Tools
**File**: `database_tools.py`

Provides database operations:
- `validate_insertion_position()` - Validates insertion position
- `fetch_presentation_context()` - Gets global theme and metadata
- `fetch_neighboring_slides()` - Gets context from adjacent slides
- `save_presentation_outline()` - Updates presentation outline

### 3. Slide Renumbering Tool
**File**: `slide_renumbering_tool.py`

Handles automatic renumbering:
- Increments `slide_number` and `slide_index` for slides after insertion point
- Ensures atomic operation (all or nothing)
- Provides debugging tools for slide numbering

### 4. Content Generator
**File**: `content_generator.py`

Generates slide content when user provides only topic/type:
- Creates slide plans based on topic and slide type
- Integrates with research tools for content enhancement
- Considers presentation context and neighboring slides

### 5. Main Orchestrator
**File**: `insertion_orchestrator.py`

Coordinates the entire insertion process:
1. Parse user request
2. Validate insertion position
3. Fetch presentation context
4. Generate content (if needed)
5. Renumber existing slides
6. Select template and generate slide
7. Save new slide to database
8. Update presentation outline
9. Return success confirmation

## Database Schema

### New Collection: `presentation_outlines`
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

### Updated Collections
- `slide_html` - New slides inserted with correct `slide_index` and `slide_number`
- `presentations` - Global theme and metadata accessed for new slides

## Integration

### Main Agent Integration
The slide insertion orchestrator is integrated into the main `SlideOrchestrationAgent`:

```python
# In root_agent/agent.py
sub_agents=[
    pipeline,                               # name="SlideCreationPipeline"
    multi_slide_modification_orchestrator,  # name="multi_slide_modification_orchestrator"
    slide_insertion_orchestrator            # name="slide_insertion_orchestrator"
]
```

### Query Classification
The query classifier now recognizes insertion requests:
- Keywords: "add", "insert", "between", "after slide X", "before slide Y"
- Returns `insert_slide` intent for routing

## Error Handling

The system handles various error scenarios:
- Invalid insertion positions
- Missing presentation context
- Database connection issues
- Content generation failures
- Template selection errors

All errors are returned with clear messages to guide user actions.

## Dependencies

### Reused Components
- `template_selector_agent` from slide creation
- `enhanced_slide_generator` from slide creation
- `save_slide_to_database` function
- `qdrant_retrieval_tool` for research context

### New Dependencies
- MongoDB client for database operations
- BeautifulSoup for HTML parsing
- JSON for data serialization

## Testing Scenarios

1. **Insert with full content** - User provides complete slide content
2. **Insert with topic only** - System generates content using research
3. **Insert with type only** - System determines content based on slide type
4. **Insert at beginning** - After slide 0 (becomes slide 1)
5. **Insert in middle** - Between existing slides
6. **Insert at end** - After last slide
7. **Verify renumbering** - All subsequent slides properly renumbered
8. **Verify theme application** - New slide uses presentation's global theme
9. **Verify outline update** - Presentation outline reflects new slide

## Future Enhancements

- Bulk slide insertion
- Slide reordering (move existing slides)
- Template consistency checking
- Advanced content research integration
- Slide dependency management
