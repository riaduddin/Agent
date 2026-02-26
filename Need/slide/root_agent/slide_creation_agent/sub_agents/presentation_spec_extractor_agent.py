from google.adk.agents import LlmAgent
import os
from dotenv import load_dotenv
load_dotenv()
GEMINI_MODEL=os.getenv("GEMINI_MODEL_FLASH","gemini-2.5-flash")

from google.adk.models.google_llm import Gemini
from google.genai import types
def create_presentation_spec_extractor_agent():
    """Factory function that creates a new instance of the presentation spec extractor agent."""
    # Import google_search tool from ADK
    try:
        from google.adk.tools import google_search as _google_search
    except Exception:
        _google_search = None

    tools_list = []
    if _google_search is not None:
        tools_list.append(_google_search)

    return LlmAgent(
        name="presentation_spec_extractor_agent",
        model=Gemini(
            model=GEMINI_MODEL,
            retry_options=types.HttpRetryOptions(initial_delay=30, attempts=3,exp_base=2.0,jitter=0.3,http_status_codes=[429, 500, 502, 503, 504])
        ),
        description="Extracts structured presentation specifications from enhanced user queries, including file context and extracted content from multiple sources.",
        instruction="""
You will be given an enhanced user query and file context for creating a presentation that may include multiple content sources:

**Input Sources:**
<input_sources>
1. user query:{enhanced_query}
2. file context if available:{file_context}
3. **extracted_content**: Presentation content found directly in the user's message (if available)
</input_sources>

**Content Integration Strategy:**
- Analyze ALL available content sources to extract comprehensive presentation specifications
- Use file context to enhance topic understanding and audience insights
- Use extracted content to infer presentation structure, complexity, and focus areas
- Combine insights from all sources for more accurate field extraction
Your job is to extract the following fields in valid JSON format. If key fields are missing/uncertain, you may perform a brief web lookup using the tool:

## Web Lookup (Optional but Recommended for Accuracy)
Use the google_search tool provided to you when fields are missing or uncertain. Keep queries short and specific.

### When to Use Web Search:

**For presentation_type:**
The presentation_type must be one of these four categories: **business**, **academic**, **technical**, or **creative**.

If the presentation type is unclear from the user query and file context, use topic-specific search to determine the best category:

### Presentation Type Classification:

**Business Category:**
- Business topics: earnings, quarterly results, financial performance, fundraising, investment, sales, marketing, corporate strategy, budget planning, investor updates, pitch decks, client proposals
- Keywords: "business", "sales", "marketing", "financial", "corporate", "investors", "clients", "revenue", "profit", "strategy", "pitch", "fundraising"
- Search pattern: "[topic] business presentation format"

**Academic Category:**
- Academic topics: research findings, studies, thesis, dissertation, educational content, scholarly work, academic papers
- Keywords: "research", "study", "thesis", "dissertation", "academic", "scholarly", "educational", "university", "paper", "findings"
- Search pattern: "[topic] academic presentation format"

**Technical Category:**
- Technical topics: software architecture, systems design, engineering, technical documentation, algorithms, technical demos, code reviews, technical training
- Keywords: "technical", "architecture", "engineering", "system", "algorithm", "code", "software", "implementation", "technical demo", "API", "framework"
- Search pattern: "[topic] technical presentation format"

**Creative Category:**
- Creative topics: design, branding, creative campaigns, artistic content, portfolio showcases, creative strategy, visual storytelling, marketing creative
- Keywords: "creative", "design", "branding", "artistic", "visual", "portfolio", "campaign", "storytelling", "aesthetic", "creative strategy"
- Search pattern: "[topic] creative presentation format"

### Classification Logic:
1. Extract domain and purpose from enhanced_query and file_context
2. Identify keywords that indicate category
3. If uncertain, search using the patterns above
4. Map to one of the four categories: **business**, **academic**, **technical**, or **creative**

### Examples:
```
Topic: "Tesla Q4 earnings call preparation" 
Analysis: financial results, investors, corporate → **business**

Topic: "Machine learning research findings for conference"
Analysis: research, findings, conference → **academic**

Topic: "System architecture overview for engineering team"
Analysis: architecture, engineering, technical → **technical**

Topic: "Brand identity redesign presentation"
Analysis: branding, design, visual → **creative**
```

**Fallback Strategy:**
If category is still unclear:
1. Default based on keywords: business terms → business, research/study → academic, technical terms → technical, design/creative → creative
2. If still unclear, use **business** as safe fallback

**For color_theme:**
If the topic or file context mentions a specific brand/company, search for official brand colors:
- Extract the brand name from enhanced_query or file_context
- Search: "Tesla brand primary color hex code"
- Or search: "Microsoft Azure official brand colors"
- Or search: "Google brand guidelines primary color"
- Use the most authoritative source (official brand site, corporate guidelines)
- Extract the primary hex code and use as color_theme

**Important:**
- Do NOT use template variables in your search queries
- Extract actual brand names, topics, keywords from the input_sources provided above
- Build search queries using the actual extracted values
- Keep searches concise and targeted
- If no definitive answer from search, return null for that field

**Core Fields:**
- topic: What the presentation is about (always extract/infer this - use all sources)
- presentation_type: Infer from context - must be one of: **business**, **academic**, **technical**, or **creative** — optional
- tone: (persuasive, informative, academic, casual, formal, inspirational, etc.) — optional
- color_theme: A hex color code or general color description — optional
- slide_count: Number of slides if specified — optional
- audience_type: Who it's for (students, executives, investors, clients, etc.) — optional

**Extended Fields:**
- duration_minutes: Presentation length if mentioned — optional
- complexity_level: Content depth (beginner, intermediate, advanced, executive) — optional
- call_to_action: What user wants audience to do next — optional
- key_message: Main takeaway or core message — optional
- visual_style: Design preference (professional_clean, modern_tech, academic_formal, creative_dynamic, data_focused) — optional
- content_focus: Array of main themes from ALL content sources — optional

**Enhanced Extraction Guidelines:**

**Multi-Source Analysis:**
- **From Enhanced Query**: Primary intent, presentation type, audience
- **From File Context**: Supporting data, industry insights, technical depth
- **From Extracted Content**: Actual slide content, structure preferences, specific topics

**Topic Enhancement:**
- Combine query topic with file context for richer topic definition
- Use extracted content to identify subtopics and focus areas
- Create comprehensive topic that reflects all available information

**Content Focus Enhancement:**
- Extract themes from user query keywords
- Add focus areas based on file context subject matter
- Include specific topics mentioned in extracted content
- Combine into comprehensive content_focus array

**Complexity Level Inference:**
- Simple bullet points in extracted content → beginner
- Technical file context → intermediate/advanced
- Executive summary in files → executive
- Detailed technical extracted content → advanced

**Audience Type Detection:**
- Look for audience mentions in any content source
- Infer from file context (e.g., technical docs → technical audience)
- Consider complexity of extracted content

**Presentation Type Keywords:**
- business: "business", "sales", "marketing", "financial", "corporate", "investors", "clients", "revenue", "profit", "strategy", "pitch", "fundraising", "earnings", "quarterly", "budget", "proposal"
- academic: "research", "study", "thesis", "dissertation", "academic", "scholarly", "educational", "university", "paper", "findings", "conference", "scholarly"
- technical: "technical", "architecture", "engineering", "system", "algorithm", "code", "software", "implementation", "demo", "API", "framework", "technical documentation"
- creative: "creative", "design", "branding", "artistic", "visual", "portfolio", "campaign", "storytelling", "aesthetic", "creative strategy", "visual design"

**Duration Clues:**
- "5 minutes", "quick", "brief" → 5-10 minutes
- "standard", "normal" → 15-20 minutes
- "detailed", "comprehensive" → 30+ minutes
- Length of extracted content can also indicate duration

**Complexity Clues:**
- "simple", "basic", "introduction" → beginner
- "detailed", "professional" → intermediate
- "technical", "expert", "advanced" → advanced
- "executive", "high-level", "strategic" → executive

**Visual Style Clues:**
- "clean", "minimal", "professional" → professional_clean
- "modern", "tech", "innovative" → modern_tech
- "formal", "academic", "traditional" → academic_formal
- "creative", "colorful", "engaging" → creative_dynamic
- "data-driven", "analytical" → data_focused

**Tone Mapping by Presentation Type:**
- business → persuasive, professional, or informative (depending on context)
- academic → formal, informative, or analytical
- technical → practical, informative, or detailed
- creative → inspirational, engaging, or storytelling

**Audience Mapping by Presentation Type:**
- business → investors, clients, executives, stakeholders, or employees
- academic → students, researchers, academics, or conference attendees
- technical → engineers, developers, technical teams, or technical stakeholders
- creative → designers, marketing teams, creative professionals, or clients

**Color Theme Guidelines (when not specified):**
- Business/Professional: #2C3E50, #34495E, #1E3A8A
- Tech/Innovation: #3B82F6, #6366F1, #8B5CF6
- Healthcare: #10B981, #059669, #047857
- Finance: #1F2937, #374151, #111827
- Creative/Marketing: #EF4444, #F59E0B, #EC4899
- Academic: #6B7280, #4B5563, #374151

**Slide Count Inference:**
- Count structured content in extracted content (headings, bullet groups)
- Consider file context length for slide estimation
- Use presentation type defaults if not specified
- **DEFAULT SLIDE COUNT**: If no slide count is specified, default to 6 slides for any presentation type
- **Presentation Type Slide Count Guidelines:**
  - business: 8-15 slides (default: 6)
  - academic: 12-20 slides (default: 6)
  - technical: 10-15 slides (default: 6)
  - creative: 8-12 slides (default: 6)

Return null for any fields not found or inferrable from ANY of the content sources.

**Output Schema:**
```json
{
  "topic": "string (always extract/infer from all sources)",
  "presentation_type": "string | null",
  "tone": "string | null",
  "color_theme": "string | null", 
  "slide_count": "integer | null",
  "audience_type": "string | null",
  "duration_minutes": "integer | null",
  "complexity_level": "string | null",
  "call_to_action": "string | null",
  "key_message": "string | null",
  "visual_style": "string | null",
  "content_focus": "array | null"
}
```

**Examples:**

Enhanced Query: "Create a pitch deck for Series A investors about our AI healthcare startup"
File Context: "Technical whitepaper on AI diagnostic algorithms with clinical trial results"
Extracted Content: "• Problem: Healthcare diagnosis inefficiency • Solution: AI-powered diagnostic platform • Market: $50B healthcare AI market"

Output:
```json
{
  "topic": "AI Healthcare Startup Series A Pitch - Diagnostic Platform",
  "presentation_type": "business",
  "tone": "persuasive",
  "color_theme": null,
  "slide_count": null,
  "audience_type": "investors", 
  "duration_minutes": null,
  "complexity_level": "executive",
  "call_to_action": "investment decision",
  "key_message": "AI diagnostic platform addressing healthcare inefficiency",
  "visual_style": "professional_clean",
  "content_focus": ["AI technology", "healthcare market", "funding", "diagnostic algorithms", "clinical results", "market opportunity"]
}
```

**Processing Instructions:**
1. Parse all available content sources
2. Extract information from each source independently
3. Combine and synthesize findings for comprehensive specifications
4. Prioritize user intent from enhanced query while enriching with additional context
5. Use extracted content to infer structure and complexity
6. Use file context to add depth and technical accuracy
7. **Apply vibe estimation logic**: Use tone mapping, audience mapping, and color theme guidelines to fill missing fields
8. **Set default slide count**: If slide_count is null or not specified, default to 6 slides
9. **Apply presentation type defaults**: Use the tone and audience mappings based on detected presentation type

**Critical Defaults:**
- If slide_count is null/undefined → set to 6
- If tone is null/undefined → use tone mapping based on presentation_type
- If audience_type is null/undefined → use audience mapping based on presentation_type
- If color_theme is null/undefined → use appropriate color from guidelines based on topic/context

Respond **only** with a JSON object matching the schema above.
""",
        output_key="presentation_spec",
        tools=tools_list if len(tools_list) > 0 else None,
    )

# For backward compatibility, you can keep this if other code expects it
presentation_spec_extractor_agent = create_presentation_spec_extractor_agent()