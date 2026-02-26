# Agent Catalog - Complete Agent Reference

This document catalogs all agents in the system with their purposes, inputs, outputs, and tools.

---

## 📋 Agent Categories

1. **Orchestration Agents** - Coordinate workflows
2. **Research Agents** - Gather information
3. **Planning Agents** - Structure presentations
4. **Generation Agents** - Create slide content
5. **Quality Agents** - Improve output
6. **Utility Agents** - Support functions

---

## 1️⃣ ORCHESTRATION AGENTS

### **SlideOrchestrationAgent**
- **File**: `root_agent/agent.py`
- **Purpose**: Main entry point, coordinates entire presentation generation
- **Type**: `SequentialAgent`
- **Input**: User query, file URLs, session state
- **Output**: Complete presentation with HTML slides
- **Sub-agents**:
  - `VibeEstimatorAgent`
  - `PresentationSpecExtractorAgent`
  - `ContentSynthesizerAgent` or `LightweightSlideGenerationAgent`
- **Key Logic**: Routes between full content synthesis vs lightweight planning

---

## 2️⃣ RESEARCH AGENTS

### **KeywordResearchAgent**
- **File**: `root_agent/slide_creation_agent/keyword_research_agent/agent.py`
- **Purpose**: Generate comprehensive search queries
- **Type**: `SequentialAgent`
- **Input**: Presentation specs, file context
- **Output**: JSON array of 8-10 search queries
- **Sub-agents**:
  - `keyword_agent` - Extract keywords, topics, goals
  - `search_query_agent` - Generate targeted search queries
- **Tools**: None (uses sub-agents)
- **Example Output**:
```json
{
  "search_queries": [
    "AI in healthcare current applications 2025",
    "machine learning medical diagnosis statistics",
    ...
  ]
}
```

### **keyword_agent**
- **File**: `root_agent/slide_creation_agent/keyword_research_agent/keyword_research/agent.py`
- **Purpose**: Extract keywords, topics, and goals
- **Type**: `LlmAgent`
- **Model**: `gemini-2.5-flash`
- **Input**: Presentation specs
- **Output**: JSON with keywords (8-12), topics (6-8), goals (4-6)

### **search_query_agent**
- **File**: `root_agent/slide_creation_agent/keyword_research_agent/search_query/agent.py`
- **Purpose**: Generate targeted search queries from keywords
- **Type**: `LlmAgent`
- **Model**: `gemini-2.5-flash`
- **Input**: Keywords, topics, goals
- **Output**: JSON array of 10 search queries
- **Query Types**: Broad overview, specific data, comparative, case studies, trends, challenges, expert opinions, technical, industry, regional

### **BrowserAgent**
- **File**: `root_agent/slide_creation_agent/slide_creation_agent.py`
- **Purpose**: Orchestrate parallel web searches and store in Qdrant
- **Type**: `CustomAgent` (custom `_run_async_impl`)
- **Input**: Search queries
- **Output**: Research results + Qdrant storage
- **Sub-agents**: Multiple `browser_worker_{idx}` via `ParallelAgent`
- **Key Features**:
  - Extracts grounding metadata (sources)
  - Stores research in Qdrant with embeddings
  - Yields sources to SSE (title + URL)
  - Tracks keyword associations

### **browser_worker_{idx}**
- **File**: `root_agent/slide_creation_agent/browser_agent/make_browser_worker.py`
- **Purpose**: Execute individual web search
- **Type**: `LlmAgent`
- **Model**: `gemini-2.5-flash`
- **Tools**: `brave_search_tool`, `content_scrapper`
- **Input**: Search query
- **Output**: Comprehensive research text
- **Instruction**: Search for data and information, NOT create slides

### **comprehensive_research_agent**
- **File**: `root_agent/slide_creation_agent/browser_agent/agent.py`
- **Purpose**: Deep research on a topic (alternative to browser workers)
- **Type**: `LlmAgent`
- **Model**: `gemini-2.5-flash`
- **Tools**: `brave_search_tool`, `content_scrapper`, `google_search`
- **Instruction**: Gather and summarize information, NOT create presentations

---

## 3️⃣ PLANNING AGENTS

### **VibeEstimatorAgent**
- **File**: `root_agent/slide_creation_agent/sub_agents/vibe_estimator_agent.py`
- **Purpose**: Infer complete presentation specifications from user query
- **Type**: `LlmAgent`
- **Model**: `gemini-2.5-flash`
- **Tools**: `google_search`
- **Input**: User query, file context
- **Output**: JSON with `presentation_type`, `audience_type`, `tone`, `slide_count`, `color_theme`
- **Google Search Usage**:
  - Brand color lookup (e.g., "Tesla brand colors")
  - Audience type clarification
  - Tone suggestions

### **PresentationSpecExtractorAgent**
- **File**: `root_agent/slide_creation_agent/sub_agents/presentation_spec_extractor_agent.py`
- **Purpose**: Extract structured specifications from user query
- **Type**: `LlmAgent`
- **Model**: `gemini-2.5-flash`
- **Tools**: `google_search`
- **Input**: User query, file context, vibe estimation
- **Output**: JSON with detailed specs (topic, goals, audience, constraints, etc.)
- **Google Search Usage**:
  - Topic-specific presentation type detection
  - Industry-standard format lookup

### **LightweightPlanningAgent**
- **File**: `root_agent/slide_creation_agent/sub_agents/lightweight_planning_agent.py`
- **Purpose**: Generate slide outline with search queries (NOT full content)
- **Type**: `LlmAgent`
- **Model**: `gemini-2.5-flash`
- **Tools**: None
- **Input**: Presentation specs, file context
- **Output**: JSON array of slide outlines
- **Each Slide Outline**:
  - `slide_number`
  - `slide_purpose`
  - `slide_title`
  - `suggested_type`
  - `search_query` (for Qdrant retrieval)
  - `content_guidance`
  - `required_elements`
  - `fallback_keywords`
- **Critical**: Returns ONLY raw JSON, no markdown or extra text

---

## 4️⃣ GENERATION AGENTS

### **EnhancedSlideGenerator**
- **File**: `root_agent/slide_creation_agent/sub_agents/enhanced_slide_generator.py`
- **Purpose**: Generate individual HTML slide with Qdrant retrieval
- **Type**: `LlmAgent`
- **Model**: `gemini-2.5-flash`
- **Tools**: `qdrant_retrieval_tool`, `search_images_tool`
- **Input**: Slide outline (from planning), presentation specs
- **Output**: Complete HTML with inline CSS
- **Process**:
  1. Retrieve relevant research from Qdrant
  2. Analyze research and extract key info
  3. Optionally search for images
  4. Generate HTML slide
  5. Apply theme colors and styles
- **HTML Requirements**:
  - 1280x720px dimensions
  - Inline CSS only
  - Responsive font sizing
  - Theme colors applied

### **SlideGeneratorAgent** (Legacy)
- **File**: `root_agent/slide_creation_agent/sub_agents/slide_generator_agent.py`
- **Purpose**: Generate slide from complete specifications
- **Type**: `LlmAgent`
- **Model**: `gemini-2.0-flash`
- **Input**: Full slide specification (content, visual, layout)
- **Output**: HTML slide
- **Status**: Used in full planning approach (alternative to lightweight)

---

## 5️⃣ QUALITY AGENTS

### **SlideQualityVerifierAgent**
- **File**: `root_agent/slide_creation_agent/sub_agents/slide_quality_verifier.py`
- **Purpose**: Analyze and enhance generated HTML slides
- **Type**: `LlmAgent`
- **Model**: `gemini-2.5-flash`
- **Tools**: None
- **Input**: Generated HTML slide
- **Output**: Enhanced HTML with fixes
- **Analysis Areas**:
  - **Content Quality**: Accuracy, completeness, clarity
  - **Visual Quality**: Layout, readability, color contrast
  - **Technical Issues**: Text overflow, missing elements, broken CSS
  - **Accessibility**: Alt text, semantic HTML, font sizes
- **Key Fixes**:
  - Text overflow → `clamp()`, `overflow: hidden`, `word-wrap`
  - Long text → Responsive font sizing
  - Layout issues → Flexbox/grid adjustments
- **Requirements**: All fixes within 1280x720px constraint

### **ContentRefinerAgent**
- **File**: `root_agent/slide_creation_agent/sub_agents/content_refiner_agent.py`
- **Purpose**: Refine and improve content quality
- **Type**: `LlmAgent`
- **Input**: Raw content
- **Output**: Refined content
- **Status**: Part of full content synthesis pipeline

---

## 6️⃣ UTILITY AGENTS

### **query_enhancer_agent**
- **File**: `root_agent/sub_agents.py`
- **Purpose**: Enhance vague or unclear user queries
- **Type**: `LlmAgent`
- **Model**: `gemini-2.5-flash`
- **Tools**: `google_search`
- **Input**: Raw user query
- **Output**: Enhanced, specific query
- **Google Search Usage**:
  - Specific brands/products
  - Technical topics
  - People/events
  - Adding current context

---

## 🔄 PIPELINE AGENTS

### **LightweightSlideGenerationAgent**
- **File**: `root_agent/slide_creation_agent/sub_agents/lightweight_slide_pipeline.py`
- **Purpose**: Orchestrate lightweight planning → parallel generation
- **Type**: Custom class (not ADK agent)
- **Process**:
  1. Run `LightweightPlanningAgent`
  2. Parse JSON (4 fallback methods)
  3. Create `ParallelAgent` with `EnhancedSlideGenerator` per slide
  4. Track slide generation progress
  5. Detect missing slides
- **Output**: Array of HTML slides
- **Key Features**:
  - Robust JSON parsing
  - Progress tracking
  - Missing slide detection

### **EnhancedSlidePipeline**
- **File**: `root_agent/slide_creation_agent/sub_agents/enhanced_slide_pipeline.py`
- **Purpose**: Add quality verification to generation pipeline
- **Type**: Custom class
- **Process**:
  1. Run `LightweightSlideGenerationAgent`
  2. For each slide → `SlideQualityVerifierAgent`
  3. Return enhanced slides
- **Output**: Array of quality-verified HTML slides

### **CustomContentSynthesizerAgent** (Legacy)
- **File**: `root_agent/slide_creation_agent/slide_creation_agent.py`
- **Purpose**: Full content synthesis (research + planning + generation)
- **Type**: `SequentialAgent`
- **Status**: Alternative to lightweight approach

---

## 🛠️ TOOL REFERENCE

### **brave_search_tool**
- **Purpose**: Web search via Brave API
- **Input**: Query string
- **Output**: Search results JSON

### **google_search**
- **Purpose**: Web search via Google API
- **Input**: Query string
- **Output**: Search results

### **content_scrapper**
- **Purpose**: Extract content from URL
- **Input**: URL
- **Output**: Cleaned text content

### **qdrant_retrieval_tool**
- **File**: `qdrant_retrieval_tool.py`
- **Purpose**: Semantic search from Qdrant
- **Input**: Query, user_id, p_id
- **Output**: Top 10 relevant text chunks
- **Key Feature**: Filters by user_id and p_id

### **search_images_tool**
- **File**: `image_search_tool.py`
- **Purpose**: Fetch relevant image URLs
- **Input**: Query
- **Output**: Array of image URLs
- **API**: `http://192.168.68.144:8000/search_images`

---

## 📊 Agent Selection Guide

| Need | Agent | Why |
|------|-------|-----|
| Full orchestration | `SlideOrchestrationAgent` | Main entry point |
| Generate search queries | `KeywordResearchAgent` | 8-10 optimized queries |
| Web research | `BrowserAgent` | Parallel search + Qdrant storage |
| Infer presentation specs | `VibeEstimatorAgent` | Auto-detect from query |
| Extract requirements | `PresentationSpecExtractorAgent` | Structured specs |
| Create slide outline | `LightweightPlanningAgent` | Lightweight approach |
| Generate slide HTML | `EnhancedSlideGenerator` | With Qdrant retrieval |
| Fix rendering issues | `SlideQualityVerifierAgent` | Text overflow, CSS fixes |
| Enhance query | `query_enhancer_agent` | Make queries specific |

---

## 🔄 Typical Agent Flow

```
SlideOrchestrationAgent
  ├─ VibeEstimatorAgent
  ├─ PresentationSpecExtractorAgent
  ├─ (Optional) query_enhancer_agent
  └─ LightweightSlideGenerationAgent
      ├─ KeywordResearchAgent
      │   ├─ keyword_agent
      │   └─ search_query_agent
      ├─ BrowserAgent
      │   └─ ParallelAgent
      │       ├─ browser_worker_0
      │       ├─ browser_worker_1
      │       └─ browser_worker_N
      └─ EnhancedSlidePipeline
          ├─ LightweightPlanningAgent
          ├─ ParallelAgent
          │   ├─ EnhancedSlideGenerator_0
          │   ├─ EnhancedSlideGenerator_1
          │   └─ EnhancedSlideGenerator_N
          └─ SlideQualityVerifierAgent (per slide)
```

---

## 🎯 Agent Best Practices

1. **Always name agents uniquely** - Helps with event attribution
2. **Keep instructions clear** - Specify what agent should NOT do
3. **Use appropriate models** - Flash for speed, Pro for quality
4. **Limit tool access** - Only give tools the agent needs
5. **Validate output format** - Especially for JSON responses
6. **Handle errors gracefully** - Fallback mechanisms
7. **Log agent execution** - Track performance and issues

---

**Related Documents**:
- `CODE_PATTERNS.md` - How to create new agents
- `TROUBLESHOOTING.md` - Common agent issues

