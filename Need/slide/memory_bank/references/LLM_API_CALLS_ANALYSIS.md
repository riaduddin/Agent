# 🔢 LLM API Calls Analysis - Complete Breakdown

This document provides a detailed analysis of how many LLM API calls are made during a single presentation generation run, broken down by agent and slide count.

---

## 📊 Pipeline Overview

The presentation generation pipeline follows this sequential flow:

```
1. Presentation Spec Extractor Agent
2. Vibe Estimator Agent  
3. Keyword Research Agent
   ├─ keyword_agent
   └─ search_query_agent
4. Browser Agent (Parallel)
   ├─ browser_worker_0
   ├─ browser_worker_1
   └─ ... (8-10 workers typically)
5. Lightweight Planning Agent
6. Lightweight Slide Generation Agent
   For each slide:
   ├─ template_selector_agent
   └─ enhanced_slide_generator
```

---

## 🧮 LLM API Call Count Formula

### Base Pipeline (Fixed) = **5-7 calls**

| Agent | LLM Calls | Purpose |
|-------|-----------|---------|
| **PresentationSpecExtractorAgent** | 1 | Extract structured specs from user query |
| **VibeEstimatorAgent** | 1 | Infer presentation specifications |
| **KeywordResearchAgent** | 2 | Generate search queries |
| ├─ keyword_agent | 1 | Extract keywords, topics, goals |
| └─ search_query_agent | 1 | Generate targeted search queries |
| **BrowserAgent** | 8-10 | Web research (one per search query) |
| ├─ browser_worker_0 | 1 | |
| ├─ browser_worker_1 | 1 | |
| └─ ... | 1 each | |
| **LightweightPlanningAgent** | 1 | Generate slide outline |

**Base Total:** `5 + (8 to 10) = 13-15 LLM calls`

### Per-Slide Pipeline = **2 calls per slide**

For each slide in the presentation:

| Agent | LLM Calls | Purpose |
|-------|-----------|---------|
| **template_selector_agent** | 1 | Select best HTML template |
| **enhanced_slide_generator** | 1 | Generate HTML slide with content |

**Per Slide:** `2 LLM calls`

---

## 📈 Total LLM API Calls by Slide Count

### Formula

```
Total LLM Calls = Base Pipeline + (Slides × 2)
Total LLM Calls = 13-15 + (N × 2)

Where N = number of slides
```

### Detailed Breakdown Table

| Slides | Base Calls | Slide Calls | **Total LLM Calls** | Cost Estimate* |
|--------|------------|-------------|---------------------|----------------|
| **5** | 13-15 | 10 | **23-25** | $0.23-$0.25 |
| **7** | 13-15 | 14 | **27-29** | $0.27-$0.29 |
| **10** | 13-15 | 20 | **33-35** | $0.33-$0.35 |
| **12** | 13-15 | 24 | **37-39** | $0.37-$0.39 |
| **15** | 13-15 | 30 | **43-45** | $0.43-$0.45 |
| **20** | 13-15 | 40 | **53-55** | $0.53-$0.55 |
| **25** | 13-15 | 50 | **63-65** | $0.63-$0.65 |
| **30** | 13-15 | 60 | **73-75** | $0.73-$0.75 |

*Cost estimate based on $0.01 per LLM call (approximate, varies by model and token usage)

---

## 🎯 Minimum API Calls (Most Efficient Case)

### Absolute Minimum for Slide Generation

If you wanted to generate slides with the **minimum possible** LLM calls, bypassing research and planning:

| Agent | LLM Calls | Required? |
|-------|-----------|-----------|
| Presentation Spec Extraction | 1 | ✅ Required |
| Planning/Outline | 1 | ✅ Required |
| **Per Slide Generation** | 1 | ✅ Required |

**Absolute Minimum Formula:**
```
Minimum = 2 + (N × 1)
```

**Example for 10 slides:** `2 + 10 = 12 LLM calls`

**However, this would produce:**
- ❌ No research data (generic content)
- ❌ No template selection (random styling)
- ❌ No vibe estimation (default theme)
- ❌ Poor quality, generic slides

---

## 📊 Current System Analysis

### Why Current Approach Uses More Calls

The current pipeline uses **13-15 base calls + 2N per slide** because:

1. **Quality over Quantity**: Multiple research calls ensure high-quality, data-driven content
2. **Template Selection**: Each slide gets a carefully selected template (1 call per slide)
3. **Enhanced Generation**: Each slide is generated with research context (1 call per slide)
4. **Professional Output**: Results in presentation-ready slides with real data

### Trade-off Analysis

| Approach | LLM Calls (10 slides) | Quality | Time |
|----------|----------------------|---------|------|
| **Current (Full)** | 33-35 | ⭐⭐⭐⭐⭐ | 60-90s |
| **Minimal Research** | 20-22 | ⭐⭐⭐ | 30-45s |
| **No Research** | 12 | ⭐⭐ | 15-20s |
| **Bare Minimum** | 12 | ⭐ | 10-15s |

---

## 🔍 Detailed Agent-by-Agent Breakdown

### Phase 1: Preparation (5 calls)

```
1️⃣ PresentationSpecExtractorAgent (1 call)
   ├─ Input: User query + file context
   ├─ Output: Structured specs (topic, goals, audience, etc.)
   └─ Model: gemini-2.5-flash

2️⃣ VibeEstimatorAgent (1 call)  
   ├─ Input: User query + specs
   ├─ Output: presentation_type, tone, slide_count, color_theme
   └─ Model: gemini-2.5-flash

3️⃣ KeywordResearchAgent (2 calls total)
   ├─ keyword_agent (1 call)
   │  ├─ Input: Presentation specs
   │  ├─ Output: Keywords (8-12), topics (6-8), goals (4-6)
   │  └─ Model: gemini-2.5-flash
   │
   └─ search_query_agent (1 call)
      ├─ Input: Keywords, topics, goals
      ├─ Output: 10 targeted search queries
      └─ Model: gemini-2.5-flash

4️⃣ LightweightPlanningAgent (1 call)
   ├─ Input: Specs + research context
   ├─ Output: Slide outline with search queries
   └─ Model: gemini-2.5-flash
```

### Phase 2: Research (8-10 calls)

```
5️⃣ BrowserAgent (8-10 calls in parallel)
   ├─ browser_worker_0 (1 call) - Search query 1
   ├─ browser_worker_1 (1 call) - Search query 2
   ├─ browser_worker_2 (1 call) - Search query 3
   ├─ browser_worker_3 (1 call) - Search query 4
   ├─ browser_worker_4 (1 call) - Search query 5
   ├─ browser_worker_5 (1 call) - Search query 6
   ├─ browser_worker_6 (1 call) - Search query 7
   ├─ browser_worker_7 (1 call) - Search query 8
   ├─ browser_worker_8 (1 call) - Search query 9 (optional)
   └─ browser_worker_9 (1 call) - Search query 10 (optional)
   
   Each worker:
   ├─ Tools: brave_search_tool, content_scrapper
   ├─ Output: Comprehensive research text
   └─ Model: gemini-2.5-flash
```

### Phase 3: Slide Generation (2N calls, where N = slide count)

```
6️⃣ LightweightSlideGenerationAgent (2 calls per slide)
   
   For each slide:
   ├─ template_selector_agent (1 call)
   │  ├─ Input: Slide outline + all templates
   │  ├─ Output: Best template selection + reasoning
   │  └─ Model: gemini-2.5-flash
   │
   └─ enhanced_slide_generator (1 call)
      ├─ Input: Slide outline + selected template + theme
      ├─ Tools: qdrant_retrieval_tool, search_images_tool
      ├─ Output: Complete HTML slide
      └─ Model: gemini-2.5-flash
```

---

## 💰 Cost Analysis

### By Model (Gemini 2.5 Flash Pricing)

Assuming Gemini 2.5 Flash pricing:
- Input: $0.000125 per 1K tokens
- Output: $0.0005 per 1K tokens

### Average Token Usage Per Agent

| Agent Type | Avg Input Tokens | Avg Output Tokens | Cost per Call |
|------------|-----------------|-------------------|---------------|
| Spec Extractor | 2,000 | 1,000 | $0.0008 |
| Vibe Estimator | 1,500 | 800 | $0.0006 |
| Keyword Agent | 1,000 | 500 | $0.0004 |
| Search Query | 1,000 | 300 | $0.0003 |
| Browser Worker | 3,000 | 2,000 | $0.0014 |
| Planning Agent | 4,000 | 3,000 | $0.0020 |
| Template Selector | 2,000 | 500 | $0.0005 |
| Slide Generator | 3,500 | 4,000 | $0.0024 |

### Total Cost Estimate

For a **10-slide presentation** (33-35 LLM calls):

| Phase | Calls | Estimated Cost |
|-------|-------|----------------|
| Preparation (5 calls) | 5 | $0.0041 |
| Research (8 workers) | 8 | $0.0112 |
| Slide Generation (20 calls) | 20 | $0.0580 |
| **Total** | **33** | **~$0.073** |

**Cost per presentation (10 slides): ~$0.07 - $0.10**

### Scaling Cost

| Presentations/Month | Avg Slides | Total LLM Calls | Estimated Cost |
|---------------------|------------|-----------------|----------------|
| 100 | 10 | 3,300-3,500 | $7-10 |
| 500 | 10 | 16,500-17,500 | $35-50 |
| 1,000 | 10 | 33,000-35,000 | $70-100 |
| 5,000 | 10 | 165,000-175,000 | $350-500 |
| 10,000 | 10 | 330,000-350,000 | $700-1,000 |

---

## ⚡ Optimization Opportunities

### Current System

✅ **Already Optimized:**
- Parallel browser workers (8-10 simultaneous calls)
- Batch slide generation (3 slides at a time to avoid API limits)
- Reuse of research data (stored in Qdrant)
- Single planning call for all slides

### Potential Optimizations

1. **Reduce Browser Workers**
   - Current: 8-10 calls
   - Optimized: 5-6 calls
   - Savings: 3-4 calls per presentation
   - Trade-off: Less research data

2. **Skip Template Selection** (Not Recommended)
   - Current: 1 call per slide
   - Optimized: Use default template
   - Savings: N calls (where N = slide count)
   - Trade-off: Lower visual quality

3. **Cache Common Queries**
   - Cache presentation specs for similar queries
   - Reuse keyword research for same topics
   - Potential savings: 2-3 calls per presentation

4. **Batch Multiple Slides in One Call** (Complex)
   - Generate 2-3 slides in single LLM call
   - Potential savings: 50% reduction in slide generation calls
   - Trade-off: More complex parsing, potential quality issues

---

## 📉 Minimum Viable Pipeline

If you absolutely need to **minimize LLM calls**, here's the bare minimum:

### Ultra-Minimal Pipeline (2 + N calls)

```
1. Spec Extraction (1 call) - Required
2. Planning/Outline (1 call) - Required  
3. Slide Generation (N calls) - 1 per slide

Total: 2 + N calls
```

**For 10 slides: 12 LLM calls**

**What you lose:**
- ❌ No research (generic, non-data-driven content)
- ❌ No vibe estimation (default themes)
- ❌ No keyword research (basic content)
- ❌ No template selection (random styling)
- ❌ Lower quality overall

**When to use:**
- Quick prototypes
- Internal drafts
- Time/cost extremely critical
- Content quality not important

---

## 🎯 Recommended Configuration

### By Use Case

#### **High Quality (Current)** - 33-35 calls for 10 slides
```yaml
use_research: true
browser_workers: 8-10
template_selection: true
quality_verification: false  # Can add +10 calls if enabled
```
✅ Best for: Client presentations, sales decks, important meetings

#### **Balanced** - 20-22 calls for 10 slides
```yaml
use_research: true
browser_workers: 4-5
template_selection: false
quality_verification: false
```
✅ Best for: Internal meetings, team updates, drafts

#### **Fast Draft** - 12-15 calls for 10 slides
```yaml
use_research: false
browser_workers: 0
template_selection: false
quality_verification: false
```
✅ Best for: Quick prototypes, brainstorming, placeholders

---

## 📊 Real-World Examples

### Example 1: Small Business Pitch (7 slides)

```
Base Pipeline:
├─ Spec Extraction: 1 call
├─ Vibe Estimator: 1 call
├─ Keyword Research: 2 calls
├─ Browser Workers: 8 calls
└─ Planning: 1 call
Total Base: 13 calls

Slide Generation (7 slides × 2):
├─ Template Selection: 7 calls
└─ Slide Generation: 7 calls
Total Slides: 14 calls

TOTAL: 27 LLM calls
Cost: ~$0.20
Time: ~45 seconds
```

### Example 2: Sales Deck (15 slides)

```
Base Pipeline: 13 calls

Slide Generation (15 slides × 2):
├─ Template Selection: 15 calls
└─ Slide Generation: 15 calls  
Total Slides: 30 calls

TOTAL: 43 LLM calls
Cost: ~$0.30
Time: ~75 seconds
```

### Example 3: Conference Presentation (25 slides)

```
Base Pipeline: 13 calls

Slide Generation (25 slides × 2):
├─ Template Selection: 25 calls
└─ Slide Generation: 25 calls
Total Slides: 50 calls

TOTAL: 63 LLM calls
Cost: ~$0.45
Time: ~120 seconds
```

---

## 🔮 Future Optimizations

### Smart Caching
- Cache research data for 24-48 hours
- Reuse keyword research for similar topics
- Share vibe estimations across users
- **Potential savings: 30-40%**

### Batch Processing
- Generate multiple slides in single call
- Batch template selections
- **Potential savings: 40-50%**

### Model Tier Selection
- Use faster models for simple tasks
- Reserve premium models for complex generation
- **Potential savings: 20-30% cost**

---

## 📋 Summary Table

| Presentation Size | Base Calls | Slide Calls | Total Calls | Cost | Time |
|------------------|------------|-------------|-------------|------|------|
| **Small (5 slides)** | 13 | 10 | **23** | $0.15 | 30s |
| **Medium (10 slides)** | 13 | 20 | **33** | $0.25 | 60s |
| **Large (15 slides)** | 13 | 30 | **43** | $0.35 | 90s |
| **Extra Large (25 slides)** | 13 | 50 | **63** | $0.50 | 150s |

---

## ✅ Key Takeaways

1. **Base pipeline is fixed:** 13-15 LLM calls regardless of slide count
2. **Per-slide cost:** 2 LLM calls per slide (template + generation)
3. **Formula:** `Total = 13-15 + (Slides × 2)`
4. **Minimum possible:** 2 + N calls (poor quality)
5. **Current system:** Optimized for quality, not call count
6. **Cost:** ~$0.02-0.03 per slide, ~$0.25 per 10-slide presentation
7. **Time:** ~60-90 seconds for 10 slides

---

**Last Updated:** October 22, 2025  
**Version:** 1.0.0  
**Pipeline Version:** Lightweight with Enhanced Slide Generation

For detailed agent documentation, see [AGENT_CATALOG.md](memory_bank/references/AGENT_CATALOG.md)

