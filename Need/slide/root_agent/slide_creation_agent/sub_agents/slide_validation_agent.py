"""
Slide Validation Agent
Validates and fixes HTML alignment issues by taking screenshots and generating corrected HTML
that fits perfectly in a 1280x720px 16:9 container.
"""
from google.adk.agents import BaseAgent, LlmAgent
from google.adk.events import Event
from google.genai import types
from google import genai
import os
import logging
import asyncio
import sys
from pathlib import Path
from typing import Optional, Tuple, List
from dotenv import load_dotenv
from fastapi.concurrency import run_in_threadpool
from core.database import get_db

# Add root directory to path for imports (ROOT is 4 levels up)
ROOT_PATH = str(Path(__file__).parent.parent.parent.parent)
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from core.token_logger import log_token_usage

load_dotenv()
logger = logging.getLogger(__name__)

GEMINI_MODEL = os.getenv("GEMINI_MODEL_FLASH", "gemini-2.5-flash")
SCREENSHOTS_DIR = os.getenv("SCREENSHOTS_DIR", "screenshots")
VALIDATION_MAX_RETRIES = int(os.getenv("VALIDATION_MAX_RETRIES", "5"))
VALIDATION_RETRY_DELAY = int(os.getenv("VALIDATION_RETRY_DELAY", "45"))




async def take_slide_screenshot(html_content: str, p_id: str, slide_index: int, width: int = 1280, height: int = 720) -> Optional[str]:
    """
    Take a screenshot of the HTML slide using Playwright.
    
    Args:
        html_content: The HTML content to render
        p_id: Presentation ID
        slide_index: 0-based slide index
        width: Screenshot width (default: 1280)
        height: Screenshot height (default: 720)
    
    Returns:
        Path to saved screenshot, or None if failed
    """
    browser = None
    page = None
    max_retries = 2
    retry_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            from playwright.async_api import async_playwright
            
            # Create directory structure
            screenshot_dir = Path(SCREENSHOTS_DIR) / p_id
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            
            screenshot_path = screenshot_dir / f"slide_{slide_index}.png"
            
            async with async_playwright() as p:
                try:
                    # Launch browser with timeout
                    browser = await asyncio.wait_for(
                        p.chromium.launch(headless=True, args=['--disable-web-security', '--disable-features=VizDisplayCompositor']),
                        timeout=30.0
                    )
                    
                    # Create page with timeout
                    page = await asyncio.wait_for(
                        browser.new_page(),
                        timeout=10.0
                    )
                    
                    # Set viewport size
                    await asyncio.wait_for(
                        page.set_viewport_size({"width": width, "height": height}),
                        timeout=5.0
                    )
                    
                    # Load HTML content with less strict wait condition
                    # Use "load" instead of "networkidle" to avoid timeout on external resources
                    await asyncio.wait_for(
                        page.set_content(html_content, wait_until="load", timeout=15000),
                        timeout=20.0
                    )
                    
                    # Wait a bit for any animations/rendering
                    await asyncio.sleep(1.0)
                    
                    # Take screenshot with timeout
                    await asyncio.wait_for(
                        page.screenshot(path=str(screenshot_path), full_page=False, timeout=15000),
                        timeout=20.0
                    )
                    
                    logger.info(f"📸 Screenshot saved: {screenshot_path}")
                    return str(screenshot_path)
                    
                except asyncio.TimeoutError as timeout_error:
                    logger.warning(f"⚠️ Screenshot timeout (attempt {attempt + 1}/{max_retries}): {timeout_error}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        raise
                except Exception as browser_error:
                    logger.warning(f"⚠️ Browser error (attempt {attempt + 1}/{max_retries}): {browser_error}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        raise
                finally:
                    # Ensure browser is closed even on error
                    if page:
                        try:
                            await asyncio.wait_for(page.close(), timeout=5.0)
                        except:
                            pass
                    if browser:
                        try:
                            await asyncio.wait_for(browser.close(), timeout=5.0)
                        except:
                            pass
        
        except ImportError:
            logger.error("❌ Playwright not installed. Run: pip install playwright && playwright install chromium")
            return None
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"⚠️ Screenshot failed (attempt {attempt + 1}/{max_retries}), retrying: {e}")
                await asyncio.sleep(retry_delay * (attempt + 1))
                continue
            else:
                logger.error(f"❌ Failed to take screenshot after {max_retries} attempts: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return None
    
    return None


class SlideValidationAgentWithImage(BaseAgent):
    """
    Custom agent that wraps Gemini model to pass screenshot image for vision analysis.
    """
    def __init__(self, html_content: str, screenshot_path: Optional[str], p_id: str, slide_index: int):
        super().__init__(name=f"slide_validation_agent_{slide_index}")
        # Use object.__setattr__ to bypass Pydantic validation for custom attributes
        object.__setattr__(self, "html_content", html_content)
        object.__setattr__(self, "screenshot_path", screenshot_path)
        object.__setattr__(self, "p_id", p_id)
        object.__setattr__(self, "slide_index", slide_index)
    
    def _create_instruction(self) -> str:
        """Create the instruction text for the validation agent"""
        instruction = f"""
        You are an expert HTML/CSS slide validator and fixer. Your task is to analyze a slide HTML and its screenshot, then generate a NEW HTML that renders perfectly in a FIXED 1280x720px iframe with NO overflow and NO cut-off text.

        ## CRITICAL REQUIREMENTS:

        1. **PRESERVE STRUCTURE COMPLETELY**: 
        - **DO NOT change the HTML structure** - keep all HTML tags, classes, IDs, and element hierarchy exactly as in the original
        - Understand the slide structure from the provided screenshot image
        - Preserve all container divs, sections, and layout elements exactly as they are
        - Only modify CSS properties (styles), NEVER add, remove, or reorder HTML elements
        - Keep all attributes (class, id, data-*) exactly as they appear in the original

        2. **PRESERVE ALL TEXT EXACTLY**: 
        - Extract ALL text content from the original HTML <body> section
        - DO NOT generate, modify, or change ANY text content
        - Keep all text exactly as it appears in the original HTML
        - Only modify CSS styles, NEVER modify text content

        3. **FIXED 1280x720px CONTAINER (MANDATORY)**:
        - The slide MUST be exactly 1280px x 720px - NOT responsive, NOT scalable
        - Set `html, body` to: `width: 1280px; height: 720px; margin: 0; padding: 0; overflow: hidden;`
        - Set the main slide container (`.slide-container` or equivalent) to: `width: 1280px; height: 720px;` (FIXED pixels, NOT responsive)
        - Remove ALL `aspect-ratio`, `max-width`, `max-height` from the main container
        - Remove ALL `vw`, `vh`, `clamp()`, `rem` units from container dimensions - use FIXED pixels only
        - Use box-sizing: border-box on ALL elements
        - The container must be exactly 1280px x 720px with NO responsive units

        4. **CONVERT RESPONSIVE UNITS TO FIXED PIXELS**:
        - Convert ALL responsive units to fixed pixels based on 1280x720px:
            * `vw` units: Multiply by 12.8 (e.g., 2.5vw = 32px, 4vw = 51.2px ≈ 50px, 10vw = 128px)
            * `vh` units: Multiply by 7.2 (e.g., 4vh = 28.8px ≈ 30px, 8vh = 57.6px ≈ 60px, 10vh = 72px)
            * `clamp()` values: Calculate based on 1280x720px and convert to fixed px (e.g., clamp(2rem, 3.5vw, 4rem) → calculate for 1280px width → use fixed px)
            * `rem` units: Convert to px (1rem = 16px typically, so 2rem = 32px, 1.5rem = 24px, 1rem = 16px)
        - For font sizes: Use fixed pixel values (e.g., 14px, 16px, 18px, 20px, 24px, 28px, 32px, 36px, 42px) instead of clamp() or rem
        - For spacing (padding, margins, gaps): Use fixed pixel values (e.g., 10px, 15px, 20px, 30px, 40px, 50px) instead of vw/vh
        - **CRITICAL**: Remove ALL responsive units from the entire CSS - convert everything to fixed pixels

        5. **REMOVE NESTED PADDING**:
        - Identify nested containers with padding: If parent has padding AND child has padding, remove child padding
        - Keep padding ONLY on the outermost container (`.slide-container` or main container)
        - Typical padding: 30px-40px on all four sides of the main container
        - Calculate available space: If container has 30px padding, content area = 1220px x 660px (1280-60 x 720-60)
        - Remove padding from inner content wrappers (like `.slide-content`, `.content-wrapper`) if parent already has padding
        - **CRITICAL**: Only the outermost container should have padding - all inner containers should have padding: 0

        6. **CALCULATE AND FIT ALL CONTENT**:
        - Calculate total vertical space needed:
            * Header/logo area height (typically 60-80px)
            * Title height + margin (typically 60-100px)
            * Gap between sections (typically 20-40px)
            * Content sections (columns, cards, etc.) height
            * Footer/slide number height (typically 30-50px)
            * Total must be ≤ 720px minus container padding (e.g., if padding is 30px, max content height = 660px)
        - If content doesn't fit:
            * Reduce container padding: 40px → 30px (if needed)
            * Reduce header margin-bottom: 40px → 30px → 25px → 20px
            * Reduce icon/image placeholder heights PROPORTIONALLY (e.g., all 250px → all 180px, all 180px → all 150px)
            * Reduce font sizes proportionally (32px → 28px, 16px → 14px) - apply same reduction to all similar elements
            * Reduce gaps between elements CONSISTENTLY (from 50px to 30px, from 40px to 25px, from 30px to 20px)
            * **CRITICAL FOR GRIDS**: Grid gaps should be 20px-30px (NOT 15px or less which creates huge visual gaps)
            * Reduce line-height: 1.6 → 1.5 → 1.4 (saves vertical space)
            * Reduce padding/margins on inner elements CONSISTENTLY (all 30px → all 25px → all 20px)
        - **CRITICAL**: When reducing sizes, reduce ALL similar elements by the SAME amount to maintain visual consistency
        - **CRITICAL FOR CARDS**: If reducing card sizes, reduce ALL cards equally - never create huge differences
        - Ensure ALL text is visible: If screenshot shows cut-off text, reduce element sizes until all text fits
        - **CRITICAL**: All content must fit within the calculated available space (1280px x 720px minus padding)

        7. **FIX OVERFLOW AND OVERLAP**:
        - **If overflow exists**: Fix by adjusting font sizes, padding, margins, or spacing (using fixed pixels)
        - **If overlap exists**: Fix by adjusting positioning, margins, or spacing (using fixed pixels)
        - **If NO overflow/overlap**: Still optimize by adjusting font sizes, spacing, or other CSS properties for better fit
        - Set overflow: hidden on html, body, and main container
        - Ensure no content exceeds 1280px width or 720px height

        8. **OPTIMIZE FONT SIZES (FIXED PIXELS)**:
        - **Always optimize**: Even if there's no overflow/overlap, adjust font sizes for optimal fit within 1280x720px
        - Use FIXED pixel values for ALL font sizes, NOT clamp() or rem:
            * Main titles: 32px-42px (reduce to 28px-36px if content is tight)
            * Section headings: 20px-28px (reduce to 18px-24px if needed)
            * Body text: 14px-18px (reduce to 13px-16px if needed)
            * Small text: 12px-14px
        - Reduce line-height when content is tight: 1.6 → 1.5 → 1.4 (saves vertical space)
        - Adjust font sizes, padding, margins, and gaps to ensure perfect fit
        - Maintain readable font sizes - do NOT make fonts too small (minimum 12px for small text, 14px for body)

        9. **OPTIMIZE ELEMENT SIZES (MAINTAIN PROPORTIONS)**:
            - Icon/image placeholders: Typically 150px-200px height (NOT 250px+, NOT less than 120px)
            - **For card layouts**: Card image heights should be 120px-180px (maintain consistency across all cards)
            - Calculate available height: Total 720px - padding (60px) - title area (80-100px) - gaps (40-60px) = ~500-550px for content
            - If content sections are too tall: Reduce icon heights, font sizes, or spacing PROPORTIONALLY
            - Ensure images/icons don't exceed container bounds
            - **CRITICAL**: Icon/image placeholders should be 150px-200px height maximum, not 250px+. But do NOT reduce below 120px as it looks unprofessional
            - **CRITICAL FOR CARDS**: All cards in a grid must have the same image height, padding, and spacing - maintain visual consistency

        10. **FIX TEXT CUT-OFF ISSUES (HIGHEST PRIORITY)**:
            - Check screenshot for cut-off text - this is CRITICAL and HIGHEST PRIORITY
            - If text is truncated or not fully visible, IMMEDIATELY apply these fixes in order:
            1. Add `min-height: 0` to all flex containers (allows content to shrink)
            2. Reduce container padding: 40px → 30px (if currently 40px)
            3. Reduce header margin-bottom: 40px → 25px → 20px
            4. Reduce ALL gaps CONSISTENTLY: 40px → 30px → 25px → 20px (reduce progressively, but same value for all)
            5. Reduce font sizes PROPORTIONALLY: 32px → 28px, 16px → 14px, 18px → 16px (apply same reduction to all similar elements)
            6. Reduce line-height: 1.6 → 1.5 → 1.4
            7. Reduce icon/image heights PROPORTIONALLY: all 250px → all 180px → all 150px (NOT below 120px, and all same height)
            8. Reduce inner padding CONSISTENTLY: all 30px → all 25px → all 20px (same value for all cards/elements)
            - **CRITICAL**: When applying reductions, maintain visual consistency - reduce ALL similar elements by the SAME amount
            - **CRITICAL FOR CARDS**: If reducing card image heights, reduce ALL cards equally (e.g., all 180px → all 150px, not 70px, 120px, 180px)
            - Prioritize text visibility over decorative element sizes, but maintain consistency
            - **CRITICAL**: All text must be fully visible - if any text is cut off, apply multiple reductions until all text fits, but maintain consistency

        11. **UNDERSTAND STRUCTURE FROM IMAGE**:
            - Analyze the screenshot image to understand the visual structure and layout
            - Identify all visual elements (titles, text blocks, images, logos, cards, grids, etc.)
            - Match the visual structure to the HTML structure
            - Preserve the visual hierarchy and relationships between elements
            - **Check for cut-off text**: If screenshot shows text that's partially visible or cut off, this is a critical issue that must be fixed

        12. **POSITIONING FOR FIXED DIMENSIONS**:
            - Absolute positioning: Use fixed pixel values (e.g., `top: 30px; right: 40px;`)
            - Logo positioning: Match container padding (e.g., if container padding is 30px, logo `top: 30px; right: 40px;`)
            - Slide number: Match container padding (e.g., `bottom: 30px; right: 40px;`)
            - Ensure absolute elements don't exceed 1280x720px bounds
            - **CRITICAL**: Use fixed pixel values for absolute positioning, matching container padding

        13. **GRID AND FLEXBOX FOR FIXED DIMENSIONS**:
            - Use fixed pixel gaps (e.g., `gap: 30px;` or `gap: 50px;` NOT `gap: 4vw;`)
            - Calculate column widths properly using fixed pixels
            - Ensure grid/flex items fit within available space
            - **CRITICAL**: Add `min-height: 0` to ALL flex containers (`.slide-container`, `.content-area`, `.article-section`, `.team-section`, etc.) to allow flex items to shrink below their content size
            - Without `min-height: 0`, flex items won't shrink and will cause overflow
            - **CRITICAL**: Use fixed pixel values for grid gaps and flexbox spacing, NOT responsive units
            - **CRITICAL FOR GRID LAYOUTS**: For grid layouts with multiple cards/columns (e.g., 3-column grids):
            * Use consistent gap between all cards: 20px-30px (NOT 15px or less, which creates huge visual gaps)
            * All cards in the same grid MUST have the same height, padding, and internal spacing
            * If reducing sizes, reduce ALL cards proportionally - never reduce one card more than others
            * Maintain visual balance: if original has 3 equal cards, keep them equal after optimization
            * Card image heights should be consistent: 120px-180px (NOT 70px which is too small)
            * Card padding should be consistent: 15px-20px (NOT 10px which makes cards look cramped)
            * **CRITICAL**: Equal spacing and sizing creates professional appearance - huge differences between cards look unprofessional

        14. **PREVENT POINTER EVENT ISSUES**:
            - **CRITICAL**: Prevent decorative/background elements from intercepting click events
            - Add CSS property "pointer-events: none" to decorative/background elements:
            * Background images (`.hero-image`, `.background-image`, etc.)
            * Decorative shapes (`.image-side`, `.decorative-element`, etc.)
            * Pattern overlays or background graphics
            - Add `pointer-events: auto` to interactive content areas:
            * Text containers (`.content-side`, `.article-section`, etc.)
            * Logos (`.logo`)
            * Buttons or clickable elements (if any)
            - **Why this matters**: Without this, background images/elements can block clicks and trigger unwanted edit modes in preview tools
            - **Example**: Use CSS like: .image-side {{ pointer-events: none; }} and .content-side {{ pointer-events: auto; }}
            - **CRITICAL**: Always set "pointer-events: none" on absolutely positioned background/decorative elements

        15. **MAINTAIN VISUAL CONSISTENCY AND BALANCE**:
            - **CRITICAL FOR GRID/CARD LAYOUTS**: When slides have multiple cards, columns, or blocks in a grid:
            * All cards/blocks MUST have identical sizing (same height, padding, margins)
            * All cards MUST have the same image/icon heights (e.g., all 150px, not 70px, 120px, 180px)
            * All cards MUST have the same padding (e.g., all 15px, not 10px, 15px, 20px)
            * Grid gaps MUST be consistent: 20px-30px between cards (NOT 15px which creates huge visual gaps)
            * If original has 3 equal cards, keep them equal after optimization
            * **CRITICAL**: Huge differences in card sizes or gaps create unprofessional appearance
            - **Proportional reduction**: When reducing sizes, reduce ALL similar elements by the same amount
            * If reducing image heights: reduce all images by same percentage (e.g., all 180px → all 150px)
            * If reducing padding: reduce all card padding by same amount (e.g., all 20px → all 15px)
            * If reducing gaps: use same gap value for all (e.g., gap: 25px for all cards)
            - **Visual balance**: Maintain the original visual hierarchy and proportions
            * If original has balanced layout, preserve that balance
            * Don't create huge gaps between elements that should be close together
            * Ensure cards look like they belong to the same design system

        16. **BALANCE PADDING AND SPACING**:
            - Main container padding: 30px-40px on all four sides (fixed pixels)
            - Inner element spacing: Use fixed pixel values (10px, 15px, 20px, 25px, 30px)
            - **For card layouts**: Card padding should be 15px-20px (consistent across all cards)
            - **For grid gaps**: Use 20px-30px between cards (consistent, not too small)
            - Ensure content does not touch borders
            - Adjust spacing as needed for optimal layout (using fixed pixels only)
            - **CRITICAL**: Maintain consistent spacing - don't create huge differences between similar elements

        ## ORIGINAL HTML:
        ```html
        {self.html_content}
        ```

        ## YOUR TASK:
        1. **Analyze the screenshot image** to understand the visual structure and identify any issues (especially cut-off text and inconsistent sizing)
        2. **Check for cut-off text FIRST**: If screenshot shows any text that's not fully visible, this is CRITICAL - apply fixes from section 10 immediately
        3. **Check for visual consistency**: If screenshot shows cards/blocks with huge size differences or gaps, ensure all similar elements have identical sizing and spacing
        4. **Add min-height: 0 to flex containers**: Add `min-height: 0` to all flex containers to allow content to shrink
        5. **Fix pointer events**: Add "pointer-events: none" to decorative/background elements and "pointer-events: auto" to content areas
        6. **Check for overflow/overlap**: If present, fix by adjusting CSS properties (using fixed pixels)
        7. **Convert all responsive units**: Convert ALL vw, vh, clamp(), rem units to fixed pixels
        8. **Set fixed container**: Ensure html, body, and main container are exactly 1280px x 720px
        9. **Remove nested padding**: Keep padding only on outermost container
        10. **Calculate and fit content**: Ensure all content fits within available space (1280px x 720px minus padding)
        11. **Maintain visual consistency**: For grid/card layouts, ensure all cards have identical sizing, padding, and spacing
        12. **Preserve structure**: Keep all HTML elements, classes, IDs, and hierarchy exactly as in the original
        13. Generate a NEW HTML that:
        - Contains EXACTLY the same text content as the original
        - Has the SAME HTML structure (same tags, classes, IDs, hierarchy)
        - Has optimized CSS with FIXED pixel values (NO responsive units)
        - Container is exactly 1280px x 720px (NOT responsive)
        - All font sizes use fixed pixels (e.g., 14px, 16px, 18px, 24px, 28px, 32px)
        - All spacing uses fixed pixels (e.g., 15px, 20px, 25px, 30px, 40px)
        - Has `min-height: 0` on all flex containers (critical for flexbox shrinking)
        - Has "pointer-events: none" on decorative/background elements (prevents click interception)
        - Has "pointer-events: auto" on content areas (ensures proper interaction)
        - Has NO overflow (horizontal or vertical)
        - Has NO overlaps
        - Has NO cut-off text (all text fully visible) - HIGHEST PRIORITY
        - Has proper padding on outermost container only (30px-40px, reduce to 30px if content is tight)
        - Is production-ready

        ## OUTPUT FORMAT:
        - Return ONLY the complete, fixed HTML code
        - Include full <!DOCTYPE html>, <head>, and <body> tags
        - Include all CSS in <style> tag in <head>
        - Preserve all original text content exactly
        - Preserve all HTML structure exactly (same elements, classes, IDs)
        - Use production-ready, clean HTML + CSS

        ## FINAL OUTPUT REQUIREMENTS:
        ✅ Same HTML structure (no element changes)
        ✅ Same text content (no text modifications)
        ✅ Fixed container: exactly 1280px x 720px (NOT responsive)
        ✅ All responsive units converted to fixed pixels
        ✅ No nested padding (padding only on outermost container)
        ✅ All text visible (no cut-off text) - HIGHEST PRIORITY
        ✅ Visual consistency maintained (all cards/blocks have identical sizing, padding, and spacing)
        ✅ Grid gaps consistent (20px-30px between cards, not too small creating huge visual gaps)
        ✅ Card image heights consistent (120px-180px, all cards same height)
        ✅ Card padding consistent (15px-20px, all cards same padding)
        ✅ min-height: 0 on all flex containers (critical for flexbox)
        ✅ pointer-events: none on decorative/background elements (prevents click interception)
        ✅ pointer-events: auto on content areas (ensures proper interaction)
        ✅ Optimized element sizes (icons/images fit within available space, but maintain proportions)
        ✅ Optimized CSS (font sizes, spacing, padding adjusted using fixed pixels)
        ✅ Font sizes using fixed pixels (14px, 16px, 18px, 24px, 28px, 32px, etc.)
        ✅ Spacing optimized (reduced gaps, margins, padding when needed, but consistently)
        ✅ Strict 16:9 safe-zone layout (1280x720px fixed)
        ✅ No overflow (vertical or horizontal)
        ✅ No overlaps
        ✅ Perfect alignment & spacing
        ✅ Professional visual balance (no huge differences between similar elements)
        ✅ Ready for production slide generation

        Output the complete fixed HTML code.
        """
        return instruction
    
    async def _run_async_impl(self, ctx):
        """Run the agent with image support - passes screenshot to Gemini vision model"""
        try:
            # Build parts list for Gemini API call
            parts = []
            
            # Add screenshot image if available
            if self.screenshot_path and os.path.exists(self.screenshot_path):
                try:
                    with open(self.screenshot_path, 'rb') as img_file:
                        img_bytes = img_file.read()
                    parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))
                    logger.info(f"✅ Added screenshot image to validation agent for slide {self.slide_index + 1}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not read screenshot: {e}")
            
            # Add instruction text
            instruction_text = self._create_instruction()
            parts.append(types.Part(text=instruction_text))
            
            # Create Gemini client and call directly with image support
            genai_client = genai.Client()
            try:
                # Retry configuration
                max_retries = VALIDATION_MAX_RETRIES
                initial_delay = VALIDATION_RETRY_DELAY
                exp_base = 2.0
                jitter = 0.3
                retry_status_codes = [429, 500, 502, 503, 504]
                
                result = None
                last_exception = None
                
                for attempt in range(max_retries):
                    try:
                        result = genai_client.models.generate_content(
                            model=GEMINI_MODEL,
                            contents=parts,
                            config=types.GenerateContentConfig(
                                thinking_config=types.ThinkingConfig(include_thoughts=False)
                            )
                        )
                        # Success - break out of retry loop
                        break
                    except Exception as e:
                        last_exception = e
                        error_code = getattr(e, 'status_code', None) or getattr(e, 'code', None)
                        
                        # Check if error is retryable
                        is_retryable = (
                            error_code in retry_status_codes or
                            isinstance(e, (ConnectionError, TimeoutError)) or
                            '429' in str(e) or '500' in str(e) or '502' in str(e) or '503' in str(e) or '504' in str(e)
                        )
                        
                        if attempt < max_retries - 1 and is_retryable:
                            # Calculate delay with exponential backoff and jitter
                            delay = initial_delay * (exp_base ** attempt)
                            jitter_amount = delay * jitter * (0.5 - (hash(str(e)) % 100) / 200)
                            total_delay = delay + jitter_amount
                            
                            logger.warning(f"⚠️ Validation API call failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {total_delay:.1f}s...")
                            await asyncio.sleep(total_delay)
                        else:
                            if not is_retryable:
                                logger.error(f"❌ Validation API call failed with non-retryable error: {e}")
                            raise
                
                if result is None:
                    raise last_exception or Exception("Failed to generate content after retries")
                
                # Extract token counts from result if available
                # The result object from genai_client.models.generate_content has usage_metadata
                if hasattr(result, 'usage_metadata') and result.usage_metadata:
                    usage = result.usage_metadata
                    input_tokens = 0
                    output_tokens = 0
                    thoughts_tokens = 0
                    
                    if hasattr(usage, 'prompt_token_count') and usage.prompt_token_count:
                        input_tokens = usage.prompt_token_count
                    
                    if hasattr(usage, 'candidates_token_count') and usage.candidates_token_count:
                        output_tokens = usage.candidates_token_count
                    
                    if hasattr(usage, 'thoughts_token_count') and usage.thoughts_token_count:
                        thoughts_tokens = usage.thoughts_token_count
                        output_tokens += thoughts_tokens
                    
                    # Store tokens in session state for later aggregation
                    if (input_tokens > 0 or output_tokens > 0) and hasattr(ctx, 'session') and hasattr(ctx.session, 'state'):
                        p_id = ctx.session.state.get('p_id', 'unknown')
                        state_exists = 'token_counts' in ctx.session.state
                        
                        if not state_exists:
                            msg = f"Initializing token_counts in ctx.session.state for p_id: {p_id}"
                            logger.info(f"📝 {msg}")
                            ctx.session.state['token_counts'] = {
                                'input_tokens': 0,
                                'output_tokens': 0,
                                'thoughts_tokens': 0
                            }
                            # Log initialization to token auditor
                            log_token_usage(p_id=p_id, author=self.name, message=msg)
                        msg=f"Updating token_counts in ctx.session.state for Slide Validation agent p_id: {p_id}"
                        # Update session state
                        ctx.session.state['token_counts']['input_tokens'] += input_tokens
                        ctx.session.state['token_counts']['output_tokens'] += output_tokens
                        ctx.session.state['token_counts']['thoughts_tokens'] += thoughts_tokens
                        
                        # Log to centralized token auditor (file)
                        log_token_usage(
                            p_id=p_id,
                            author=self.name,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            thoughts_tokens=thoughts_tokens,
                            message=msg
                        )
                        
                        logger.debug(f"📊 Validation Token Usage ({self.name}): In={input_tokens}, Out={output_tokens}, StateExists={state_exists}")
                
                # Extract response
                response_text = result.text or ""
                
                # Yield event similar to LlmAgent
                yield Event(
                    author=self.name,
                    content=types.Content(parts=[types.Part(text=response_text)]),
                    usage_metadata=result.usage_metadata if hasattr(result, 'usage_metadata') else None
                )
            finally:
                genai_client.close()
                
        except Exception as e:
            logger.error(f"❌ Error in validation agent with image: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Fallback: yield error event
            yield Event(
                author=self.name,
                content=types.Content(parts=[types.Part(text=f"Error during validation: {str(e)}")])
            )


def create_slide_validation_agent(html_content: str, screenshot_path: Optional[str], p_id: str, slide_index: int) -> BaseAgent:
    """
    Creates a slide validation agent that fixes HTML alignment issues.
    Uses custom BaseAgent to pass screenshot directly to Gemini vision model.
    
    Args:
        html_content: Original HTML content with all text
        screenshot_path: Path to screenshot image (optional)
        p_id: Presentation ID
        slide_index: Slide index
    
    Returns:
        BaseAgent configured to validate and fix HTML with image support
    """
    return SlideValidationAgentWithImage(html_content, screenshot_path, p_id, slide_index)


async def validate_and_fix_slide(
    html_content: str,
    p_id: str,
    slide_index: int,
    ctx,
    max_retries: int = 2,
    max_validation_iterations: int = 3
    ) -> Tuple[str, Optional[str], int]:
    """
    Validate and fix a slide's HTML by taking screenshot and passing to LLM agent.
    The LLM agent will handle all overflow and alignment issues.
    No iteration or re-validation is performed - the agent fixes it once.
    
    Args:
        html_content: Original HTML content
        p_id: Presentation ID
        slide_index: Slide index
        ctx: ADK context
        max_retries: Maximum number of retry attempts (kept for compatibility, but only used once)
        max_validation_iterations: Not used anymore, kept for compatibility
    
    Returns:
        Tuple of (fixed_html, thinking_text, validation_iterations_count)
        If validation fails, returns (original_html, None, 0)
        validation_iterations_count: Always 1 if validation ran, 0 if skipped
    """
    try:
        logger.info(f"🔍 Starting validation for slide {slide_index + 1}...")
        
        # Step 1: Take screenshot of HTML
        screenshot_path = await take_slide_screenshot(html_content, p_id, slide_index)
        
        if not screenshot_path:
            logger.warning(f"⚠️ Failed to take screenshot for slide {slide_index + 1}, skipping validation")
            return html_content, None, 0
        
        # Step 2: Create validation agent with HTML and screenshot
        validation_agent = create_slide_validation_agent(
            html_content, screenshot_path, p_id, slide_index
        )
        
        # Step 3: Run validation agent with retries
        thinking_text = None
        fixed_html = None
        
        for attempt in range(max_retries + 1):
            try:
                logger.info(f"🔍 Validating slide {slide_index + 1} (attempt {attempt + 1}/{max_retries + 1})")
                
                # Run agent (image is handled internally by the custom agent)
                async for ev in validation_agent.run_async(ctx):
                    if ev.content and ev.content.parts:
                            
                            for part in ev.content.parts:
                                #print(f"validation_agent_text: {part.text}")
                                # Extract thinking text (check thought attribute first)
                                if hasattr(part, 'thought') and part.thought:
                                    if hasattr(part, 'text') and part.text:
                                        thinking_text = part.text.strip()
                                
                                # Extract HTML from text parts
                            if hasattr(part, 'text') and part.text:
                                text = part.text.strip()
                                
                                # Extract HTML from response
                                # Look for HTML code blocks or direct HTML
                                if '<!DOCTYPE html>' in text or '<html' in text:
                                    # Extract HTML (remove markdown code blocks if present)
                                    html_text = text
                                    if '```html' in html_text:
                                        html_text = html_text.split('```html')[1].split('```')[0].strip()
                                    elif '```' in html_text:
                                        html_text = html_text.split('```')[1].split('```')[0].strip()
                                    
                                    fixed_html = html_text
                                    logger.info(f"✅ Validation agent returned fixed HTML ({len(fixed_html)} chars)")
                            
                            # Extract token counts and update database
                            if hasattr(ev, 'usage_metadata') and ev.usage_metadata:
                                usage = ev.usage_metadata
                                i_t = getattr(usage, 'prompt_token_count', 0) or 0
                                o_t = getattr(usage, 'candidates_token_count', 0) or 0
                                th_t = getattr(usage, 'thoughts_token_count', 0) or 0
                                
                                if i_t > 0 or o_t > 0:
                                    try:
                                        db = get_db()
                                        await run_in_threadpool(
                                            db.presentations.update_one,
                                            {"p_id": p_id},
                                            {"$inc": {
                                                "token_counts.input_tokens": i_t,
                                                "token_counts.output_tokens": o_t + th_t,
                                                "token_counts.thoughts_tokens": th_t
                                            }}
                                        )
                                        logger.info(f"📊 Updated DB token counts for {p_id}: +{i_t}i, +{o_t+th_t}o")
                                    except Exception as db_err:
                                        logger.warning(f"⚠️ Failed to update DB token counts: {db_err}")
                
                # If we got fixed HTML, break retry loop
                if fixed_html:
                    break
                    
            except Exception as e:
                logger.warning(f"⚠️ Validation attempt {attempt + 1} failed: {e}")
                if attempt < max_retries:
                    await asyncio.sleep(1)  # Brief delay before retry
                    continue
                else:
                    logger.error(f"❌ All validation attempts failed for slide {slide_index + 1}")
        
        # Take final screenshot of fixed HTML (or original if validation failed)
        final_html = fixed_html if fixed_html else html_content
        await take_slide_screenshot(final_html, p_id, slide_index)
        
        # Return fixed HTML or original if validation failed
        if fixed_html and fixed_html != html_content:
            logger.info(f"✅ Slide {slide_index + 1} validated and fixed successfully")
            return fixed_html, thinking_text, 1
        else:
            logger.warning(f"⚠️ Validation did not modify HTML for slide {slide_index + 1}, using original")
            return html_content, None, 0
            
    except Exception as e:
        logger.error(f"❌ Error during slide validation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return html_content, None, 0

