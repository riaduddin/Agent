"""
Test fixes for slide rendering and missing slide issues
"""

import asyncio
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from root_agent.slide_creation_agent.sub_agents.slide_quality_verifier import create_slide_quality_verifier
from root_agent.slide_creation_agent.sub_agents.lightweight_slide_pipeline import create_lightweight_slide_generation_agent

# Sample HTML with rendering issues (like your quantum computing example)
RENDERING_ISSUE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Embracing the Quantum Era</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <style>
        body {
            width: 1280px;
            height: 720px;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #FFFFFF;
            color: #1F2937;
            display: flex;
            justify-content: center;
            align-items: center;
            text-align: center;
            overflow: hidden;
            position: relative;
        }

        .slide-container {
            width: 90%;
            max-width: 1000px;
            position: relative;
            z-index: 2;
            padding: 40px;
            background: rgba(255, 255, 255, 0.85);
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        }

        .background-image {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: url('https://cdn.prod.website-files.com/643d2eea03135260bdaca209/6585b1365545f8efad49a7a5_Future%20of%20computing%20is.webp');
            background-size: cover;
            background-position: center;
            opacity: 0.15;
            z-index: 1;
        }

        h1 {
            color: #3B82F6;
            font-size: 3.5em;  /* This might cause overflow */
            margin-bottom: 25px;
            line-height: 1.1;
        }

        .body-content {
            font-size: 1.5em;  /* This might also cause overflow */
            line-height: 1.6;
            margin-bottom: 40px;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
        }

        .body-content ul {
            list-style: none;
            padding: 0;
            margin: 0;
            display: flex;
            flex-direction: column;
            gap: 15px;
            text-align: left;
            max-width: 700px;
            margin-left: auto;
            margin-right: auto;
        }

        .body-content ul li {
            padding-left: 30px;
            position: relative;
        }

        .body-content ul li::before {
            content: "\\f00c";
            font-family: "Font Awesome 6 Free";
            font-weight: 900;
            color: #F6823B;
            position: absolute;
            left: 0;
            top: 5px;
            font-size: 0.8em;
        }

        .action-items {
            margin-top: 30px;
            font-size: 1.6em;  /* This might also cause overflow */
            font-weight: bold;
            color: #2F68C5;
        }

        .action-items p {
            margin: 10px 0;
        }

        .action-items .fa-question-circle,
        .action-items .fa-search-plus {
            color: #F6823B;
            margin-right: 10px;
        }
    </style>
</head>
<body>
    <div class="background-image"></div>
    <div class="slide-container">
        <h1>Embracing the Quantum Era</h1>
        <div class="body-content">
            <p>Quantum computing is rapidly moving from theoretical promise to a tangible future, reshaping industries and creating unprecedented opportunities:</p>
            <ul>
                <li>A market poised for significant growth, reaching <strong>$1.2B to $1.85B by 2025</strong>.</li>
                <li>Global investments exceed <strong>$55.7 billion</strong>, fueling innovation and deployment.</li>
                <li>Anticipated to unlock <strong>trillions in economic value</strong> by 2035-2040 across various sectors.</li>
                <li>Focus on niche applications and hybrid solutions as mainstream adoption evolves.</li>
            </ul>
        </div>
        <div class="action-items">
            <p><i class="fas fa-question-circle"></i> Questions?</p>
            <p><i class="fas fa-search-plus"></i> Explore the future with us!</p>
        </div>
    </div>
</body>
</html>
"""

async def test_rendering_issues_fix():
    """Test the slide quality verifier with rendering issues"""
    
    print("🧪 Testing Rendering Issues Fix...\n")
    
    # Create verification agent
    verifier = create_slide_quality_verifier()
    
    # Create mock context with rendering issue HTML
    class MockContext:
        def __init__(self):
            self.session = type('Session', (), {
                'state': {
                    'slide_html': RENDERING_ISSUE_HTML,
                    'slide_title': 'Embracing the Quantum Era',
                    'search_query': 'quantum computing market growth investment opportunities'
                }
            })()
    
    ctx = MockContext()
    
    try:
        print("🔍 Analyzing slide for rendering issues...")
        result = await verifier.run_async(ctx)
        
        if result.content and result.content.parts:
            result_text = result.content.parts[0].text
            print("✅ Analysis completed!")
            print(f"📊 Result length: {len(result_text)} characters")
            
            # Try to parse JSON result
            try:
                verification_data = json.loads(result_text)
                
                print(f"\n📈 Quality Analysis:")
                analysis = verification_data.get('analysis', {})
                
                # Check technical quality for rendering issues
                technical = analysis.get('technical_quality', {})
                print(f"  Technical Score: {technical.get('score', 'N/A')}/10")
                
                # Check for rendering-related issues
                issues = verification_data.get('improvements', [])
                rendering_issues = [issue for issue in issues if any(keyword in issue.lower() for keyword in ['overflow', 'fit', 'visible', 'font', 'size', 'container'])]
                
                if rendering_issues:
                    print(f"\n🔧 Rendering Issues Fixes Applied:")
                    for issue in rendering_issues:
                        print(f"  ✅ {issue}")
                else:
                    print(f"\n⚠️ No specific rendering fixes detected in improvements")
                
                # Show enhanced HTML length
                enhanced_html = verification_data.get('enhanced_html', '')
                print(f"\n📏 Enhanced HTML length: {len(enhanced_html)} characters")
                
                # Check if enhanced HTML has rendering fixes
                rendering_fixes = [
                    'overflow: hidden' in enhanced_html,
                    'clamp(' in enhanced_html,
                    'max-width: 100%' in enhanced_html,
                    'box-sizing: border-box' in enhanced_html
                ]
                
                if any(rendering_fixes):
                    print("✅ Enhanced HTML contains rendering fixes!")
                    print(f"  - Overflow handling: {'overflow: hidden' in enhanced_html}")
                    print(f"  - Responsive sizing: {'clamp(' in enhanced_html}")
                    print(f"  - Container fixes: {'max-width: 100%' in enhanced_html}")
                    print(f"  - Box sizing: {'box-sizing: border-box' in enhanced_html}")
                else:
                    print("⚠️ Enhanced HTML may not contain rendering fixes")
                
                # Show overall score
                overall_score = verification_data.get('overall_score', 'N/A')
                print(f"\n🎯 Overall Quality Score: {overall_score}/10")
                
            except json.JSONDecodeError:
                print("⚠️ Result is not valid JSON, showing raw text:")
                print(result_text[:1000] + "..." if len(result_text) > 1000 else result_text)
        
        else:
            print("❌ No content in verification result")
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()

async def test_missing_slide_detection():
    """Test the lightweight slide pipeline for missing slide detection"""
    
    print("\n🧪 Testing Missing Slide Detection...\n")
    
    # Create pipeline agent
    pipeline = create_lightweight_slide_generation_agent()
    
    # Create mock context with all required data
    class MockContext:
        def __init__(self):
            self.session = type('Session', (), {
                'state': {
                    'presentation_spec': {
                        'topic': 'Quantum Computing Future',
                        'slide_count': 6,  # Request 6 slides
                        'tone': 'professional',
                        'audience': 'business_owners',
                        'primary_color': '#3B82F6',
                        'background_color': '#ffffff',
                        'text_color': '#1F2937'
                    },
                    'keywords': ['quantum computing', 'market growth', 'investment opportunities', 'AI integration', 'business applications', 'future technology'],
                    'user_id': 'test_user_123',
                    'p_id': 'test_pres_456'
                }
            })()
    
    ctx = MockContext()
    
    try:
        print("🚀 Running lightweight slide pipeline...")
        print("📊 Expected: 6 slides")
        
        slide_count = 0
        async for ev in pipeline.run_async(ctx):
            if ev.content and ev.content.parts:
                for part in ev.content.parts:
                    if hasattr(part, 'text'):
                        text = part.text
                        if 'Generated slide' in text and '/' in text:
                            # Extract slide count
                            try:
                                current = int(text.split('Generated slide ')[1].split('/')[0])
                                total = int(text.split('/')[1].split()[0])
                                slide_count = current
                                print(f"  📈 {text}")
                            except:
                                pass
                        elif 'slides generated' in text:
                            print(f"  🎯 {text}")
        
        print(f"\n📊 Final Results:")
        print(f"  Expected: 6 slides")
        print(f"  Generated: {slide_count} slides")
        
        if slide_count == 6:
            print("✅ All slides generated successfully!")
        elif slide_count > 6:
            print(f"⚠️ More slides than expected: {slide_count}")
        else:
            print(f"❌ Missing slides: {6 - slide_count} slides not detected")
            
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Run all tests"""
    
    print("=" * 60)
    print("🧪 SLIDE ISSUES FIX TESTING")
    print("=" * 60)
    
    # Test 1: Rendering Issues Fix
    await test_rendering_issues_fix()
    
    # Test 2: Missing Slide Detection
    await test_missing_slide_detection()
    
    print("\n" + "=" * 60)
    print("✅ Testing complete!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
