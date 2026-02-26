"""
Test the text overflow fix for slide quality verification
"""

import asyncio
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from root_agent.slide_creation_agent.sub_agents.slide_quality_verifier import create_slide_quality_verifier

# Sample HTML with text overflow issue (like your example)
OVERFLOW_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Strategic Imperatives for AI-Powered Healthcare</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <style>
        body {
            margin: 0;
            padding: 0;
            width: 1280px;
            height: 720px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #FFFFFF;
            color: #1F2937;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
        }

        .slide-container {
            width: 90%;
            height: 90%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }

        .headline {
            color: #10B981;
            font-size: 3.5em;  /* This might be too large and cause overflow */
            margin-bottom: 30px;
            font-weight: bold;
        }

        .body-content {
            font-size: 1.8em;  /* This might also be too large */
            line-height: 1.6;
            margin-bottom: 40px;
            max-width: 80%;
            color: #1F2937;
        }

        .body-content ul {
            list-style: none;
            padding: 0;
            margin: 0;
        }

        .body-content ul li {
            margin-bottom: 15px;
            display: flex;
            align-items: flex-start;
            justify-content: center;
        }

        .body-content ul li i {
            color: #10B981;
            margin-right: 15px;
            font-size: 1.2em;
            flex-shrink: 0;
            padding-top: 5px;
        }

        .action-items {
            font-size: 1.6em;  /* This might also cause overflow */
            color: #B91055;
            font-weight: 600;
            margin-top: 30px;
        }

        .action-items ul {
            list-style: none;
            padding: 0;
            margin: 20px 0 0 0;
        }

        .action-items ul li {
            margin-bottom: 10px;
        }

        .visual-icon {
            margin-top: 40px;
            max-width: 250px;
            height: auto;
        }
    </style>
</head>
<body>
    <div class="slide-container">
        <h1 class="headline">Strategic Imperatives for AI-Powered Healthcare</h1>
        <div class="body-content">
            <ul>
                <li><i class="fas fa-microchip"></i>AI as a "Co-Pilot" for enhanced clinical decision-making and efficiency.</li>
                <li><i class="fas fa-cogs"></i>Prioritize high-impact AI "Game Changers" for strategic pilots and partnerships.</li>
                <li><i class="fas fa-shield-alt"></i>Address ethical considerations, data privacy, and evolving regulatory landscapes.</li>
                <li><i class="fas fa-users-cog"></i>Foster cross-functional collaboration for scalable AI implementation and adoption.</li>
            </ul>
        </div>
        <div class="action-items">
            <h3>Discussion Prompt:</h3>
            <ul>
                <li>What 'Game Changer' AI initiatives should we prioritize to drive both clinical impact and measurable ROI?</li>
            </ul>
        </div>
        <img class="visual-icon" src="https://www.advisory.com/content/dam/advisory/en/public/images/4-imperatives-for-the-future-of-healthcare-innovation.jpg" alt="Strategic healthcare innovation" />
    </div>
</body>
</html>
"""

async def test_text_overflow_fix():
    """Test the slide quality verifier with text overflow issues"""
    
    print("🧪 Testing Text Overflow Fix...\n")
    
    # Create verification agent
    verifier = create_slide_quality_verifier()
    
    # Create mock context with overflow HTML
    class MockContext:
        def __init__(self):
            self.session = type('Session', (), {
                'state': {
                    'slide_html': OVERFLOW_HTML,
                    'slide_title': 'Strategic Imperatives for AI-Powered Healthcare',
                    'search_query': 'AI healthcare strategic imperatives clinical decision making'
                }
            })()
    
    ctx = MockContext()
    
    try:
        print("🔍 Analyzing slide for text overflow issues...")
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
                
                # Check technical quality for overflow issues
                technical = analysis.get('technical_quality', {})
                print(f"  Technical Score: {technical.get('score', 'N/A')}/10")
                
                # Check for overflow-related issues
                issues = verification_data.get('improvements', [])
                overflow_issues = [issue for issue in issues if 'overflow' in issue.lower() or 'fit' in issue.lower() or 'visible' in issue.lower()]
                
                if overflow_issues:
                    print(f"\n🔧 Text Overflow Fixes Applied:")
                    for issue in overflow_issues:
                        print(f"  ✅ {issue}")
                else:
                    print(f"\n⚠️ No specific overflow fixes detected in improvements")
                
                # Show enhanced HTML length
                enhanced_html = verification_data.get('enhanced_html', '')
                print(f"\n📏 Enhanced HTML length: {len(enhanced_html)} characters")
                
                # Check if enhanced HTML has overflow fixes
                if 'overflow: hidden' in enhanced_html or 'clamp(' in enhanced_html or 'max-height' in enhanced_html:
                    print("✅ Enhanced HTML contains overflow fixes!")
                else:
                    print("⚠️ Enhanced HTML may not contain overflow fixes")
                
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

async def main():
    """Run the text overflow test"""
    
    print("=" * 60)
    print("🧪 TEXT OVERFLOW FIX TESTING")
    print("=" * 60)
    
    await test_text_overflow_fix()
    
    print("\n" + "=" * 60)
    print("✅ Text overflow testing complete!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
