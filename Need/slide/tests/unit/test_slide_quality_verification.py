"""
Test the slide quality verification system
"""

import asyncio
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from root_agent.slide_creation_agent.sub_agents.slide_quality_verifier import create_slide_quality_verifier
from root_agent.slide_creation_agent.sub_agents.enhanced_slide_pipeline import create_enhanced_slide_pipeline

# Sample HTML slide for testing
SAMPLE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Benefits for Business Growth</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
        body {
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            width: 1280px;
            height: 720px;
            overflow: hidden;
        }
        .slide-container {
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            padding: 40px;
            box-sizing: border-box;
        }
        .title {
            font-size: 48px;
            font-weight: bold;
            color: #ff6b35;
            margin-bottom: 60px;
            text-align: center;
        }
        .benefits-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 40px;
            width: 100%;
            max-width: 1200px;
        }
        .benefit-card {
            background: white;
            border-radius: 12px;
            padding: 30px;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }
        .benefit-card:hover {
            transform: translateY(-5px);
        }
        .benefit-icon {
            font-size: 48px;
            color: #ff6b35;
            margin-bottom: 20px;
        }
        .benefit-title {
            font-size: 24px;
            font-weight: bold;
            color: #ff6b35;
            margin-bottom: 15px;
        }
        .benefit-description {
            font-size: 16px;
            color: #333;
            line-height: 1.5;
            margin-bottom: 20px;
        }
        .benefit-takeaway {
            font-size: 14px;
            color: #007bff;
            font-weight: 600;
            background: #f8f9fa;
            padding: 10px;
            border-radius: 6px;
        }
    </style>
</head>
<body>
    <div class="slide-container">
        <h1 class="title">Benefits for Business Growth</h1>
        
        <div class="benefits-grid">
            <div class="benefit-card">
                <div class="benefit-icon">
                    <i class="fas fa-bullhorn"></i>
                </div>
                <h3 class="benefit-title">Increased Reach</h3>
                <p class="benefit-description">
                    Expand your customer base globally, overcoming geographical limits of traditional marketing.
                </p>
                <div class="benefit-takeaway">
                    Digital advertising can boost brand awareness by up to 80%.
                </div>
            </div>
            
            <div class="benefit-card">
                <div class="benefit-icon">
                    <i class="fas fa-dollar-sign"></i>
                </div>
                <h3 class="benefit-title">Cost-Effectiveness</h3>
                <p class="benefit-description">
                    Achieve higher returns on investment with optimized spending and targeted campaigns.
                </p>
                <div class="benefit-takeaway">
                    For every $1 invested, businesses typically see a $5 return.
                </div>
            </div>
            
            <div class="benefit-card">
                <div class="benefit-icon">
                    <i class="fas fa-chart-line"></i>
                </div>
                <h3 class="benefit-title">Measurability & Optimization</h3>
                <p class="benefit-description">
                    Track performance precisely to refine strategies for continuous improvement and informed adjustments.
                </p>
                <div class="benefit-takeaway">
                    Successful campaigns emphasize tracking key metrics.
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

async def test_slide_quality_verifier():
    """Test the slide quality verification agent"""
    
    print("🧪 Testing Slide Quality Verifier...\n")
    
    # Create verification agent
    verifier = create_slide_quality_verifier()
    
    # Create mock context
    class MockContext:
        def __init__(self):
            self.session = type('Session', (), {
                'state': {
                    'slide_html': SAMPLE_HTML,
                    'slide_title': 'Benefits for Business Growth',
                    'search_query': 'digital marketing benefits business growth ROI statistics'
                }
            })()
    
    ctx = MockContext()
    
    try:
        # Run verification
        print("🔍 Analyzing slide quality...")
        result = await verifier.run_async(ctx)
        
        if result.content and result.content.parts:
            result_text = result.content.parts[0].text
            print("✅ Verification completed!")
            print(f"📊 Result length: {len(result_text)} characters")
            
            # Try to parse JSON result
            try:
                verification_data = json.loads(result_text)
                print(f"📈 Overall Quality Score: {verification_data.get('overall_score', 'N/A')}")
                print(f"🔧 Improvements: {len(verification_data.get('improvements', []))}")
                
                # Show analysis breakdown
                analysis = verification_data.get('analysis', {})
                print("\n📊 Quality Analysis:")
                for category, score in analysis.items():
                    if isinstance(score, dict) and 'score' in score:
                        print(f"  {category}: {score['score']}/10")
                    elif isinstance(score, (int, float)):
                        print(f"  {category}: {score}/10")
                
                print(f"\n🎯 Enhanced HTML length: {len(verification_data.get('enhanced_html', ''))}")
                
            except json.JSONDecodeError:
                print("⚠️ Result is not valid JSON, showing raw text:")
                print(result_text[:500] + "..." if len(result_text) > 500 else result_text)
        
        else:
            print("❌ No content in verification result")
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()

async def test_enhanced_pipeline():
    """Test the enhanced slide pipeline"""
    
    print("\n🧪 Testing Enhanced Slide Pipeline...\n")
    
    # Create pipeline agent
    pipeline = create_enhanced_slide_pipeline()
    
    # Create mock context with all required data
    class MockContext:
        def __init__(self):
            self.session = type('Session', (), {
                'state': {
                    'presentation_spec': {
                        'topic': 'Digital Marketing Benefits',
                        'slide_count': 3,
                        'tone': 'professional',
                        'audience': 'business_owners',
                        'primary_color': '#007bff',
                        'background_color': '#ffffff',
                        'text_color': '#333333'
                    },
                    'keywords': ['digital marketing benefits', 'business growth', 'ROI statistics'],
                    'user_id': 'test_user_123',
                    'p_id': 'test_pres_456'
                }
            })()
    
    ctx = MockContext()
    
    try:
        print("🚀 Running enhanced slide pipeline...")
        result = await pipeline.run_async(ctx)
        
        if result.content and result.content.parts:
            result_text = result.content.parts[0].text
            print("✅ Pipeline completed!")
            print(f"📊 Result length: {len(result_text)} characters")
            
            # Try to parse JSON result
            try:
                pipeline_data = json.loads(result_text)
                enhanced_slides = pipeline_data.get('enhanced_slides', [])
                print(f"🎨 Generated {len(enhanced_slides)} enhanced slides")
                print(f"📈 Overall Quality: {pipeline_data.get('overall_quality', 'N/A')}")
                print(f"🔧 Total Improvements: {pipeline_data.get('improvements_applied', 0)}")
                
            except json.JSONDecodeError:
                print("⚠️ Result is not valid JSON, showing raw text:")
                print(result_text[:500] + "..." if len(result_text) > 500 else result_text)
        
        else:
            print("❌ No content in pipeline result")
            
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Run all tests"""
    
    print("=" * 60)
    print("🧪 SLIDE QUALITY VERIFICATION TESTING")
    print("=" * 60)
    
    # Test 1: Quality Verifier
    await test_slide_quality_verifier()
    
    # Test 2: Enhanced Pipeline
    await test_enhanced_pipeline()
    
    print("\n" + "=" * 60)
    print("✅ Testing complete!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
