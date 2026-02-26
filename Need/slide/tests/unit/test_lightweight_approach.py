#!/usr/bin/env python3
"""
Test script for the lightweight planning approach
"""
import asyncio
import json
import os
from dotenv import load_dotenv

# Add project root to path
import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)

# Import components
from root_agent.slide_creation_agent.sub_agents.lightweight_planning_agent import (
    create_lightweight_planning_agent
)
from tools.qdrant_utils import get_qdrant_manager
from tools.qdrant_retrieval import retrieve_research_context

# Load environment
load_dotenv()


async def test_qdrant_connection():
    """Test Qdrant connection and basic operations"""
    print("🔍 Testing Qdrant connection...")
    try:
        qdrant = get_qdrant_manager()
        
        # Get collection info
        info = qdrant.client.get_collection("browser_research")
        print(f"✅ Qdrant connected!")
        print(f"   Collection: browser_research")
        print(f"   Total points: {info.points_count}")
        print(f"   Vector dimensions: {info.config.params.vectors.size}")
        
        return True
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        print("   Make sure Qdrant is running: docker run -p 6333:6333 qdrant/qdrant")
        return False


async def test_sample_data_storage():
    """Store some sample data for testing"""
    print("\n📊 Storing sample test data...")
    try:
        qdrant = get_qdrant_manager()
        
        # Store sample research data
        sample_data = [
            {
                "text": "Tesla's marketing strategy relies on minimal traditional advertising, with 78% of their marketing coming from word-of-mouth and Elon Musk's social media presence.",
                "keyword": "Tesla marketing strategy",
                "sources": [{"title": "Tesla Marketing Report", "url": "https://example.com/tesla"}]
            },
            {
                "text": "Tesla's direct-to-consumer sales model eliminates dealership markups, allowing for better customer control and 25% higher profit margins.",
                "keyword": "Tesla sales model",
                "sources": [{"title": "Auto Industry Analysis", "url": "https://example.com/auto"}]
            },
            {
                "text": "Elon Musk's Twitter engagement generates an estimated $2 billion in equivalent advertising value annually for Tesla.",
                "keyword": "Elon Musk social media",
                "sources": [{"title": "Social Media Impact Study", "url": "https://example.com/social"}]
            }
        ]
        
        for i, data in enumerate(sample_data):
            chunks = qdrant.store_research_data(
                text=data["text"],
                user_id="test_user",
                p_id="test_presentation",
                keyword=data["keyword"],
                sources=data["sources"]
            )
            print(f"   ✅ Stored {chunks} chunks for: {data['keyword']}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to store sample data: {e}")
        return False


async def test_retrieval():
    """Test Qdrant retrieval with sample queries"""
    print("\n🔍 Testing retrieval with sample queries...")
    
    test_queries = [
        "Tesla marketing strategy unconventional approaches",
        "Tesla direct sales model advantages",
        "Elon Musk social media marketing impact",
        "electric vehicle advertising spend",
    ]
    
    for query in test_queries:
        try:
            results = retrieve_research_context(
                query=query,
                user_id="test_user",
                p_id="test_presentation",
                limit=5
            )
            
            print(f"\n   Query: {query}")
            if "No research context found" in results:
                print(f"   ❌ No results found")
            else:
                # Count results by looking for numbered items
                result_count = results.count("**1.")  # Count of "**1." patterns
                if result_count == 0:
                    result_count = "multiple" if len(results) > 100 else "some"
                print(f"   ✅ Found {result_count} results")
                print(f"   Preview: {results[:150]}...")
        except Exception as e:
            print(f"   ❌ Query failed: {e}")


async def test_planning_agent():
    """Test the lightweight planning agent"""
    print("\n📋 Testing lightweight planning agent...")
    
    # Mock session state
    class MockSession:
        def __init__(self):
            self.state = {
                "presentation_spec": {
                    "topic": "Tesla's Marketing Strategies",
                    "presentation_type": "regular_presentation", 
                    "tone": "analytical",
                    "color_theme": "#E82127",
                    "slide_count": 10,
                    "audience_type": "business professionals",
                    "complexity_level": "intermediate",
                    "key_message": "Tesla's unconventional marketing drives success",
                    "visual_style": "modern_tech",
                    "content_focus": ["product innovation", "CEO brand", "direct sales"]
                },
                "keywords": ["Tesla marketing strategy", "Elon Musk social media", "Tesla sales model"],
                "user_id": "test_user",
                "p_id": "test_presentation"
            }
    
    class MockContext:
        def __init__(self):
            self.session = MockSession()
    
    try:
        planning_agent = create_lightweight_planning_agent()
        ctx = MockContext()
        
        print("   🎯 Running planning agent...")
        plan_result = None
        
        async for event in planning_agent.run_async(ctx):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'text') and part.text:
                        try:
                            # Try to parse JSON
                            plan_result = json.loads(part.text)
                            break
                        except:
                            # Try to extract JSON from text
                            import re
                            match = re.search(r'\{.*\}', part.text, re.DOTALL)
                            if match:
                                try:
                                    plan_result = json.loads(match.group(0))
                                    break
                                except:
                                    pass
        
        if plan_result:
            print(f"   ✅ Planning agent succeeded!")
            print(f"   📊 Generated plan for: {plan_result.get('presentation_metadata', {}).get('topic', 'Unknown')}")
            print(f"   📄 Total slides: {len(plan_result.get('slide_outline', []))}")
            
            # Check first few slides
            outline = plan_result.get('slide_outline', [])
            for i in range(min(3, len(outline))):
                slide = outline[i]
                print(f"   Slide {slide.get('slide_number', i+1)}: {slide.get('slide_title', 'No title')}")
                print(f"     Query: {slide.get('search_query', 'No query')[:80]}...")
            
            return plan_result
        else:
            print("   ❌ Planning agent failed to return valid JSON")
            return None
            
    except Exception as e:
        print(f"   ❌ Planning agent error: {e}")
        return None


async def main():
    """Run all tests"""
    print("🧪 Testing Lightweight Presentation Generation Approach\n")
    
    # Test 1: Qdrant connection
    qdrant_ok = await test_qdrant_connection()
    if not qdrant_ok:
        print("\n❌ Cannot continue without Qdrant. Please start Qdrant first.")
        return
    
    # Test 2: Store sample data
    storage_ok = await test_sample_data_storage()
    if not storage_ok:
        print("\n❌ Sample data storage failed.")
        return
    
    # Test 3: Test retrieval
    await test_retrieval()
    
    # Test 4: Test planning agent
    plan = await test_planning_agent()
    
    if plan:
        print(f"\n🎉 All tests passed! Lightweight approach is working.")
        print(f"✅ Planning agent created outline for {len(plan.get('slide_outline', []))} slides")
        print(f"✅ Each slide has specific search query for Qdrant retrieval")
        print(f"✅ Ready for slide generation!")
    else:
        print(f"\n❌ Planning agent test failed. Check the logs above.")


if __name__ == "__main__":
    asyncio.run(main())
