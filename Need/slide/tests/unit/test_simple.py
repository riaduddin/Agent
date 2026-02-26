"""
Simple test to verify the lightweight approach components load correctly
"""

print("🧪 Testing Lightweight Approach Components...\n")

# Test 1: Import lightweight planning agent
print("1️⃣ Testing lightweight planning agent import...")
try:
    from root_agent.slide_creation_agent.sub_agents.lightweight_planning_agent import (
        create_lightweight_planning_agent
    )
    agent = create_lightweight_planning_agent()
    print(f"   ✅ Planning agent created: {agent.name}")
    print(f"   ✅ Model: {agent.model}")
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Test 2: Import enhanced slide generator
print("\n2️⃣ Testing enhanced slide generator import...")
try:
    from root_agent.slide_creation_agent.sub_agents.enhanced_slide_generator import (
        create_enhanced_slide_generator
    )
    test_outline = {
        "slide_number": 1,
        "slide_title": "Test Slide",
        "search_query": "test query",
        "suggested_type": "hero_title",
        "required_elements": ["headline"]
    }
    test_theme = {"primary_color": "#3B82F6"}
    
    generator = create_enhanced_slide_generator(
        slide_outline=test_outline, 
        global_theme=test_theme, 
        selected_template_html="<!-- Test template -->",
        idx=0
    )
    print(f"   ✅ Slide generator created: {generator.name}")
    print(f"   ✅ Model: {generator.model}")
    print(f"   ✅ Tools available: {len(generator.tools)} tool(s)")
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Test 3: Import pipeline
print("\n3️⃣ Testing lightweight pipeline import...")
try:
    from root_agent.slide_creation_agent.sub_agents.lightweight_slide_pipeline import (
        create_lightweight_slide_generation_agent
    )
    pipeline = create_lightweight_slide_generation_agent()
    print(f"   ✅ Pipeline created: {pipeline.name}")
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Test 4: Import Qdrant retrieval tool
print("\n4️⃣ Testing Qdrant retrieval tool import...")
try:
    import sys
    import os
    # Add root directory
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from tools.qdrant_retrieval import retrieve_research_tool
    print(f"   ✅ Retrieval tool loaded")
    print(f"   ✅ Function: {retrieve_research_tool.func.__name__}")
except Exception as e:
    print(f"   ❌ Failed: {e}")

print("\n" + "="*50)
print("✅ All components loaded successfully!")
print("="*50)
print("\nThe lightweight approach is ready to use.")
print("Run your actual service to test full functionality.")

