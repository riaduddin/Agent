
import sys
import os
import importlib
import traceback

# Add root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def verify_imports():
    print("🔍 Verifying imports after reorganization...")
    
    modules_to_check = [
        "core.database",
        "core.socketio_manager",
        "middleware.auth",
        "routers.socketio",
        "routers.templates",
        "tools.qdrant_utils",
        "tools.qdrant_retrieval",
        "tools.image_search",
        "tools.template_retrieval",
        "utils.logging",
        "main"
    ]
    
    failed = []
    
    for module_name in modules_to_check:
        try:
            print(f"   Importing {module_name}...", end=" ")
            importlib.import_module(module_name)
            print("✅ OK")
        except Exception as e:
            print(f"❌ FAILED: {e}")
            failed.append(module_name)
            traceback.print_exc()
            
    if failed:
        print(f"\n❌ Verification failed for {len(failed)} modules: {failed}")
        sys.exit(1)
    else:
        print("\n✅ All imports verified successfully!")
        sys.exit(0)

if __name__ == "__main__":
    verify_imports()
