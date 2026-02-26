# template_service_router.py
from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from bson import ObjectId
import httpx
import asyncio
import os
from dotenv import load_dotenv
import logging
from core.database import get_mongo_client

load_dotenv()
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")
NUM_RESULTS = 5

# Create router
router = APIRouter(prefix="/templates", tags=["templates"])

# MongoDB connection for templates - using the same client as main app
def get_template_collection():
    """Get the templates collection using the shared MongoDB client"""
    try:
        client = get_mongo_client()
        template_db = client["slide_template"]
        return template_db["templates"]
    except Exception as e:
        logger.error(f"❌ Failed to get template collection: {e}")
        raise HTTPException(status_code=503, detail="Template database not available")

# Pydantic models (same as your service)
class ItemResponse(BaseModel):
    id: str
    description: str
    html_code: str

class CreateItemRequest(BaseModel):
    description: str = Field(..., min_length=1)
    html_code: str = Field(..., min_length=1)
    category: str = Field(..., min_length=1)
    type: str = Field(..., min_length=1)
    sub_category: Optional[str] = None

class UpdateItemRequest(BaseModel):
    description: Optional[str] = None
    html_code: Optional[str] = None
    category: Optional[str] = None
    type: Optional[str] = None
    sub_category: Optional[str] = None

class ImageResult(BaseModel):
    title: str
    link: str
    snippet: str

class QueryResult(BaseModel):
    query: str
    images: List[ImageResult]

class SearchResponse(BaseModel):
    results: List[QueryResult]

class SearchOptions(BaseModel):
    count: Optional[int] = 5
    imageSize: Optional[str] = "large"
    imageType: Optional[str] = "photo"
    colorType: Optional[str] = "color"
    safeSearch: Optional[str] = "active"
    country: Optional[str] = "us"
    language: Optional[str] = "en"

class SearchRequest(BaseModel):
    search_images: List[str]
    options: Optional[SearchOptions] = None

class DeleteResponse(BaseModel):
    message: str
    deleted_id: str

class BulkDeleteResponse(BaseModel):
    message: str
    deleted_count: int

class TypeStatistics(BaseModel):
    type: str
    count: int

class CategoryStatistics(BaseModel):
    category: str
    types: List[TypeStatistics]
    total_types: int
    total_templates: int

class StatisticsResponse(BaseModel):
    categories: List[CategoryStatistics]

# Helper functions
def validate_object_id(id: str) -> ObjectId:
    try:
        return ObjectId(id)
    except:
        raise HTTPException(status_code=400, detail="Invalid ID format")

def document_to_item_response(document: Dict) -> ItemResponse:
    return ItemResponse(
        id=str(document["_id"]),
        description=document.get("description", ""),
        html_code=document.get("html_code", "")
    )

async def fetch_google_images(
    enhanced_query: str, 
    options: Optional[SearchOptions] = None,
    num: int = NUM_RESULTS
) -> List[ImageResult]:
    """Call Google Custom Search API and return a list of ImageResult."""
    url = "https://www.googleapis.com/customsearch/v1"
    
    if options:
        num = options.count or num
    
    params = {
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "q": enhanced_query,
        "searchType": "image",
        "num": num
    }
    
    if options:
        if options.imageSize:
            size_mapping = {
                "xlarge": "huge",
                "large": "large",
                "medium": "medium",
                "small": "small"
            }
            params["imgSize"] = size_mapping.get(options.imageSize, "large")
        
        if options.imageType:
            type_mapping = {
                "photo": "photo",
                "clipart": "clipart",
                "lineart": "lineart",
                "face": "face"
            }
            params["imgType"] = type_mapping.get(options.imageType, "photo")
        
        if options.colorType:
            color_mapping = {
                "color": "color",
                "blackandwhite": "gray",
                "transparent": "trans"
            }
            params["imgColorType"] = color_mapping.get(options.colorType, "color")
        
        if options.safeSearch:
            safe_mapping = {
                "active": "high",
                "moderate": "medium",
                "off": "off"
            }
            params["safe"] = safe_mapping.get(options.safeSearch, "high")
        
        if options.country:
            params["cr"] = f"country{options.country.upper()}"
        if options.language:
            params["lr"] = f"lang_{options.language}"

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.get(url, params=params)
            
            if response.status_code != 200:
                error_detail = f"Google API request failed for query: {enhanced_query}. Status: {response.status_code}"
                try:
                    error_data = response.json()
                    if 'error' in error_data:
                        error_detail += f", Error: {error_data['error'].get('message', 'Unknown error')}"
                except:
                    error_detail += f", Response: {response.text}"
                
                raise HTTPException(status_code=500, detail=error_detail)

            data = response.json()
            items = data.get("items", [])

            results = [
                ImageResult(
                    title=item.get("title", ""),
                    link=item.get("link", ""),
                    snippet=item.get("snippet", "")
                )
                for item in items
            ]
            return results
            
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail=f"Google API timeout for query: {enhanced_query}")
        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=500, detail=f"Google API error for query {enhanced_query}: {str(e)}")

# CREATE endpoints
@router.post("/items", response_model=ItemResponse, status_code=201)
async def create_item(item: CreateItemRequest):
    """Create a new template item"""
    template_collection = get_template_collection()
    
    try:
        # Support both Pydantic v1 and v2
        if hasattr(item, 'model_dump'):
            item_dict = item.model_dump()
        else:
            item_dict = item.dict()
        result = template_collection.insert_one(item_dict)
        created_item = template_collection.find_one({"_id": result.inserted_id})
        
        if not created_item:
            raise HTTPException(status_code=500, detail="Failed to create item")
            
        return document_to_item_response(created_item)
        
    except Exception as e:
        logger.error(f"Failed to create item: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create item: {str(e)}")

# READ endpoints
@router.get("/items", response_model=List[ItemResponse])
async def get_items(
    category: Optional[str] = None,
    type: Optional[str] = None,
    sub_category: Optional[str] = None
):
    """Query items from MongoDB based on category, type, and optional sub_category"""
    template_collection = get_template_collection()
    
    query = {}
    if category:
        query["category"] = category
    if type:
        query["type"] = type
    if sub_category:
        query["sub_category"] = sub_category
    
    try:
        results = list(template_collection.find(query))
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    if not results:
        return []
    
    items = [document_to_item_response(item) for item in results]
    return items

@router.get("/items/all", response_model=List[ItemResponse])
async def get_all_items():
    """Retrieve all items from the 'templates' collection."""
    template_collection = get_template_collection()
    
    try:
        results = list(template_collection.find({}))
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    if not results:
        return []

    items = [document_to_item_response(item) for item in results]
    return items

@router.get("/items/{id}", response_model=ItemResponse)
async def get_item_by_id(id: str):
    """Get a specific template by ID"""
    template_collection = get_template_collection()
    
    object_id = validate_object_id(id)
    result = template_collection.find_one({"_id": object_id})
    
    if not result:
        raise HTTPException(status_code=404, detail="Item not found")
    
    return document_to_item_response(result)

@router.get("/template/{id}")
async def get_template_by_id(id: str):
    """Get HTML code by template ID"""
    template_collection = get_template_collection()
    
    object_id = validate_object_id(id)
    result = template_collection.find_one({"_id": object_id})
    
    if not result:
        raise HTTPException(status_code=404, detail="Template not found")
    
    return {"html_code": result.get("html_code", "")}

@router.get("/statistics", response_model=StatisticsResponse)
async def get_statistics():
    """
    Get statistics about templates grouped by category.
    Returns:
    - For each category: list of types with counts, total distinct types, total templates
    """
    template_collection = get_template_collection()
    
    # Valid categories
    valid_categories = ["creative", "technical", "academic", "business"]
    
    # Valid types (as specified by user)
    valid_types = [
        "title", "problem", "solution", "content", "data", "testimonial", 
        "comparison", "benefits", "process", "overview", "demo", "timeline", 
        "features", "objectives", "Quote", "contact", "analysis", "thank you"
    ]
    
    try:
        # Query all templates
        all_templates = list(template_collection.find({}))
        
        # Initialize stats dictionary: category -> {type -> count}
        # Start with all valid types set to 0 for each category
        category_stats = {
            category: {type_name: 0 for type_name in valid_types}
            for category in valid_categories
        }
        
        # Count templates by category and type
        for template in all_templates:
            category_raw = template.get("category", "").strip().lower()
            type_name = template.get("type", "").strip()
            
            # Only count if category is valid (case-insensitive match)
            # Count all types, even if not in valid_types list (for completeness)
            if category_raw in valid_categories and type_name:
                if type_name in category_stats[category_raw]:
                    category_stats[category_raw][type_name] += 1
        
        # Build response
        response_categories = []
        for category in valid_categories:
            types_dict = category_stats[category]
            
            # Create TypeStatistics list with ALL valid types, sorted by type name
            # This ensures all types are included, even with count 0
            types_list = [
                TypeStatistics(type=type_name, count=types_dict.get(type_name, 0))
                for type_name in sorted(valid_types)
            ]
            
            # Count only types with templates (count > 0)
            total_types = sum(1 for count in types_dict.values() if count > 0)
            total_templates = sum(types_dict.values())
            
            response_categories.append(
                CategoryStatistics(
                    category=category,
                    types=types_list,
                    total_types=total_types,
                    total_templates=total_templates
                )
            )
        
        return StatisticsResponse(categories=response_categories)
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

# UPDATE endpoints
@router.put("/items/{id}", response_model=ItemResponse)
async def update_item(id: str, item_update: UpdateItemRequest):
    """Update an existing template item"""
    template_collection = get_template_collection()
    
    object_id = validate_object_id(id)
    existing_item = template_collection.find_one({"_id": object_id})
    if not existing_item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    # Support both Pydantic v1 and v2
    if hasattr(item_update, 'model_dump'):
        update_dict = item_update.model_dump()
    else:
        update_dict = item_update.dict()
    update_data = {k: v for k, v in update_dict.items() if v is not None}
    
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")
    
    try:
        result = template_collection.update_one(
            {"_id": object_id},
            {"$set": update_data}
        )
        
        if result.modified_count == 0:
            raise HTTPException(status_code=500, detail="Failed to update item")
        
        updated_item = template_collection.find_one({"_id": object_id})
        return document_to_item_response(updated_item)
        
    except Exception as e:
        logger.error(f"Failed to update item: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update item: {str(e)}")

@router.patch("/items/{id}", response_model=ItemResponse)
async def partial_update_item(id: str, item_update: UpdateItemRequest):
    """Partially update an existing template item"""
    return await update_item(id, item_update)

# DELETE endpoints
@router.delete("/items/{id}", response_model=DeleteResponse)
async def delete_item(id: str):
    """Delete a specific template item by ID"""
    template_collection = get_template_collection()
    
    object_id = validate_object_id(id)
    existing_item = template_collection.find_one({"_id": object_id})
    if not existing_item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    try:
        result = template_collection.delete_one({"_id": object_id})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=500, detail="Failed to delete item")
        
        return DeleteResponse(
            message="Item deleted successfully",
            deleted_id=id
        )
        
    except Exception as e:
        logger.error(f"Failed to delete item: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete item: {str(e)}")

@router.delete("/items", response_model=BulkDeleteResponse)
async def bulk_delete_items(ids: List[str] = Query(...)):
    """Delete multiple template items by their IDs"""
    template_collection = get_template_collection()
    
    if not ids:
        raise HTTPException(status_code=400, detail="No IDs provided")
    
    object_ids = []
    invalid_ids = []
    
    for id in ids:
        try:
            object_ids.append(ObjectId(id))
        except:
            invalid_ids.append(id)
    
    if invalid_ids:
        raise HTTPException(status_code=400, detail=f"Invalid ID format(s): {invalid_ids}")
    
    try:
        result = template_collection.delete_many({"_id": {"$in": object_ids}})
        
        return BulkDeleteResponse(
            message=f"Successfully deleted {result.deleted_count} items",
            deleted_count=result.deleted_count
        )
        
    except Exception as e:
        logger.error(f"Failed to delete items: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete items: {str(e)}")

@router.delete("/items/all", response_model=BulkDeleteResponse)
async def delete_all_items():
    """Delete all template items (use with caution!)"""
    template_collection = get_template_collection()
    
    try:
        result = template_collection.delete_many({})
        
        return BulkDeleteResponse(
            message=f"Successfully deleted all {result.deleted_count} items",
            deleted_count=result.deleted_count
        )
        
    except Exception as e:
        logger.error(f"Failed to delete all items: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete all items: {str(e)}")

# SEARCH IMAGES ENDPOINT
@router.post("/search_images", response_model=SearchResponse)
async def search_images(request: SearchRequest):
    """Search Google Images using multiple enhanced queries in parallel."""
    options = request.options or SearchOptions()
    
    tasks = [fetch_google_images(query, options) for query in request.search_images]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    query_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            query_results.append(QueryResult(
                query=request.search_images[i],
                images=[]
            ))
        else:
            query_results.append(QueryResult(
                query=request.search_images[i],
                images=result
            ))
    
    return SearchResponse(results=query_results)

