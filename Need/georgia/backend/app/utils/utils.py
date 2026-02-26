from functools import wraps
from flask import jsonify
from flask_jwt_extended import jwt_required, get_jwt
from app import db
from app.utils.debug_logger import debug_log

docs_ref = db.collection("document_metadata")  # 🔁 use your collection name

def admin_required(fn):
    @wraps(fn)
    @jwt_required()
    def wrapper(*args, **kwargs):
        claims = get_jwt()
        if claims.get("role") != "admin":
            return jsonify(msg="Admins only!"), 403
        return fn(*args, **kwargs)
    return wrapper

def admin_or_superadmin_required(fn):
    """
    Decorator that requires the user to have either 'admin' or 'superadmin' role.
    """
    @wraps(fn)
    @jwt_required()
    def wrapper(*args, **kwargs):
        claims = get_jwt()
        user_role = claims.get("role")
        if user_role not in ["admin", "superadmin"]:
            return jsonify(msg="Admin or Superadmin access required!"), 403
        return fn(*args, **kwargs)
    return wrapper

def superadmin_required(fn):
    """
    Decorator that requires the user to have 'superadmin' role.
    """
    @wraps(fn)
    @jwt_required()
    def wrapper(*args, **kwargs):
        claims = get_jwt()
        user_role = claims.get("role")
        if user_role != "superadmin":
            return jsonify(msg="Superadmin access required!"), 403
        return fn(*args, **kwargs)
    return wrapper

def generate_keywords(filename: str) -> list[str]:
    """
    Generate lowercase substrings of filename for substring search.
    """
    filename = filename.lower()
    keywords = set()
    for i in range(len(filename)):
        for j in range(i + 1, len(filename) + 1):
            keywords.add(filename[i:j])
    return list(keywords)

def update_search_keywords_by_doc_id(doc_id: str) -> tuple[bool, str]:
    """
    Updates search_keywords field for a specific document using its doc_id.
    Returns (success, message).
    """
    try:
        doc_ref = docs_ref.document(doc_id)
        doc_snapshot = doc_ref.get()

        if not doc_snapshot.exists:
            return False, f"Document with ID '{doc_id}' not found."

        data = doc_snapshot.to_dict()
        filename = data.get("original_filename")

        if not filename:
            return False, f"Document '{doc_id}' does not contain 'original_filename'."

        search_keywords = generate_keywords(filename)
        doc_ref.update({"search_keywords": search_keywords})

        return True, f"search_keywords updated for document '{doc_id}'."

    except Exception as e:
        return False, f"Error updating document '{doc_id}': {str(e)}"



def backfill_search_keywords(batch_size: int = 20):
    """
    Scans all documents in the collection, generates search_keywords,
    and updates documents with the new field.
    """
    debug_log("🔁 Starting backfill of search_keywords...")

    last_doc = None
    updated_count = 0

    while True:
        query = docs_ref.limit(batch_size)
        if last_doc:
            query = query.start_after(last_doc)

        docs = query.stream()
        docs = list(docs)
        if not docs:
            break

        batch = db.batch()

        for doc in docs:
            data = doc.to_dict()
            if not data:
                continue  # Skip if somehow the document has no data
            filename = data.get("original_filename")
            if not filename:
                continue
            search_keywords = generate_keywords(filename)
            doc_ref = docs_ref.document(doc.id)

            # ✅ OPTIONAL: Skip if already present
            if "search_keywords" in data:
                continue

            batch.update(doc_ref, {"search_keywords": search_keywords})
            updated_count += 1

        batch.commit()
        debug_log(f"✅ Updated {updated_count} documents so far...")

        last_doc = docs[-1]

    debug_log(f"🎉 Backfill complete. Total documents updated: {updated_count}")
