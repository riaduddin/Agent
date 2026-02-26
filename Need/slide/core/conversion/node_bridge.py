import asyncio
import re
import json
import os
import logging
import tempfile
import uuid
from typing import List, Dict, Any, Callable, Optional

logger = logging.getLogger(__name__)

# Path to the conversion service directory
CONVERSION_SERVICE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "conversion_service")
BRIDGE_SCRIPT = os.path.join(CONVERSION_SERVICE_DIR, "bridge.js")

class NodeConversionBridge:
    """
    Executes Node.js conversion scripts via asyncio subprocess.
    Captured logs are parsed to stream progress updates.
    """

    @staticmethod
    async def convert_slides(
        slides: List[Dict[str, Any]], 
        format: str, 
        job_id: str = None, 
        progress_callback: Optional[Callable[[int, str], Any]] = None
    ) -> bytes:
        """
        Convert slides to PDF or PPTX using the embedded Node.js service.
        Streams logs and updates progress if a callback is provided.
        """
        if format not in ['pdf', 'pptx']:
            raise ValueError(f"Unsupported format: {format}")
            
        if not slides:
            raise ValueError("No slides provided")

        if not job_id:
            job_id = str(uuid.uuid4())

        # Create temporary input and output files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as input_file:
            json.dump(slides, input_file)
            input_path = input_file.name
            
        output_path = os.path.join(tempfile.gettempdir(), f"output_{job_id}.{format}")
        
        try:
            cmd = [
                "node",
                BRIDGE_SCRIPT,
                "--format", format,
                "--input", input_path,
                "--output", output_path,
                "--jobId", job_id
            ]
            
            logger.info(f"Executing Node.js conversion: {' '.join(cmd)}")
            
            # Create subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=CONVERSION_SERVICE_DIR
            )
            
            # Read stdout line by line for progress
            total_slides = len(slides)
            
            # Regex to match: info: Processing slide 1/12
            slide_progress_pattern = re.compile(r"Processing slide (\d+)/(\d+)")
            
            last_logs = []
            
            while True:
                line_bytes = await process.stdout.readline()
                if not line_bytes:
                    break
                    
                line = line_bytes.decode('utf-8').strip()
                if not line:
                    continue
                
                logger.debug(f"[NodeJS] {line}")
                last_logs.append(line)
                if len(last_logs) > 50:
                    last_logs.pop(0)
                
                # Check for progress updates
                if progress_callback:
                    # Match "Processing slide X/Y"
                    match = slide_progress_pattern.search(line)
                    if match:
                        current = int(match.group(1))
                        # total = int(match.group(2))
                        # Calc percent (allocating 90% for processing, 10% for merge)
                        percent = int((current / total_slides) * 90)
                        await _safe_callback(progress_callback, percent, line)
                        
                    # Match "Merging..." (PPTX specific)
                    elif "Merging" in line:
                        await _safe_callback(progress_callback, 95, "Merging output files...")
                        
            # Wait for process to exit
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode('utf-8') if stderr else ""
                if not error_msg.strip() and last_logs:
                    error_msg = f"Last logs:\n" + "\n".join(last_logs[-20:])
                
                logger.error(f"Node.js conversion failed. Error: {error_msg}")
                raise RuntimeError(f"Conversion failed: {error_msg}")
                
            logger.info("Node.js conversion completed successfully.")
            
            # Read output file
            if not os.path.exists(output_path):
                raise FileNotFoundError("Output file was not created by the conversion script")
                
            with open(output_path, "rb") as f:
                result_bytes = f.read()
                
            return result_bytes

        finally:
            # Cleanup temp files
            try:
                if os.path.exists(input_path):
                    os.remove(input_path)
                if os.path.exists(output_path):
                    os.remove(output_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp files: {e}")

async def _safe_callback(callback, *args):
    """Helper to call sync or async callbacks safely"""
    try:
        if asyncio.iscoroutinefunction(callback):
            await callback(*args)
        else:
            callback(*args)
    except Exception as e:
        logger.error(f"Error in progress callback: {e}")

