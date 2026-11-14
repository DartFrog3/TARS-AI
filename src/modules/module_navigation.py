"""
module_navigation.py

Vision-based navigation module for TARS-AI.
Enables TARS to navigate to objects using vision and LLM planning.
"""

import json
import re
import time
from modules.module_config import load_config
from modules.module_messageQue import queue_message
from modules.module_vision import capture_image_as_base64
from modules.module_llm import get_vision_completion

# === Constants ===
CONFIG = load_config()

def navigate_to_object(target_name: str) -> str:
    """
    Main navigation function that iteratively moves TARS toward a target object.
    
    Parameters:
    - target_name (str): Name of the target object to navigate to (e.g., "fridge", "door").
    
    Returns:
    - str: Status message indicating success or failure.
    """
    nav_config = CONFIG.get('NAVIGATION', {})
    if not nav_config.get('enabled', False):
        return "Navigation is disabled in configuration."
    
    max_iterations = int(nav_config.get('max_iterations', 20))
    max_steps_per_iteration = int(nav_config.get('max_steps_per_iteration', 5))
    stuck_threshold = int(nav_config.get('stuck_threshold', 3))
    use_vision_api = nav_config.get('use_vision_api', True)
    
    queue_message(f"NAV: Starting navigation to {target_name}")
    
    stuck_count = 0
    previous_analysis = None
    
    for iteration in range(max_iterations):
        queue_message(f"NAV: Iteration {iteration + 1}/{max_iterations}")
        
        try:
            # Capture and analyze current view
            image_base64 = capture_image_as_base64()
            analysis = _capture_and_analyze(target_name, image_base64, use_vision_api)
            
            if not analysis:
                queue_message(f"NAV: Failed to analyze image at iteration {iteration + 1}")
                continue
            
            # Check if target is reached
            if _is_target_reached(analysis, target_name):
                queue_message(f"NAV: Target '{target_name}' reached!")
                return f"Successfully navigated to {target_name}."
            
            # Check for stuck condition (same analysis as previous)
            if previous_analysis and analysis.get('scene_description') == previous_analysis.get('scene_description'):
                stuck_count += 1
                if stuck_count >= stuck_threshold:
                    queue_message(f"NAV: Stuck detected after {stuck_count} iterations. Trying alternative approach...")
                    # Try turning to find alternative path
                    execute_movement("turnLeft", 2)
                    time.sleep(1)
                    stuck_count = 0
                    continue
            else:
                stuck_count = 0
            
            previous_analysis = analysis
            
            # Generate movement plan
            movement_plan = _generate_movement_plan(analysis, target_name, max_steps_per_iteration)
            
            if not movement_plan:
                queue_message(f"NAV: Failed to generate movement plan at iteration {iteration + 1}")
                continue
            
            # Execute movement plan
            success = _execute_movement_plan(movement_plan)
            
            if not success:
                queue_message(f"NAV: Movement execution failed at iteration {iteration + 1}")
                continue
            
            # Wait for movement to complete
            time.sleep(2)
            
        except Exception as e:
            queue_message(f"NAV: Error at iteration {iteration + 1}: {e}")
            continue
    
    return f"Navigation to {target_name} timed out after {max_iterations} iterations."


def _capture_and_analyze(target: str, image_base64: str, use_vision_api: bool) -> dict:
    """
    Analyze the current view using vision-LLM to understand the scene and locate the target.
    
    Parameters:
    - target (str): Name of the target object.
    - image_base64 (str): Base64-encoded image.
    - use_vision_api (bool): Whether to use vision API or BLIP fallback.
    
    Returns:
    - dict: Analysis containing target visibility, direction, distance, and scene description.
    """
    prompt = f"""Analyze this image and provide information about locating a {target}.

Respond with a JSON object in this exact format:
{{
    "target_visible": true/false,
    "direction": "left"/"right"/"center"/"forward"/"behind"/"not_visible",
    "distance": "close"/"medium"/"far"/"unknown",
    "scene_description": "brief description of what you see",
    "obstacles": "description of any obstacles in the path",
    "recommended_action": "turnLeft"/"turnRight"/"stepForward"/"stop"
}}

If the {target} is not visible, estimate the most likely direction based on the scene context.
Be specific about the direction and distance if the target is visible."""
    
    try:
        if use_vision_api:
            response = get_vision_completion(prompt, image_base64)
        else:
            # Fallback to BLIP + text LLM
            from modules.module_vision import get_image_caption_from_base64
            caption = get_image_caption_from_base64(image_base64)
            enhanced_prompt = f"{prompt}\n\nImage description: {caption}"
            from modules.module_llm import raw_complete_llm
            response = raw_complete_llm(enhanced_prompt, istext=True)
        
        if not response:
            return None
        
        # Parse JSON from response
        analysis = _parse_analysis_response(response)
        return analysis
        
    except Exception as e:
        queue_message(f"NAV: Error in vision analysis: {e}")
        return None


def _parse_analysis_response(response: str) -> dict:
    """
    Parse the LLM response to extract analysis JSON.
    
    Parameters:
    - response (str): LLM response text.
    
    Returns:
    - dict: Parsed analysis dictionary.
    """
    try:
        # Try to extract JSON from response
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            analysis = json.loads(json_str)
            
            # Validate required fields
            required_fields = ['target_visible', 'direction', 'distance', 'scene_description']
            for field in required_fields:
                if field not in analysis:
                    analysis[field] = None
            
            # Set defaults for optional fields
            if 'obstacles' not in analysis:
                analysis['obstacles'] = "none"
            if 'recommended_action' not in analysis:
                analysis['recommended_action'] = "stepForward"
            
            return analysis
        else:
            # Fallback: create basic analysis from text
            return {
                "target_visible": False,
                "direction": "unknown",
                "distance": "unknown",
                "scene_description": response[:100],
                "obstacles": "unknown",
                "recommended_action": "stepForward"
            }
    except json.JSONDecodeError as e:
        queue_message(f"NAV: Failed to parse JSON from response: {e}")
        return {
            "target_visible": False,
            "direction": "unknown",
            "distance": "unknown",
            "scene_description": response[:100] if response else "unknown",
            "obstacles": "unknown",
            "recommended_action": "stepForward"
        }


def _generate_movement_plan(analysis: dict, target: str, max_steps: int) -> dict:
    """
    Generate a movement plan based on the vision analysis.
    
    Parameters:
    - analysis (dict): Vision analysis results.
    - target (str): Target object name.
    - max_steps (int): Maximum steps per iteration.
    
    Returns:
    - dict: Movement plan with movement type and number of times.
    """
    direction = analysis.get('direction', 'unknown')
    distance = analysis.get('distance', 'unknown')
    recommended_action = analysis.get('recommended_action', 'stepForward')
    
    # Map direction to movement
    movement_map = {
        'left': 'turnLeft',
        'right': 'turnRight',
        'center': 'stepForward',
        'forward': 'stepForward',
        'behind': 'turnLeft',  # Turn around
        'not_visible': 'stepForward'  # Move forward to explore
    }
    
    # Determine movement type
    if recommended_action in ['turnLeft', 'turnRight', 'stepForward']:
        movement = recommended_action
    else:
        movement = movement_map.get(direction, 'stepForward')
    
    # Determine number of steps/turns based on distance and direction
    if movement == 'stepForward':
        if distance == 'close':
            times = min(1, max_steps)
        elif distance == 'medium':
            times = min(2, max_steps)
        elif distance == 'far':
            times = min(3, max_steps)
        else:
            times = min(2, max_steps)
    else:  # turnLeft or turnRight
        if direction in ['left', 'right']:
            times = 1  # Small turn
        elif direction == 'behind':
            times = 2  # 180 degree turn
        else:
            times = 1
    
    return {
        "movement": movement,
        "times": times
    }


def _execute_movement_plan(plan: dict) -> bool:
    """
    Execute the movement plan.
    
    Parameters:
    - plan (dict): Movement plan with 'movement' and 'times' keys.
    
    Returns:
    - bool: True if execution started successfully, False otherwise.
    """
    try:
        # Import here to avoid circular dependency
        from modules.module_engine import execute_movement
        
        movement = plan.get('movement')
        times = plan.get('times', 1)
        
        if movement not in ['turnLeft', 'turnRight', 'stepForward', 'poseaction', 'unposeaction']:
            queue_message(f"NAV: Invalid movement type: {movement}")
            return False
        
        queue_message(f"NAV: Executing {movement} x {times}")
        execute_movement(movement, times)
        return True
        
    except Exception as e:
        queue_message(f"NAV: Error executing movement plan: {e}")
        return False


def _is_target_reached(analysis: dict, target: str) -> bool:
    """
    Check if the target has been reached based on the analysis.
    
    Parameters:
    - analysis (dict): Vision analysis results.
    - target (str): Target object name.
    
    Returns:
    - bool: True if target is reached, False otherwise.
    """
    nav_config = CONFIG.get('NAVIGATION', {})
    stopping_criteria = nav_config.get('stopping_criteria', 'llm_based')
    
    if stopping_criteria == 'llm_based':
        # Use LLM's assessment
        target_visible = analysis.get('target_visible', False)
        distance = analysis.get('distance', 'unknown')
        
        # Consider reached if target is visible and close
        if target_visible and distance == 'close':
            return True
        
        # Also check if scene description indicates we're at the target
        scene_desc = analysis.get('scene_description', '').lower()
        if target.lower() in scene_desc and distance in ['close', 'medium']:
            return True
        
        return False
    else:
        # Distance-based (would require additional sensors)
        distance = analysis.get('distance', 'unknown')
        return distance == 'close' and analysis.get('target_visible', False)

