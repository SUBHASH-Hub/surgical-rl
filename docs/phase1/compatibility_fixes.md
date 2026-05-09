# LapGym Compatibility Fixes for SOFA v25.12 + NumPy 2.x

## Problem
LapGym was written in 2022 against SOFA v22 and NumPy 1.x.
Running on SOFA v25.12 and NumPy 2.2.6 requires the following fixes.

## Fix 1: np.NaN removed in NumPy 2.0
**File:** sofa_env/scenes/search_for_point/search_for_point_env.py  
**Change:** `np.NaN` → `np.nan`  
**Command:** `sed -i 's/np\.NaN/np.nan/g' <file>`

## Fix 2: DefaultPipeline alias removed in SOFA v24.12
**File:** sofa_env/sofa_templates/scene_header.py  
**Change:** `"DefaultPipeline"` → `"CollisionPipeline"`  
**Command:** `sed -i 's/"DefaultPipeline"/"CollisionPipeline"/g' <file>`  
**Affects:** pick_and_place, ligating_loop, rope_threading, and all scenes
using add_scene_header() with collisions enabled.

## Fix 3: gym vs gymnasium API
**Approach:** Compatibility shim in all demo scripts  
**Code:** `sys.modules['gym'] = gymnasium`

## Notes
- These fixes are applied directly to LapGym source in lap_gym/
- lap_gym/ is not committed to surgical-rl — it is a workspace dependency
- If lap_gym is re-cloned, these fixes must be reapplied
- tissue_retraction is unaffected by Fix 2 (uses scene_has_collisions=False)