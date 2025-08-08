import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import mujoco.viewer
import os
import time
from pathlib import Path
from typing import Optional, Tuple, Dict
from bddl_utils.bddl_utils import prbench_parse_problem

def find_available_objects(assets_path: str) -> dict:
    """Scans the asset directory to find all available objects and their XML paths."""
    print("Discovering available objects...")
    available_objects = {}
    
    # Directories within libero_assets to search for objects
    search_dirs = ["articulated_objects", "stable_scanned_objects", "stable_hope_objects"]
    
    for subdir in search_dirs:
        search_path = Path(assets_path) / subdir
        if not search_path.exists():
            continue
        
        for item in search_path.iterdir():
            if item.is_dir():
                object_name = item.name
                # The primary XML file usually shares the same name as the directory
                object_xml = item / f"{object_name}.xml"
                if object_xml.exists():
                    available_objects[object_name] = str(object_xml)
    
    print(f"Found {len(available_objects)} objects.")
    return available_objects

def create_scene_with_objects(base_xml_path: str, objects_to_add: dict, object_regions: dict, fixture_names: list, output_filename: str, libero_assets_path: str) -> str:
    """
    Parses a base XML, merges assets and bodies from specified object XMLs,
    places them on the table according to region data, and saves to a new file.
    """
    print(f"\nLoading base arena from: {base_xml_path}")
    tree = ET.parse(base_xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    asset_root = root.find("asset")

    if worldbody is None or asset_root is None:
        raise ValueError("Base XML must contain <worldbody> and <asset> tags.")

    # Determine paths for asset reconstruction
    output_dir = Path(base_xml_path).parent
    relative_assets_path = Path(os.path.relpath(libero_assets_path, output_dir))

    # Table properties from table_arena.xml
    table_pos = np.array([0, 0, 0.4])
    table_half_size = np.array([0.5, 0.6, 0.025])
    table_top_z = table_pos[2] + table_half_size[2]

    print(f"Processing {len(objects_to_add)} objects to add to the scene...")
    for obj_name, obj_xml_path in objects_to_add.items():
        print(f" -> Adding '{obj_name}' from {obj_xml_path}")
        
        obj_tree = ET.parse(obj_xml_path)
        obj_root = obj_tree.getroot()
        rename_map = {}

        # --- 1. Merge Assets with corrected paths and names ---
        obj_asset_root = obj_root.find("asset")
        if obj_asset_root is not None:
            for asset_elem in obj_asset_root:
                original_name = asset_elem.get("name")
                if original_name:
                    rename_map[original_name] = f"{obj_name}_{original_name}"
            
            for asset_elem in obj_asset_root:
                original_name = asset_elem.get("name")
                if original_name in rename_map:
                    asset_elem.set("name", rename_map[original_name])
                
                for attr_ref in ['texture', 'material', 'mesh']:
                    original_ref = asset_elem.get(attr_ref)
                    if original_ref in rename_map:
                        asset_elem.set(attr_ref, rename_map[original_ref])

                if 'file' in asset_elem.attrib:
                    original_file = asset_elem.attrib['file']
                    obj_xml_dir = Path(obj_xml_path).parent
                    abs_asset_file_path = (obj_xml_dir / original_file).resolve()
                    rel_to_assets_root = abs_asset_file_path.relative_to(Path(libero_assets_path).resolve())
                    final_path = relative_assets_path / rel_to_assets_root
                    asset_elem.set('file', str(final_path))
                
                asset_root.append(asset_elem)

        # --- 2. Get placement bounds from BDDL region ---
        region_name = next((r_name for r_name, r_data in object_regions.items() if obj_name in r_data["objects"]), None)
        if region_name and 'ranges' in object_regions.get(region_name, {}):
            bounds = object_regions[region_name]['ranges'][0] # Take the first range
            x_min, y_min, x_max, y_max = bounds
            x_pos = np.random.uniform(x_min, x_max)
            y_pos = np.random.uniform(y_min, y_max)
            pos = np.array([x_pos, y_pos])
        else:
             # Fallback to random placement if no region is defined
            margin = 0.1
            x_pos = np.random.uniform(-table_half_size[0] + margin, table_half_size[0] - margin)
            y_pos = np.random.uniform(-table_half_size[1] + margin, table_half_size[1] - margin)
            pos = np.array([x_pos, y_pos])
        
        # --- 3. Copy Object Body contents into a new, clean Body ---
        obj_worldbody = obj_root.find("worldbody")
        if obj_worldbody is not None:
            source_body = obj_worldbody.find(".//body[@name='object']") or obj_worldbody.find("body")

            if source_body is not None:
                new_body = ET.SubElement(worldbody, "body", 
                                         name=f"{obj_name}_instance", 
                                         pos=f"{pos[0]} {pos[1]} {table_top_z + 0.02}")
                
                # Add a freejoint if the object is not a fixture
                if obj_name not in fixture_names:
                    ET.SubElement(new_body, "freejoint")

                for child in list(source_body):
                    if child.tag in ['geom', 'site', 'joint']:
                        _update_element_references(child, rename_map)
                        if child.tag == 'site' and 'name' in child.attrib:
                            child.attrib['name'] = f"{obj_name}_{child.attrib['name']}"
                        new_body.append(child)

    # Write the modified tree to a new file
    output_path = os.path.join(os.path.dirname(base_xml_path), output_filename)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"\nSuccessfully generated dynamic scene at: {output_path}")
    
    return os.path.abspath(output_path)

def _update_element_references(element, rename_map):
    """Recursively update asset references in an element and its children."""
    for attr_ref in ['material', 'mesh', 'texture']:
        original_ref = element.get(attr_ref)
        if original_ref and original_ref in rename_map:
            element.set(attr_ref, rename_map[original_ref])
    
    for child in element:
        _update_element_references(child, rename_map)


def merge_additional_model_into_scene(scene_xml_path: str, additional_model_xml_path: str, output_filename: str, body_pos: Optional[Tuple[float, float, float]] = None, name_prefix: Optional[str] = None) -> str:
    """
    Merge another MuJoCo model XML (e.g., tidybot) into an existing scene XML by:
    - Merging <default> classes
    - Merging <asset> entries while prefixing names to avoid collisions and rewriting file paths relative to the scene
    - Appending the top-level body from the additional model's <worldbody> into the scene's <worldbody>
    - Merging <contact>, <tendon>, <equality>, and <actuator> sections

    Returns absolute path to the newly written merged XML.
    """
    scene_tree = ET.parse(scene_xml_path)
    scene_root = scene_tree.getroot()

    scene_output_dir = Path(scene_xml_path).parent

    # Ensure required roots exist in the scene
    scene_default = scene_root.find('default')
    if scene_default is None:
        scene_default = ET.SubElement(scene_root, 'default')

    scene_asset = scene_root.find('asset')
    if scene_asset is None:
        scene_asset = ET.SubElement(scene_root, 'asset')

    scene_worldbody = scene_root.find('worldbody')
    if scene_worldbody is None:
        scene_worldbody = ET.SubElement(scene_root, 'worldbody')

    # Parse the additional model
    add_tree = ET.parse(additional_model_xml_path)
    add_root = add_tree.getroot()

    model_name = Path(additional_model_xml_path).stem
    prefix = name_prefix if name_prefix is not None else model_name

    # 1) Merge defaults (append all children of <default>)
    add_default = add_root.find('default')
    if add_default is not None:
        for child in list(add_default):
            scene_default.append(child)

    # 2) Merge assets with name prefixing and file path rewriting
    rename_map: Dict[str, str] = {}

    add_asset = add_root.find('asset')
    if add_asset is not None:
        add_xml_dir = Path(additional_model_xml_path).parent
        # Resolve meshdir if specified in additional model compiler
        add_compiler = add_root.find('compiler')
        meshdir = add_compiler.get('meshdir') if add_compiler is not None else None
        base_asset_dir = (add_xml_dir / meshdir).resolve() if meshdir else add_xml_dir.resolve()
        for asset_elem in list(add_asset):
            # Determine original asset name
            original_name = asset_elem.get('name')
            inferred = False
            if original_name is None:
                # Infer from file basename if available
                file_attr = asset_elem.get('file')
                if file_attr:
                    original_name = Path(file_attr).stem
                    inferred = True
            # If we still don't have a name, skip renaming for this asset
            if original_name:
                new_name = f"{prefix}_{original_name}"
                rename_map[original_name] = new_name
                asset_elem.set('name', new_name)
            
            # Rewrite file path to be relative to the scene output dir
            if 'file' in asset_elem.attrib:
                original_file = asset_elem.attrib['file']
                abs_asset_file_path = (base_asset_dir / original_file).resolve()
                # Always compute relative path from the scene output directory, without requiring Python 3.12's Path.is_relative_to
                try:
                    rel_path = abs_asset_file_path.relative_to(scene_output_dir.resolve())
                except Exception:
                    rel_path = Path(os.path.relpath(abs_asset_file_path, scene_output_dir))
                asset_elem.set('file', str(rel_path))

            # Update intra-asset references (e.g., texture/material/mesh) within the asset element itself
            for attr_ref in ['texture', 'material', 'mesh']:
                ref = asset_elem.get(attr_ref)
                if ref and ref in rename_map:
                    asset_elem.set(attr_ref, rename_map[ref])

            scene_asset.append(asset_elem)

    # 3) Merge worldbody: copy the first top-level body from additional model, update references, set pos
    add_worldbody = add_root.find('worldbody')
    if add_worldbody is not None:
        add_top_body = add_worldbody.find('body')
        if add_top_body is not None:
            # Deep-copy the body
            new_body = ET.fromstring(ET.tostring(add_top_body))
            # Update material/mesh/texture references according to rename_map
            _update_element_references(new_body, rename_map)
            # Set position if provided
            if body_pos is not None:
                new_body.set('pos', f"{body_pos[0]} {body_pos[1]} {body_pos[2]}")
            scene_worldbody.append(new_body)

    # 4) Merge contact / tendon / equality / actuator sections
    def _merge_section(tag_name: str):
        scene_sec = scene_root.find(tag_name)
        add_sec = add_root.find(tag_name)
        if add_sec is None:
            return
        if scene_sec is None:
            scene_sec = ET.SubElement(scene_root, tag_name)
        for child in list(add_sec):
            # These sections typically reference bodies/joints by name; no renaming needed
            scene_sec.append(child)

    for tag in ['contact', 'tendon', 'equality', 'actuator']:
        _merge_section(tag)

    # Write merged model
    merged_output_path = scene_output_dir / output_filename
    ET.indent(scene_tree, space="  ")
    scene_tree.write(str(merged_output_path), encoding="utf-8", xml_declaration=True)

    print(f"\nSuccessfully merged additional model into scene: {merged_output_path}")
    return str(merged_output_path.resolve())


def main():
    """Main function to run the script."""
    # Construct paths relative to the script's location
    script_dir = Path(__file__).parent
    libero_assets_path = str((script_dir / "../models/libero_assets").resolve())
    base_arena_xml = str((script_dir / "../models/stanford_tidybot/table_arena.xml").resolve())
    bddl_file = str((script_dir / "example_suites/example_2.bddl").resolve())
    output_filename = "generated_scene_from_bddl.xml"

    # 1. Parse the BDDL file
    print(f"Parsing BDDL file: {bddl_file}")
    try:
        problem_data = prbench_parse_problem(bddl_file)
    except Exception as e:
        print(f"Error parsing BDDL file: {e}")
        return

    # 2. Find all available object assets
    available_objects = find_available_objects(libero_assets_path)
    if not available_objects:
        print("Error: No objects found. Please check the 'libero_assets' path.")
        return

    # 3. Determine which objects to add from the BDDL file
    objects_to_add = {}
    bddl_objects = problem_data.get("objects", {})
    for obj_type, obj_names in bddl_objects.items():
        for obj_name in obj_names:
            # We assume the BDDL object name can be mapped to an asset type.
            # E.g., 'ketchup' object uses 'ketchup' asset.
            if obj_type in available_objects:
                 objects_to_add[obj_name] = available_objects[obj_type]
            else:
                 print(f"Warning: Asset for object type '{obj_type}' not found. Cannot add '{obj_name}'.")

    if not objects_to_add:
        print("No valid objects from BDDL could be added. Exiting.")
        return

    # 4. Extract fixture names
    bddl_fixtures = problem_data.get("fixtures", {})
    fixture_names = [name for names in bddl_fixtures.values() for name in names]

    # 5. Map object instances to their initial regions
    object_regions = {}
    init_states = problem_data.get("initial_state", [])
    regions_data = problem_data.get("regions", {})

    for state in init_states:
        if state[0].lower() == 'on':
            obj_instance, region_name = state[1], state[2]
            if region_name in regions_data:
                if region_name not in object_regions:
                     object_regions[region_name] = regions_data[region_name]
                     object_regions[region_name]["objects"] = []
                object_regions[region_name]["objects"].append(obj_instance)


    # 6. Generate the new scene XML (objects on table)
    dynamic_xml_path = create_scene_with_objects(base_arena_xml, objects_to_add, object_regions, fixture_names, output_filename, libero_assets_path)

    # 7. Merge tidybot into the generated scene and produce a second XML
    tidybot_xml_path = str((script_dir / "../models/stanford_tidybot/tidybot.xml").resolve())
    merged_output_filename = "generated_scene_with_tidybot.xml"
    # Place the robot slightly behind the table along -x
    final_xml_path = merge_additional_model_into_scene(dynamic_xml_path, tidybot_xml_path, merged_output_filename, body_pos=(-1.6, 0.0, 0.0), name_prefix="tidybot")

    # 8. Load and render the merged scene
    print("\nLoading and rendering the generated scene with tidybot...")
    try:
        model = mujoco.MjModel.from_xml_path(final_xml_path)
        data = mujoco.MjData(model)
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("Viewer launched. Running for 60 seconds...")
            start_time = time.time()
            while viewer.is_running() and time.time() - start_time < 60:
                step_start = time.time()
                mujoco.mj_step(model, data)
                viewer.sync()
                # Rudimentary time keeping to run sim close to real time
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
        
    except Exception as e:
        print(f"\nError loading/rendering dynamic scene: {e}")
    finally:
        pass


if __name__ == "__main__":
    main() 