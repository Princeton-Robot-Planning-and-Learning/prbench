import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import mujoco.viewer
import os
import time
from pathlib import Path

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

def create_scene_with_objects(base_xml_path: str, objects_to_add: dict, output_filename: str, libero_assets_path: str) -> str:
    """
    Parses a base XML, merges assets and bodies from specified object XMLs,
    places them on the table, and saves to a new file.
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
    
    generated_positions = []
    margin = 0.1  # Margin from the table edge

    print(f"Processing {len(objects_to_add)} objects to add to the scene...")
    for obj_name, obj_xml_path in objects_to_add.items():
        print(f" -> Adding '{obj_name}' from {obj_xml_path}")
        
        obj_tree = ET.parse(obj_xml_path)
        obj_root = obj_tree.getroot()
        rename_map = {}

        # --- 1. Merge Assets with corrected paths and names ---
        obj_asset_root = obj_root.find("asset")
        if obj_asset_root is not None:
            # First pass: find all named assets and create a renaming map
            for asset_elem in obj_asset_root:
                original_name = asset_elem.get("name")
                if original_name:
                    rename_map[original_name] = f"{obj_name}_{original_name}"

            # Second pass: apply renames and fix file paths
            for asset_elem in obj_asset_root:
                # Rename the asset itself
                original_name = asset_elem.get("name")
                if original_name in rename_map:
                    asset_elem.set("name", rename_map[original_name])
                
                # Update references to other renamed assets (e.g., material referencing a texture)
                for attr_ref in ['texture', 'material', 'mesh']:
                    original_ref = asset_elem.get(attr_ref)
                    if original_ref in rename_map:
                        asset_elem.set(attr_ref, rename_map[original_ref])

                # Fix file paths to be relative to the new scene file
                if 'file' in asset_elem.attrib:
                    original_file = asset_elem.attrib['file']
                    obj_xml_dir = Path(obj_xml_path).parent
                    abs_asset_file_path = (obj_xml_dir / original_file).resolve()
                    rel_to_assets_root = abs_asset_file_path.relative_to(Path(libero_assets_path).resolve())
                    final_path = relative_assets_path / rel_to_assets_root
                    asset_elem.set('file', str(final_path))
                
                asset_root.append(asset_elem)

        # --- 2. Find a non-overlapping position on the table ---
        while True:
            x_pos = np.random.uniform(-table_half_size[0] + margin, table_half_size[0] - margin)
            y_pos = np.random.uniform(-table_half_size[1] + margin, table_half_size[1] - margin)
            pos = np.array([x_pos, y_pos])
            
            is_overlapping = any(np.linalg.norm(pos - ex_pos) < 0.15 for ex_pos in generated_positions)
            if not is_overlapping:
                generated_positions.append(pos)
                break
        
        # --- 3. Copy Object Body contents into a new, clean Body ---
        obj_worldbody = obj_root.find("worldbody")
        if obj_worldbody is not None:
            # Find the nested structure: body > body[@name='object']
            outer_body = obj_worldbody.find("body")
            if outer_body is not None:
                source_body = outer_body.find("body[@name='object']")
                if source_body is None:
                    # If no nested object body, use the outer body directly
                    source_body = outer_body

                if source_body is not None:
                    new_body = ET.SubElement(worldbody, "body", 
                                             name=f"{obj_name}_instance", 
                                             pos=f"{pos[0]} {pos[1]} {table_top_z + 0.02}")

                    # Copy children (geoms, joints, sites) from the source body
                    # But exclude nested body elements to flatten the structure
                    for child in list(source_body):
                        if child.tag in ['geom', 'site', 'joint']:
                            # Update references in all elements recursively
                            _update_element_references(child, rename_map)
                            # Also rename sites to avoid conflicts between object instances
                            if child.tag == 'site' and 'name' in child.attrib:
                                original_site_name = child.attrib['name']
                                child.attrib['name'] = f"{obj_name}_{original_site_name}"
                            new_body.append(child)
                    
                    # Don't copy sites from outer body to avoid duplicates
                    # Sites should already be included from the inner object body

    # Write the modified tree to a new file
    output_path = os.path.join(os.path.dirname(base_xml_path), output_filename)
    # Use pretty_print for readability
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"\nSuccessfully generated dynamic scene at: {output_path}")
    
    return os.path.abspath(output_path)

def _update_element_references(element, rename_map):
    """Recursively update asset references in an element and its children."""
    # Update references in the current element
    for attr_ref in ['material', 'mesh', 'texture']:
        original_ref = element.get(attr_ref)
        if original_ref and original_ref in rename_map:
            element.set(attr_ref, rename_map[original_ref])
    
    # Recursively update children
    for child in element:
        _update_element_references(child, rename_map)

def main():
    """Main function to run the script."""
    # Construct paths relative to the script's location
    script_dir = Path(__file__).parent
    libero_assets_path = str((script_dir / "../models/libero_assets").resolve())
    base_arena_xml = str((script_dir / "../models/stanford_tidybot/table_arena.xml").resolve())
    output_filename = "generated_scene_with_objects.xml"
    
    available_objects = find_available_objects(libero_assets_path)
    if not available_objects:
        print("Error: No objects found. Please check the 'libero_assets' path.")
        return

    print("\nAvailable objects:")
    for name in sorted(available_objects.keys()):
        print(f"- {name}")
    
    try:
        object_names_str = input("\nEnter the names of objects to add (comma-separated): ")
        selected_names = [name.strip() for name in object_names_str.split(',') if name.strip()]
        
        objects_to_add = {}
        for name in selected_names:
            if name in available_objects:
                objects_to_add[name] = available_objects[name]
            else:
                print(f"Warning: Object '{name}' not found and will be skipped.")
        
        if not objects_to_add:
            print("No valid objects selected. Exiting.")
            return

    except Exception as e:
        print(f"An error occurred during input processing: {e}")
        return

    # Generate the new scene
    dynamic_xml_path = create_scene_with_objects(base_arena_xml, objects_to_add, output_filename, libero_assets_path)

    # Load and render
    print("\nLoading and rendering the generated scene...")
    try:
        model = mujoco.MjModel.from_xml_path(dynamic_xml_path)
        data = mujoco.MjData(model)
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("Viewer launched. Running for 60 seconds...")
            start_time = time.time()
            while viewer.is_running() and time.time() - start_time < 60:
                viewer.sync()
        
    except Exception as e:
        print(f"\nError loading/rendering dynamic scene: {e}")
    finally:
        pass
        # if os.path.exists(dynamic_xml_path):
        #     os.remove(dynamic_xml_path)
        #     print(f"Cleaned up temporary file: {dynamic_xml_path}")

if __name__ == "__main__":
    main() 