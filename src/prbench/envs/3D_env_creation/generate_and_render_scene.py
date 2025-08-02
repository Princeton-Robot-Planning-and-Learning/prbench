import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import mujoco.viewer
import os
import time

def generate_dynamic_scene(base_xml_path: str, num_objects: int, output_filename: str) -> str:
    """
    Parses a base XML, adds a number of objects at random positions on a table,
    and saves the result to a new XML file.

    Args:
        base_xml_path (str): Path to the template XML file (e.g., table_arena.xml).
        num_objects (int): The number of objects to add to the scene.
        output_filename (str): The filename for the generated XML.

    Returns:
        str: The absolute path to the newly created XML file.
    """
    print(f"Loading base arena from: {base_xml_path}")
    tree = ET.parse(base_xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    if worldbody is None:
        raise ValueError("Could not find <worldbody> in the XML file.")

    # Table properties from table_arena.xml:
    # <body name="table" pos="0 0 0.4">
    #   <geom name="table_collision" size="0.5 0.6 0.025" ... />
    table_pos = np.array([0, 0, 0.4])
    table_half_size = np.array([0.5, 0.6, 0.025])
    table_top_z = table_pos[2] + table_half_size[2]

    # Define a list of possible object shapes and their properties
    object_templates = [
        {"type": "sphere", "size": "0.04"},
        {"type": "box", "size": "0.03 0.03 0.03"},
        {"type": "cylinder", "size": "0.03 0.05"},
    ]

    print(f"Generating {num_objects} random objects...")
    generated_positions = []
    margin = 0.05  # Keep objects away from the table edge

    for i in range(num_objects):
        # Choose a random object shape
        template = np.random.choice(object_templates)
        
        # Keep trying to find a non-overlapping position
        while True:
            # Random position on the table
            x_pos = np.random.uniform(-table_half_size[0] + margin, table_half_size[0] - margin)
            y_pos = np.random.uniform(-table_half_size[1] + margin, table_half_size[1] - margin)
            pos = np.array([x_pos, y_pos])
            
            # Check for overlap with other generated objects
            is_overlapping = False
            for existing_pos in generated_positions:
                if np.linalg.norm(pos - existing_pos) < 0.1:  # Minimum distance between objects
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                generated_positions.append(pos)
                break
        
        # Create the object body
        obj_name = f"object_{i+1}"
        obj_pos_str = f"{pos[0]} {pos[1]} {table_top_z + 0.05}"
        
        body = ET.SubElement(worldbody, "body", name=obj_name, pos=obj_pos_str)
        ET.SubElement(body, "freejoint")
        
        # Add a random color material
        color = f"{np.random.rand()} {np.random.rand()} {np.random.rand()} 1"
        material_name = f"mat_{obj_name}"
        asset = root.find("asset")
        if asset is None: # Should not happen with the provided file
            asset = ET.SubElement(root, "asset")
        ET.SubElement(asset, "material", name=material_name, rgba=color, specular="0.5", shininess="0.5")
        
        # Add the geom
        ET.SubElement(
            body,
            "geom",
            type=template["type"],
            size=template["size"],
            material=material_name,
            density="100",
            solimp="0.99 0.99 0.01",
            solref="0.01 1"
        )

    # Write the modified tree to a new file
    output_path = os.path.join(os.path.dirname(base_xml_path), output_filename)
    tree.write(output_path)
    print(f"Successfully generated dynamic scene at: {output_path}")
    
    return os.path.abspath(output_path)


def main():
    """Main function to run the script."""
    base_xml = "../models/stanford_tidybot/table_arena.xml"
    dynamic_xml_filename = "dynamic_scene.xml"
    
    try:
        num_objects_str = input("Enter the number of objects to generate on the table: ")
        num_objects = int(num_objects_str)
        if num_objects < 0:
            raise ValueError("Number of objects cannot be negative.")
    except (ValueError, TypeError):
        print("Invalid input. Defaulting to 5 objects.")
        num_objects = 5

    # Generate the new scene
    dynamic_xml_path = generate_dynamic_scene(base_xml, num_objects, dynamic_xml_filename)

    # Load and render the generated scene
    print("\nLoading and rendering the generated scene...")
    try:
        model = mujoco.MjModel.from_xml_path(dynamic_xml_path)
        data = mujoco.MjData(model)
        
        print("Dynamic scene loaded successfully!")
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("Viewer launched. Running for 30 seconds...")
            start_time = time.time()
            while viewer.is_running() and time.time() - start_time < 30:
                step_start = time.time()
                mujoco.mj_step(model, data)
                viewer.sync()
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
        
        print("Viewer closed.")
    except Exception as e:
        print(f"\nAn error occurred while loading or rendering the dynamic scene: {e}")
    finally:
        # Clean up the generated file
        if os.path.exists(dynamic_xml_path):
            os.remove(dynamic_xml_path)
            print(f"Cleaned up temporary file: {dynamic_xml_path}")


if __name__ == "__main__":
    main() 