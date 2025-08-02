import json
import os
import sys

# Add the directory containing the bddl_utils package to sys.path
bddl_utils_dir = os.path.abspath("prbench/src/prbench/envs/BDDL_utils.py")
sys.path.insert(0, bddl_utils_dir)

try:
    from bddl_utils import bddl_utils
    print("bddl_utils module imported from:", bddl_utils.__file__)
    print("bddl_utils attributes:", dir(bddl_utils))
except (ImportError, ModuleNotFoundError) as e:
    print(f"Error: Could not import the bddl_utils module from {bddl_utils_dir}")
    print(f"Import failed with error: {e}")
    sys.exit(1)

def test_bddl_parsing():
    """
    Parses the example BDDL file and creates a JSON output file.
    """
    # Define the path to the input BDDL file and the output JSON file.
    bddl_file = "example_suites/example_1.bddl"
    output_json_file = "example_1_parsed.json"

    # Verify that the BDDL file exists before trying to parse it.
    if not os.path.exists(bddl_file):
        print(f"Error: The specified BDDL file does not exist at '{bddl_file}'")
        return

    print(f"Parsing BDDL file: {bddl_file}")

    # Call the parsing function from the bddl_utils module.
    parsed_data = bddl_utils.prbench_parse_problem(bddl_file)

    # Write the resulting dictionary to a JSON file.
    print(f"Writing parsed data to: {output_json_file}")
    with open(output_json_file, "w") as f:
        json.dump(parsed_data, f, indent=4)

    print("\nScript completed successfully.")
    print(f"The parsed JSON file has been saved to: {os.path.abspath(output_json_file)}")


if __name__ == "__main__":
    test_bddl_parsing() 