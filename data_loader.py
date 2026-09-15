"""
Data loader for NFT review inputs.
Reads and returns contents of required input files as strings.
"""

import json
import os


def load_inputs():
    """
    Load input files for NFT review.

    Returns:
        dict: Contains the following keys:
            - application_overview: str (markdown content)
            - architecture: str (markdown content)
            - business_volumes: str (JSON content as string)
            - user_stories: str (markdown content)
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "input")

    # Read application overview
    with open(os.path.join(input_dir, "application_overview.md"), "r") as f:
        application_overview = f.read()

    # Read architecture
    with open(os.path.join(input_dir, "architecture.md"), "r") as f:
        architecture = f.read()

    # Read business volumes (JSON file, convert to string)
    with open(os.path.join(input_dir, "business_volumes.json"), "r") as f:
        business_volumes = f.read()

    # Read user stories
    with open(os.path.join(input_dir, "user_stories.md"), "r") as f:
        user_stories = f.read()

    return {
        "application_overview": application_overview,
        "architecture": architecture,
        "business_volumes": business_volumes,
        "user_stories": user_stories,
    }


if __name__ == "__main__":
    # Simple test to verify data loading
    data = load_inputs()
    print("✓ Data loaded successfully")
    print(f"  - application_overview: {len(data['application_overview'])} chars")
    print(f"  - architecture: {len(data['architecture'])} chars")
    print(f"  - business_volumes: {len(data['business_volumes'])} chars")
    print(f"  - user_stories: {len(data['user_stories'])} chars")
