import sys
import json
import os
from jsonschema import validate, ValidationError
from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for

def validate_json_with_details(data, schema, data_file_path=""):
    """
    Validate JSON data against a schema with detailed error reporting.
    
    Args:
        data: The JSON data to validate
        schema: The JSON schema to validate against
        data_file_path: Optional path to the data file for better error messages
    """
    try:
        # Get the appropriate validator for the schema
        validator_class = validator_for(schema)
        validator = validator_class(schema)
        
        # Collect all validation errors
        errors = list(validator.iter_errors(data))
        
        if not errors:
            print("✅ JSON is valid against the schema!")
            return True
        
        print(f"❌ Found {len(errors)} validation error(s):\n")
        
        for i, error in enumerate(errors, 1):
            print(f"Error #{i}:")
            print(f"  Message: {error.message}")
            
            # Show the path to the failing element
            if error.path:
                path_str = " → ".join(str(p) for p in error.path)
                print(f"  Path: {path_str}")
                
                # Try to show the actual value at that path
                try:
                    current = data
                    for key in error.path:
                        current = current[key]
                    print(f"  Value: {json.dumps(current, indent=2)}")
                except (KeyError, TypeError, IndexError):
                    pass
            
            # Show the schema rule that failed
            if error.schema_path:
                schema_path_str = " → ".join(str(p) for p in error.schema_path)
                print(f"  Schema Rule: {schema_path_str}")
                try:
                    # Try to show the relevant schema constraint
                    current_schema = schema
                    for key in error.schema_path:
                        current_schema = current_schema[key]
                    print(f"  Constraint: {json.dumps(current_schema, indent=2)}")
                except (KeyError, TypeError, IndexError):
                    pass
            
            # Show context for relative errors
            if error.context:
                print(f"  Details:")
                for sub_error in error.context:
                    print(f"    - {sub_error.message}")
                    if sub_error.path:
                        sub_path = " → ".join(str(p) for p in sub_error.path)
                        print(f"      Path: {sub_path}")
            
            # Show the absolute JSON pointer if available
            if hasattr(error, 'absolute_path'):
                abs_path = list(error.absolute_path)
                if abs_path:
                    print(f"  JSON Pointer: /{'/'.join(str(p) for p in abs_path)}")
            
            print("-" * 60)
        
        # Additional help for common issues
        print("\n💡 Tips for debugging:")
        print("1. Check if required fields are missing")
        print("2. Verify data types match schema expectations")
        print("3. Check enum values against allowed values")
        print("4. Verify array items match the specified item schema")
        print("5. Check string formats (email, date, uri, etc.)")
        
        return False
        
    except SchemaError as e:
        print(f"❌ Invalid schema: {e.message}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {type(e).__name__}: {str(e)}")
        return False

def display_json_structure(data, indent=0, max_depth=3, current_depth=0):
    """Display a simplified structure of the JSON data for reference."""
    if current_depth >= max_depth:
        print("  " * indent + "...")
        return
    
    if isinstance(data, dict):
        for key, value in data.items():
            print("  " * indent + f"📁 {key}: ", end="")
            if isinstance(value, (dict, list)):
                print()
                display_json_structure(value, indent + 1, max_depth, current_depth + 1)
            else:
                print(f"{type(value).__name__} = {repr(str(value)[:50])}")
    elif isinstance(data, list):
        print("  " * indent + f"📋 List with {len(data)} items")
        if data and current_depth < max_depth - 1:
            # Show first item as example
            print("  " * indent + "  Example item:")
            display_json_structure(data[0], indent + 2, max_depth, current_depth + 1)
    else:
        print("  " * indent + f"{type(data).__name__} = {repr(str(data)[:50])}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python validate_json.py <data_file.json> <schema_file.json>")
        print("Example: python validate_json.py data.json schema.json")
        sys.exit(1)
    
    data_file = sys.argv[1]
    schema_file = sys.argv[2]
    
    # Check if files exist
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        sys.exit(1)
    if not os.path.exists(schema_file):
        print(f"❌ Schema file not found: {schema_file}")
        sys.exit(1)
    
    try:
        # Load schema
        with open(schema_file, 'r') as f:
            schema = json.load(f)
        
        # Load data
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        print(f"📄 Validating {data_file} against {schema_file}")
        print("=" * 60)
        
        # Display data structure for reference
        print("\n📊 JSON Structure Overview:")
        display_json_structure(data)
        print()
        
        # Perform validation with detailed error reporting
        is_valid = validate_json_with_details(data, schema, data_file)
        
        if is_valid:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in {data_file if 'data' in str(e) else schema_file}:")
        print(f"   Line {e.lineno}, Column {e.colno}: {e.msg}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {type(e).__name__}: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
