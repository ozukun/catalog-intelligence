
import json

class schema_loader:

    def load_schema(self, json_file_path):
        with open(json_file_path, "r") as json_file:
            extraction_schema = json.load(json_file)
        return extraction_schema
    

if __name__ == "__main__":

    json_file_path = "context/extraction_schema.json"

    loader = schema_loader()
    extraction_schema = loader.load_schema(json_file_path)

    #print("Extraction Schema:", json.dumps(extraction_schema, indent=4))

    print(extraction_schema["attributes"].items()  )

    for attribute_name, definition in extraction_schema["attributes"].items() :
        print(attribute_name, definition["type"])

    if "schema_version" not in extraction_schema:
        raise ValueError("Missing schema_version")
    
    if  extraction_schema["domain"] != "beverages":
        raise ValueError("Missing beverages")
    
    if len(extraction_schema["attributes"]) == 0:
        raise ValueError("Missing attributes")