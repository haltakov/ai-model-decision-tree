import csv
import os
import math
from collections import Counter

try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False

def load_models(filename="models.csv"):
    """Loads model data from a CSV file."""
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        return None, None
    
    models = []
    features = []
    try:
        with open(filename, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            if not reader.fieldnames:
                print(f"Error: CSV file '{filename}' is empty or missing headers.")
                return None, None
            
            # Ensure 'Model Name' is the first column conceptually
            if 'Model Name' not in reader.fieldnames:
                 print(f"Error: 'Model Name' column not found in '{filename}'.")
                 return None, None
                 
            features = [f for f in reader.fieldnames if f != 'Model Name']
            if not features:
                print(f"Error: No feature columns found (besides 'Model Name') in '{filename}'.")
                return None, None

            for i, row in enumerate(reader):
                 # Basic validation: Ensure 'Model Name' exists and is not empty
                if 'Model Name' not in row or not row['Model Name']:
                    print(f"Warning: Skipping row {i+2} due to missing or empty 'Model Name'.")
                    continue
                
                # Normalize feature values and check for missing ones
                valid_row = True
                processed_row = {'Model Name': row['Model Name']}
                for feature in features:
                    if feature not in row or row[feature] is None:
                         print(f"Warning: Skipping model '{row.get('Model Name', 'Unknown')}' due to missing data for feature '{feature}'.")
                         valid_row = False
                         break
                    processed_row[feature] = str(row[feature]).lower().strip()
                    if processed_row[feature] not in ['yes', 'no', 'partially']:
                         print(f"Warning: Model '{row.get('Model Name', 'Unknown')}', Feature '{feature}': Invalid value '{processed_row[feature]}'. Treating as 'unknown/no'.")
                         # Decide how to handle invalid values, e.g., treat as 'no' or skip row
                         processed_row[feature] = 'no' # Or set valid_row = False and break

                if valid_row:
                    models.append(processed_row)
                
        if not models:
            print(f"Error: No valid model data found in '{filename}'.")
            return None, None
            
        return models, features
    except Exception as e:
        print(f"Error reading CSV file '{filename}': {e}")
        return None, None

def calculate_split_score(models, feature):
    """
    Calculates a score for a feature based on how well it splits the models.
    A simple heuristic: maximize the minimum number of models eliminated 
    by either a 'yes' or 'no' answer.
    'yes' preference includes 'yes' and 'partially'.
    """
    if not models:
        return 0
        
    counts = {'yes_partial': 0, 'no': 0}
    for model in models:
        value = model.get(feature, 'no') # Default to 'no' if missing unexpectedly
        if value in ['yes', 'partially']:
            counts['yes_partial'] += 1
        elif value == 'no':
            counts['no'] += 1
            
    # If all models have the same value for this feature, it's not useful for splitting
    if counts['yes_partial'] == len(models) or counts['no'] == len(models):
        return 0
        
    # Score: Higher is better - rewards features that split decisively.
    # We want to maximize the number of models we *know* we can eliminate.
    # If user says 'yes', we eliminate 'no' models. If user says 'no', we eliminate 'yes_partial' models.
    # The minimum of these two counts represents the guaranteed minimum elimination.
    return min(counts['yes_partial'], counts['no'])


def find_best_feature_to_ask(models, available_features):
    """Finds the best feature to ask about next based on the split score."""
    best_feature = None
    max_score = -1

    # Shuffle features slightly to break ties arbitrarily but consistently-ish
    # Or sort them alphabetically for perfect consistency
    sorted_features = sorted(list(available_features))

    for feature in sorted_features:
        score = calculate_split_score(models, feature)
        if score > max_score:
            max_score = score
            best_feature = feature
            
    # If no feature provides any split (score > 0), return None
    return best_feature if max_score > 0 else None


def interactive_selection(all_models, all_features):
    """Guides the user through an interactive model selection process."""
    current_models = list(all_models)
    available_features = set(all_features)
    
    print("--- Interactive Model Selection ---")
    print("Answer 'yes', 'no', or 'any' for each feature.")

    while len(current_models) > 1 and available_features:
        best_feature = find_best_feature_to_ask(current_models, available_features)

        if best_feature is None:
            # No single feature can further distinguish the remaining models
            print("Cannot narrow down further based on remaining features.")
            break
            
        available_features.remove(best_feature) # Ask about each feature at most once

        while True:
            response = input(f"Need '{best_feature}'? (yes/no/any) [{len(current_models)} models remain]: ").lower().strip()
            if response in ['yes', 'y']:
                preference = 'yes'
                break
            elif response in ['no', 'n']:
                preference = 'no'
                break
            elif response in ['any', 'a', '']:
                preference = 'any'
                break
            else:
                print("Invalid input. Please enter 'yes', 'no', or 'any'.")

        if preference == 'any':
            continue # Don't filter, move to the next best question

        # Filter models based on the answer
        new_filtered_models = []
        for model in current_models:
            model_feature_value = model.get(best_feature, 'no')
            
            if preference == 'yes':
                 # Accept 'yes' or 'partially' if user wants the feature
                if model_feature_value in ['yes', 'partially']:
                    new_filtered_models.append(model)
            elif preference == 'no':
                 # Accept only 'no' if user doesn't want the feature
                if model_feature_value == 'no':
                    new_filtered_models.append(model)
                    
        current_models = new_filtered_models
        
        if not current_models:
            print("No models match your criteria after this step.")
            return [] # Early exit

    return current_models


node_counter = 0 # Global counter for unique node IDs

def build_diagram_recursive(dot, models, available_features, parent_node_id, edge_label):
    """Recursively builds the Graphviz diagram structure."""
    global node_counter
    
    # Base cases
    if not models:
        node_id = f"leaf_{node_counter}"
        dot.node(node_id, label="No Match", shape="box", style="filled", fillcolor="lightcoral")
        if parent_node_id is not None:
             dot.edge(parent_node_id, node_id, label=edge_label)
        node_counter += 1
        return

    if len(models) == 1:
        model_name = models[0]['Model Name']
        node_id = f"leaf_{node_counter}"
        dot.node(node_id, label=f"Result: {model_name}", shape="box", style="filled", fillcolor="lightgreen")
        if parent_node_id is not None:
            dot.edge(parent_node_id, node_id, label=edge_label)
        node_counter += 1
        return

    best_feature = find_best_feature_to_ask(models, available_features)

    if best_feature is None:
         # Multiple models left, but no more features to distinguish
        model_names = "\n".join([m['Model Name'] for m in models])
        node_id = f"leaf_{node_counter}"
        label = f"Ambiguous ({len(models)} models): {model_names}"
        # Add truncated list if too long
        if len(model_names) > 100:
             label = f"Ambiguous ({len(models)} models): {model_names[:100]}..."

        dot.node(node_id, label=label, shape="box", style="filled", fillcolor="lightblue")
        if parent_node_id is not None:
            dot.edge(parent_node_id, node_id, label=edge_label)
        node_counter += 1
        return

    # Recursive step: Create node for the question
    current_node_id = f"node_{node_counter}"
    node_label = f"Need '{best_feature}'?\n({len(models)} models)"
    dot.node(current_node_id, label=node_label, shape="ellipse")
    if parent_node_id is not None:
        dot.edge(parent_node_id, current_node_id, label=edge_label)
    node_counter += 1

    remaining_features = available_features - {best_feature}

    # Branch for 'yes'
    yes_models = [m for m in models if m.get(best_feature, 'no') in ['yes', 'partially']]
    build_diagram_recursive(dot, yes_models, remaining_features, current_node_id, "yes")

    # Branch for 'no'
    no_models = [m for m in models if m.get(best_feature, 'no') == 'no']
    build_diagram_recursive(dot, no_models, remaining_features, current_node_id, "no")


def generate_tree_diagram(models, features, filename="model_decision_tree"):
    """Generates a Graphviz diagram of the potential decision process."""
    global node_counter
    if not GRAPHVIZ_AVAILABLE:
        print("Graphviz library not found. Skipping diagram generation.")
        print("To install: pip install graphviz (and potentially system package)")
        return
        
    if not models or not features:
        print("Cannot generate diagram: No valid model data or features loaded.")
        return

    print(f"Generating decision tree diagram to '{filename}.gv' and '{filename}.gv.png'...")
    
    dot = graphviz.Digraph(comment='Model Selection Decision Tree', format='png')
    dot.attr(rankdir='TB', size='8,8') # Top-to-bottom layout
    dot.attr('node', shape='ellipse', style='filled', fillcolor='gray95')
    dot.attr('edge', fontsize='10')

    node_counter = 0 # Reset counter for each generation
    build_diagram_recursive(dot, list(models), set(features), None, "") 
    
    try:
        dot.render(filename, view=False, cleanup=True)
        print(f"Diagram successfully generated: '{filename}.gv.png'")
    except graphviz.backend.execute.ExecutableNotFound:
        print("Error: Graphviz executable not found.")
        print("Please ensure Graphviz is installed and in your system's PATH.")
        print("(e.g., 'brew install graphviz' on macOS, 'sudo apt-get install graphviz' on Debian/Ubuntu)")
    except Exception as e:
        print(f"Error generating diagram: {e}")


def main():
    """Main function to run the model selector."""
    models, features = load_models() # Uses default "models.csv"
    
    if models is None or features is None:
        return # Exit if loading failed

    print(f"Loaded {len(models)} models with {len(features)} features: {', '.join(features)}")
    
    # Generate the diagram based on the initial data
    generate_tree_diagram(models, features)

    # Start interactive selection
    matching_models = interactive_selection(models, features)
    
    print("--- Final Matching Models ---")
    if matching_models:
        if len(matching_models) == 1:
            print(f"The best matching model is: {matching_models[0]['Model Name']}")
        else:
             print("The following models match your criteria:")
             for model in matching_models:
                 print(f"- {model['Model Name']}")
    else:
        # Specific message was already printed during interaction if list became empty
        if len(models) > 0 : # Avoid printing this if loading failed initially
             print("No models ultimately matched all your specified criteria.")

if __name__ == "__main__":
    main() 