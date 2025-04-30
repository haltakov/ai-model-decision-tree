import csv
import graphviz
import os

def load_models(filename='models.csv'):
    """Loads model data from a CSV file."""
    models = []
    try:
        with open(filename, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            features = [field.strip() for field in reader.fieldnames if field.strip() != 'Model Name']
            for row in reader:
                # Clean up whitespace in model names and feature values
                cleaned_row = {k.strip(): v.strip() for k, v in row.items()}
                models.append(cleaned_row)
        if not models:
            print(f"Error: No data found in {filename}.")
            return None, None
        if not features:
            print(f"Error: No feature columns found in {filename} (besides 'Model Name').")
            return None, None
        return models, features
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return None, None
    except Exception as e:
        print(f"An error occurred while reading {filename}: {e}")
        return None, None

def build_tree(dot, models, features, current_feature_index=0, parent_node_id="root", decision_path="", max_models_leaf=3):
    """Recursively builds the decision tree with pruning."""
    # Pruning or Leaf node condition
    if current_feature_index >= len(features) or not models or len(models) <= max_models_leaf:
        model_names = "\n".join(sorted([m['Model Name'] for m in models])) if models else "None"
        leaf_id = f"leaf_{parent_node_id}_{decision_path.replace(' ', '_').replace('->','_')}"
        # Use a different color for leaf nodes
        dot.node(leaf_id, label=f"Models ({len(models)}):\n{model_names}", shape='box', style='filled', fillcolor='lightgrey')
        # Get the label for the edge (Yes/Maybe/No)
        edge_label = decision_path.split('->')[-1].strip() if '->' in decision_path else ""
        dot.edge(parent_node_id, leaf_id, label=edge_label)
        return

    feature = features[current_feature_index]
    # Sanitize feature name for node ID
    sanitized_feature = "".join(c if c.isalnum() else "_" for c in feature)
    node_id = f"node_{current_feature_index}_{parent_node_id}_{sanitized_feature}_{decision_path.replace(' ', '_').replace('->','_')}"
    dot.node(node_id, label=f"Care about\n{feature}?") # Default node style is applied

    # Get the label for the edge leading to this node
    edge_label = decision_path.split('->')[-1].strip() if '->' in decision_path else ""
    if parent_node_id != "root":
        dot.edge(parent_node_id, node_id, label=edge_label)

    # Define filters for branches
    filters = {
        "Yes": lambda m: m.get(feature, '').lower() == 'yes',
        "Maybe": lambda m: m.get(feature, '').lower() in ['yes', 'partially'],
        "No": lambda m: m.get(feature, '').lower() == 'no' # Only models explicitly marked 'no'
    }

    for decision, filter_func in filters.items():
        filtered_models = [m for m in models if filter_func(m)]
        # Only create a branch if it leads to a non-empty set of models
        # AND if that set is different from the parent set (unless it's the only option)
        # Avoid creating redundant branches if filtering doesn't change the model list
        if filtered_models and (len(filtered_models) < len(models) or len(filters) == 1):
             build_tree(dot, filtered_models, features, current_feature_index + 1, node_id, f"{decision_path}->{decision}", max_models_leaf)
        elif not filtered_models:
             # Optional: Add a node indicating no models match this path
             no_match_id = f"no_match_{node_id}_{decision}"
             dot.node(no_match_id, label="No matching models", shape='box', style='filled', fillcolor='salmon')
             dot.edge(node_id, no_match_id, label=decision)


def main():
    # Read the potentially larger CSV
    models, features = load_models()
    if models is None or features is None:
        return

    # Initialize the graph
    dot = graphviz.Digraph(comment='AI Model Decision Tree', format='png')
    dot.attr(rankdir='TB') # Top-to-bottom might be better for deeper trees
    dot.attr('node', shape='ellipse', style='filled', fillcolor='lightblue', fontsize='10')
    dot.attr('edge', fontsize='10')
    dot.attr(label='AI Model Selector\n(Branches pruned when <= 3 models remain)', fontsize='12')


    # Start building the tree with pruning enabled (e.g., max_models_leaf=3)
    build_tree(dot, models, features, max_models_leaf=3)

    # Render the graph
    output_filename = 'model_decision_tree_pruned'
    try:
        dot.render(output_filename, view=False) # Set view=False to not open automatically
        print(f"Decision tree saved as '{output_filename}.gv' and rendered to '{output_filename}.gv.png'")
        print("You may need to install Graphviz (https://graphviz.org/download/) to view the .gv file or render it to other formats (e.g., pdf, svg).")
        print(f"To render manually: dot -Tpng {output_filename}.gv -o {output_filename}.png")
    except graphviz.exceptions.ExecutableNotFound:
        print(f"Error: 'dot' command not found. Please install Graphviz and ensure it's in your system's PATH.")
        print(f"Graphviz source saved as '{output_filename}.gv'. You can render it manually after installing Graphviz.")
        dot.save(output_filename + '.gv') # Save the source file even if rendering fails
    except Exception as e:
        print(f"An error occurred during graph rendering: {e}")
        print(f"Graphviz source saved as '{output_filename}.gv'.")
        dot.save(output_filename + '.gv')


if __name__ == "__main__":
    main()
