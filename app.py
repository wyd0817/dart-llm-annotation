# dart-llm-annotation/app.py

import json
import gradio as gr
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import os
import io
from PIL import Image
import numpy as np
import warnings

# Check for optional dependencies
try:
    import pygraphviz
    HAS_PYGRAPHVIZ = True
except ImportError:
    HAS_PYGRAPHVIZ = False
    warnings.warn("PyGraphviz not found. Will use NetworkX's spring layout for visualization.")

try:
    from networkx.drawing.nx_pydot import graphviz_layout
    HAS_PYDOT = True
except ImportError:
    HAS_PYDOT = False
    if not HAS_PYGRAPHVIZ:
        warnings.warn("Neither PyGraphviz nor pydot found. Graph visualization will be limited.")

# Try importing huggingface_hub
try:
    from huggingface_hub import hf_hub_download
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False
    warnings.warn("huggingface_hub not found. HuggingFace loading will not be available.")

class DartLLMAnnotator:
    def __init__(self):
        """Initialize the annotator"""
        self.data = []
        self.current_index = 0
        self.graph = None
        
    def load_data(self, file_obj):
        """Load JSON data from file or JSON string"""
        if isinstance(file_obj, str):
            try:
                # Try to parse as direct JSON string
                self.data = [json.loads(line) for line in file_obj.strip().split('\n') if line.strip()]
                return f"Loaded {len(self.data)} tasks from JSON string."
            except json.JSONDecodeError:
                # Try to read as file path
                try:
                    with open(file_obj, 'r') as f:
                        self.data = [json.loads(line) for line in f.readlines() if line.strip()]
                    return f"Loaded {len(self.data)} tasks from file: {file_obj}"
                except (FileNotFoundError, json.JSONDecodeError):
                    return "Error: Invalid JSON or file not found."
        else:
            # Handle file upload from Gradio
            content = file_obj.decode('utf-8')
            self.data = [json.loads(line) for line in content.strip().split('\n') if line.strip()]
            self.current_index = 0
            return f"Loaded {len(self.data)} tasks from uploaded file."
    
    def load_from_huggingface(self, repo_id, filename="data/dart_llm_tasks.jsonl"):
        """Load data directly from HuggingFace repository"""
        if not HAS_HF_HUB:
            return "Error: huggingface_hub package not installed. Run 'pip install huggingface_hub' first."
        
        try:
            # Create a temporary directory for downloaded files
            os.makedirs("temp", exist_ok=True)
            
            # Download the file from HuggingFace
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset"
            )
            
            # Process JSONL file - read line by line
            self.data = []
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            task = json.loads(line)
                            self.data.append(task)
                        except json.JSONDecodeError as e:
                            print(f"Error parsing line: {str(e)}")
                            continue
            
            self.current_index = 0
            return f"Loaded {len(self.data)} tasks from HuggingFace: {repo_id}/{filename}"
        except Exception as e:
            return f"Error loading from HuggingFace: {str(e)}"
    
    def get_current_task(self):
        """Get the current task data"""
        if not self.data or self.current_index >= len(self.data):
            return None
        return self.data[self.current_index]
    
    def navigate(self, direction):
        """Navigate through tasks"""
        if not self.data:
            return "No data loaded.", None, None, None
            
        if direction == "Next" and self.current_index < len(self.data) - 1:
            self.current_index += 1
        elif direction == "Previous" and self.current_index > 0:
            self.current_index -= 1
        
        current_task = self.get_current_task()
        if current_task:
            task_info = f"Task {self.current_index + 1}/{len(self.data)}: {current_task['ID']}"
            task_text = current_task['text']
            # Format the output JSON with indentation for better readability
            output_json = json.dumps(current_task['output'], indent=2)
            dag_image = self.visualize_dag(current_task)
            return task_info, task_text, output_json, dag_image
        
        return "No task available.", "", "", None
    
    def visualize_dag(self, task_data):
        """Visualize the task DAG using NetworkX"""
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes and edges
        tasks = task_data['output']['tasks']
        
        # Add all tasks as nodes first
        for task in tasks:
            task_name = task['task']
            G.add_node(task_name)
            
            # Add node attributes
            function_name = task['instruction_function']['name']
            G.nodes[task_name]['function'] = function_name
            
            # Add robot information
            if 'robot_ids' in task['instruction_function']:
                robots = ', '.join(task['instruction_function']['robot_ids'])
                G.nodes[task_name]['robots'] = robots
            elif 'robot_type' in task['instruction_function']:
                robots = ', '.join(task['instruction_function']['robot_type'])
                G.nodes[task_name]['robots'] = f"All {robots}s"
                
            # Add target information
            if 'object_keywords' in task['instruction_function'] and task['instruction_function']['object_keywords']:
                targets = ', '.join(task['instruction_function']['object_keywords'])
                G.nodes[task_name]['targets'] = targets
        
        # Add edges based on dependencies
        for task in tasks:
            task_name = task['task']
            dependencies = task['instruction_function']['dependencies']
            
            for dep in dependencies:
                G.add_edge(dep, task_name)
        
        # Draw the graph
        plt.figure(figsize=(12, 8))
        
        # Try to use graphviz_layout, but fall back to spring_layout if not available
        try:
            # Check if we have pygraphviz 
            if hasattr(nx, 'nx_agraph') and hasattr(nx.nx_agraph, 'graphviz_layout'):
                pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
            else:
                # Try to use pydot instead
                if hasattr(nx, 'drawing') and hasattr(nx.drawing, 'nx_pydot') and hasattr(nx.drawing.nx_pydot, 'graphviz_layout'):
                    pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
                else:
                    # Fall back to spring_layout
                    pos = nx.spring_layout(G, seed=42)
        except Exception:
            # Fall back to spring_layout
            pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue', alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=2, alpha=0.7, edge_color='gray', arrowsize=20)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
        
        # Draw node attributes as additional text
        for node, (x, y) in pos.items():
            attrs = []
            if 'function' in G.nodes[node]:
                attrs.append(f"Function: {G.nodes[node]['function']}")
            if 'robots' in G.nodes[node]:
                attrs.append(f"Robots: {G.nodes[node]['robots']}")
            if 'targets' in G.nodes[node]:
                attrs.append(f"Targets: {G.nodes[node]['targets']}")
                
            plt.text(x, y-0.15, '\n'.join(attrs), 
                     bbox=dict(facecolor='white', alpha=0.7),
                     horizontalalignment='center', fontsize=8)
        
        plt.title(f"Task DAG: {task_data['ID']}")
        plt.axis('off')
        
        # Save the figure to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        # Convert to numpy array for Gradio
        image = Image.open(buf)
        return np.array(image)
    
    def edit_task(self, task_text, output_json):
        """Edit the task text and output"""
        if not self.data or self.current_index >= len(self.data):
            return "No task to edit.", None
            
        current_task = self.data[self.current_index]
        current_task['text'] = task_text
        
        # Try to parse and update the output JSON
        try:
            output_data = json.loads(output_json)
            current_task['output'] = output_data
            dag_image = self.visualize_dag(current_task)
            return f"Updated task {current_task['ID']}.", dag_image
        except json.JSONDecodeError as e:
            return f"Error parsing JSON: {str(e)}", None
    
    def save_data(self, output_path):
        """Save the current data to a file"""
        if not self.data:
            return "No data to save."
        
        # Check if the save path is empty    
        if not output_path or output_path.strip() == "":
            return "Error: Please provide a valid save path."
            
        try:
            # Create directory if it doesn't exist
            if os.path.dirname(output_path):
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                for task in self.data:
                    f.write(json.dumps(task) + '\n')
            return f"Saved {len(self.data)} tasks to {output_path}"
        except Exception as e:
            return f"Error saving file: {str(e)}"

# Create the Gradio interface
def create_interface():
    """Create the Gradio web interface"""
    annotator = DartLLMAnnotator()
    
    # Create static directory if it doesn't exist
    os.makedirs("static", exist_ok=True)
    
    # Create CSS file if it doesn't exist
    css_path = os.path.join("static", "styles.css")
    if not os.path.exists(css_path):
        with open(css_path, "w") as f:
            f.write("""
/* Overall compact styling */
.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
}

/* Compact header */
h1 {
    margin-top: 0.5rem !important;
    margin-bottom: 0.5rem !important;
    font-size: 1.5rem !important;
}

/* Compact elements */
.file-loader input, .file-loader button {
    height: 30px !important;
    line-height: 30px !important;
}

button {
    min-width: 60px !important;
    height: 30px !important;
    line-height: 30px !important;
}

/* Compact text areas */
textarea {
    min-height: 60px !important;
}

/* Compact code editor */
.cm-editor {
    max-height: 300px !important;
}
""")
    
    # Define interface components
    with gr.Blocks(title="DART-LLM Annotation Tool", css=css_path) as interface:
        gr.Markdown("# DART-LLM Annotation Tool")
        
        # Compact file loading row similar to the COIN tool
        with gr.Row():
            file_input = gr.File(label="", elem_id="file_input", visible=False)
            json_input = gr.Textbox(label="", placeholder="JSON file path", scale=3)
            load_btn = gr.Button("Load", size="sm")
        
        # Add HuggingFace loading option
        with gr.Row():
            hf_repo_input = gr.Textbox(label="", placeholder="HuggingFace repo", 
                                     scale=3, value="YongdongWang/dart_llm_tasks")
            hf_filename_input = gr.Textbox(label="", placeholder="Filename with path", 
                                         scale=2, value="data/dart_llm_tasks.jsonl")
            hf_load_btn = gr.Button("Load from HF", size="sm")
        
        with gr.Row():
            with gr.Column(scale=1):
                task_info = gr.Textbox(label="Task Info", interactive=False)
                
                with gr.Tabs():
                    with gr.TabItem("Task Text"):
                        task_text = gr.TextArea(label="", placeholder="Task description", lines=3)
                    
                    with gr.TabItem("Task Output"):
                        output_json = gr.Code(label="", language="json", lines=10)
                
                with gr.Row():
                    prev_btn = gr.Button("Previous", size="sm")
                    next_btn = gr.Button("Next", size="sm")
                    update_btn = gr.Button("Update", size="sm")
                
                update_status = gr.Textbox(label="Status", lines=1)
                
                with gr.Row():
                    save_path = gr.Textbox(label="", placeholder="Save path", scale=3, value="annotated_tasks.jsonl")
                    save_btn = gr.Button("Save", size="sm")
                
                save_output = gr.Textbox(label="", lines=1)
            
            with gr.Column(scale=2):
                dag_output = gr.Image(label="Task DAG Visualization")
        
        # Define interactions
        load_btn.click(
            fn=lambda f, t: annotator.load_data(f) if f else annotator.load_data(t), 
            inputs=[file_input, json_input], 
            outputs=task_info
        )
        
        load_btn.click(
            fn=lambda: annotator.navigate("Next"), 
            inputs=None, 
            outputs=[task_info, task_text, output_json, dag_output]
        )
        
        # HuggingFace loading
        hf_load_btn.click(
            fn=annotator.load_from_huggingface,
            inputs=[hf_repo_input, hf_filename_input],
            outputs=task_info
        )
        
        hf_load_btn.click(
            fn=lambda: annotator.navigate("Next"),
            inputs=None,
            outputs=[task_info, task_text, output_json, dag_output]
        )
        
        prev_btn.click(
            fn=lambda: annotator.navigate("Previous"), 
            inputs=None, 
            outputs=[task_info, task_text, output_json, dag_output]
        )
        
        next_btn.click(
            fn=lambda: annotator.navigate("Next"), 
            inputs=None, 
            outputs=[task_info, task_text, output_json, dag_output]
        )
        
        update_btn.click(
            fn=annotator.edit_task,
            inputs=[task_text, output_json],
            outputs=[update_status, dag_output]
        )
        
        save_btn.click(
            fn=annotator.save_data,
            inputs=[save_path],
            outputs=[save_output]
        )
    
    return interface

# Run the application
if __name__ == "__main__":
    interface = create_interface()
    interface.launch()