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

class DartLLMAnnotator:
    def __init__(self):
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
        
        # Use a hierarchical layout
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot') if nx.nx_agraph.graphviz_layout is not None else nx.spring_layout(G)
        
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
            
        try:
            with open(output_path, 'w') as f:
                for task in self.data:
                    f.write(json.dumps(task) + '\n')
            return f"Saved {len(self.data)} tasks to {output_path}"
        except Exception as e:
            return f"Error saving file: {str(e)}"

# Create the Gradio interface
def create_interface():
    annotator = DartLLMAnnotator()
    
    # Define interface components
    with gr.Blocks(title="DART-LLM Annotation Tool") as interface:
        gr.Markdown("# DART-LLM Annotation Tool")
        gr.Markdown("Visualize and annotate DART-LLM task decompositions")
        
        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(label="Upload JSON Data")
                json_input = gr.Textbox(label="Or paste JSON file path or content", lines=1)
                load_btn = gr.Button("Load Data")
                
                save_path = gr.Textbox(label="Save Path", value="output.json")
                save_btn = gr.Button("Save Data")
                save_output = gr.Textbox(label="Save Status")
                
            with gr.Column(scale=2):
                task_info = gr.Textbox(label="Task Info", interactive=False)
                
                with gr.Tabs():
                    with gr.TabItem("Task Text"):
                        task_text = gr.TextArea(label="Task Text", interactive=True, lines=3)
                    
                    with gr.TabItem("Task Output"):
                        output_json = gr.Code(label="Task Output (JSON)", language="json", interactive=True, lines=15)
                
                update_btn = gr.Button("Update Task")
                update_status = gr.Textbox(label="Update Status")
                
                with gr.Row():
                    prev_btn = gr.Button("Previous Task")
                    next_btn = gr.Button("Next Task")
                
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