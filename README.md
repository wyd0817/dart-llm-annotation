# DART-LLM Annotation Tool

A visualization and annotation tool for DART-LLM tasks, inspired by the [COIN annotation tool](https://github.com/coin-dataset/annotation-tool).

## 🔧 Features

* Load DART-LLM task data from JSON files
* Visualize task decompositions as Directed Acyclic Graphs (DAGs)
* Navigate through task collections
* Edit task descriptions in-browser
* Save modified task collections to new JSON

## 📋 Prerequisites

* **Python 3.8+**
* **System dependencies** (for advanced graph layouts):

  * Graphviz C library and headers

### Install Graphviz

* **macOS** (Homebrew):

  ```bash
  brew install graphviz
  ```

* **Ubuntu/Debian**:

  ```bash
  sudo apt-get update
  sudo apt-get install graphviz libgraphviz-dev
  ```

* **Conda** (cross-platform):

  ```bash
  conda install -c conda-forge graphviz
  ```

*If Graphviz is not available, the tool will fall back to NetworkX’s default layout.*

## ⚙️ Installation

We recommend using **Conda** for a smooth setup.

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/dart-llm-annotation.git
   cd dart-llm-annotation
   ```

2. **Create and activate environment**:

   ```bash
   conda create -n dart-llm-annotation python=3.10
   conda activate dart-llm-annotation
   ```

3. **Install Python dependencies**:

   ```bash
   # Core dependencies
   conda install -c conda-forge gradio networkx matplotlib pillow numpy

   # Optional: PyGraphviz for richer layouts
   conda install -c conda-forge pygraphviz

   # Or, using pip
   pip install -r requirements.txt
   ```

## 🚀 Usage

Run the web application:

```bash
python app.py
```

Open your browser to `http://localhost:7860` to access the Gradio UI.

### Workflow

1. **Load Data**: Upload a JSON file or specify a file path containing DART-LLM tasks.
2. **Navigate**: Use Previous / Next buttons to browse tasks.
3. **Visualize**: View each task’s decomposition as an interactive DAG.
4. **Edit**: Click on text fields to modify task descriptions.
5. **Save**: Export the updated collection to disk.

## 📂 Data Format

Input files should be newline-delimited JSON, where each line follows:

```json
{
  "ID": "dart_llm-L1-T1-001",
  "text": "Move Excavator 1 and Dump Truck 1 to the puddle area, then command all robots to avoid the puddle.",
  "output": {
    "tasks": [
      {
        "task": "move_excavator_and_dump_to_puddle",
        "instruction_function": {
          "name": "target_area_for_specific_robots",
          "robot_ids": ["robot_excavator_01", "robot_dump_truck_01"],
          "dependencies": [],
          "object_keywords": ["puddle1"]
        }
      },
      {
        "task": "avoid_puddle_all_robots",
        "instruction_function": {
          "name": "avoid_areas_for_all_robots",
          "robot_type": ["dump_truck", "excavator"],
          "dependencies": ["move_excavator_and_dump_to_puddle"],
          "object_keywords": ["puddle1"]
        }
      }
    ]
  }
}
```

Each line must be a valid JSON object matching this schema.

## 🔗 Dataset

The DART-LLM task collections are hosted on Hugging Face:

* [YongdongWang/dart\_llm\_tasks](https://huggingface.co/datasets/YongdongWang/dart_llm_tasks)

## ⚖️ License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
