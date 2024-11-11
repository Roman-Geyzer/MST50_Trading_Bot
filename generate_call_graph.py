from pycallgraph2 import PyCallGraph
from pycallgraph2.output import GraphvizOutput

# Import your main script or modules
from MST50.main import main  

# Define the main function to profile
def main_gen():
    # Place any initial setup here if needed
    main()  # replace 'run' with your main function if needed

# Configure PyCallGraph with Graphviz
graphviz = GraphvizOutput()
graphviz.output_file = 'call_graph.png'  # Output file for the call graph

# Generate the call graph
with PyCallGraph(output=graphviz):
    main_gen()