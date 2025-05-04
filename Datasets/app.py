from flask import Flask, render_template_string, flash
import pandas as pd
import networkx as nx
from pyvis.network import Network
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for flash messages

def load_graph(csv_file):
    df = pd.read_csv(csv_file)
    G = nx.Graph()
    
    # Print columns for debugging
    print("Available columns:", df.columns.tolist())
    
    # Use statement as source and label as target
    for _, row in df.iterrows():
        # Convert label to string description for better visualization
        label_map = {
            0: "False",
            1: "Mostly False",
            2: "Half True",
            3: "Mostly True", 
            4: "True",
            5: "Pants on Fire"
        }
        label_str = label_map.get(row['label'], str(row['label']))
        
        # Add edge with the statement as source and the label as target
        G.add_edge(str(row['statement'])[:50] + "...", label_str)
    
    return G

def generate_graph_html(G):
    net = Network(height='700px', width='100%', notebook=True)
    net.from_nx(G)
    return net.generate_html()

@app.route('/')
def index():
    csv_path = 'train.csv'  # Ensure train.csv is in the same directory
    
    if not os.path.exists(csv_path):
        return "Error: CSV file not found"
    
    try:
        G = load_graph(csv_path)
        graph_content = generate_graph_html(G)
        
        html_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Graph Visualization</title>
        </head>
        <body>
            <h1>Graph Visualization</h1>
            {{ graph_content | safe }}
        </body>
        </html>
        """
        
        return render_template_string(html_template, graph_content=graph_content)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)