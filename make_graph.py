from graphviz import Digraph

# Create a Digraph object
dot = Digraph(comment='Project Flow')

# Nodes
dot.node('A', 'Camera / Motion Input')
dot.node('B', 'Motion Detection')
dot.node('C', 'Dynamic Queue\n(groups of frames with motion & timestamps)')
dot.node('D', 'Fall Detection\n- Detect Person\n- How long on ground')
dot.node('E', 'YOLO + Pose + ViT\nFace Detection & Person ID')
dot.node('F', 'Person ID')
dot.node('G', 'Main Processing')
dot.node('H', 'Count People')
dot.node('I', 'Action Module')
dot.node('J', 'New Action -> Timestamp\n- Change in Presence/Location\n- Sitting/Walking/Standing Duration')
dot.node('K', 'Output Filter\nTasks')

# Edges
dot.edges(['AB'])
dot.edge('B', 'C')
dot.edge('B', 'D')
dot.edge('D', 'E')
dot.edge('E', 'F')
dot.edge('F', 'G', label='Person Change?')
dot.edge('G', 'H')
dot.edge('H', 'I', label='> 0 People')
dot.edge('I', 'J')
dot.edge('C', 'G')
dot.edge('G', 'K')

# Save and render
file_path = 'vision-architecture'
dot.render(file_path, format='png', cleanup=True)

file_path + '.png'