import streamlit as st

# Set a title for the web app
st.title("Simple Streamlit App")

# Add a text input box
user_input = st.text_input("Enter your name")

    
from InstructorEmbedding import INSTRUCTOR
model = INSTRUCTOR('hkunlp/instructor-large')

# Define the story
story = "Once upon a time, there was a brave knight named Sir Arthur. He embarked on a quest to rescue a princess from a wicked dragon."

# Define instructions for different story segments
instructions = [
    "Capture the bravery and courage of the knight:",
    "Describe the quest and its purpose:",
    "Portray the princess as in need of rescue:",
    "Highlight the wickedness of the dragon:"
]

# Encode story segments with corresponding instructions
embeddings = model.encode([[instruction, segment] for instruction, segment in zip(instructions, story.split("."))])

# Print the embeddings for each segment
for i, (instruction, segment) in enumerate(zip(instructions, story.split("."))):
    print(f"Instruction: {instruction}")
    print(f"Segment: {segment.strip()}")
    print(f"Embeddings: {embeddings[i]}")
    print()
