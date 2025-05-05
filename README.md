# Ai-end-sem-project
import argparse
import torch
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import requests
from torchvision import transforms
import openai
from puzzle_model import PuzzleMLP  # Your model from Exercise 1

# Set your OpenAI API key
openai.api_key = "YOUR_API_KEY"  # Replace with your actual API key

class PuzzleSolver:
    def __init__(self, checkpoint_path, config_path='config.json'):
        self.load_model(checkpoint_path, config_path)
        
    def load_model(self, checkpoint_path, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.model = PuzzleMLP(self.config.get("model", {}))
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.eval()
        
    def preprocess_image(self, image_path):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        image = Image.open(image_path).convert('L')
        return transform(image).unsqueeze(0)
    
    def extract_puzzle_state(self, image_tensor):
        with torch.no_grad():
            # Process the image to extract the 3x3 grid of digits
            # This implementation depends on your model's architecture
            output = self.model(image_tensor)
            
            # Placeholder - implement based on your model
            # Should return a 3x3 grid representing the puzzle state
            puzzle_state = self.process_model_output(output)
            
        return puzzle_state
    
    def process_model_output(self, output):
        # Placeholder - implement based on your model's output format
        # In a real implementation, this would convert the model's output
        # to a 3x3 grid representing the puzzle state
        return output
    
    def query_llm_for_states(self, source_state, goal_state):
        # Format the states for the prompt
        source_str = self.format_state_for_prompt(source_state)
        goal_str = self.format_state_for_prompt(goal_state)
        
        # Create the prompt for the LLM
        prompt = f"""
        I'm solving an 8-puzzle problem using A* search.
        
        <tag>
        Source state:
        {source_str}
        
        Goal state:
        {goal_str}
        </tag>
        
        Please provide all intermediate states at each step of the solution path using uninformed A* search.
        Only return the states, not the explanation or reasoning.
        """
        
        # Call the OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Extract and parse the response
        llm_response = response.choices[0].message.content
        states = self.parse_llm_response(llm_response, source_state, goal_state)
        
        return states
    
    def format_state_for_prompt(self, state):
        # Convert the state to a string format for the prompt
        state_str = ""
        for row in state:
            state_str += " ".join(str(cell) for cell in row) + "\n"
        return state_str
    
    def parse_llm_response(self, response, source_state, goal_state):
        # Parse the LLM response to extract intermediate states
        # Add source and goal states to the sequence
        states = [source_state]
        
        # Extract states from the response
        # This is a simplified parser - adjust based on your LLM's output format
        lines = response.strip().split('\n')
        current_state = []
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_state:
                    states.append(current_state)
                    current_state = []
                continue
                
            # Parse a line of the state
            row = [int(cell) if cell.isdigit() else 0 for cell in line.split()]
            if len(row) == 3:  # Ensure it's a valid row
                current_state.append(row)
                
            # If we have a complete state, add it
            if len(current_state) == 3:
                states.append(current_state)
                current_state = []
        
        # Add goal state if it's not already included
        if states[-1] != goal_state:
            states.append(goal_state)
            
        return states
    
    def solve_puzzle(self, source_path, goal_path):
        # Load and preprocess images
        source_image = self.preprocess_image(source_path)
        goal_image = self.preprocess_image(goal_path)
        
        # Extract puzzle states using the neural network
        source_state = self.extract_puzzle_state(source_image)
        goal_state = self.extract_puzzle_state(goal_image)
        
        # Query LLM for intermediate states
        states = self.query_llm_for_states(source_state, goal_state)
        
        # Save states to JSON
        with open('states.json', 'w') as f:
            json.dump(states, f, indent=4)
        
        print(f"Solution saved to states.json with {len(states)} intermediate states.")
        
        return states, source_path, goal_path

class PuzzleVisualizer:
    def __init__(self):
        pass
    
    def visualize_state(self, state):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)
        ax.set_xticks([])
        ax.set_yticks([])
        
        for i in range(3):
            for j in range(3):
                if state[i][j] != 0:
                    ax.text(j + 0.5, 2.5 - i, str(state[i][j]), 
                            fontsize=24, ha='center', va='center')
                else:
                    ax.add_patch(plt.Rectangle((j, 2-i), 1, 1, fill=True, color='lightgray'))
        
        for i in range(4):
            ax.axhline(y=i, color='black', linewidth=2)
            ax.axvline(x=i, color='black', linewidth=2)
        
        return fig
    
    def create_animation(self, states, output_path='puzzle_solution.mp4'):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)
        ax.set_xticks([])
        ax.set_yticks([])
        
        def update(frame):
            ax.clear()
            ax.set_xlim(0, 3)
            ax.set_ylim(0, 3)
            ax.set_xticks([])
            ax.set_yticks([])
            
            state = states[frame]
            for i in range(3):
                for j in range(3):
                    if state[i][j] != 0:
                        ax.text(j + 0.5, 2.5 - i, str(state[i][j]), 
                                fontsize=24, ha='center', va='center')
                    else:
                        ax.add_patch(plt.Rectangle((j, 2-i), 1, 1, fill=True, color='lightgray'))
            
            for i in range(4):
                ax.axhline(y=i, color='black', linewidth=2)
                ax.axvline(x=i, color='black', linewidth=2)
            
            ax.set_title(f'Step {frame}')
        
        ani = animation.FuncAnimation(fig, update, frames=len(states), interval=1000)
        ani.save(output_path, writer='ffmpeg')
        return ani
    
    def visualize_solution(self, states, source_path, goal_path, output_dir='./results'):
        os.makedirs(output_dir, exist_ok=True)
        
        # Create visualization
        plt.figure(figsize=(15, 5))
        
        # Display source image and model inference
        plt.subplot(1, 3, 1)
        source_img = Image.open(source_path)
        plt.imshow(source_img, cmap='gray')
        plt.title('Source State Image')
        plt.axis('off')
        
        # Display goal image and model inference
        plt.subplot(1, 3, 2)
        goal_img = Image.open(goal_path)
        plt.imshow(goal_img, cmap='gray')
        plt.title('Goal State Image')
        plt.axis('off')
        
        # Create and display animation
        animation_path = os.path.join(output_dir, 'puzzle_solution.mp4')
        self.create_animation(states, animation_path)
        
        # Display a static image of the solution path
        plt.subplot(1, 3, 3)
        plt.imshow(plt.imread(source_path), cmap='gray')
        plt.title('Solution Path (See Animation)')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'visualization.png'))
        plt.show()
        
        print(f"Visualization saved to {output_dir}")
        print(f"Animation saved to {animation_path}")

def main():
    parser = argparse.ArgumentParser(description='8-Puzzle Solver with Neural Networks and LLM')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to neural network checkpoint')
    parser.add_argument('--config', type=str, default='config.json', help='Path to model config file')
    parser.add_argument('--source', type=str, required=True, help='Path to source state image')
    parser.add_argument('--goal', type=str, required=True, help='Path to goal state image')
    parser.add_argument('--output', type=str, default='./results', help='Output directory for results')
    args = parser.parse_args()
    
    # Create solver and visualizer
    solver = PuzzleSolver(args.checkpoint, args.config)
    visualizer = PuzzleVisualizer()
    
    # Solve the puzzle
    states, source_path, goal_path = solver.solve_puzzle(args.source, args.goal)
    
    # Visualize the solution
    visualizer.visualize_solution(states, source_path, goal_path, args.output)

if __name__ == "_main__":
    main()
