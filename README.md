# Unitree G1 Apple Toss Simulation (ME206)

In this project, we built a MuJoCo physics simulation of a Unitree G1 humanoid robot tossing an apple back and forth between its hands. Both palms stay facing upward throughout the motion. One hand dips down, scoops, and tosses the apple into the air, while the other hand catches it, cushions the impact, and closes its fingers around it before tossing it back.

The robot model files are not uploaded in this repository because they are very large. They are taken from Google DeepMind's MuJoCo Menagerie: https://github.com/google-deepmind/mujoco_menagerie

Youtube link of simulation : https://youtube.com/shorts/bWUeG0htUq8?si=jGpstovOSMzI3bdd

# Files in This Repository
- `g1_apple_toss.py` — The main simulation script.
- `requirements.txt` — Lists the required Python packages (`mujoco` and `numpy`).
- `README.md` — Setup instructions and explanation of how the simulation works.

# Software Requirements
- Python 3.10 or higher
- Visual Studio Code (with the Microsoft Python extension) or any standard terminal
- Python libraries: `mujoco` and `numpy`
- The `unitree_g1` robot model from `mujoco_menagerie`

# How to Set Up and Run in VS Code
1. Download or clone this repository into a new folder on your computer.
2. Download MuJoCo Menagerie from the GitHub link above (**Code > Download ZIP**) and extract it inside the same project folder, or clone it using the terminal:
   git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git
3. Open your project folder in VS Code (`File > Open Folder...`).
4. Open the integrated terminal in VS Code and install the required Python packages:
   pip install -r requirements.txt
5. Run the code: g1_apple_toss.py

When the MuJoCo viewer opens, the robot will continuously toss the apple between both hands. You can left-click and drag to rotate the view, right-click and drag to pan, and use the scroll wheel to zoom in and out. Close the viewer window to stop the simulation.

# How the Code Works
When you run the script, it sets up the scene by fixing the robot's waist in place so it stays standing upright without falling over, and places an apple in its hand. The code moves the arms to follow target points while keeping both palms facing upward, and opens or closes the fingers to hold the apple. For every throw, the holding hand dips down a little, moves quickly upward, and releases the apple at the exact speed and angle needed for it to land in the other palm. When the apple arrives, the catching hand moves down slightly to soften the impact, closes its fingers around the apple, and holds it steady so it does not bounce out. After a short pause, the two hands switch roles and repeat the toss, while MuJoCo handles all the gravity and movement physics in the background.