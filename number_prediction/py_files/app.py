import os
import pygame
import torch
import numpy as np
from model_training import CNN

# ----------------------------
# Pygame Drawing Constants
# ----------------------------
GRID_WIDTH = 560         
WIDTH, HEIGHT = 890, 560 
PANEL_START = GRID_WIDTH  

ROWS, COLS = 28, 28
CELL_SIZE = GRID_WIDTH // COLS

BG_COLOR = (255, 255, 255)
PIXEL_ON = (0, 0, 0)
PIXEL_OFF = (220, 220, 220)
GRID_COLOR = (180, 180, 180)

BRUSH_SIZE = 2  # square brush

# ----------------------------
# Helper Functions
# ----------------------------
def load_model():
    """
    Load the trained CNN model from file and set it to evaluation mode.

    Returns:
        model (CNN): The loaded CNN model ready for inference.
    """
    model = CNN()
    current_folder = os.path.dirname(os.path.abspath(__file__))
    parent_folder = os.path.dirname(current_folder)
    model_path = os.path.join(parent_folder, "model.pth")

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    print(f"Model loaded from {model_path}")
    return model

def recenter_grid(grid):
    """
    Recenter a 28x28 drawing grid so that its barycenter is at the center.
    This improves prediction when the digit is drawn off-center.

    Args:
        grid (list[list[int]]): 28x28 grid of 0/1 representing the drawing.

    Returns:
        shifted (np.ndarray): 28x28 grid shifted so that the drawing's center of mass is centered.
    """
    arr = np.array(grid, dtype=np.float32)

    if np.sum(arr) == 0:
        # nothing drawn
        return arr

    ys, xs = np.nonzero(arr)
    cx = np.mean(xs)
    cy = np.mean(ys)
    shift_x = int(arr.shape[1] / 2 - cx)
    shift_y = int(arr.shape[0] / 2 - cy)

    shifted = np.zeros_like(arr)
    for y, x in zip(ys, xs):
        new_y = y + shift_y
        new_x = x + shift_x
        if 0 <= new_y < arr.shape[0] and 0 <= new_x < arr.shape[1]:
            shifted[new_y, new_x] = arr[y, x]

    return shifted

def predict(grid, model):
    """
    Predict the digit drawn in the grid using the CNN model.

    Args:
        grid (list[list[int]]): 28x28 grid representing the drawing.
        model (CNN): The trained CNN model for digit classification.

    Returns:
        pred (int or None): The predicted digit (0-9) or None if grid is empty.
        probs (np.ndarray): Array of length 10 with probabilities for each digit.
    """
    centered_grid = recenter_grid(grid)
    arr = np.array(centered_grid, dtype="float32").reshape(1, 1, 28, 28)
    tensor = torch.from_numpy(arr)

    with torch.no_grad():
        if np.sum(arr) == 0:
            probs = np.zeros(10, dtype=np.float32)
            pred = None
        else:
            output = model(tensor)
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()
            pred = int(probs.argmax())

    return pred, probs

def probability_to_color(p):
    """
    Convert a probability value into a color gradient from red → yellow → green.

    Args:
        p (float): Probability value between 0 and 1.

    Returns:
        (tuple): RGB color corresponding to the probability.
    """
    if p < 0.5:
        ratio = p / 0.5
        r = 255
        g = int(255 * ratio)
        b = 0
    else:
        ratio = (p - 0.5) / 0.5
        r = int(255 * (1 - ratio))
        g = 255
        b = 0
    return (r, g, b)

def apply_brush(grid, row, col):
    """
    Apply a square brush to the grid to mark pixels as drawn.

    Args:
        grid (list[list[int]]): The 28x28 drawing grid.
        row (int): Row index where brush is applied.
        col (int): Column index where brush is applied.

    Returns:
        None (modifies grid in-place)
    """
    for dr in range(BRUSH_SIZE):
        for dc in range(BRUSH_SIZE):
            r, c = row + dr, col + dc
            if 0 <= r < ROWS and 0 <= c < COLS:
                grid[r][c] = 1

# ----------------------------
# Event Handling
# ----------------------------
def handle_events(grid):
    """
    Handle user events: quitting and right-click to clear the grid.

    Args:
        grid (list[list[int]]): The current 28x28 grid.

    Returns:
        running (bool): Whether the app should keep running.
        mx (int): Current mouse x-coordinate.
        my (int): Current mouse y-coordinate.
        grid (list[list[int]]): Updated grid (cleared if right-clicked).
    """
    running = True
    mx, my = pygame.mouse.get_pos()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
            grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]
    return running, mx, my, grid

# ----------------------------
# Draw 28x28 Grid
# ----------------------------
def draw_grid(screen, grid):
    """
    Draw the 28x28 pixel drawing grid on the Pygame screen.

    Args:
        screen (pygame.Surface): The Pygame screen to draw on.
        grid (list[list[int]]): The 28x28 grid representing drawn pixels.

    Returns:
        None
    """
    for r in range(ROWS):
        for c in range(COLS):
            val = grid[r][c]
            color = (
                int(PIXEL_OFF[0] + (PIXEL_ON[0] - PIXEL_OFF[0]) * val),
                int(PIXEL_OFF[1] + (PIXEL_ON[1] - PIXEL_OFF[1]) * val),
                int(PIXEL_OFF[2] + (PIXEL_ON[2] - PIXEL_OFF[2]) * val),
            )
            rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, GRID_COLOR, rect, 1)

# ----------------------------
# Grid Update
# ----------------------------
def update_grid(grid, mx, my):
    """
    Update the grid by drawing with the left mouse button.

    Args:
        grid (list[list[int]]): The current 28x28 drawing grid.
        mx (int): Current mouse x-coordinate.
        my (int): Current mouse y-coordinate.

    Returns:
        grid (list[list[int]]): Updated grid with brush applied.
    """
    if pygame.mouse.get_pressed()[0]:
        row, col = my // CELL_SIZE, mx // CELL_SIZE
        apply_brush(grid, row, col)
    return grid

# ----------------------------
# Draw Probability Panel
# ----------------------------
def draw_probability_panel(screen, probabilities, predicted_digit, font):
    """
    Draw the right-hand panel showing probabilities for each digit
    and the current predicted digit at the bottom.

    Args:
        screen (pygame.Surface): The Pygame screen to draw on.
        probabilities (np.ndarray): Array of length 10 with probabilities for each digit.
        predicted_digit (int or None): The predicted digit or None if grid empty.
        font (pygame.font.Font): Font for rendering text.

    Returns:
        None
    """
    bar_x = PANEL_START + 20
    bar_y = 20
    bar_width = 200
    bar_height = 30
    spacing = 10

    for digit in range(10):
        prob = probabilities[digit]  # 0 if empty grid
        percent = prob * 100

        # Background bar
        pygame.draw.rect(
            screen,
            (200, 200, 200),
            (bar_x, bar_y + digit * (bar_height + spacing), bar_width, bar_height)
        )

        # Gradient-filled bar
        bar_color = probability_to_color(prob)
        pygame.draw.rect(
            screen,
            bar_color,
            (bar_x, bar_y + digit * (bar_height + spacing), int(bar_width * prob), bar_height)
        )

        # Text label
        text = pygame.font.SysFont(None, 32).render(
            f"{digit}: {percent:.1f}%", True, (0, 0, 0)
        )
        screen.blit(text, (bar_x + bar_width + 15, bar_y + digit * (bar_height + spacing)))

    # Draw current prediction at bottom
    if predicted_digit is None:
        pred_text_str = "N/A"
        pred_percent = 0
    else:
        pred_text_str = str(predicted_digit)
        pred_percent = probabilities[predicted_digit] * 100

    pred_text = font.render(
        f"{pred_text_str}: {pred_percent:.1f}%", True, (0, 0, 0)
    )
    screen.blit(pred_text, (bar_x, HEIGHT - 50))

# ----------------------------
# Main Application Loop
# ----------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("28x28 Digit Drawer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 48)

    model = load_model()
    grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]
    predicted_digit = None

    running = True
    while running:
        # 1) Handle events
        running, mx, my, grid = handle_events(grid)

        # 2) Update grid
        grid = update_grid(grid, mx, my)

        # 3) Prediction
        predicted_digit, probabilities = predict(grid, model)

        # 4) Draw everything
        screen.fill(BG_COLOR)
        draw_grid(screen, grid)
        draw_probability_panel(screen, probabilities, predicted_digit, font)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
