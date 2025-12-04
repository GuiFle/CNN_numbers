import os
import pygame
import torch
import numpy as np
from model_training import CNN
from scipy.ndimage import label, find_objects
import cv2

# ----------------------------
# CONFIGURATION
# ----------------------------
GRID_WIDTH = 1500
WIDTH, HEIGHT = 1920, 1080
PANEL_START = GRID_WIDTH

ROWS, COLS = 100, 200
CELL_WIDTH = GRID_WIDTH / COLS
CELL_HEIGHT = HEIGHT / ROWS

BG_COLOR = (255, 255, 255)
BRUSH_SIZE = 2
last_pos = None

# ----------------------------
# MODEL LOADING
# ----------------------------
def load_model():
    model = CNN()
    current_folder = os.path.dirname(os.path.abspath(__file__))
    parent_folder = os.path.dirname(current_folder)
    model_path = os.path.join(parent_folder, "model.pth")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print(f"Model loaded from {model_path}")
    return model

# ----------------------------
# PREPROCESSING HELPERS
# ----------------------------
def recenter_grid(grid):
    arr = np.array(grid, dtype=np.float32)
    if np.sum(arr) == 0:
        return arr
    ys, xs = np.nonzero(arr)
    cx, cy = np.mean(xs), np.mean(ys)
    shift_x = int(arr.shape[1] / 2 - cx)
    shift_y = int(arr.shape[0] / 2 - cy)
    shifted = np.zeros_like(arr)
    for y, x in zip(ys, xs):
        ny, nx = y + shift_y, x + shift_x
        if 0 <= ny < arr.shape[0] and 0 <= nx < arr.shape[1]:
            shifted[ny, nx] = arr[y, x]
    return shifted

def find_components(binary_grid):
    arr = np.array(binary_grid, dtype=np.uint8)
    labeled, num = label(arr)
    components = []
    if num == 0:
        return components
    slices = find_objects(labeled)
    for slc in slices:
        y1, y2 = slc[0].start, slc[0].stop
        x1, x2 = slc[1].start, slc[1].stop
        comp = arr[y1:y2, x1:x2]
        components.append({"bbox": (y1, y2, x1, x2), "mask": comp})
    return components

def resize_to_28(mask):
    h, w = mask.shape
    scale = 20.0 / max(h, w)
    resized = cv2.resize(mask.astype(np.float32), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((28, 28), dtype=np.float32)
    y = (28 - resized.shape[0]) // 2
    x = (28 - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas

def extract_number(grid, model):
    """Extract multiple digits from the grid and return predictions and their probabilities."""
    components = find_components(grid)
    if not components:
        return "", []

    components.sort(key=lambda c: c["bbox"][2])  # Sort left to right
    digits = []
    all_probs = []

    for comp in components:
        mask = comp["mask"]
        digit_img = resize_to_28(mask)
        digit_img = recenter_grid(digit_img)
        arr = digit_img.reshape(1, 1, 28, 28).astype("float32")
        tensor = torch.from_numpy(arr)
        with torch.no_grad():
            out = model(tensor)
            prob = torch.softmax(out, dim=1).numpy()[0]
            pred = int(torch.argmax(out, dim=1))
        digits.append(str(pred))
        all_probs.append(prob)

    return "".join(digits), all_probs

# ----------------------------
# DRAWING HELPERS
# ----------------------------
def apply_brush(grid, row, col, radius=3):
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr*dr + dc*dc <= radius*radius:
                r, c = row + dr, col + dc
                if 0 <= r < ROWS and 0 <= c < COLS:
                    grid[r][c] = 1

def draw_line(grid, r1, c1, r2, c2, radius=3):
    dr = abs(r2 - r1)
    dc = abs(c2 - c1)
    sr = 1 if r1 < r2 else -1
    sc = 1 if c1 < c2 else -1
    err = dr - dc
    while True:
        apply_brush(grid, r1, c1, radius)
        if r1 == r2 and c1 == c2:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r1 += sr
        if e2 < dr:
            err += dr
            c1 += sc

def update_grid(grid, mx, my):
    global last_pos
    if pygame.mouse.get_pressed()[0]:
        row = int(my // CELL_HEIGHT)
        col = int(mx // CELL_WIDTH)
        if last_pos is not None:
            draw_line(grid, last_pos[0], last_pos[1], row, col, radius=BRUSH_SIZE)
        else:
            apply_brush(grid, row, col, radius=BRUSH_SIZE)
        last_pos = (row, col)
    else:
        last_pos = None
    return grid

def draw_grid(screen, grid):
    for r in range(ROWS):
        for c in range(COLS):
            val = grid[r][c]
            color = (220 - int(220 * val), 220 - int(220 * val), 220 - int(220 * val))
            rect = pygame.Rect(c * CELL_WIDTH, r * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT)
            pygame.draw.rect(screen, color, rect)

# ----------------------------
# PROBABILITY PANEL
# ----------------------------
def probability_to_color(prob):
    return (int((1-prob)*255), int(prob*255), 0)

def draw_probability_panel(screen, probs_list, font):
    """Draw the right-hand panel showing only the predicted digit and its probability for each component."""
    bar_x = PANEL_START + 50
    bar_y = 50
    bar_width = 200
    bar_height = 30
    spacing = 10

    for i, probs in enumerate(probs_list):
        pred_digit = int(np.argmax(probs))
        pred_prob = probs[pred_digit]

        # Background bar
        pygame.draw.rect(screen, (200, 200, 200),
                         (bar_x, bar_y + i * (bar_height + spacing), bar_width, bar_height))
        # Filled bar
        bar_color = (int((1 - pred_prob) * 255), int(pred_prob * 255), 0)
        pygame.draw.rect(screen, bar_color,
                         (bar_x, bar_y + i * (bar_height + spacing), int(bar_width * pred_prob), bar_height))
        # Label
        text = font.render(f"{pred_digit}: {pred_prob*100:.1f}%", True, (0, 0, 0))
        screen.blit(text, (bar_x + bar_width + 15, bar_y + i * (bar_height + spacing)))


# ----------------------------
# MAIN LOOP
# ----------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Multi-Digit Drawer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 36)

    model = load_model()
    grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]
    predicted_number = ""
    probabilities_list = []

    running = True
    while running:
        mx, my = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                grid = [[0 for _ in range(COLS)] for _ in range(ROWS)]
                predicted_number = ""
                probabilities_list = []

        if mx < GRID_WIDTH:
            grid = update_grid(grid, mx, my)

        predicted_number, probabilities_list = extract_number(grid, model)

        screen.fill(BG_COLOR)
        draw_grid(screen, grid)

        # Show predicted digits
        pred_text = font.render(f"Predicted: {predicted_number}", True, (0, 0, 0))
        screen.blit(pred_text, (PANEL_START + 20, 10))

        # Show probabilities for each digit
        draw_probability_panel(screen, probabilities_list, font)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
