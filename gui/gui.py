import pygame
import sys
from game.board import Board
from game.board_utils import CARDS, MOVES, Move
import glob
import json
import os
from network.model import OnitamaNet
import yaml
import torch
from mcts.batched_mcts import BatchedMCTS
import torch.nn.functional as F
import collections
import random


def extract_gen_idx(str : str):
    try:
        idx = int(str.strip("v").split(".")[0].split("_")[0])
        return idx
    except:
        return -1

# ==========================================
# 1. CONFIGURATION & VISUAL CONSTANTS
# ==========================================
WINDOW_WIDTH, WINDOW_HEIGHT = 1000, 750
SQUARE_SIZE = 90
BOARD_OFFSET_X, BOARD_OFFSET_Y = 50, 150
MENU_HEIGHT = 40

WINDOW_WIDTH, WINDOW_HEIGHT = 1000, 750
SQUARE_SIZE = 90
BOARD_OFFSET_X, BOARD_OFFSET_Y = 50, 150

# Colors
C_BG = (245, 240, 225)
C_GRID = (50, 40, 30)
C_P1 = (40, 100, 160)       # Blue
C_P2 = (160, 50, 50)        # Red
C_SIDE = (100, 100, 100)    # Grey (Side card)
C_HIGHLIGHT = (100, 200, 100)
C_SELECTED = (255, 215, 0)
C_MENU_BG = (50, 50, 60)
C_MENU_PANEL = (255, 255, 255)
C_TEXT = (0, 0, 0)
C_BTN_ACTIVE = (70, 150, 70)
C_BTN_DISABLED = (150, 150, 150)

# ==========================================
# 3. NETWORK & MODEL MANAGER
# ==========================================


class ModelManager:
    def __init__(self, base_path="./models"):
        self.config_path = os.path.join(base_path, "configs")
        self.weight_path = os.path.join(base_path, "weights")
        self.available_models = [] 
        self.active_model = None
        self.active_model_name = "None"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.active_mcts = None

    def scan_models(self):
            """
            Scans for configs, then looks for a matching FOLDER in weights/
            and picks the latest .pt file inside that folder.
            """
            self.available_models = []
            if not os.path.exists(self.config_path):
                print(f"Warning: Config path not found at {self.config_path}")
                return

            # 1. Find all YAML config files
            config_files = glob.glob(os.path.join(self.config_path, "*.yaml"))
            
            for cfg_file in config_files:
                try:
                    with open(cfg_file, 'r') as f:
                        data = yaml.safe_load(f)
                        
                    model_name = data.get("model", {}).get("name")
                    if not model_name: continue
                    
                    # 2. Construct path to the model's specific weight folder
                    # Structure: ./models/weights/{model_name}/
                    target_folder = os.path.join(self.weight_path, model_name)
                    
                    if os.path.isdir(target_folder):
                        # 3. Find .pt files inside this folder
                        # We search for *.pt. You can add *.pth if needed.
                        weight_files = glob.glob(os.path.join(target_folder, "*.pt"))
                        
                        if weight_files:
                            # Optional: Sort to find the "latest" version.
                            # Assuming names like v0_15000.pt, alphabetical sort works well.
                            weight_files = sorted(weight_files, key = lambda x : extract_gen_idx(x))
                            best_weight = weight_files[-1] # Take the last one (highest version)

                            self.available_models.append({
                                "name": model_name,
                                "config_path": cfg_file,
                                "weight_path": best_weight,
                                "config_data": data
                            })
                except Exception as e:
                    print(f"Error scanning {cfg_file}: {e}")

    def load_model(self, index, num_simulations = 100):
        if 0 <= index < len(self.available_models):
            item = self.available_models[index]
            try:
                self.active_config = item['config_data']
                self.active_model = OnitamaNet(self.active_config).to(self.device)
                state_dict = torch.load(item["weight_path"], weights_only = False)
                model_state_dict = state_dict["model_state_dict"]
                self.active_model.load_state_dict(model_state_dict)
                self.active_model.eval()

                self.active_mcts = BatchedMCTS(self.active_model, self.active_config, self.device)
                self.active_mcts.num_simulations = num_simulations

                self.active_model_name = item['name']
                return True
            except Exception as e:
                print(f"Failed to load model: {e}")
        return False
# ==========================================
# 4. GUI CLASS
# ==========================================
class OnitamaGUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Onitama Configurator")
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font = pygame.font.SysFont('Segoe UI', 16)
        self.bold_font = pygame.font.SysFont('Segoe UI', 16, bold=True)
        self.title_font = pygame.font.SysFont('Segoe UI', 32, bold=True)
        
        # Logic Modules
        self.model_manager = ModelManager()
        self.model_manager.scan_models()
        self.board = None 
        
        # State Management
        self.state = "MENU" 
        self.human_is_blue = True 
        
        # --- MENU STATE VARIABLES ---
        self.menu_random_cards = True
        self.menu_selected_model_idx = -1
        self.menu_card_assignments = {i: None for i in range(16)}
        
        # Slider State
        self.menu_mcts_sims = 800     # Default value
        self.slider_dragging = False
        self.slider_rect = pygame.Rect(400, 520, 200, 20) # Hitbox for the slider track
        
        # --- GAME STATE VARIABLES ---
        self.selected_card_slot = None
        self.selected_piece_idx = None
        self.valid_targets = []
        self.ui_rects = {} 

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                
                if self.state == "MENU":
                    self.handle_menu_input(event)
                else:
                    self.handle_game_input(event)

            self.screen.fill(C_BG)
            if self.state == "MENU":
                self.draw_menu()
            else:
                self.draw_game()
                # Trigger AI if playing and it's not Human's turn
                if not self.board.game_over:
                    is_human_turn = (self.board.turn == self.human_is_blue)
                    if not is_human_turn:
                        self.handle_ai_turn()

            pygame.display.flip()
            self.clock.tick(60)

    # --- UPDATED MENU INPUT ---
    def handle_menu_input(self, event):
        # 1. Handle Dragging (Mouse Motion)
        if event.type == pygame.MOUSEMOTION:
            if self.slider_dragging:
                self.update_slider_value(event.pos[0])
                return

        # 2. Handle Mouse Up (Stop Dragging)
        if event.type == pygame.MOUSEBUTTONUP:
            self.slider_dragging = False
            return

        # 3. Handle Clicks
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = event.pos
            
            # A. Slider Click
            # Expand hit area slightly for easier clicking
            hit_rect = self.slider_rect.inflate(10, 20) 
            if hit_rect.collidepoint(pos):
                self.slider_dragging = True
                self.update_slider_value(pos[0])
                return

            # B. Start Button
            # Moved down to y=580 to make room for slider
            start_rect = pygame.Rect(400, 580, 200, 60) 
            if start_rect.collidepoint(pos):
                if self.can_start_game():
                    self.start_game()
                else:
                    print("Cannot start: Invalid configuration")
                return 

            # C. Model Selection
            list_rect = pygame.Rect(50, 100, 300, 400)
            if list_rect.collidepoint(pos):
                idx = (pos[1] - 100) // 40
                if 0 <= idx < len(self.model_manager.available_models):
                    self.menu_selected_model_idx = idx
                    print(f"Selected Model: {self.model_manager.available_models[idx]['name']}")
                return

            # D. Color Toggle
            toggle_rect = pygame.Rect(400, 100, 200, 40)
            if toggle_rect.collidepoint(pos):
                self.human_is_blue = not self.human_is_blue
                return

            # E. Random Checkbox
            rand_rect = pygame.Rect(400, 160, 200, 40) 
            if rand_rect.collidepoint(pos):
                self.menu_random_cards = not self.menu_random_cards
                if self.menu_random_cards:
                    self.menu_card_assignments = {i: None for i in range(16)}
                return

            # F. Card Grid
            if not self.menu_random_cards:
                for i in range(16):
                    r, c = i // 4, i % 4
                    cx, cy = 400 + c * 110, 225 + r * 60
                    card_rect = pygame.Rect(cx, cy, 100, 50)
                    if card_rect.collidepoint(pos):
                        self.cycle_card_assignment(i)
                        return

    def update_slider_value(self, mouse_x):
        # Clamp mouse x to slider width
        min_x = self.slider_rect.left
        max_x = self.slider_rect.right
        x = max(min_x, min(mouse_x, max_x))
        
        # Normalize 0.0 to 1.0
        ratio = (x - min_x) / self.slider_rect.width
        
        # Map to range [50, 1500]
        val = 50 + ratio * (1500 - 50)
        self.menu_mcts_sims = int(val)

    # --- UPDATED START LOGIC ---
    def start_game(self):
        # Pass MCTS Sims to load_model
        success = self.model_manager.load_model(self.menu_selected_model_idx, self.menu_mcts_sims)
        
        if not success:
            print("Error: Could not load model. Game not started.")
            return

        final_cards = []
        if self.menu_random_cards:
            all_c = list(range(16))
            random.shuffle(all_c)
            final_cards = all_c[:5]
        else:
            blues = [k for k,v in self.menu_card_assignments.items() if v == 'blue']
            reds = [k for k,v in self.menu_card_assignments.items() if v == 'red']
            side = [k for k,v in self.menu_card_assignments.items() if v == 'side']
            final_cards = blues + reds + side

        try:
            self.board = Board(final_cards)
            self.state = "PLAYING"
        except Exception as e:
            print(f"Error instantiating Board: {e}")

    # --- UPDATED MENU DRAWING ---
    def draw_menu(self):
        title = self.title_font.render("Onitama - Game Setup", True, C_MENU_BG)
        self.screen.blit(title, (50, 30))

        # --- LEFT: MODEL LIST ---
        pygame.draw.rect(self.screen, (230, 230, 230), (50, 100, 300, 400))
        pygame.draw.rect(self.screen, (0,0,0), (50, 100, 300, 400), 2)
        
        if not self.model_manager.available_models:
            self.screen.blit(self.font.render("No models found!", True, (200, 0, 0)), (60, 120))
        
        for i, model in enumerate(self.model_manager.available_models):
            y = 100 + i * 40
            if y + 40 > 500: break 
            color = (180, 220, 255) if i == self.menu_selected_model_idx else (255, 255, 255)
            pygame.draw.rect(self.screen, color, (50, y, 300, 40))
            pygame.draw.rect(self.screen, (200, 200, 200), (50, y, 300, 40), 1)
            self.screen.blit(self.font.render(model.get('name', 'Unknown'), True, (0,0,0)), (60, y + 10))

        # --- RIGHT: CONFIG ---
        # 1. Color
        col_txt = f"Play as: {'BLUE (Bottom)' if self.human_is_blue else 'RED (Bottom)'}"
        col_color = C_P1 if self.human_is_blue else C_P2
        pygame.draw.rect(self.screen, col_color, (400, 100, 200, 40), border_radius=5)
        self.screen.blit(self.bold_font.render(col_txt, True, (255,255,255)), (420, 110))

        # 2. Random
        pygame.draw.rect(self.screen, (255, 255, 255), (400, 160, 30, 30), 2)
        if self.menu_random_cards:
            pygame.draw.rect(self.screen, (50, 50, 50), (405, 165, 20, 20))
        self.screen.blit(self.font.render("Use Random Cards", True, (0,0,0)), (440, 165))

        # 3. Cards
        if not self.menu_random_cards:
            self.screen.blit(self.bold_font.render("Select Cards:", True, (0,0,0)), (400, 200))
            for i in range(16):
                r, c = i // 4, i % 4
                cx, cy = 400 + c * 110, 225 + r * 60
                assign = self.menu_card_assignments[i]
                bg = (240, 240, 240)
                if assign == 'blue': bg = (180, 200, 255)
                elif assign == 'red': bg = (255, 180, 180)
                elif assign == 'side': bg = (200, 200, 200)
                pygame.draw.rect(self.screen, bg, (cx, cy, 100, 50), border_radius=4)
                pygame.draw.rect(self.screen, (100,100,100), (cx, cy, 100, 50), 1, border_radius=4)
                self.screen.blit(self.font.render(CARDS[i]['name'], True, (0,0,0)), (cx + 5, cy + 15))

        # 4. MCTS Slider
        # Label
        self.screen.blit(self.bold_font.render(f"MCTS Simulations: {self.menu_mcts_sims}", True, (0,0,0)), (400, 490))
        
        # Track line
        pygame.draw.rect(self.screen, (150, 150, 150), self.slider_rect, border_radius=10)
        
        # Calculate knob position
        ratio = (self.menu_mcts_sims - 50) / (1500 - 50)
        knob_x = self.slider_rect.left + ratio * self.slider_rect.width
        knob_rect = pygame.Rect(knob_x - 10, self.slider_rect.centery - 10, 20, 20)
        
        # Draw Knob
        pygame.draw.circle(self.screen, C_MENU_BG, knob_rect.center, 10)
        pygame.draw.circle(self.screen, (200, 200, 200), knob_rect.center, 10, 1)

        # 5. Start Button (Moved to 580)
        can_start = self.can_start_game()
        btn_col = C_BTN_ACTIVE if can_start else C_BTN_DISABLED
        pygame.draw.rect(self.screen, btn_col, (400, 580, 200, 60), border_radius=8)
        self.screen.blit(self.title_font.render("START GAME", True, (255,255,255)), (415, 590))

    # [Rest of methods: cycle_card_assignment, can_start_game, handle_ai_turn, etc. remain as before]
    def cycle_card_assignment(self, card_id):
        current_state = self.menu_card_assignments[card_id]
        counts = {'blue': 0, 'red': 0, 'side': 0}
        for v in self.menu_card_assignments.values():
            if v in counts: counts[v] += 1
        cycle_order = [None, 'blue', 'red', 'side']
        try: current_idx = cycle_order.index(current_state)
        except ValueError: current_idx = 0
        found_valid = False
        next_idx = current_idx
        for _ in range(3):
            next_idx = (next_idx + 1) % 4
            candidate = cycle_order[next_idx]
            if candidate is None: found_valid = True; break
            is_already_this = (current_state == candidate)
            count = counts[candidate]
            limit = 1 if candidate == 'side' else 2
            if count < limit or is_already_this: found_valid = True; break
        if found_valid: self.menu_card_assignments[card_id] = cycle_order[next_idx]
        else: self.menu_card_assignments[card_id] = None

    def can_start_game(self):
        if self.menu_selected_model_idx == -1: return False
        if self.menu_random_cards: return True
        b_count, r_count, s_count = 0, 0, 0
        for v in self.menu_card_assignments.values():
            if v == 'blue': b_count += 1
            elif v == 'red': r_count += 1
            elif v == 'side': s_count += 1
        return b_count == 2 and r_count == 2 and s_count == 1




    def handle_game_input(self, event):
        # Allow input only if it is Human's Turn
        is_human_turn = (self.board.turn == self.human_is_blue)
        
        if not is_human_turn or self.board.game_over:
            return

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = event.pos
            # 1. Card Selection
            for key, rect in self.ui_rects.items():
                if rect.collidepoint(pos) and "p1_card" in key:
                    slot = int(key.split("_")[-1])
                    self.selected_card_slot = slot
                    self.update_valid_moves()
                    return

            # 2. Board Selection
            mx, my = pos
            if BOARD_OFFSET_X <= mx <= BOARD_OFFSET_X + 5*SQUARE_SIZE:
                if BOARD_OFFSET_Y <= my <= BOARD_OFFSET_Y + 5*SQUARE_SIZE:
                    c = (mx - BOARD_OFFSET_X) // SQUARE_SIZE
                    r = (my - BOARD_OFFSET_Y) // SQUARE_SIZE
                    # Visual Index clicked
                    vis_idx = r * 5 + c
                    
                    # Logic Index mapping
                    # Because we rotate visuals 180 if it's NOT human turn, 
                    # but here we are inside "is_human_turn", visuals match logic 1:1.
                    # Human is at bottom -> Logic is at bottom -> No Flip.
                    logic_idx = vis_idx

                    is_own = ((self.board.player_disciples >> logic_idx) & 1) or \
                             ((self.board.player_master >> logic_idx) & 1)

                    if is_own:
                        self.selected_piece_idx = logic_idx
                        self.update_valid_moves()
                    elif logic_idx in self.valid_targets:
                        move = Move(self.selected_piece_idx, logic_idx, self.selected_card_slot)
                        self.board.play_move(move)
                        self.reset_selection()
    
    def update_valid_moves(self):
        if self.selected_card_slot is None or self.selected_piece_idx is None:
            self.valid_targets = []
            return
        card_id = self.board.player_cards[self.selected_card_slot]
        target_mask = MOVES[card_id][self.selected_piece_idx]
        my_pieces = self.board.player_disciples | self.board.player_master
        target_mask &= ~my_pieces
        self.valid_targets = []
        for i in range(25):
            if (target_mask >> i) & 1: self.valid_targets.append(i)

            
    def reset_selection(self):
            self.selected_card_slot = None
            self.selected_piece_idx = None
            self.valid_targets = []

    # --- PERSPECTIVE FIX IN DRAW_GAME ---
    def draw_game(self):
        # 1. Grid
        for r in range(5):
            for c in range(5):
                rect = pygame.Rect(BOARD_OFFSET_X + c*SQUARE_SIZE, 
                                   BOARD_OFFSET_Y + r*SQUARE_SIZE, 
                                   SQUARE_SIZE, SQUARE_SIZE)
                pygame.draw.rect(self.screen, C_GRID, rect, 2)
                if (r==0 and c==2) or (r==4 and c==2):
                     pygame.draw.rect(self.screen, (220, 210, 190), rect)

        # 2. Pieces
        # RULE: Board class ALWAYs puts "Current Turn Player" at Logic Bottom (Row 4).
        # We want Human at Visual Bottom (Row 4).
        
        is_human_turn = (self.board.turn == self.human_is_blue)
        
        # If Human Turn: Current Player (Logic Bottom) is Human. Visual Bottom is Human. -> Match.
        # If AI Turn:    Current Player (Logic Bottom) is AI.    Visual Bottom is Human. -> Mismatch (Rotate).
        
        should_rotate = not is_human_turn

        def get_rotated_bits(b):
            s = f"{b:025b}"[::-1] # Reverse string bits
            return int(s, 2)

        # Get Raw Bits (Logic: P=Bottom, O=Top)
        p_bits = self.board.player_disciples
        m_bits = self.board.player_master
        o_d_bits = self.board.opponent_disciples
        o_m_bits = self.board.opponent_master

        if should_rotate:
            # We are drawing the AI turn.
            # Logic Bottom (AI) -> needs to be Visual Top.
            # Logic Top (Human) -> needs to be Visual Bottom.
            
            # AI (P_Bits) -> Rotate to Top
            ai_p = get_rotated_bits(p_bits)
            ai_m = get_rotated_bits(m_bits)
            
            # Human (O_Bits) -> Rotate to Bottom
            hum_p = get_rotated_bits(o_d_bits)
            hum_m = get_rotated_bits(o_m_bits)
            
            # Draw AI (Red/P2 if Human is Blue)
            ai_col = C_P2 if self.human_is_blue else C_P1
            hum_col = C_P1 if self.human_is_blue else C_P2
            
            self._draw_bits(ai_p, ai_m, ai_col)   # Visual Top
            self._draw_bits(hum_p, hum_m, hum_col) # Visual Bottom
            
        else:
            # Human Turn. Logic Bottom is Human. Visual Bottom is Human.
            hum_col = C_P1 if self.human_is_blue else C_P2
            ai_col = C_P2 if self.human_is_blue else C_P1
            
            self._draw_bits(p_bits, m_bits, hum_col)    # Visual Bottom
            self._draw_bits(o_d_bits, o_m_bits, ai_col) # Visual Top

        # 3. Cards
        # If Human Turn: player_cards = Human.
        # If AI Turn:    player_cards = AI (because board logic flipped).
        
        if is_human_turn:
            human_cards = self.board.player_cards
            ai_cards = self.board.opponent_cards
        else:
            human_cards = self.board.opponent_cards
            ai_cards = self.board.player_cards
            
        self.render_card(human_cards[0], 600, 500, True, "p1_card_0")
        self.render_card(human_cards[1], 760, 500, True, "p1_card_1")
        self.render_card(ai_cards[0], 600, 70, False)
        self.render_card(ai_cards[1], 760, 70, False)
        self.render_card(self.board.side_card, 680, 285, True, tag="side")

        # 4. Highlights
        if self.selected_piece_idx is not None:
            vis_idx = self.selected_piece_idx
            # If we rotated the board visuals, we must rotate the highlight index to match
            if should_rotate: vis_idx = 24 - vis_idx
            
            r, c = vis_idx // 5, vis_idx % 5
            rect = pygame.Rect(BOARD_OFFSET_X + c*SQUARE_SIZE, BOARD_OFFSET_Y + r*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(self.screen, C_SELECTED, rect, 4)

        for t_idx in self.valid_targets:
            vis_idx = t_idx
            if should_rotate: vis_idx = 24 - t_idx
            
            r, c = vis_idx // 5, vis_idx % 5
            cx = BOARD_OFFSET_X + c*SQUARE_SIZE + SQUARE_SIZE//2
            cy = BOARD_OFFSET_Y + r*SQUARE_SIZE + SQUARE_SIZE//2
            s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            pygame.draw.circle(s, (100, 255, 100, 100), (SQUARE_SIZE//2, SQUARE_SIZE//2), 15)
            self.screen.blit(s, (BOARD_OFFSET_X + c*SQUARE_SIZE, BOARD_OFFSET_Y + r*SQUARE_SIZE))

    def _draw_bits(self, pawns, master, color):
        for i in range(25):
            if (pawns >> i) & 1: self._draw_circ(i, color, False)
            if (master >> i) & 1: self._draw_circ(i, color, True)

    def _draw_circ(self, i, color, is_master):
        r, c = i // 5, i % 5
        cx = BOARD_OFFSET_X + c*SQUARE_SIZE + SQUARE_SIZE//2
        cy = BOARD_OFFSET_Y + r*SQUARE_SIZE + SQUARE_SIZE//2
        radius = 32 if is_master else 22
        pygame.draw.circle(self.screen, color, (cx, cy), radius)
        if is_master: pygame.draw.circle(self.screen, (255, 215, 0), (cx, cy), radius-4, 3)

    def render_card(self, card_id, x, y, is_interactable, tag=None):
        rect = pygame.Rect(x, y, 140, 140)
        if tag and is_interactable: self.ui_rects[tag] = rect
        color = C_MENU_PANEL
        if is_interactable and self.selected_card_slot is not None and tag and "p1_card" in tag:
            slot = int(tag.split("_")[-1])
            if slot == self.selected_card_slot: color = (255, 230, 150)
        pygame.draw.rect(self.screen, color, rect, border_radius=5)
        pygame.draw.rect(self.screen, (0,0,0), rect, 2, border_radius=5)
        self.screen.blit(self.font.render(CARDS[card_id]["name"], True, C_TEXT), (x+10, y+10))
        cx, cy = x + 70, y + 80
        gw = 15
        for r in range(-2, 3):
            for c in range(-2, 3):
                cell = pygame.Rect(cx + c*gw - gw//2, cy + r*gw - gw//2, gw, gw)
                pygame.draw.rect(self.screen, (200,200,200), cell, 1)
                if r==0 and c==0: pygame.draw.rect(self.screen, (50,50,50), (cell.x+2, cell.y+2, gw-3, gw-3))
                if (r, c) in CARDS[card_id]["pattern"]:
                    pygame.draw.rect(self.screen, (50,50,50), (cell.x+2, cell.y+2, gw-3, gw-3))
    def handle_ai_turn(self):
        pygame.display.set_caption(f"AI ({self.model_manager.active_model_name}) is thinking...")
        pygame.event.pump()
        

        policy = self.model_manager.active_mcts.search_batch([self.board])[0]

        if self.model_manager.active_config["training"].get("mask_illegal_moves", False):
            policy = F.relu(policy)
        action_idx = torch.multinomial(policy, 1).item()
        move = self.board.action_index_to_move(action_idx)

        self.board.play_move(move)

if __name__ == "__main__":
    gui = OnitamaGUI()
    gui.run()