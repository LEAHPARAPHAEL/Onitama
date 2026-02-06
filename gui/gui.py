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
# SCALED DOWN DIMENSIONS (Height 850 -> 700)
WINDOW_WIDTH, WINDOW_HEIGHT = 990, 700 
SQUARE_SIZE = 74
BOARD_OFFSET_X, BOARD_OFFSET_Y = 82, 165

# Card Dimensions
CARD_W, CARD_H = 165, 165 
CARD_MINI_GRID = 23 

# Colors
C_BG = (245, 240, 225)
C_GRID = (50, 40, 30)
C_P1 = (40, 100, 160)       # Blue
C_P2 = (160, 50, 50)        # Red
C_HIGHLIGHT = (100, 200, 100)
C_SELECTED = (255, 215, 0)
C_MENU_BG = (50, 50, 60)
C_MENU_PANEL = (255, 255, 255)
C_TEXT = (0, 0, 0)
C_BTN_ACTIVE = (70, 150, 70)
C_BTN_DISABLED = (150, 150, 150)
C_OVERLAY = (0, 0, 0, 180) # Semi-transparent black

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
                state_dict = torch.load(item["weight_path"], weights_only = False,map_location=self.device)
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
        pygame.display.set_caption("Onitama AlphaZero")
        self.clock = pygame.time.Clock()
        
        # Scaled Fonts
        self.font = pygame.font.SysFont('Segoe UI', 13)
        self.bold_font = pygame.font.SysFont('Segoe UI', 15, bold=True)
        self.title_font = pygame.font.SysFont('Segoe UI', 26, bold=True)
        self.large_font = pygame.font.SysFont('Segoe UI', 40, bold=True)
        
        self.model_manager = ModelManager()
        self.model_manager.scan_models()
        self.board = None 
        
        # State
        self.state = "MENU" # MENU, PLAYING, GAMEOVER
        self.human_is_blue = True 
        
        # Menu Data (Preserved between games)
        self.menu_random_cards = True
        self.menu_selected_model_idx = -1
        self.menu_card_assignments = {i: None for i in range(16)}
        self.menu_mcts_sims = 800
        self.slider_dragging = False
        # Initial placeholder, updated in draw_menu/handle_input
        self.slider_rect = pygame.Rect(330, 430, 165, 16) 
        
        # Game Data
        self.selected_card_slot = None
        self.selected_piece_idx = None
        self.valid_targets = []
        self.ui_rects = {} 

    def run(self):
        while True:
            # --- 1. AI Logic (Moved to start) ---
            # We check for AI turn here so that the previous frame (with Human's move)
            # has already been flipped to the display.
            if self.state == "PLAYING" and self.board and not self.board.game_over:
                is_human_turn = (self.board.turn == self.human_is_blue)
                if not is_human_turn:
                    self.handle_ai_turn()

            # --- 2. Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                
                if self.state == "MENU":
                    self.handle_menu_input(event)
                elif self.state == "PLAYING":
                    self.handle_game_input(event)
                elif self.state == "GAMEOVER":
                    self.handle_gameover_input(event)

            # --- 3. Drawing ---
            self.screen.fill(C_BG)
            if self.state == "MENU":
                self.draw_menu()
            else:
                self.draw_game()
                
                # Check for Game Over logic update
                if self.board and self.board.game_over:
                    self.state = "GAMEOVER"
                
                # Draw Overlay if Game Over
                if self.state == "GAMEOVER":
                    self.draw_gameover()

            pygame.display.flip()
            self.clock.tick(60)

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

    # --- PERSPECTIVE FIX IN DRAW_GAME (SCALED) ---
    def draw_game(self):
        # Grid
        for r in range(5):
            for c in range(5):
                rect = pygame.Rect(BOARD_OFFSET_X + c*SQUARE_SIZE, BOARD_OFFSET_Y + r*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
                pygame.draw.rect(self.screen, C_GRID, rect, 2)
                if (r==0 or r==4) and c==2: pygame.draw.rect(self.screen, (220, 210, 190), rect)

        # Pieces
        is_human_turn = (self.board.turn == self.human_is_blue)
        should_rotate = not is_human_turn

        def get_rot(b): return int(f"{b:025b}"[::-1], 2)

        pb, mb = self.board.player_disciples, self.board.player_master
        ob, om = self.board.opponent_disciples, self.board.opponent_master

        if should_rotate:
            self._draw_bits(get_rot(pb), get_rot(mb), C_P2 if self.human_is_blue else C_P1)
            self._draw_bits(get_rot(ob), get_rot(om), C_P1 if self.human_is_blue else C_P2)
        else:
            self._draw_bits(pb, mb, C_P1 if self.human_is_blue else C_P2)
            self._draw_bits(ob, om, C_P2 if self.human_is_blue else C_P1)

        # Cards
        # Scaled Y Positions
        ai_y = 58  
        hum_y = 478 
        
        # Side Card Scaled
        side_x, side_y = 618, 263

        if is_human_turn:
            hum_c, ai_c = self.board.player_cards, self.board.opponent_cards
        else:
            hum_c, ai_c = self.board.opponent_cards, self.board.player_cards

        # Draw Cards (Scaled X positions)
        self.render_card(ai_c[0], 519, ai_y, False, rotate_pattern=True)
        self.render_card(ai_c[1], 725, ai_y, False, rotate_pattern=True)
        
        self.render_card(hum_c[0], 519, hum_y, True, "p1_card_0")
        self.render_card(hum_c[1], 725, hum_y, True, "p1_card_1")
        
        self.render_card(self.board.side_card, side_x, side_y, True, "side")

        # Selection/Target Highlights
        if self.selected_piece_idx is not None:
            vis = self.selected_piece_idx if not should_rotate else 24 - self.selected_piece_idx
            r, c = vis // 5, vis % 5
            pygame.draw.rect(self.screen, C_SELECTED, (BOARD_OFFSET_X + c*SQUARE_SIZE, BOARD_OFFSET_Y + r*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 4)
        
        for t in self.valid_targets:
            vis = t if not should_rotate else 24 - t
            r, c = vis // 5, vis % 5
            s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            pygame.draw.circle(s, (100,255,100,100), (SQUARE_SIZE//2, SQUARE_SIZE//2), 12) # Scaled radius
            self.screen.blit(s, (BOARD_OFFSET_X + c*SQUARE_SIZE, BOARD_OFFSET_Y + r*SQUARE_SIZE))

    def draw_gameover(self):
        # Overlay
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill(C_OVERLAY)
        self.screen.blit(overlay, (0,0))
        
        if self.board.result == -1:
            if self.board.turn == self.human_is_blue:
                msg = "YOU LOST"
                color = (255, 100, 100)
            else:
                msg = "YOU WON!"
                color = (100, 255, 100)
        else:
            msg = "DRAW (Repetition)"
            color = (200, 200, 255)

        # Draw Text
        txt = self.large_font.render(msg, True, color)
        outline = self.large_font.render(msg, True, (0,0,0))
        
        rc = txt.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2 - 40))
        self.screen.blit(outline, (rc.x-2, rc.y-2))
        self.screen.blit(outline, (rc.x+2, rc.y+2))
        self.screen.blit(txt, rc)
        
        # New Game Button (Scaled)
        btn_w, btn_h = 200, 58
        bx = (WINDOW_WIDTH - btn_w) // 2
        by = (WINDOW_HEIGHT - btn_h) // 2 + 40
        
        if pygame.Rect(bx, by, btn_w, btn_h).collidepoint(pygame.mouse.get_pos()):
            b_col = (90, 180, 90)
        else:
            b_col = C_BTN_ACTIVE
            
        pygame.draw.rect(self.screen, b_col, (bx, by, btn_w, btn_h), border_radius=10)
        pygame.draw.rect(self.screen, (255,255,255), (bx, by, btn_w, btn_h), 2, border_radius=10)
        
        btxt = self.title_font.render("New Game", True, (255,255,255))
        brc = btxt.get_rect(center=(bx + btn_w//2, by + btn_h//2))
        self.screen.blit(btxt, brc)

    def _draw_bits(self, pawns, master, color):
        for i in range(25):
            if (pawns >> i) & 1: self._draw_circ(i, color, False)
            if (master >> i) & 1: self._draw_circ(i, color, True)

    def _draw_circ(self, i, color, is_master):
        r, c = i // 5, i % 5
        cx = BOARD_OFFSET_X + c*SQUARE_SIZE + SQUARE_SIZE//2
        cy = BOARD_OFFSET_Y + r*SQUARE_SIZE + SQUARE_SIZE//2
        # Scaled Radii
        radius = 26 if is_master else 18
        pygame.draw.circle(self.screen, color, (cx, cy), radius)
        if is_master: pygame.draw.circle(self.screen, (255, 215, 0), (cx, cy), radius-3, 2)


    def handle_menu_input(self, event):
        # SCALED LAYOUT CONSTANTS
        PANEL_Y = 100
        L_CX = 350
        R_PANEL_X = 580
        R_CX = 705

        if event.type == pygame.MOUSEMOTION and self.slider_dragging:
            self.update_slider_value(event.pos[0]); return
        if event.type == pygame.MOUSEBUTTONUP:
            self.slider_dragging = False; return

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = event.pos
            
            # --- LEFT PANEL CONTROLS ---
            # 1. Color Toggle
            if pygame.Rect(L_CX - 82, PANEL_Y, 164, 32).collidepoint(pos):
                self.human_is_blue = not self.human_is_blue
                return

            # 2. Slider
            slider_hitbox = pygame.Rect(L_CX - 100, PANEL_Y + 65, 200, 16).inflate(10, 20)
            if slider_hitbox.collidepoint(pos):
                self.slider_dragging = True
                self.slider_rect = pygame.Rect(L_CX - 100, PANEL_Y + 65, 200, 16) 
                self.update_slider_value(pos[0])
                return

            # 3. Random Checkbox
            if pygame.Rect(L_CX - 82, PANEL_Y + 107, 164, 32).collidepoint(pos):
                self.menu_random_cards = not self.menu_random_cards
                if self.menu_random_cards: self.menu_card_assignments = {i: None for i in range(16)}
                return

            # 4. Card Grid
            if not self.menu_random_cards:
                start_x = L_CX - 180
                start_y = PANEL_Y + 156
                for i in range(16):
                    r, c = i // 4, i % 4
                    cx, cy = start_x + c * 90, start_y + r * 50
                    if pygame.Rect(cx, cy, 82, 40).collidepoint(pos):
                        self.cycle_card_assignment(i); return

            # --- RIGHT PANEL MODELS & START ---
            # 5. Model List
            if pygame.Rect(R_PANEL_X, PANEL_Y, 250, 330).collidepoint(pos):
                idx = (pos[1] - PANEL_Y) // 32
                if 0 <= idx < len(self.model_manager.available_models):
                    self.menu_selected_model_idx = idx
                return

            # 6. Start Button
            if pygame.Rect(R_CX - 82, PANEL_Y + 345, 164, 50).collidepoint(pos):
                if self.can_start_game(): self.start_game()
                return

    def draw_menu(self):
        # SCALED LAYOUT CONSTANTS
        PANEL_Y = 100
        L_CX = 350
        R_PANEL_X = 580
        R_PANEL_W = 250
        R_CX = R_PANEL_X + R_PANEL_W // 2

        # Main Title
        title = self.title_font.render("Onitama Setup", True, C_MENU_BG)
        self.screen.blit(title, title.get_rect(center=(WINDOW_WIDTH//2, 40)))

        # --- LEFT PANEL: CONTROLS ---
        
        # 1. Player Color Toggle
        col_c = C_P1 if self.human_is_blue else C_P2
        tog_rect = pygame.Rect(L_CX - 82, PANEL_Y, 164, 32)
        pygame.draw.rect(self.screen, col_c, tog_rect, border_radius=5)
        pygame.draw.rect(self.screen, (0,0,0), tog_rect, 2, border_radius=5)
        lbl = f"Human: {'BLUE' if self.human_is_blue else 'RED'}"
        ts = self.bold_font.render(lbl, True, (255,255,255))
        self.screen.blit(ts, ts.get_rect(center=tog_rect.center))

        # 2. Simulations Slider
        sim_lbl = self.bold_font.render(f"AI Simulations: {self.menu_mcts_sims}", True, C_TEXT)
        self.screen.blit(sim_lbl, sim_lbl.get_rect(center=(L_CX, PANEL_Y + 48)))
        
        self.slider_rect = pygame.Rect(L_CX - 100, PANEL_Y + 65, 200, 16)
        pygame.draw.rect(self.screen, (200,200,200), self.slider_rect, border_radius=8)
        pygame.draw.rect(self.screen, (100,100,100), self.slider_rect, 2, border_radius=8)
        ratio = (self.menu_mcts_sims - 50)/(1500-50)
        kx = self.slider_rect.left + ratio * self.slider_rect.width
        pygame.draw.circle(self.screen, C_MENU_BG, (int(kx), int(self.slider_rect.centery)), 8)

        # 3. Random Cards Checkbox
        rnd_rect = pygame.Rect(L_CX - 82, PANEL_Y + 107, 164, 32)
        cb_rect = pygame.Rect(rnd_rect.x, rnd_rect.centery - 8, 16, 16)
        pygame.draw.rect(self.screen, (255,255,255), cb_rect)
        pygame.draw.rect(self.screen, (0,0,0), cb_rect, 2)
        if self.menu_random_cards: 
            pygame.draw.rect(self.screen, (50,50,50), cb_rect.inflate(-4,-4))
        rnd_txt = self.font.render("Random Cards", True, C_TEXT)
        self.screen.blit(rnd_txt, (cb_rect.right + 8, cb_rect.y - 2))

        # 4. Card Grid
        if not self.menu_random_cards:
            start_x = L_CX - 180
            start_y = PANEL_Y + 156
            for i in range(16):
                r, c = i//4, i%4
                cx, cy = start_x + c*90, start_y + r*50
                asn = self.menu_card_assignments[i]
                bg = (180,200,255) if asn=='blue' else (255,180,180) if asn=='red' else (200,200,200) if asn=='side' else (240,240,240)
                pygame.draw.rect(self.screen, bg, (cx,cy,82,40), border_radius=4)
                pygame.draw.rect(self.screen, (100,100,100), (cx,cy,82,40), 1, border_radius=4)
                c_name = CARDS[i]['name']
                if len(c_name) > 10: c_name = c_name[:9] + "."
                self.screen.blit(self.font.render(c_name, True, C_TEXT), (cx+4, cy+12))

        # --- RIGHT PANEL: MODELS & START ---
        
        # Title for model list
        mod_title = self.bold_font.render("Choose your opponent", True, C_TEXT)
        self.screen.blit(mod_title, (R_PANEL_X, PANEL_Y - 24))

        # Model List
        pygame.draw.rect(self.screen, (230,230,230), (R_PANEL_X, PANEL_Y, R_PANEL_W, 330))
        pygame.draw.rect(self.screen, (0,0,0), (R_PANEL_X, PANEL_Y, R_PANEL_W, 330), 2)
        for i, m in enumerate(self.model_manager.available_models):
            y = PANEL_Y + i*32
            # Ensure we don't draw outside the box
            if y + 32 > PANEL_Y + 330: break 
            col = (180,220,255) if i == self.menu_selected_model_idx else (255,255,255)
            pygame.draw.rect(self.screen, col, (R_PANEL_X, y, R_PANEL_W, 32))
            pygame.draw.rect(self.screen, (200,200,200), (R_PANEL_X, y, R_PANEL_W, 32), 1)
            self.screen.blit(self.font.render(m.get('name','?'), True, C_TEXT), (R_PANEL_X + 8, y+8))

        # 5. Start Button
        can = self.can_start_game()
        btn_rect = pygame.Rect(R_CX - 82, PANEL_Y + 345, 164, 50)
        pygame.draw.rect(self.screen, C_BTN_ACTIVE if can else C_BTN_DISABLED, btn_rect, border_radius=8)
        pygame.draw.rect(self.screen, (0,0,0), btn_rect, 2, border_radius=8)
        st_txt = self.title_font.render("START", True, (255,255,255))
        self.screen.blit(st_txt, st_txt.get_rect(center=btn_rect.center))

    def render_card(self, card_id, x, y, interact, tag=None, rotate_pattern=False):
        rect = pygame.Rect(x, y, CARD_W, CARD_H)
        if interact and tag: self.ui_rects[tag] = rect
        
        col = C_MENU_PANEL
        if interact and self.selected_card_slot is not None and tag and "p1_card" in tag:
            if int(tag.split("_")[-1]) == self.selected_card_slot: 
                col = (255, 230, 150)
            
        pygame.draw.rect(self.screen, col, rect, border_radius=8)
        pygame.draw.rect(self.screen, (0,0,0), rect, 2, border_radius=8)
        
        name_surf = self.bold_font.render(CARDS[card_id]["name"], True, C_TEXT)
        name_rect = name_surf.get_rect(center=(x + CARD_W//2, y + 16))
        self.screen.blit(name_surf, name_rect)
        
        gw = CARD_MINI_GRID
        grid_center_x = x + CARD_W // 2
        grid_center_y = y + CARD_H // 2 + 8 
        
        for r in range(-2, 3):
            for c in range(-2, 3):
                draw_r = -r if rotate_pattern else r
                draw_c = -c if rotate_pattern else c
                
                cell_x = grid_center_x + c*gw - gw//2
                cell_y = grid_center_y + r*gw - gw//2
                
                pygame.draw.rect(self.screen, (200,200,200), (cell_x, cell_y, gw, gw), 1)
                
                # Piece Contrast
                # Center Piece (Darker - almost black)
                if draw_r == 0 and draw_c == 0:
                    pygame.draw.rect(self.screen, (30, 30, 30), (cell_x+2, cell_y+2, gw-4, gw-4))
                
                # Move Piece (Lighter - Grey)
                elif (draw_r, draw_c) in CARDS[card_id]["pattern"]:
                    pygame.draw.rect(self.screen, (160, 160, 160), (cell_x+2, cell_y+2, gw-4, gw-4))

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