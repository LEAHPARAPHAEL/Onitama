import json

def generate_latex_table(json_log_path):
    # 1. Load the JSON data
    with open(json_log_path, 'r') as f:
        log_data = json.load(f)

    # 2. Define the base competitors
    '''
    competitors = [
        "resnet", "resnet_aug", "resnet_aug_mask", "resnet_aug_wdl", 
        "resnet_aug_mask_wdl", "resnet_mask", "resnet_mask_wdl", "resnet_wdl",
        "senet", "senet_aug", "senet_aug_mask", "senet_aug_wdl", 
        "senet_aug_mask_wdl", "senet_mask", "senet_mask_wdl", "senet_wdl"
    ]'''

    competitors = [
        "senet", "senet_all"
    ]

    # 3. Extract win rates and calculate averages
    win_rates = {}
    averages = {}

    for row_comp in competitors:
        win_rates[row_comp] = {}
        row_key = f"{row_comp}_v100"
        total_rate = 0.0
        matchups_counted = 0
        
        for col_comp in competitors:
            if row_comp == col_comp:
                win_rates[row_comp][col_comp] = None
                continue
                
            col_key = f"{col_comp}_v100"
            rate = 0.0
            
            try:
                stats = log_data.get(row_key, {}).get(col_key, {})
                played = stats.get("played", 0)
                if played > 0:
                    wins = stats.get("wins", 0)
                    draws = stats.get("draws", 0)
                    rate = ((wins + (draws / 2.0)) / played) * 100
            except KeyError:
                pass # Defaults to 0.0
                
            win_rates[row_comp][col_comp] = rate
            total_rate += rate
            matchups_counted += 1
            
        # Calculate the average win rate against all opponents
        averages[row_comp] = total_rate / matchups_counted if matchups_counted > 0 else 0.0

    # Helper function to apply the color logic
    def format_score(score, is_avg=False):
        if score < 40:
            color = "scoreRed"
        elif score < 50:
            color = "scoreOrange"
        elif score < 60:
            color = "scoreBlue"
        else:
            color = "scoreGreen"
            
        # Format to 1 decimal place
        text = f"{score:.1f}"
        if is_avg:
            text = f"\\textbf{{{text}}}" # Keep the average bold
            
        return f"\\textcolor{{{color}}}{{{text}}}"

    # 4. Sort competitors descending by their average win rate
    sorted_competitors = sorted(competitors, key=lambda c: averages[c], reverse=True)

    # 5. Build the LaTeX string with color definitions
    latex = [
        "% Ensure \\usepackage{xcolor} is in your preamble",
        "\\definecolor{scoreRed}{HTML}{C0392B}    % Muted Red",
        "\\definecolor{scoreOrange}{HTML}{D4AC0D} % Muted Orange",
        "\\definecolor{scoreBlue}{HTML}{2980B9}   % Muted Blue",
        "\\definecolor{scoreGreen}{HTML}{27AE60}  % Muted Green",
        "",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Tournament with 100 simulations per move}",
        "\\label{tab:arena100}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l c *{16}{c}}",
        "\\toprule"
    ]

    # 6. Build the header row (now with rotated Average)
    header_elements = [" "] + ["\\rotatebox{90}{average score}"]
    for comp in sorted_competitors:
        comp_latex = comp.replace('_', '\\_')
        header_elements.append(f"\\rotatebox{{90}}{{{comp_latex}}}")
    latex.append(" & ".join(header_elements) + " \\\\")
    latex.append("\\midrule")

    # 7. Build the data rows
    for row_comp in sorted_competitors:
        row_name_latex = row_comp.replace("_", "\\_")
        
        # Format the average score with color and bolding
        avg_str = format_score(averages[row_comp], is_avg=True)
        row_data = [row_name_latex, avg_str]
        
        for col_comp in sorted_competitors:
            if row_comp == col_comp:
                row_data.append("-")
            else:
                rate = win_rates[row_comp][col_comp]
                row_data.append(format_score(rate, is_avg=False))
                
        latex.append(" & ".join(row_data) + " \\\\")

    # 8. Close the LaTeX table
    latex.extend([
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table}"
    ])

    return "\n".join(latex)

# --- Example Usage ---
#latex_string = generate_latex_table("tournaments/logs/arena_0.json")
latex_string = generate_latex_table("tournaments/logs/standard.json")
print(latex_string)