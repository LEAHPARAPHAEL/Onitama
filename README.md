# Structure of the code 

The code is organized into 7 main modules :
- game : all the code related to the representation of the board as a bitboard, the rules of the game and the cards.
- network : description of the architecture of the torch model and of the input processing.
- mcts : logic for the search engine.
- train : scripts for self-play data generation and supervised learning.
- benchmark : scripts to perform various benchmarks on the training pipeline and helper functions to plot graphics.
- tournaments : scripts related to the organization of tournaments between different models.
- gui : code for the Graphical User Interface.

# Train a model

To train a new model, you need to write the configuration file in the models/configs directory. The model name should be specified, as well as all the desired options. The other configurations can serve as examples to show the list of tunable parameters.
Then, once a my_model.yaml configuration file is accessible, you can launch the training of the model with the command :

python -m train.end_to_end -c my_model.yaml

The logs of each step of this training will be available in models/logs/my_model.json. 

# Play against the AI

To play against our models, run the command :

python -m gui.gui

Then, select your opponent in the list of available models on the right. On the left, choose the number of simulations, and the starting state of the game. You can choose the 5 starting cards, or pick a random set of 5 cards from the original game, the extension Sensei's Path, or both. Click "START" when you are ready to play.

When it is your turn to play, choose one of your cards (the two at the bottom), one of your pieces, and click on a valid move on the board combining the two. 



