# Advanced AI HW 5: Deep Q-Learning

Run `sh setup_linux.sh` for setup.

The only file that you need to modify is `dqn.py`.

To run experiment, execute `python3.5 run_dqn_atari.py`. It creates a new log directory with a name formatted like `2018_12_05_04_52_28_424073_KST__12dca7b8-270b-4f8f-a038-e7379fa57be4`.

To create a learning curve plot, run `plot.py` passing the log directory as an argument, like `python3.5 plot.py 2018_12_05_04_52_28_424073_KST__12dca7b8-270b-4f8f-a038-e7379fa57be4`.

See the given PDF for further instructions.

The starter code was based on an implementation of Q-learning for Atari generously provided by Szymon Sidor from OpenAI.
