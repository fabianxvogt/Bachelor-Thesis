import numpy as np
import matplotlib.pyplot as plt

class PlotManager:
    def __init__(self)-> None:
        self.error_rate_history = []
        self.action_rate_history = []
        self.errors_corrected_history = []
        self.errors_found_history = []
        self.errors_missed_history = []
        self.errors_made_history = []
        self.corrects_found_history = []


    def load_historical_plot_data(self, file_path):
        try:
            history = np.load(file_path)
            self.error_rate_history = history[0].tolist()
            self.action_rate_history = history[1].tolist()
            self.errors_corrected_history = history[2].tolist()
            self.errors_found_history = history[3].tolist()
            self.errors_missed_history = history[4].tolist()
            self.errors_made_history = history[5].tolist()
            self.corrects_found_history = history[6].tolist()
            return True
            
        except (Exception):
            print("Historical plot data not loaded.")
            self.error_rate_history = []
            self.action_rate_history = []
            self.errors_corrected_history = []
            self.errors_found_history = []
            self.errors_missed_history = []
            self.errors_made_history = []
            self.corrects_found_history = []
            return False

    def save_historical_plot_data(self, file_path):
        history = (
            self.error_rate_history, 
            self.action_rate_history, 
            self.errors_corrected_history, 
            self.errors_found_history,
            self.errors_missed_history, 
            self.errors_made_history, 
            self.corrects_found_history
        )

        with open(file_path, 'wb') as f:
            np.save(f, history)
            f.close()

    def append_plot_data(self, plot_data):

        error_rate = plot_data[0]
        action_rate = plot_data[1]
        no_of_errors_corrected = plot_data[2]
        no_of_errors_missed = plot_data[3]
        no_of_errors_made = plot_data[4]
        no_of_corrects_found = plot_data[5]

        self.error_rate_history.append(error_rate)
        self.action_rate_history.append(action_rate)
        self.errors_corrected_history.append(no_of_errors_corrected)
        self.errors_missed_history.append(no_of_errors_missed)
        self.errors_made_history.append(no_of_errors_made)
        self.corrects_found_history.append(no_of_corrects_found)

    def create_plot_and_save(self, file_path):
        total_actions_history = np.array(self.errors_found_history) + np.array(self.errors_made_history)
        total_errors_history = np.array(self.errors_found_history) + np.array(self.errors_missed_history)
        
        plt.close() # close existing plot
        
        [plt.plot(data, label=label) for (data, label) in [
            (np.array(self.error_rate_history), "Error rate"),
            (np.array(self.action_rate_history), "Action rate"),
            (np.array(self.errors_made_history) / total_actions_history, "Errors made out of total actions"),
            (np.array(self.errors_missed_history) / total_errors_history, "Errors missed out of total errors"),
            (np.array(self.errors_found_history) / total_actions_history, "Errors found out of total actions"),
            (np.array(self.errors_found_history) / total_errors_history, "Errors found out of total errors"),
        ]]
        
        plt.legend(loc="lower left", bbox_to_anchor=(0,0))
        plt.ylim(0, 1.0)

        plt.grid(color='grey', linestyle='-', linewidth=0.5)
        plt.xlabel("Timesteps")
        plt.ylabel("Probability")
        
        plt.savefig(file_path)
    