from pathlib import Path
import pandas as pd
import re
import numpy as np
from .datareader import DataReader

class FileDataReader:
    def __init__(self, 
    parent_path, 
    lang_dir_names={'fr': "HW-FRENCH", 'ar': "HW-ARAB"},
    info_file_name="Info.txt",
    tasks_file_names={
        'fr': ["Test1.txt", "Test2.txt", "Test3.txt", "Test4.txt", "Test5.txt", "Test6.txt", "Test7.txt"], 'ar': ["Test1.txt", "Test2.txt", "Test3.txt"]
        },
    data_header=['Time', 'X', 'Y', 'P', 'Az', 'Al'],
    header_reg=r'^[ \t]*Time[ \t]+X'):
        """
        Initializes a new FileDataReader.

        Args:
            
        """
        self.parent_path = Path(parent_path)

        self.lang_paths = {}
        for key, pathname in lang_dir_names.items():
            self.lang_paths[key] =  self.parent_path / pathname

        self.info_file_name = info_file_name
        self.tasks_file_names = tasks_file_names
        self.data_header = data_header
        self.header_reg = header_reg


    def load_ml_pd_data(self, tasks_per_lang):
        """
        Load and return images of handwriting data, and their PD/HC labels of all participants, in specific languages, for specific tasks.

        Args:
            tasks_per_lang (dict): A dictionary with languages as keys, and a list of task numbers from (0, n - 1) where n the number of tasks for the chosen language as values for each key.

        Returns:
            X (list(numpy.ndarray)): A list of HW images with respect to the selection criteria.
            y (numpy.ndarray): The array of labels, 1 for PD, 0 for HC.
        """
        print('Loading the data, please be patient, this may take a few minutes.')
        X = list()
        y = list()

        for lang, tasks in tasks_per_lang.items():
            for p_dir in self.lang_paths[lang].iterdir():
                if not p_dir.is_dir():
                    continue

                pathology = None
                dementia = None
                other_dementia = None

                is_pd = None
                
                for t_dir in p_dir.iterdir():
                    if t_dir.is_dir():
                        for task in tasks:
                            try:
                                f = open(t_dir / self.tasks_file_names[lang][task], "r", encoding='ISO-8859-1')
                            except:
                                continue

                            hw_data = np.zeros((1, len(self.data_header)))

                            header_found = None

                            for line in f:
                                if not header_found:
                                    header_found = re.match(self.header_reg, line)
                                    
                                    if not ((dementia and other_dementia) or pathology):
                                        if re.match(r'^[ \t]*[pP]athology', line):
                                            pathology = line.split(':')[1].strip()
                                            continue
                                        
                                        if re.match(r'^[ \t]*[dD]ementia[ \t]*:', line):
                                            dementia = line.split(':')[1].strip()
                                            continue

                                        if re.match(r'^[ \t]*[oO]ther[ \t]*[dD]ementia[ \t]*:', line):
                                            other_dementia = line.split(':')[1].strip()
                                            continue

                                    elif is_pd is None and (is_pd := is_parkinsonian(pathology, dementia, other_dementia) == -1):
                                        break

                                    continue

                                try:
                                    line_data = np.array(line.split(" ")).astype(np.int32)
                                except:
                                    print("Problem in line:",  line, " so it was ignored the line because it couldn't be converted into a number.")
                                    continue
                                
                                try:
                                    hw_data = np.vstack([hw_data, line_data])
                                except:
                                    print('There was a problem with adding this line to the image:', line, "so it was ignored.")

                            f.close()

                            if is_pd == -1:
                                break

                            X.append(hw_data[1:,:])
                            y.append(is_pd)
                            
                        break

        y = np.array(y)

        print('Data loaded.')

        return X, y


