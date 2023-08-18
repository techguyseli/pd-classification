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
    data_header=['Time', 'X', 'Y', 'P', 'Az', 'Al']):
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


    def __postprocess_info_df(self, df):
        """
        Perform post-processing on the participants' info dataframe.

        Args:
            df (pandas.core.frame.DataFrame): A DataFrame containing the participants info.

        Returns:
            df (pandas.core.frame.DataFrame): The post-processed df.
        """
        df.set_index('ID', inplace=True)

        return df


    def __postprocess_tasks_df(self, df):
        """
        Perform post-processing on the tasks dataframe.

        Args:
            df (pandas.core.frame.DataFrame): A DataFrame containing the tasks data.

        Returns:
            df (pandas.core.frame.DataFrame): The post-processed df.
        """
        df.set_index('Participant', inplace=True)

        return df


    def load_participant_info(self, participant_id):
        """
        Load the info of a participant into a series and return it.

        Args:
            participant_id (str or pathlib.Path): The name of the directory, or the directory object of the participant.

        Returns:
            participant_info (pandas.core.series.Series): A series containing the information of the participant.
        """
        if not isinstance(participant_id, Path):
            for key in self.lang_paths.keys():
                lang = key
                break
            participant_info_file = self.lang_paths[lang] / participant_id / self.info_file_name

        else:
            participant_info_file = participant_id / self.info_file_name

        f = open(participant_info_file, "r", encoding='ISO-8859-1')
        
        participant_info = pd.Series(dtype='object')
        
        for line in f.readlines():
            line = re.subn('[ ]*:[ ]*', ':', line.strip(), count=1)
            key, value = line[0].split(':', 1)
            participant_info[key] = value
        
        f.close()

        try:
            participant_info["ID"]
        except:
            participant_info["ID"] = str(participant_id.absolute()).split("/")[-1] if isinstance(participant_id, Path) else participant_id

        return participant_info

    
    def load_info(self):
        """
        Load and return the info of all participants in a dataframe.

        Returns:
            df (pandas.core.frame.DataFrame): A dataframe containing the info of all the participants.
        """
        dl = []

        for value in self.lang_paths.values():
                lang_dir = value
                break

        for d in lang_dir.iterdir():
            if not d.is_dir():
                continue

            participant_info = self.load_participant_info(d)

            dl.append(participant_info)

        df = pd.DataFrame(dl)
        df = self.__postprocess_info_df(df)

        return df


    def load_hw(self, participant_id, lang, task):
        """
        Load and return a single image of handwriting data, of a certain participant, in a certain language, for a certain task.

        Args:
            participant_id (str): The name of the directory of the participant (the participant's name in uppercase).
            lang (str): The language.
            task (int): The task number from (0, n - 1) where n the number of tasks for the chosen language.

        Returns:
            hw_data (np.ndarray): An array of the handwriting data.
        """
        hw_file = self.lang_paths[lang] / participant_id
        for d in hw_file.iterdir():
            if d.is_dir():
                hw_file = d / self.tasks_file_names[lang][task]
                break

        f = open(hw_file, "r", encoding='ISO-8859-1')
        header = ' '.join(self.data_header)
        in_data = False

        hw_data = np.zeros((1, len(self.data_header)))

        for line in f:
            line = re.sub(r"[ \t]+", " ", line).strip()
            
            if not in_data:
                in_data = line == header
                continue
            
            try:
                line_data = np.array(line.split(" ")).astype(np.int32)
            except:
                print("Problem in line:",  line)
                print("Ignored the line because it couldn't be converted into a number.")
                continue
            
            hw_data = np.vstack([hw_data, line_data])

        f.close()

        hw_data = pd.DataFrame(hw_data[1:,:], columns=self.data_header, dtype=int)
        hw_data['Participant'] = participant_id
        hw_data['Language'] = lang
        hw_data['Task'] = task

        return hw_data


    def load_participant_hw(self, participant_id, tasks_per_lang):
        """
        Load and return a dataframe containing images of handwriting data, of a specific participant, in specific languages, for specific tasks.

        Args:
            participant_id (str): The name of the directory of the participant (the participant's name in uppercase).
            tasks_per_lang (dict): A dictionary with languages as keys, and a list of task numbers from (0, n - 1) where n the number of tasks for the chosen language as values for each key.

        Returns:
            df (pandas.core.frame.DataFrame): A dataframe containing the results of the selection criteria.
        """
        df = pd.DataFrame()
        for lang, tasks in tasks_per_lang.items():
            for task in tasks:
                try:
                    hw = self.load_hw(participant_id, lang, task)
                except:
                    continue
                df = pd.concat((df, hw))

        return df
        
        
    def load_data(self, tasks_per_lang):
        """
        Load and return a dataframe containing images of handwriting data of all participants, in specific languages, for specific tasks.

        Args:
            tasks_per_lang (dict): A dictionary with languages as keys, and a list of task numbers from (0, n - 1) where n the number of tasks for the chosen language as values for each key.

        Returns:
            df (pandas.core.frame.DataFrame): A dataframe containing the results of the selection criteria.
        """
        print('Loading the data, this may take a few minutes, please be patient.')

        df = pd.DataFrame()

        for value in self.lang_paths.values():
                lang_dir = value
                break

        for d in lang_dir.iterdir():
            if not d.is_dir():
                continue
            
            participant_id = d.absolute().name

            df = pd.concat((df, self.load_participant_hw(participant_id, tasks_per_lang)))

        df = self.__postprocess_tasks_df(df)
        
        print("Data fully loaded.")

        return df

    