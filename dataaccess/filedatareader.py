from pathlib import Path
import pandas as pd
import re
import numpy as np

class FileDataReader:
    """
    A class used to access data about participants.

    Attributes:
        parentDir (pathlib.Path): The parent directory of the data files.
        frenchDir (pathlib.Path): The directory of french data.
    """


    def __init__(self, parent_dir, french_dir_name="HW-FRENCH", info_filename="Info.txt",
    tasks_filenames=["Test1.txt", "Test2.txt", "Test3.txt", "Test4.txt", "Test5.txt", "Test6.txt", "Test7.txt"],
    data_header=['Time', 'X', 'Y', 'P', 'Az', 'Al'],
    header_reg=r'.*[tT]ime.*[xX].*[yY].*[pP].*[aA]z.*[aA]l.*'):
        """
        Initializes a new FileDataReader.

        Args:
            parent_dir (str): The parent directory of the data files.
            french_dir_name (str): The french data directory name, defaults to 'HW-FRENCH'.
            info_filename (str): The name of the info file for each participant, defaults to 'Info.txt'.
            tasks_filenames (list): The ordered list of tasks' files' names, defaults to ['Test1.txt', 'Test2.txt', 'Test3.txt', 'Test4.txt', 'Test5.txt', 'Test6.txt', 'Test7.txt'].
            data_header (list): The ordered headers of the tasks' data, defaults to ['Time', 'X', 'Y', 'P', 'Az', 'Al'].
            header_reg (regex str): The regular expression used to capture the header of the data in a task file, defaults to r'.*[tT]ime.*[xX].*[yY].*[pP].*[aA]z.*[aA]l.*'.
        """
        self.parent_dir = Path(parent_dir)
        self.french_dir = self.parent_dir / french_dir_name
        self.info_filename = info_filename
        self.tasks_filenames = tasks_filenames
        self.data_header = data_header
        self.header_reg = header_reg


    def _fetch_info(self, patient_dir):
        """
        Maps the key/value info for a participant into a dictionary.

        Args:
            patient_dir (pathlib.Path): Path of the files of the patient.

        Returns:
            patient_info (dict): A dictionary of the patient's data.
        """
        infofile_path = patient_dir / self.info_filename
        f = open(infofile_path, "r", encoding='ISO-8859-1')
        
        patient_info = dict()

        for line in f.readlines():
            line = re.sub(':[ ]*', ':', line.strip())
            keyvalue = line.split(':')
            patient_info[keyvalue[0]] = ": ".join(keyvalue[1:])
        
        f.close()

        return patient_info

    
    def _fetch_data(self, patient_info, patient_dir, tasks):
        """
        Gather tasks data for a participant into a array.

        Args:
            patient_info (dict): Patient's information.
            patient_dir (pathlib.Path): Path of the files of the patient.
            tasks (list): An array of the numbers of tasks.

        Returns:
            tasks_data (pandas): A dictionary of the tasks data of the participant.
        """
        indexes = np.array(tasks) - 1

        tasks_dir = None
        for d in patient_dir.iterdir():
            if not d.is_dir():
                continue
            tasks_dir = d
            break
        if not tasks_dir:
            return list()

        tasks_data = list()
        for i in indexes:
            task_file_path = tasks_dir / self.tasks_filenames[i]

            if not task_file_path.exists():
                continue

            f = open(task_file_path, "r", encoding='ISO-8859-1')
            lines = f.readlines()
            reg_match = None

            for line in lines:
                line = re.sub(r"[ \t]+", " ", line).strip()
                
                if not reg_match:
                    reg_match = re.match(self.header_reg, line)
                    continue
                
                try:
                    line_data = np.array(line.split(" ")).astype(np.int32).tolist()
                except:
                    print("Problem in line:",  line)
                    print("Ignored the line because it couldn't be converted into a float.")
                    continue
                
                data = dict()
                for header, value in zip(self.data_header, line_data):
                    data[header] = value
                data["ID"] = patient_info['ID']
                data["Task"] = i+1

                tasks_data.append(data)

            f.close()

        return tasks_data


    def _generate_info_dataframe(self, info):
        """
        Turn the python object info into a pandas dataframe.

        Args:
            info (list): A list containing the participants info.

        Returns:
            info (pandas.core.frame.DataFrame): A DataFrame of the information of all the participants.
        """
        info = pd.DataFrame(info)
        info.set_index('ID', inplace=True)

        return info


    def _generate_tasks_dataframe(self, data):
        """
        Turn the python object data into a pandas dataframe.

        Args:
            data (list): A list containing the participants' tasks' data.

        Returns:
            data (pandas.core.frame.DataFrame): A DataFrame conversion of the input.
        """
        data = pd.DataFrame(data)
        data.set_index('ID', inplace=True)

        return data


    def load_french(self, tasks=[1, 2, 3, 4, 5, 6, 7], info_only=False, data_only=False):
        """
        Loads the french handwriting data for all participants.

        Args:
            tasks (list[int]): A list of the numbers of the tasks to load, in the range of [1-7], by default it loads all the tasks.
            info_only (bool): default value False, Set to True if you want only the info data.
            data_only (bool): default value False, Set to True if you want only the tasks' data.

        Returns:
            info, data (pandas.core.frame.DataFrame, pandas.core.frame.DataFrame): Default return, a tuple of 2 dataFrames, containing participants' data, and the tasks' data of all the participants, respectively, from the french directory.
            info (pandas.core.frame.DataFrame): A DataFrame of the information of all the participants in the french directory, if infoOnly is set to True.
            data (pandas.core.frame.DataFrame): A DataFrame containing the tasks' data of all the participants in the french directory, if dataOnly is set to True.
        """
        print("Loading the data, please wait.")

        assert not (info_only and data_only), "The infoOnly and dataOnly arguments can't bith be True."

        if not info_only:
            unique_tasks = [i for i in set(tasks)]
            alien_tasks = [task for task in unique_tasks if task < 1 or task > 7]
            assert len(alien_tasks) == 0, "The following tasks don't exist: " + str(alien_tasks)

        info = list() if not data_only else None
        data = list() if not info_only else None

        for d in self.french_dir.iterdir():
            if not d.is_dir():
                continue

            participant_info = self._fetch_info(d)
            participant_info["ID"] = str(d.absolute()).split("/")[-1]

            participant_data = self._fetch_data(participant_info, d, tasks) if not info_only else None

            info.append(participant_info) if not data_only else None

            data = data + participant_data if not info_only else None

        info = self._generate_info_dataframe(info) if not data_only else None
        data = self._generate_tasks_dataframe(data) if not info_only else None

        print("Data loaded successfully.")

        return info if info_only else (data if data_only else (info, data))

