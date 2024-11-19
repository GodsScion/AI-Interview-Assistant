'''
Author:     Sai Vignesh Golla
LinkedIn:   https://www.linkedin.com/in/saivigneshgolla/

Copyright (C) 2024 Sai Vignesh Golla

License:    GNU Affero General Public License
            https://www.gnu.org/licenses/agpl-3.0.en.html
            
GitHub:     https://github.com/GodsScion/Auto_job_applier_linkedIn

'''


# Imports

import os
import json

from time import sleep
from random import randint
from datetime import datetime, timedelta
from pyautogui import alert
from pprint import pprint

from config.settings import logs_folder_path



#### Common functions ####

#< Directories related
def make_directories(paths: list[str]) -> None:
    '''
    Function to create missing directories
    '''
    for path in paths:  
        path = path.replace("//","/")
        if '/' in path and '.' in path: path = path[:path.rfind('/')]
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except Exception as e:
            print(f'Error while creating directory "{path}": ', e)

#>


#< Logging related
__logs_file_path = "logs/log.txt"


def print_lg(*msgs: str | dict, end: str = "\n", pretty: bool = False, flush: bool = False, from_critical: bool = False) -> None:
    '''
    Function to log and print. **Note that, `end` and `flush` parameters are ignored if `pretty = True`**
    '''
    try:
        for message in msgs:
            pprint(message) if pretty else print(message, end=end, flush=flush)
            with open(__logs_file_path, 'a+', encoding="utf-8") as file:
                file.write(message)
                file.write(end)
    except Exception as e:
        trail = f'Skipped saving this message: "{message}" to log.txt!' if from_critical else "We'll try one more time to log..."
        alert(f"log.txt in {__logs_file_path} is open or is occupied by another program! Please close it! {trail}", "Failed Logging")
        if not from_critical:
            critical_error_log("Log.txt is open or is occupied by another program!", e)


def critical_error_log(possible_reason: str, stack_trace: Exception) -> None:
    '''
    Function to log and print critical errors along with datetime stamp
    '''
    print_lg(possible_reason, stack_trace, datetime.now(), from_critical=True)
#>


def buffer(speed: int=0) -> None:
    '''
    Function to wait within a period of selected random range.
    * Will not wait if input `speed <= 0`
    * Will wait within a random range of 
      - `0.6 to 1.0 secs` if `1 <= speed < 2`
      - `1.0 to 1.8 secs` if `2 <= speed < 3`
      - `1.8 to speed secs` if `3 <= speed`
    '''
    if speed<=0:
        return
    elif speed <= 1 and speed < 2:
        return sleep(randint(6,10)*0.1)
    elif speed <= 2 and speed < 3:
        return sleep(randint(10,18)*0.1)
    else:
        return sleep(randint(18,round(speed)*10)*0.1)
    

def convert_to_json(data) -> dict:
    '''
    Function to convert data to JSON, if unsuccessful, returns `{"error": "Unable to parse the response as JSON", "data": data}`
    '''
    try:
        result_json = json.loads(data)
        return result_json
    except json.JSONDecodeError:
        return {"error": "Unable to parse the response as JSON", "data": data}