import pandas as pd
import re
from scipy import stats


def process_datetime(data):
    data['date'] = pd.to_datetime(data['date']).dt.date
    data['time'] = pd.to_datetime(data['time'], format='%I:%M:%S %p').dt.time
    return data

def create_lead_column(data):
    data['lead'] = data['user_id'].notnull().astype(int)
    return data

def extract_agent_info(agent_string):
    if not isinstance(agent_string, str):
        return 'Unknown', 'Unknown', 'Unknown'
    
    os_pattern = r'\(([^;]*);'
    version_pattern = r'Android (\d+)|iPhone OS (\d+_\d+)|Mac OS X (\d+_\d+_\d+)|Windows NT (\d+\.\d+)'
    device_pattern = r'; (.*?)\)'

    os_match = re.search(os_pattern, agent_string)
    version_match = re.search(version_pattern, agent_string)
    device_match = re.search(device_pattern, agent_string)
    
    OS = os_match.group(1).strip() if os_match else 'Unknown'
    
    if version_match:
        version = next((x for x in version_match.groups() if x is not None), 'Unknown')
    else:
        version = 'Unknown'
    
    device = device_match.group(1).strip() if device_match else 'Unknown'
    
    return OS, version, device

def process_agent_info(data):
    data['OS'], data['version'], data['device'] = zip(*data['agent'].apply(extract_agent_info))
    return data

def encode_categorical_variables(data):
# Perform one-hot encoding for categorical variables
    data['event_name'] = data['event_name'].astype('category')
    data['header'] = data['header'].astype('category')
    data['region'] = data['region'].astype('category')
    data['OS'] = data['OS'].astype('category')    
    return data

def aggregate_user_level_data(data):
    user_level_data = data.groupby('anonymous_id')
    return user_level_data
