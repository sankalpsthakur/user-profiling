import os
import pandas as pd
import numpy as np
from google.cloud import bigquery
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed


def get_bigquery_data():
    client = bigquery.Client()

    query = """
        SELECT
            event_name,
            page_url,
            header,
            agent,
            region,
            time,
            date,
            anonymous_id,
            itinerary_id,
            mongo_id,
            user_id
        FROM
            `your_project_id.your_dataset_id.all_events_data`
    """

    query_job = client.query(query)
    results = query_job.result()
    return results

def preprocess_data(data):
    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(data, columns=['event_name', 'page_url', 'header', 'agent', 'region', 'time', 'date', 'anonymous_id', 'itinerary_id', 'mongo_id', 'user_id'])

    # Feature engineering
    df['time'] = pd.to_datetime(df['time'])
    df['date'] = pd.to_datetime(df['date'])

    # Feature encoding
    categorical_columns = ['event_name', 'page_url', 'header', 'agent', 'region']
    for col in categorical_columns:
        label_encoder = LabelEncoder()
        df[col] = label_encoder.fit_transform(df[col])

    # Scaling numerical columns
    numerical_columns = ['time', 'date']
    for col in numerical_columns:
        scaler = MinMaxScaler()
        df[col] = scaler.fit_transform(df[[col]])

    # Group by user_id, sort by date and time, and pad sequences to the same length
    df = df.sort_values(by=['user_id', 'date', 'time'])
    df_grouped = df.groupby('user_id').agg(lambda x: x.tolist())
    max_sequence_length = df_grouped.applymap(len).max().max()
    padded_sequences = df_grouped.applymap(lambda x: x + [0] * (max_sequence_length - len(x)))

    return padded_sequences


def build_sequence_model(input_shape):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=True))
    model.add(TimeDistributed(Dense(32, activation='relu')))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    # Set your Google Cloud Project ID and the path to your JSON credentials file
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'path/to/your/credentials.json'
    os.environ['GOOGLE_CLOUD_PROJECT'] = 'your_project_id'

    all_events_data = get_bigquery_data()

    for row in all_events_data:
        print(row)

if __name__ == '__main__':
    main()
