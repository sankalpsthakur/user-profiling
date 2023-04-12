
# this code does the following - 
# Chooses numerical columns for interaction terms and polynomial features. 
# Imputs missing values in the numerical columns with the mean of each column.
# Creates polynomial features and interaction terms for the numerical columns.
# Log-normalizes the features with extreme values.
# Combines the original data, polynomial features, and interaction terms.



def create_user_features(data):

    # Calculate the total number of events per user.
    total_events_per_user = grouped_data['event_name'].count().rename('total_events')

    # Calculate the average time spent per session or per page. (Assuming sessions are distinguished by date)
    data['datetime'] = pd.to_datetime(data['date'].astype(str) + ' ' + data['time'].astype(str))
    time_spent_per_page = data.groupby(['anonymous_id', 'date', 'page_url'])['datetime'].apply(lambda x: x.max() - x.min())
    avg_time_per_session = time_spent_per_page.groupby('anonymous_id').mean().dt.total_seconds().rename('avg_time_per_session')

    # Determine the most common event, page, or region visited by the user.
    def safe_mode(series):
        mode_result = stats.mode(series)
        if mode_result.count[0] == 0:
            return None
        else:
            return mode_result.mode[0]

    filtered_data = data[data['page_url'].notnull()]

    most_common_event = filtered_data.groupby('anonymous_id')['event_name'].agg(lambda x: safe_mode(x)).rename('most_common_event')
    most_common_page = filtered_data.groupby('anonymous_id')['page_url'].agg(lambda x: safe_mode(x)).rename('most_common_page')
    most_common_region = filtered_data.groupby('anonymous_id')['region'].agg(lambda x: safe_mode(x)).rename('most_common_region')
    most_common_os = filtered_data.groupby('anonymous_id')['OS'].agg(lambda x: safe_mode(x)).rename('most_common_os')

    # Calculate the proportion of specific events (e.g., clicks, scrolls, etc.) to the total events for each user.
    event_counts = data.groupby(['anonymous_id', 'event_name']).size().unstack(level=-1).fillna(0)
    event_proportions = event_counts.div(event_counts.sum(axis=1), axis=0)

    # Add lead column to user_level_data
    lead_per_user = data.groupby('anonymous_id')['lead'].max()

    # Find out the number of sessions per user on desktop vs. mobile devices
    #data['device'] = data['OS'].apply(lambda x: 'desktop' if x in ['Windows', 'Mac OS X'] else ('mobile' if x in ['Android', 'iPhone OS'] else 'unknown'))
    sessions_per_device = data.groupby(['anonymous_id', 'OS'])['date'].nunique().unstack(level=-1).fillna(0)

    user_level_data = pd.concat([
    total_events_per_user,
    avg_time_per_session,
    most_common_event,
    most_common_page,
    most_common_region,
    most_common_os,
    event_counts,
    event_proportions,
    lead_per_user,
    sessions_per_device
    ], axis=1)

    return user_level_data