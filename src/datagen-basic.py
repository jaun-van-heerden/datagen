import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import base64

# Time increments options
TIME_INCREMENTS = ["Seconds", "Minutes", "Hours", "Days"]

COMBINATION_OPS = ["Add", "Subtract", "Multiply", "Divide", "Exponent"]


if 'series' not in st.session_state:
    st.session_state.series = {}


#@st.cache_data
def random_walk(rows, lower_bound, upper_bound, seed=None):
    np.random.seed(seed)
    start = (upper_bound + lower_bound) / 2

    step_magnitude = (upper_bound - lower_bound) / 10
    print(step_magnitude)
    series = [start]
    
    for _ in range(1, rows):
        step = np.random.normal(0, step_magnitude)
        
        next_step = series[-1] + step

        if next_step < lower_bound:
            series.append(lower_bound)
        elif next_step > upper_bound:
            series.append(upper_bound)
        else:
            series.append(next_step)

    return series




def sinusoidal(rows, lower_bound, upper_bound, frequency, offset, seed=None):
    np.random.seed(seed)
    x = np.linspace(0, frequency * 2 * np.pi, rows)
    y = np.sin(x + offset)
    y = lower_bound + (y - y.min()) / (y.max() - y.min()) * (upper_bound - lower_bound)
    return y

def uniform_random(rows, lower_bound, upper_bound, seed=None):
    np.random.seed(seed)
    return np.random.uniform(lower_bound, upper_bound, rows)

ALGORITHMS = {
    "Random Walk": random_walk,
    "Sinusoidal": sinusoidal,
    "Uniform Random": uniform_random
}

@st.cache_data
def generate_series(rows, algorithm, lower_bound, upper_bound, seed=None, frequency=1, offset=0):
    if algorithm == "Sinusoidal":
        return sinusoidal(rows, lower_bound, upper_bound, frequency, offset, seed)
    return ALGORITHMS[algorithm](rows, lower_bound, upper_bound, seed)


def combine_series(data, series_a, series_b, operation):
    
    print(series_a)
    
    if operation == "Add":
        return data[series_a] + data[series_b]
    elif operation == "Subtract":
        return data[series_a] - data[series_b]
    elif operation == "Multiply":
        return data[series_a] * data[series_b]
    elif operation == "Divide":
        return data[series_a] / data[series_b]
    elif operation == "Exponent":
        return data[series_a] ** data[series_b]
    else:
        return series_a

def to_csv_download_link(df, filename="data.csv"):
    """Generate a link to download the data as a CSV."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href



st.title('DataGen by Jaun van Heerden')


def generate_data(row_count):
    
    time_increment = st.sidebar.selectbox("Select time increment", TIME_INCREMENTS, index=1)  # Default to "Minutes"

    series_count = st.sidebar.slider("Select number of series", 1, 20, 3)

    data = {}

    series_name_store = {}

    for i in range(series_count):
        series_name = st.sidebar.text_input(f"Name of Series {i+1}", f"series_{i+1}")
        series_name_store[i] =series_name


    series_algorithm_store = {}


    # set up all ui
    for i in range(series_count):

        st.sidebar.markdown(f"### Series {i+1} Configuration")

        _series_name = series_name_store[i]
        algorithm = st.sidebar.selectbox(f"Algorithm for {_series_name}", list(ALGORITHMS.keys()) + ["Combination"])

        if algorithm == "Sinusoidal":
            frequency = st.sidebar.slider("Frequency", 1, 20, 1)
            offset = st.sidebar.slider("Offset", 0.0, 2 * np.pi, 0.1)
        else:
            frequency, offset = 1, 0

        seed = st.sidebar.number_input(f"Seed for {_series_name}", value=i+1, step=1, format="%i")
        lower_bound = st.sidebar.number_input(f"Lower Bound for {_series_name}", value=-10.0)
        upper_bound = st.sidebar.number_input(f"Upper Bound for {_series_name}", value=10.0)

        configuration = {'frequency': frequency,
                         'offset': offset,
                         'seed': seed,
                         'lower_bound': lower_bound,
                         'upper_bound': upper_bound,
                         'algorithm': algorithm}


        if algorithm == "Combination":
            configuration['ref_series_1'] = st.sidebar.selectbox(f"First Series for Combination {series_name_store[i]}", series_name_store.values())
            configuration['ref_series_2'] = st.sidebar.selectbox(f"Second Series for Combination {series_name_store[i]}", series_name_store.values())
            configuration['operation'] = st.sidebar.selectbox(f"Operation for {series_name_store[i]}", COMBINATION_OPS)
            

        series_algorithm_store[i] = configuration

    
    non_combine, combine = [], []
    for series, value in series_algorithm_store.items():
        (non_combine, combine)[value['algorithm']=="Combination"].append(series)



    for series in non_combine:

        configuration = series_algorithm_store[series]

        series_gen = generate_series(row_count, 
                                 configuration['algorithm'], 
                                 configuration['lower_bound'], 
                                 configuration['upper_bound'], 
                                 configuration['seed'], 
                                 configuration['frequency'], 
                                 configuration['offset'])

        data[series_name_store[series]] = series_gen


    for series in combine:

        configuration = series_algorithm_store[series]

        if series_name_store[series] not in [configuration['ref_series_1'], configuration['ref_series_2']]:

            series_gen = combine_series(data, 
                                    configuration['ref_series_1'], 
                                    configuration['ref_series_2'], 
                                    configuration['operation'])

            data[series_name_store[series]] = series_gen

        # Adjusting the timestamp generation according to selected time increment
    if time_increment == "Seconds":
        data["timestamp"] = [datetime.now() + timedelta(seconds=i) for i in range(row_count)]
    elif time_increment == "Minutes":
        data["timestamp"] = [datetime.now() + timedelta(minutes=i) for i in range(row_count)]
    elif time_increment == "Hours":
        data["timestamp"] = [datetime.now() + timedelta(hours=i) for i in range(row_count)]
    else:  # Days
        data["timestamp"] = [datetime.now() + timedelta(days=i) for i in range(row_count)]

    
    return pd.DataFrame(data)



with st.sidebar:
    # Preview
    st.header("Configuration")
    preview_rows = st.slider("Select number of preview rows", 10, 200, 50)


preview_df = generate_data(preview_rows)
st.subheader("Preview Data:")
st.write(preview_df)

st.line_chart(preview_df.set_index('timestamp'))


table_name = st.sidebar.text_input("SQL Table Name", "table")

# Regenerate based on user input
st.sidebar.header("Download Configuration")
#final_row_count = st.sidebar.slider("Select number of rows for download", 10, 20000, 1000)
if st.sidebar.button('Regenerate and Download Data'):
    #final_df = generate_data(final_row_count)
    st.sidebar.markdown(to_csv_download_link(preview_df), unsafe_allow_html=True)

    # Replace this with your actual DataFrame
    df = preview_df  # For example

    # Generate the SQL string to create the table based on DataFrame columns and their data types
    create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} (\n"
    for column in df.columns:
        data_type = "VARCHAR(MAX)" if df[column].dtype == object else "FLOAT"  # Adjust data types as needed
        create_table_query += f"    {column} {data_type},\n"
    create_table_query = create_table_query.rstrip(",\n") + "\n)"

    # Generate the SQL insert query string
    insert_query = f"INSERT INTO {table_name} ({', '.join(df.columns)}) VALUES "

    # Generate the VALUES portion of the query
    values = []
    for index, row in df.iterrows():
        row_values = [f"'{value}'" if isinstance(value, str) else str(value) for value in row]
        values.append("(" + ", ".join(row_values) + ")")

    insert_query += ", ".join(values)

    # Print or use the create_table_query and insert_query as needed
    create_table_query
    insert_query