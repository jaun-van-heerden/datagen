import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Time increments options
TIME_INCREMENTS = ["Seconds", "Minutes", "Hours", "Days"]

COMBINATION_OPS = ["Add", "Subtract", "Multiply", "Divide", "Exponent"]


if 'series' not in st.session_state:
    st.session_state.series = {}

def random_walk(rows, lower_bound, upper_bound, seed=None):
    np.random.seed(seed)
    start = (upper_bound + lower_bound) / 2
    step_magnitude = (upper_bound - lower_bound) / 10
    steps = np.random.normal(0, step_magnitude, rows)
    series = np.cumsum(steps) + start
    series = np.clip(series, lower_bound, upper_bound)
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
    #... [rest of your timeseries generation code, using 'row_count' instead of 'preview_rows']...
    
    #time_increment = st.sidebar.selectbox("Select time increment" + len(row_count), TIME_INCREMENTS, index=1)  # Default to "Minutes"
    time_increment = "Seconds"

    series_count = st.sidebar.slider("Select number of series", 1, 10, 3)
    

    # Adjusting the timestamp generation according to selected time increment
    if time_increment == "Seconds":
        data = {"timestamp": [datetime.now() + timedelta(seconds=i) for i in range(row_count)]}
    elif time_increment == "Minutes":
        data = {"timestamp": [datetime.now() + timedelta(minutes=i) for i in range(row_count)]}
    elif time_increment == "Hours":
        data = {"timestamp": [datetime.now() + timedelta(hours=i) for i in range(row_count)]}
    else:  # Days
        data = {"timestamp": [datetime.now() + timedelta(days=i) for i in range(row_count)]}


    name_holder = []

    for i in range(series_count):
        series_name = st.sidebar.text_input(f"Name of Series {i+1}", f"series_{i+1}")
        name_holder.append(series_name)


    for i in range(series_count):
        st.sidebar.markdown(f"### Series {i+1} Configuration")
        #series_name = st.sidebar.text_input(f"Name of Series {i+1}", f"series_{i+1}")
        series_name = name_holder[i]
        algorithm = st.sidebar.selectbox(f"Algorithm for {series_name}", list(ALGORITHMS.keys()) + ["Combination"])

        if algorithm == "Sinusoidal":
            frequency = st.sidebar.slider("Frequency", 1, 10, 1)
            offset = st.sidebar.slider("Offset", 0.0, 2 * np.pi, 0.1)
        else:
            frequency, offset = 1, 0

        seed = st.sidebar.number_input(f"Seed for {series_name}", value=i+1, step=1, format="%i")
        lower_bound = st.sidebar.number_input(f"Lower Bound for {series_name}", value=-10.0)
        upper_bound = st.sidebar.number_input(f"Upper Bound for {series_name}", value=10.0)

        if algorithm == "Combination":
            ref_series_1 = st.sidebar.selectbox(f"First Series for Combination {series_name}", name_holder)
            ref_series_2 = st.sidebar.selectbox(f"Second Series for Combination {series_name}", name_holder)
            operation = st.sidebar.selectbox(f"Operation for {series_name}", COMBINATION_OPS)
            
            series = combine_series(data, ref_series_1, ref_series_2, operation)
        else:
            series = generate_series(row_count, algorithm, lower_bound, upper_bound, seed, frequency, offset)

        data[series_name] = series
        
    
    return pd.DataFrame(data)





def data_bounds():

    'SERIES NAME'

    with st.sidebar:
        col1, col2, col3 = st.columns([4, 1, 1])

        with col1:
            series_name = st.text_input("Name", label_visibility="collapsed")
        with col2:
            series_button_add = st.button("➕")
        with col3:
            series_button_remove = st.button("🗑️")


        series_type = st.selectbox("Type", ["Distribution", "Random Walk", "Sinusoidal", "Uniform Random", "Combination"])


        col1, col2 = st.columns(2)

        with col1:
            data_min = st.number_input("Min")

        with col2:
            data_max = st.number_input("Max", value=1.00)

        
        if series_button_add:
            st.session_state.series[series_name] = {'config': 'a'}
            st.toast(f'{series_name} Updated ✅')


        if series_button_remove:
            if series_name in st.session_state.series:
                _popped = st.session_state.series.pop(series_name)
                st.toast(f'{series_name} Removed 🚮')




def time_scalar():

    with st.sidebar:
        time_unit = st.select_slider("Unit of Time", [
                    "ns",
                    "ms",
                    "s",
                    "m",
                    "h",
                    "d",
                    "w",
                    "M",
                    "Y"
                ])
        
        time_number = st.number_input('Set Number', 
                                            min_value=0,
                                            help=f'Select a number of {time_unit}')
        
        
        f'**{time_number} {time_unit}**'

        st.divider()

    data_bounds()
    

    # df = pd.DataFrame(columns=['Unit of Time', 'Number'])
    # st.sidebar.data_editor(df, 
    #                        num_rows="dynamic",
    #                         column_config={
    #     "Unit of Time": st.column_config.SelectboxColumn(
    #         "Unity of Time",
    #         help="The unit of time",
    #         width="medium",
    #         options=[
    #             "ns",
    #             "ms",
    #             "s",
    #             "m",
    #             "h",
    #             "d",
    #             "w",
    #             "M",
    #             "Y"
    #         ],
    #     )
    # },)
    # 'Add Time period'
    # st.sidebar.button('Add')




    # 'Starting a long computation...'

    # # Add a placeholder
    # latest_iteration = st.empty()
    # bar = st.progress(0)

    # for i in range(100):
    #     # Update the progress bar with each iteration.
    #     latest_iteration.text(f'Iteration {i+1}')
    #     bar.progress(i + 1)
    #     time.sleep(0.1)

    # '...and now we\'re done!'

def time_span():
    'time span'



#st.toast('Welcome - DATA GENeration time', icon='📈')

  #st.sidebar.metric("My metric", 32, 2)

with st.sidebar:
    # Preview
    st.header("Configuration")
    choice = st.selectbox("Data Type", ["Time Scalar", "Time Span"])


st.json(st.session_state.series)


if choice == "Time Scalar":
    time_scalar()

elif choice == "Time Span":
    time_span()








preview_rows = st.sidebar.slider("Select number of preview rows", 10, 200, 50)
preview_df = generate_data(preview_rows)
st.subheader("Preview Data:")
st.write(preview_df)

st.line_chart(preview_df.set_index('timestamp'))




# Regenerate based on user input
st.sidebar.header("Download Configuration")
final_row_count = st.sidebar.slider("Select number of rows for download", 10, 10000, 1000)
if st.sidebar.button('Regenerate and Download Data'):
    final_df = generate_data(final_row_count)
    st.sidebar.markdown(to_csv_download_link(final_df), unsafe_allow_html=True)