![2023-08-23_20-53-25](https://github.com/jaun-van-heerden/datagen/assets/39254276/bc2bced9-4f1d-4c96-aca7-511ac3279557)Sure, here's an example of how your GitHub README markdown file could look like to explain how to set up and run your Streamlit app named `datagen-basic.py`:

```markdown
# DataGen by Jaun van Heerden

This Streamlit app, named `datagen-basic.py`, is designed to generate synthetic time-series data with various algorithms. You can customize the parameters of each series and combine them to create unique datasets for testing and analysis.

## Setup and Usage

1. Install Required Dependencies:

   Make sure you have the required packages installed. You can install them using the following command:

   ```bash
   pip install streamlit pandas numpy
   ```

2. Run the Streamlit App:

   Run the Streamlit app by executing the following command in your terminal:

   ```bash
   streamlit run datagen-basic.py
   ```

3. Configure Data Generation:

   - Choose the time increment (seconds, minutes, hours, or days).
   - Select the number of series you want to generate.
   - Provide configuration details for each series, including algorithm, frequency, offset, seed, lower bound, and upper bound.

4. Preview Generated Data:

   After configuring the data generation settings, you will see a preview of the generated data and a line chart representation.

5. Regenerate and Download Data:

   - Adjust the number of preview rows.
   - Use the "Regenerate and Download Data" button to generate the data and download it as a CSV file.

## Features

- **Algorithms**: Choose from random walk, sinusoidal, or uniform random algorithms.
- **Combination**: Combine series using arithmetic operations (add, subtract, multiply, divide, or exponent).
- **Customization**: Configure parameters for each series, such as frequency, offset, seed, lower bound, and upper bound.
- **Timestamps**: Generate timestamps based on your chosen time increment.
- **Preview**: Preview the generated data and visualize it using a line chart.

## Example Code

```bash
# Install required dependencies
pip install streamlit pandas numpy

# Run the Streamlit app
streamlit run datagen-basic.py
```

## Screenshots

<todo>


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This app was developed by Jaun van Heerden. Feel free to contribute, provide feedback, or report issues.

```
```

Remember to replace the example placeholders and descriptions with actual information relevant to your project.
