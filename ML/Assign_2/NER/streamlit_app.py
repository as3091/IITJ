import streamlit as st
import pandas as pd
import io

def string_to_dataframe(input_string):
    """Converts a string to a pandas DataFrame."""
    try:
        # Attempt to read the string as a CSV
        df = pd.read_csv(io.StringIO(input_string))
        return df
    except Exception as e:
        st.error(f"Error converting string to DataFrame: {e}")
        return None

st.title("String to DataFrame Converter")

input_string = st.text_area("Enter your string (e.g., CSV format):", 
                           value="col1,col2\n1,a\n2,b\n3,c", 
                           height=200)  # Example CSV string

if st.button("Convert to DataFrame"):
    if input_string:
        df = string_to_dataframe(input_string)
        if df is not None:
            st.write("DataFrame:")
            st.dataframe(df)
    else:
        st.warning("Please enter a string.")