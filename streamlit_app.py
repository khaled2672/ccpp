import streamlit as st
import matplotlib.pyplot as plt

def set_theme(dark):
    plt.style.use('dark_background' if dark else 'default')

    if dark:
        st.markdown(
        """ <style>
        .stApp {
        background-image: url("https://img.freepik.com/free-photo/view-nuclear-power-plant-with-towers-letting-out-steam-from-process_23-2150957658.jpg?t=st=1746689462~exp=1746693062~hmac=71da5c1edb4e4c2bd79eda912f889934c4d11e1aeea35a5106d1bd18e53a89b4&w=1380");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        color: #f1f1f1;
        }
        /* Dark overlay for better readability */
        .stApp\:before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: rgba(0, 0, 0, 0.75);
        z-index: -1;
        }
        /* Main content area */
        .main .block-container {
        background-color: rgba(0, 0, 0, 0.7);
        padding: 2rem;
        border-radius: 10px;
        backdrop-filter: blur(4px);
        }
        /* Sidebar */
        [data-testid="stSidebar"] > div\:first-child {
        background-color: rgba(0, 0, 0, 0.8) !important;
        color: #ffffff ;
        backdrop-filter: blur(4px);
        }
        /* Text colors */
        .css-1d391kg, .css-1cpxqw2, .st-b7, .st-b8, .st-b9 {
        color: #f1f1f1 !important;
        }
        /* Widget styling */
        .st-bb, .st-at, .st-ae, .st-af, .st-ag, .st-ah, .st-ai, .st-aj {
        background-color: rgba(30, 30, 30, 0.7) !important;
        }
        /* Button styling */
        .stDownloadButton, .stButton>button {
        background-color: #4a8af4 !important;
        color: black !important;
        border: white !important;
        }
        .stDownloadButton\:hover, .stButton>button\:hover {
        background-color: #f5f6f7 !important;
        } </style>
        """,
        unsafe_allow_html=True
        )
    else:
        st.markdown(
        """ <style>
        .stApp {
        background-image: url("https://img.freepik.com/free-photo/view-nuclear-power-plant-with-towers-letting-out-steam-from-process_23-2150957658.jpg?t=st=1746689462~exp=1746693062~hmac=71da5c1edb4e4c2bd79eda912f889934c4d11e1aeea35a5106d1bd18e53a89b4&w=1380");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        color: #333333;
        }
        /* Light overlay for better readability */
        .stApp\:before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: rgba(255, 255, 255, 0.75);
        z-index: -1;
        }
        /* Main content area */
        .main .block-container {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 2rem;
        border-radius: 10px;
        backdrop-filter: blur(4px);
        }
        /* Sidebar */
        [data-testid="stSidebar"] > div\:first-child {
        background-color: rgba(255, 255, 255, 0.85) !important;
        backdrop-filter: blur(4px);
        }
        /* Text colors */
        .css-1d391kg, .css-1cpxqw2, .st-b7, .st-b8, .st-b9 {
        color: #333333 !important;
        }
        /* Widget styling */
        .st-bb, .st-at, .st-ae, .st-af, .st-ag, .st-ah, .st-ai, .st-aj {
        background-color: rgba(240, 240, 240, 0.8) !important;
        }
        /* Button styling */
        .stDownloadButton, .stButton>button {
        background-color: #4a8af4 !important;
        color: white !important;
        border: none !important;
        }
        .stDownloadButton\:hover, .stButton>button\:hover {
        background-color: #3a7ae4 !important;
        } </style>
        """,
        unsafe_allow_html=True
        )
        
# Example of how to use this theme function in your app:
if __name__ == '__main__':
    # Toggle the theme
    dark_mode = True  # Set this to False for light mode, True for dark mode
    set_theme(dark_mode)

    # Your Streamlit content goes here
    st.title("Welcome to the Streamlit App")
    st.write("This app supports both dark and light modes.")
    
    # Example content that you can place in your app
    st.markdown("### Example Markdown")
    st.write("Here is an example of some content in the app.")
    
    # You can add more widgets, charts, etc.
    st.button('Click Me')
