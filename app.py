import streamlit as st
import pandas as pd
import plotly.io as pio
import json
import time
import os
from openai import OpenAI
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI

from agents.pandas_data_analyst import PandasDataAnalyst
from agents.data_wrangling_agent import DataWranglingAgent
from agents.data_visualization_agent import DataVisualizationAgent

# App configuration
MODEL_LIST = ["gpt-4o-mini", "gpt-4o"]
APP_TITLE = "Data Analyst AI Copilot"
APP_ICON = "📊"
THEME_COLOR = "#4169E1"  # Royal Blue
ACCENT_COLOR = "#FFA500"  # Orange
BG_COLOR = "#F8F9FA"

# Set page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown(f"""
<style>
    /* Main styling */
    .stApp {{
        background-color: {BG_COLOR};
    }}
    .main .block-container {{
        padding-top: 2rem;
    }}
    
    /* Header styling */
    .custom-title {{
        font-size: 2.5rem;
        font-weight: 700;
        color: {THEME_COLOR};
        margin-bottom: 0;
    }}
    .custom-subtitle {{
        font-size: 1.2rem;
        color: #555;
        margin-top: 0;
    }}
    
    /* Card styling */
    .css-card {{
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}
    .css-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
    }}
    
    /* Button styling */
    .stButton>button {{
        background-color: {THEME_COLOR};
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: 600;
        transition: all 0.3s ease;
    }}
    .stButton>button:hover {{
        background-color: {ACCENT_COLOR};
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }}
    
    /* Sidebar styling */
    .css-sidebar {{
        background-color: #F8F9FA;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }}
    
    /* File uploader */
    .css-file-uploader {{
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }}
    .css-file-uploader:hover {{
        border-color: {THEME_COLOR};
    }}
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        white-space: pre-wrap;
        background-color: #f1f3f4;
        border-radius: 5px 5px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {THEME_COLOR};
        color: white;
    }}
    
    /* Chat container */
    .chat-container {{
        margin-bottom: 5rem;
    }}
    
    /* Custom labels */
    .custom-label {{
        font-weight: 600;
        color: {THEME_COLOR};
        margin-bottom: 0.5rem;
    }}
    
    /* Success message */
    .success-message {{
        background-color: #D4EDDA;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }}
    
    /* Error message */
    .error-message {{
        background-color: #F8D7DA;
        color: #721C24;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }}
    
    /* Info message */
    .info-message {{
        background-color: #E2F0FD;
        color: #0C5460;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }}
    
    /* Animation */
    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to {{ opacity: 1; }}
    }}
    .fade-in {{
        animation: fadeIn 0.5s ease-in-out;
    }}
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if "plots" not in st.session_state:
    st.session_state.plots = []

if "dataframes" not in st.session_state:
    st.session_state.dataframes = []

if "api_key_valid" not in st.session_state:
    st.session_state.api_key_valid = False

# Functions
def validate_api_key(api_key):
    """Validate the OpenAI API key"""
    try:
        client = OpenAI(api_key=api_key)
        models = client.models.list()
        st.session_state.api_key_valid = True
        return True
    except Exception as e:
        st.session_state.api_key_valid = False
        return False

def display_chat_history():
    """Display chat message history"""
    for msg in msgs.messages:
        with st.chat_message(msg.type):
            if "PLOT_INDEX:" in msg.content:
                plot_index = int(msg.content.split("PLOT_INDEX:")[1])
                st.plotly_chart(
                    st.session_state.plots[plot_index], 
                    key=f"history_plot_{plot_index}",
                    use_container_width=True
                )
            elif "DATAFRAME_INDEX:" in msg.content:
                df_index = int(msg.content.split("DATAFRAME_INDEX:")[1])
                st.dataframe(
                    st.session_state.dataframes[df_index],
                    key=f"history_dataframe_{df_index}",
                    use_container_width=True
                )
            else:
                st.write(msg.content)

def process_query(question, data, model_option, api_key):
    """Process user query using AI agents"""
    with st.spinner("Analyzing your data... Please wait."):
        try:
            llm = ChatOpenAI(model=model_option, api_key=api_key)
            
            # Initialize agents
            pandas_data_analyst = PandasDataAnalyst(
                model=llm,
                data_wrangling_agent=DataWranglingAgent(
                    model=llm,
                    log=False,
                    bypass_recommended_steps=True,
                    n_samples=100,
                ),
                data_visualization_agent=DataVisualizationAgent(
                    model=llm,
                    n_samples=100,
                    log=False,
                ),
            )
            
            # Invoke the agent
            pandas_data_analyst.invoke_agent(
                user_instructions=question,
                data_raw=data,
            )
            
            # Get the response
            result = pandas_data_analyst.get_response()
            return result
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            return None

# App Header
st.markdown(f'<h1 class="custom-title fade-in">{APP_TITLE}</h1>', unsafe_allow_html=True)
st.markdown('<p class="custom-subtitle fade-in">Unlock insights from your data with AI-powered analysis</p>', unsafe_allow_html=True)

# Initialize chat history
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you analyze this data?")

# Create tabs for main navigation
tab1, tab2, tab3 = st.tabs(["📈 Data Analysis", "❓ Help & Examples", "ℹ️ About"])

with tab1:
    # Sidebar configuration
    st.sidebar.markdown('<div class="css-sidebar">', unsafe_allow_html=True)
    st.sidebar.header("Configuration")
    
    # API Key Input
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        placeholder="Enter your OpenAI API key",
        type="password",
        help="Your OpenAI API key is required for the app to function."
    )
    
    # Store API key in session state
    if api_key:
        st.session_state["OPENAI_API_KEY"] = api_key
        
        # Validate API key if not already validated
        if not st.session_state.api_key_valid:
            # Create a placeholder in the sidebar for status messages
            validation_placeholder = st.sidebar.empty()
            validation_placeholder.info("Validating API key...")
            
            # Validate the API key
            is_valid = validate_api_key(api_key)
            
            # Update the placeholder with the result
            if is_valid:
                validation_placeholder.markdown('<div class="success-message">API Key is valid! ✅</div>', unsafe_allow_html=True)
            else:
                validation_placeholder.markdown('<div class="error-message">Invalid API Key. Please check and try again.</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div class="info-message">Please enter your OpenAI API Key to proceed.</div>', unsafe_allow_html=True)
        st.session_state.api_key_valid = False
    
    # Model Selection
    model_option = st.sidebar.selectbox(
        "Choose OpenAI model",
        MODEL_LIST,
        index=0,
        help="GPT-4o-mini is faster but less capable. GPT-4o is more powerful but may be slower."
    )
    
    # Advanced options expander
    with st.sidebar.expander("Advanced Options"):
        sample_size = st.slider(
            "Sample Size",
            min_value=50,
            max_value=500,
            value=100,
            step=50,
            help="Number of rows to sample for data analysis. Lower values are faster but less accurate."
        )
        
        log_enabled = st.checkbox(
            "Enable Logging",
            value=False,
            help="Log agent operations for debugging."
        )
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('<p class="custom-label">1. Upload Your Data</p>', unsafe_allow_html=True)
        
        st.markdown('<div class="css-file-uploader">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            help="Support for CSV and Excel files (.xlsx, .xls)"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                file_stats = {
                    "Rows": len(df),
                    "Columns": len(df.columns),
                    "Memory usage": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
                }
                
                st.success(f"✅ Successfully loaded: {uploaded_file.name}")
                
                # Display file stats
                st.markdown("#### File Statistics")
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                col_stats1.metric("Rows", f"{file_stats['Rows']:,}")
                col_stats2.metric("Columns", file_stats['Columns'])
                col_stats3.metric("Size", file_stats['Memory usage'])
                
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        else:
            st.info("Please upload a CSV or Excel file to get started.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('<p class="custom-label">2. Try Example Questions</p>', unsafe_allow_html=True)
        
        example_questions = [
            "Show the top 5 rows of this dataset",
            "What is the average value in each column?",
            "Show a bar chart of the top 5 values",
            "Create a scatter plot comparing two numeric columns",
            "Show the correlation heatmap between numeric columns"
        ]
        
        for question in example_questions:
            if st.button(question, key=f"btn_{question}"):
                if uploaded_file is not None and st.session_state.api_key_valid:
                    # Display user message
                    st.chat_message("human").write(question)
                    msgs.add_user_message(question)
                    
                    # Process the query
                    with st.spinner("Processing example question..."):
                        result = process_query(
                            question=question,
                            data=df,
                            model_option=model_option,
                            api_key=st.session_state["OPENAI_API_KEY"]
                        )
                        
                        if result:
                            routing = result.get("routing_preprocessor_decision")
                            
                            if routing == "chart" and not result.get("plotly_error", False):
                                # Process chart result
                                plot_data = result.get("plotly_graph")
                                if plot_data:
                                    # Convert dictionary to JSON string if needed
                                    if isinstance(plot_data, dict):
                                        plot_json = json.dumps(plot_data)
                                    else:
                                        plot_json = plot_data
                                    plot_obj = pio.from_json(plot_json)
                                    response_text = "Here's the visualization based on your request:"
                                    # Store the chart
                                    plot_index = len(st.session_state.plots)
                                    st.session_state.plots.append(plot_obj)
                                    msgs.add_ai_message(response_text)
                                    msgs.add_ai_message(f"PLOT_INDEX:{plot_index}")
                                    st.chat_message("ai").write(response_text)
                                    st.plotly_chart(plot_obj, use_container_width=True)
                                else:
                                    st.chat_message("ai").write("The agent did not return a valid chart.")
                                    msgs.add_ai_message("The agent did not return a valid chart.")
                            
                            elif routing == "table":
                                # Process table result
                                data_wrangled = result.get("data_wrangled")
                                if data_wrangled is not None:
                                    response_text = "Here's the data table you requested:"
                                    # Ensure data_wrangled is a DataFrame
                                    if not isinstance(data_wrangled, pd.DataFrame):
                                        data_wrangled = pd.DataFrame(data_wrangled)
                                    df_index = len(st.session_state.dataframes)
                                    st.session_state.dataframes.append(data_wrangled)
                                    msgs.add_ai_message(response_text)
                                    msgs.add_ai_message(f"DATAFRAME_INDEX:{df_index}")
                                    st.chat_message("ai").write(response_text)
                                    st.dataframe(data_wrangled, use_container_width=True)
                                else:
                                    st.chat_message("ai").write("No table data was returned by the agent.")
                                    msgs.add_ai_message("No table data was returned by the agent.")
                            else:
                                # Fallback
                                data_wrangled = result.get("data_wrangled")
                                if data_wrangled is not None:
                                    response_text = "I apologize. There was an issue with generating the chart. Returning the data table instead."
                                    if not isinstance(data_wrangled, pd.DataFrame):
                                        data_wrangled = pd.DataFrame(data_wrangled)
                                    df_index = len(st.session_state.dataframes)
                                    st.session_state.dataframes.append(data_wrangled)
                                    msgs.add_ai_message(response_text)
                                    msgs.add_ai_message(f"DATAFRAME_INDEX:{df_index}")
                                    st.chat_message("ai").write(response_text)
                                    st.dataframe(data_wrangled, use_container_width=True)
                                else:
                                    response_text = "An error occurred while processing your query. Please try again."
                                    msgs.add_ai_message(response_text)
                                    st.chat_message("ai").write(response_text)
                        else:
                            st.chat_message("ai").write("An error occurred while processing your query. Please try again.")
                            msgs.add_ai_message("An error occurred while processing your query. Please try again.")
                else:
                    if not uploaded_file:
                        st.warning("Please upload a file first")
                    if not st.session_state.api_key_valid:
                        st.warning("Please enter a valid API key first")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown('<p class="custom-label">3. Preview Your Data</p>', unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="css-card chat-container">', unsafe_allow_html=True)
        st.markdown('<p class="custom-label">4. Chat with Your Data</p>', unsafe_allow_html=True)
        
        # Display chat history
        display_chat_history()
        
        # Chat input
        if question := st.chat_input("Ask a question about your data...", key="query_input"):
            if not st.session_state.api_key_valid:
                st.error("Please enter a valid OpenAI API Key to proceed.")
            else:
                # Display user message
                st.chat_message("human").write(question)
                msgs.add_user_message(question)
                
                # Process the query
                with st.spinner("Thinking... This may take a moment depending on your dataset size and query complexity."):
                    result = process_query(
                        question=question,
                        data=df,
                        model_option=model_option,
                        api_key=st.session_state["OPENAI_API_KEY"]
                    )
                    
                    if result:
                        routing = result.get("routing_preprocessor_decision")
                        
                        if routing == "chart" and not result.get("plotly_error", False):
                            # Process chart result
                            plot_data = result.get("plotly_graph")
                            if plot_data:
                                # Convert dictionary to JSON string if needed
                                if isinstance(plot_data, dict):
                                    plot_json = json.dumps(plot_data)
                                else:
                                    plot_json = plot_data
                                plot_obj = pio.from_json(plot_json)
                                response_text = "Here's the visualization based on your request:"
                                # Store the chart
                                plot_index = len(st.session_state.plots)
                                st.session_state.plots.append(plot_obj)
                                msgs.add_ai_message(response_text)
                                msgs.add_ai_message(f"PLOT_INDEX:{plot_index}")
                                st.chat_message("ai").write(response_text)
                                st.plotly_chart(plot_obj, use_container_width=True)
                            else:
                                st.chat_message("ai").write("The agent did not return a valid chart.")
                                msgs.add_ai_message("The agent did not return a valid chart.")
                        
                        elif routing == "table":
                            # Process table result
                            data_wrangled = result.get("data_wrangled")
                            if data_wrangled is not None:
                                response_text = "Here's the data table you requested:"
                                # Ensure data_wrangled is a DataFrame
                                if not isinstance(data_wrangled, pd.DataFrame):
                                    data_wrangled = pd.DataFrame(data_wrangled)
                                df_index = len(st.session_state.dataframes)
                                st.session_state.dataframes.append(data_wrangled)
                                msgs.add_ai_message(response_text)
                                msgs.add_ai_message(f"DATAFRAME_INDEX:{df_index}")
                                st.chat_message("ai").write(response_text)
                                st.dataframe(data_wrangled, use_container_width=True)
                            else:
                                st.chat_message("ai").write("No table data was returned by the agent.")
                                msgs.add_ai_message("No table data was returned by the agent.")
                        else:
                            # Fallback if routing decision is unclear or if chart error occurred
                            data_wrangled = result.get("data_wrangled")
                            if data_wrangled is not None:
                                response_text = (
                                    "I apologize. There was an issue with generating the chart. "
                                    "Returning the data table instead."
                                )
                                if not isinstance(data_wrangled, pd.DataFrame):
                                    data_wrangled = pd.DataFrame(data_wrangled)
                                df_index = len(st.session_state.dataframes)
                                st.session_state.dataframes.append(data_wrangled)
                                msgs.add_ai_message(response_text)
                                msgs.add_ai_message(f"DATAFRAME_INDEX:{df_index}")
                                st.chat_message("ai").write(response_text)
                                st.dataframe(data_wrangled, use_container_width=True)
                            else:
                                response_text = (
                                    "An error occurred while processing your query. Please try again."
                                )
                                msgs.add_ai_message(response_text)
                                st.chat_message("ai").write(response_text)
                    else:
                        st.chat_message("ai").write("An error occurred while processing your query. Please try again.")
                        msgs.add_ai_message("An error occurred while processing your query. Please try again.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # No file uploaded yet
        st.markdown("""
        <div style="text-align:center; padding:4rem 2rem; background-color:#f8f9fa; border-radius:10px; margin:2rem 0;">
            <img src="https://cdn-icons-png.flaticon.com/512/4712/4712139.png" style="width:100px; margin-bottom:1rem;">
            <h3>No Data Uploaded Yet</h3>
            <p>Upload a CSV or Excel file to get started with your data analysis journey!</p>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.header("How to Use This App")
    
    st.markdown("""
    ### Getting Started
    1. **Enter your OpenAI API Key** in the sidebar
    2. **Upload your data file** (CSV or Excel format)
    3. **Ask questions** about your data using natural language
    4. View the **results** as tables or interactive charts
    
    ### Example Questions
    
    #### Data Exploration
    - "Show me the first 10 rows of this dataset"
    - "What are the columns in this dataset?"
    - "Give me summary statistics for all numeric columns"
    - "Show me the unique values in [column_name]"
    
    #### Data Analysis
    - "What's the average [column_name] grouped by [another_column]?"
    - "Calculate the correlation between [column1] and [column2]"
    - "Find the top 5 values in [column_name]"
    - "What percentage of values in [column_name] are greater than [value]?"
    
    #### Data Visualization
    - "Create a bar chart of [column_name]"
    - "Show me a scatter plot of [column1] vs [column2]"
    - "Make a pie chart showing the distribution of [column_name]"
    - "Plot a histogram of [column_name]"
    - "Create a heatmap of correlations between numeric columns"
    
    ### Tips for Better Results
    - Be specific about what you want to see
    - Mention column names exactly as they appear in your data
    - Specify the chart type if you have a preference
    - For complex analyses, break down your question into simpler steps
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.header("Example Datasets")
    
    example_datasets = {
        "Sales Data": {
            "description": "Monthly sales data with product categories and regional information",
            "example_questions": [
                "What are the top 5 products by revenue?",
                "Show monthly sales trends as a line chart",
                "Compare sales across different regions in a bar chart"
            ]
        },
        "Customer Churn": {
            "description": "Telecom customer data with churn indicators and service details",
            "example_questions": [
                "What factors correlate most strongly with customer churn?",
                "Show the distribution of customer tenure in a histogram",
                "Compare monthly charges for customers who churned vs stayed"
            ]
        },
        "Stock Market": {
            "description": "Historical stock price data with various technical indicators",
            "example_questions": [
                "Plot the closing prices over time with a trendline",
                "Calculate the average daily return and volatility",
                "Show the correlation between trading volume and price changes"
            ]
        }
    }
    
    for dataset_name, dataset_info in example_datasets.items():
        st.subheader(dataset_name)
        st.write(dataset_info["description"])
        st.markdown("**Example questions:**")
        for question in dataset_info["example_questions"]:
            st.markdown(f"- {question}")
        st.markdown("---")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.header("About This App")
    
    st.markdown("""
    ### Pandas Data Analyst AI Copilot
    
    This application combines the power of large language models with data analysis tools to help you explore, 
    analyze, and visualize your data through natural language. Built with Streamlit and backed by OpenAI's 
    advanced AI models, this tool makes data analysis accessible to everyone, regardless of their technical expertise.
    
    ### Key Features
    
    - **Natural Language Interface**: Ask questions about your data in plain English
    - **Data Wrangling**: Automatically transform, filter, and prepare your data
    - **Interactive Visualizations**: Generate insightful charts and graphs based on your queries
    - **Multi-format Support**: Works with CSV and Excel files
    - **Conversation Memory**: Maintains context throughout your analysis session
    
    ### Technology Stack
    
    - **Streamlit**: Front-end interface and application framework
    - **Pandas**: Data manipulation and analysis
    - **OpenAI LLMs**: Natural language understanding and code generation
    - **Plotly**: Interactive data visualizations
    - **LangChain**: Framework for AI-powered applications
    
    ### Created By
    
    This application was developed by Georges Chamma as part of a portfolio project showcasing the 
    intersection of data science and artificial intelligence.
    
    ### Acknowledgments
    
    Special thanks to OpenAI, the Streamlit team, and the open-source community for creating and 
    maintaining the tools that make applications like this possible.
    """)
    
    # Add links to GitHub and portfolio
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/georgeschamma)
        """)
    
    with col2:
        st.markdown("""
        [![Portfolio](https://img.shields.io/badge/Portfolio-0A0A0A?style=for-the-badge&logo=dev.to&logoColor=white)](https://georgeschamma.github.io)
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align:center; margin-top:3rem; padding:1rem; border-top:1px solid #eee;">
    <p>© 2025 Georges Chamma • Data Science & AI Innovation</p>
</div>
""", unsafe_allow_html=True)