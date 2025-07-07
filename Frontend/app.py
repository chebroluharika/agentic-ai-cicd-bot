from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import streamlit as st
import os, sys, requests, urllib, re
sys.path.append("..")

from backend.jenkins_operations import JenkinsOperations
from backend.auth.auth import authenticate
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain_community.llms import Ollama

# Load environment variables
load_dotenv("../backend/config/auth.env")

# Jenkins Configuration
JENKINS_BASE_URL = os.getenv("JENKINS_BASE_URL")
JENKINS_USER = os.getenv("JENKINS_USER")
JENKINS_API_TOKEN = os.getenv("JENKINS_API_TOKEN")

# Jenkins API Instance
jenkins_api = JenkinsOperations()

# Authentication state
if "authenticated_user" not in st.session_state:
    st.session_state.authenticated_user = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Function to handle login
def handle_login():
    username = st.session_state.username
    password = st.session_state.password
    auth_result = authenticate(username, password)

    if auth_result["status"] == "failed":
        st.session_state.authenticated_user = None
        st.error(f"❌ {auth_result['message']}")
    else:
        st.session_state.authenticated_user = {"username": username, "role": auth_result["role"]}
        st.success(f"✅ Welcome, {username} ({auth_result['role']})!")
        st.rerun()

# Function to handle logout
def handle_logout():
    st.session_state.authenticated_user = None
    st.success("✅ You have been logged out.")
    st.rerun()

# Define Tool Functions
def list_all_jobs(*args, **kwargs):
    if not st.session_state.authenticated_user:
        return "⚠️ Please log in first."
    jobs = jenkins_api.get_all_jobs(st.session_state.authenticated_user)
    if isinstance(jobs, dict) and "jobs" in jobs:
        job_list = jobs["jobs"][:10]  # Limit display to first 10 jobs
        formatted_jobs = "\n".join(f"- {job}" for job in job_list)
        return f"Here are some available Jenkins jobs:\n{formatted_jobs}\n\n(Type 'Show More' for additional jobs.)"
    return "❌ Failed to fetch job list."

# def trigger_job(job_name: str):
#     if not st.session_state.authenticated_user:
#         return "⚠️ Please log in first."
#     return jenkins_api.trigger_job(st.session_state.authenticated_user, job_name, {})

def trigger_job(job_name: str, params=None):
    """triggers the job with parameters job_name and params for job_name"""
    if not st.session_state.authenticated_user:
        return "⚠️ Please log in first."

    params = params or {}
    formatted_params = urllib.parse.urlencode(params)
    url = f"{JENKINS_BASE_URL}/job/{urllib.parse.quote(job_name)}/buildWithParameters?{formatted_params}"

    response = requests.post(url, auth=(JENKINS_USER, JENKINS_API_TOKEN))
    
    if response.status_code == 201:
        # st.success(f"✅ Job '{job_name}' triggered successfully!")
        return f"✅ Job '{job_name}' triggered successfully!"
    else:
        return f"❌ Failed to trigger job. Status: {response.status_code}, Response: {response.text}"

def get_last_build_summary(job_name: str):
    data = jenkins_api.get_last_build_summary(job_name)
    if not isinstance(data, dict):
        return "Error: Invalid API response."
    
    build_number = data.get("number", "Unknown")
    build_status = data.get("result", None)
    build_url = data.get("url", "#")

    if build_status is None or build_status == "None":
        build_status = "RUNNING"

    status_color = "green" if str(build_status).upper() == "SUCCESS" else "red"

    build_summary = (
        f"<p>Build <span style='color: blue;'>#{build_number}</span> "
        f"Status: <span style='color: {status_color};'>{build_status}</span> - "
        f"<a href='{build_url}' target='_blank'>{build_url}</a></p>"
    )
    st.markdown(build_summary, unsafe_allow_html=True)
    return build_summary

# def get_specific_build_summary(job_name: str, build_number: str):
#     """
#     Gets the build summary of specifc build of a job_name where
#     parameters to this function are job_name and build_number
#     Args:
#         job_name (str): name of the job where we need get build details
#         build_number (str): build number
#     """
#     data = jenkins_api.get_specific_build_summary(job_name, int(build_number))
#     return f"Build #{build_number} Status: {data['result']} - {data['url']}"

def get_specific_build_summary(query: str):
    """Extracts job name and build number from the query and fetches the build summary."""
    
    match = re.search(r"build summary of (\S+) with build number (\d+)", query, re.IGNORECASE)
    if not match:
        return "❌ Invalid input format. Please provide job name and build number, e.g., 'get the build summary of my-job with build number 42'."

    job_name, build_number = match.groups()
    data = jenkins_api.get_specific_build_summary(job_name, build_number)

    if not isinstance(data, dict):
        return "❌ Error fetching build data."

    return f"Build #{build_number} Status: {data.get('result', 'Unknown')} - {data.get('url', '#')}"

# def get_job_health(job_name: str):
#     return jenkins_api.get_job_health(job_name)

def get_job_health(job_name: str):
    health_status = jenkins_api.get_job_health(job_name)

    # Convert binary health values to counts
    healthy_count = sum(1 for status in health_status if status == "Good")
    unhealthy_count = len(health_status) - healthy_count

    # Define labels and sizes
    labels = ["Healthy", "Unhealthy"]
    sizes = [healthy_count, unhealthy_count]
    colors = ["#4CAF50", "#FF5733"]  # Green for healthy, Red for unhealthy

    # Create a clean and compact pie chart
    fig, ax = plt.subplots(figsize=(0.75, 0.75))  # Small but clear
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct="%0.1f%%",
        colors=colors,
        startangle=140,
        wedgeprops={"edgecolor": "black", "linewidth": 0.6},  # Thin black edge for better visibility
        textprops={"fontsize": 5}  # Small but legible text
    )

    # Improve label and percentage visibility
    for text in texts:
        text.set_fontsize(3)  # Keep labels readable
        text.set_color("black")
    for autotext in autotexts:
        autotext.set_fontsize(3)  # Keep percentages readable
        autotext.set_color("white")
        autotext.set_weight("bold")

    # Adjust title with a clear but small font
    ax.set_title(f"Job Health: {job_name}", fontsize=5, fontweight="bold", color="#222")  

    # Display chart in Streamlit
    st.pyplot(fig)

    return f"Job '{job_name}' has {healthy_count} healthy runs and {unhealthy_count} unhealthy runs."



# Define Tools
tools = [
    Tool(name="List All Jobs", func=list_all_jobs, description="Lists available Jenkins jobs.", return_direct=True),
    Tool(name="Trigger Job", func=trigger_job, description="Triggers a Jenkins job."),
    Tool(name="Get Last Build Summary", func=get_last_build_summary, description="Fetches the last build summary."),
    Tool(name="Get Specific Build Summary", 
         func=get_specific_build_summary, 
         description="Fetches the summary of a specific Jenkins build. Example query: 'get the build summary of job-name with build number 42'."),
    Tool(name="Get Job Health", func=get_job_health, description="Checks the health status of a Jenkins job."),
]

# Memory and LLM
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = Ollama(model="llama3")

# Initialize AI Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
)

# **Process Query Function**
def process_query(query: str):
    if not st.session_state.authenticated_user:
        return "⚠️ Please log in first."
    response = agent.run(query)
    st.session_state.chat_history.append((query, response))
    return response

# Streamlit UI
st.set_page_config(page_title="CICD AI Bot", page_icon="🤖", layout="wide")
st.title("🤖 Welcome to CICD AI Agentic Bot")

# Sidebar Authentication
with st.sidebar:
    st.subheader("🔑 Authentication")

    if not st.session_state.authenticated_user:
        st.text_input("Username:", key="username")
        st.text_input("Password:", type="password", key="password")

        if st.button("Login"):
            handle_login()

    else:
        st.success(f"✅ Logged in as {st.session_state.authenticated_user['username']}")

        # Logout Button
        if st.button("Logout"):
            handle_logout()

# Main Section
if not st.session_state.authenticated_user:
    st.warning("⚠️ Please authenticate via the sidebar to use the chatbot.")
else:
    st.success(f"✅ Logged in as {st.session_state.authenticated_user['username']}")
    # display the propulated commands
    st.markdown(
    """
    <p style='font-size:22px; font-weight:bold;'>Here are some available commands you can use:</p>
    """,
    unsafe_allow_html=True
    )
    if st.button("⚡ Trigger Job"):
         st.session_state.show_trigger_input = True
         st.rerun()

    if st.button("⚡ list all the jobs"):
        response = process_query("list all the available jobs")
        # st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    if st.button("📊 Last Build Summary"):
        response = process_query("last build summary")
        st.markdown(response)

    if st.button("🔍 Specific Build Summary"):
        response = process_query("specific build summary")
        st.markdown(response)

    if st.button("💡 Job Health"):
        response = process_query("job health")
        st.markdown(response)

    if st.button("❌ Exit"):
        response = process_query("exit")
        st.markdown(response)
    
    # Show input fields only if "Trigger Jobs" is clicked
    if st.session_state.get("show_trigger_input", False):
        job_name = st.text_input("Job Name:", key="job_name_input")
        raw_params = st.text_area("Parameters (Optional, enter one per line as key=value):", key="params_input")

        if st.button("Submit Job"):
            # Parse parameters from key=value format
            params = {}
            if raw_params.strip():
                try:
                    for line in raw_params.strip().split("\n"):
                        key, value = line.split("=", 1)
                        params[key.strip()] = value.strip()
                except ValueError:
                    st.error("❌ Invalid parameters format! Use 'key=value', one per line.")
                    params = {}

            response = trigger_job(job_name, params)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    # Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Type your command (e.g., trigger job, last build summary)..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = process_query(prompt)
                st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})