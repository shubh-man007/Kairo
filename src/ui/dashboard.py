# """Streamlit dashboard for Kairo Evaluation Platform."""

# import sys
# import json
# import time
# from pathlib import Path
# from typing import Optional, List, Dict, Any
# import streamlit as st
# from datetime import datetime

# # Add project root to path
# project_root = Path(__file__).parent.parent.parent
# sys.path.insert(0, str(project_root))

# from src.models import Persona, Environment, Question, AgentResponse, EvaluationResult, PersonaScore
# from src.pipeline import EvaluationOrchestrator
# from src.agents import WeatherAgent
# from src.evaluators import LLMEvaluator, EnsembleEvaluator
# from src.llm import create_llm_client
# from src.config import get_settings
# from src.utils.logging import setup_logging, get_logger

# # Setup logging
# setup_logging()
# logger = get_logger(__name__)


# # Initialize session state
# def init_session_state():
#     """Initialize session state variables."""
#     if "evaluation_running" not in st.session_state:
#         st.session_state.evaluation_running = False
#     if "evaluation_results" not in st.session_state:
#         st.session_state.evaluation_results = []
#     if "persona_score" not in st.session_state:
#         st.session_state.persona_score = None
#     if "progress_data" not in st.session_state:
#         st.session_state.progress_data = {
#             "current_phase": "Not started",
#             "environments_selected": [],
#             "questions_generated": 0,
#             "responses_generated": 0,
#             "evaluations_completed": 0,
#             "total_steps": 0,
#             "completed_steps": 0,
#         }
#     if "detailed_results" not in st.session_state:
#         st.session_state.detailed_results = []


# def load_persona(file_path: Path) -> Persona:
#     """Load persona from JSON file."""
#     with open(file_path, "r", encoding="utf-8") as f:
#         data = json.load(f)
#     return Persona(**data)


# def load_environments(file_path: Path) -> List[Environment]:
#     """Load environments from JSON file."""
#     with open(file_path, "r", encoding="utf-8") as f:
#         data = json.load(f)
#     return [Environment(**env) for env in data]


# def render_config_panel():
#     """Render configuration panel in sidebar."""
#     st.sidebar.header("⚙️ Configuration")
    
#     # Load settings
#     settings = get_settings()
    
#     st.sidebar.subheader("Model Settings")
#     st.sidebar.text(f"Generator: {settings.generator_model}")
#     st.sidebar.text(f"Evaluator 1: {settings.evaluator_model_1}")
#     st.sidebar.text(f"Evaluator 2: {settings.evaluator_model_2}")
#     st.sidebar.text(f"Gen Temp: {settings.generator_temperature}")
#     st.sidebar.text(f"Eval Temp: {settings.evaluator_temperature}")
    
#     st.sidebar.divider()
    
#     st.sidebar.subheader("Evaluation Parameters")
#     num_environments = st.sidebar.slider(
#         "Environments per Persona",
#         min_value=1,
#         max_value=10,
#         value=2,
#         help="Number of environments to select for evaluation"
#     )
    
#     num_questions = st.sidebar.slider(
#         "Questions per Task",
#         min_value=1,
#         max_value=10,
#         value=2,
#         help="Number of questions to generate per evaluation task"
#     )
    
#     st.sidebar.divider()
    
#     # API Keys status
#     st.sidebar.subheader("API Keys Status")
#     import os
#     openai_status = "✅" if settings.openai_api_key else "❌"
#     anthropic_status = "✅" if settings.anthropic_api_key else "❌"
#     weather_key = os.getenv("ACCUWEATHER_API_KEY", "")
#     weather_status = "✅" if weather_key else "❌"
    
#     st.sidebar.text(f"OpenAI: {openai_status}")
#     st.sidebar.text(f"Anthropic: {anthropic_status}")
#     st.sidebar.text(f"AccuWeather: {weather_status}")
    
#     return num_environments, num_questions


# def render_persona_info(persona: Persona):
#     """Render persona information."""
#     st.header("👤 Persona Information")
    
#     col1, col2 = st.columns([2, 1])
    
#     with col1:
#         st.subheader(persona.name)
#         st.write(persona.description)
        
#         if persona.attributes:
#             st.write("**Attributes:**")
#             attrs_text = ", ".join([f"{k}: {v}" for k, v in persona.attributes.items()])
#             st.write(attrs_text)
    
#     with col2:
#         st.metric("Status", "Ready" if not st.session_state.evaluation_running else "Evaluating")


# def render_environment_selection(environments: List[Environment], selected: List[Environment]):
#     """Render environment selection visualization."""
#     st.header("🌍 Environment Selection")
    
#     st.write(f"**Total Available:** {len(environments)} environments")
#     st.write(f"**Selected for Evaluation:** {len(selected)} environments")
    
#     # Display selected environments
#     if selected:
#         cols = st.columns(min(3, len(selected)))
#         for idx, env in enumerate(selected):
#             with cols[idx % 3]:
#                 with st.container():
#                     st.markdown(f"### {env.name}")
#                     st.caption(env.description[:100] + "..." if len(env.description) > 100 else env.description)
#                     if env.domain:
#                         st.badge(env.domain, type="secondary")


# def render_question_generation(questions_by_task: Dict[str, List[Question]]):
#     """Render question generation results."""
#     st.header("❓ Question Generation")
    
#     total_questions = sum(len(questions) for questions in questions_by_task.values())
#     st.write(f"**Total Questions Generated:** {total_questions}")
    
#     # Display questions by task
#     for task_name, questions in questions_by_task.items():
#         with st.expander(f"📋 {task_name.replace('_', ' ').title()} ({len(questions)} questions)", expanded=False):
#             for idx, question in enumerate(questions, 1):
#                 st.markdown(f"**Q{idx}:** {question.text}")
#                 if question.quality_criteria:
#                     st.caption(f"*Quality Criteria: {question.quality_criteria[:100]}...*")


# def render_agent_responses(responses_data: List[Dict[str, Any]]):
#     """Render agent responses."""
#     st.header("🤖 Agent Responses")
    
#     st.write(f"**Total Responses:** {len(responses_data)}")
    
#     for idx, response_data in enumerate(responses_data, 1):
#         with st.expander(f"Response {idx}: {response_data.get('question', '')[:50]}...", expanded=False):
#             col1, col2 = st.columns([3, 1])
            
#             with col1:
#                 st.markdown("**Response Text:**")
#                 st.text_area(
#                     "",
#                     value=response_data.get("response", ""),
#                     height=150,
#                     disabled=True,
#                     key=f"response_{idx}",
#                     label_visibility="collapsed",
#                 )
            
#             with col2:
#                 st.markdown("**Metadata:**")
#                 metadata = response_data.get("metadata", {})
#                 if metadata:
#                     if "location_used" in metadata:
#                         st.write(f"📍 Location: {metadata.get('location_used', 'N/A')}")
#                     if "weather_data_fetched" in metadata:
#                         status = "✅ Yes" if metadata.get("weather_data_fetched") else "❌ No"
#                         st.write(f"🌤️ Weather Data: {status}")
#                     if "model" in metadata:
#                         st.write(f"🤖 Model: {metadata.get('model', 'N/A')}")


# def render_evaluation_details(evaluation_data: List[Dict[str, Any]]):
#     """Render detailed evaluation information."""
#     st.header("📊 Evaluation Details")
    
#     for idx, eval_data in enumerate(evaluation_data, 1):
#         task_name = eval_data.get("task", "Unknown")
#         question = eval_data.get("question", "")
#         response = eval_data.get("response", "")
#         scores = eval_data.get("scores", {})
#         evaluator_outputs = eval_data.get("evaluator_outputs", {})
        
#         with st.expander(f"Evaluation {idx}: {task_name.replace('_', ' ').title()}", expanded=False):
#             # Question and Response
#             st.markdown("**Question:**")
#             st.write(question)
            
#             st.markdown("**Agent Response:**")
#             st.text_area("", value=response, height=100, disabled=True, key=f"eval_response_{idx}", label_visibility="collapsed")
            
#             # Scores
#             st.markdown("**Scores:**")
#             col1, col2, col3 = st.columns(3)
            
#             with col1:
#                 if "evaluator_1" in scores:
#                     st.metric("Evaluator 1", f"{scores['evaluator_1']:.2f}")
#             with col2:
#                 if "evaluator_2" in scores:
#                     st.metric("Evaluator 2", f"{scores['evaluator_2']:.2f}")
#             with col3:
#                 if "final" in scores:
#                     st.metric("Final Score", f"{scores['final']:.2f}", delta=None)
            
#             # Evaluator outputs (if available)
#             if evaluator_outputs:
#                 with st.expander("Evaluator Details", expanded=False):
#                     for eval_name, output in evaluator_outputs.items():
#                         st.markdown(f"**{eval_name}:**")
#                         output_text = str(output)
#                         st.text_area("", value=output_text[:500] + "..." if len(output_text) > 500 else output_text, 
#                                     height=100, disabled=True, key=f"eval_output_{idx}_{eval_name}", 
#                                     label_visibility="collapsed")


# def render_results_dashboard(persona_score: Optional[PersonaScore], results: List[EvaluationResult]):
#     """Render results dashboard with charts and tables."""
#     st.header("📈 Results Dashboard")
    
#     if not persona_score:
#         st.info("Run an evaluation to see results here.")
#         return
    
#     # Overall score
#     col1, col2, col3, col4 = st.columns(4)
#     with col1:
#         st.metric("Overall PersonaScore", f"{persona_score.overall_score:.2f}", "out of 5.0")
#     with col2:
#         st.metric("Total Questions", persona_score.total_questions)
#     with col3:
#         st.metric("Total Evaluations", persona_score.total_evaluations)
#     with col4:
#         avg_per_task = persona_score.overall_score / len(persona_score.task_averages) if persona_score.task_averages else 0
#         st.metric("Avg per Task", f"{avg_per_task:.2f}")
    
#     st.divider()
    
#     # Task-wise scores
#     st.subheader("Task-wise Scores")
    
#     if persona_score.task_averages:
#         # Create a bar chart
#         import pandas as pd
#         import plotly.express as px
        
#         task_data = {
#             "Task": [t.replace("_", " ").title() for t in persona_score.task_averages.keys()],
#             "Score": list(persona_score.task_averages.values())
#         }
#         df = pd.DataFrame(task_data)
        
#         fig = px.bar(
#             df,
#             x="Task",
#             y="Score",
#             title="Scores by Evaluation Task",
#             color="Score",
#             color_continuous_scale="Viridis",
#             range_y=[0, 5],
#         )
#         fig.update_layout(showlegend=False, height=400)
#         st.plotly_chart(fig, use_container_width=True)
        
#         # Table view
#         st.dataframe(df, use_container_width=True, hide_index=True)
    
#     st.divider()
    
#     # Detailed results table
#     st.subheader("Detailed Results")
#     if results:
#         results_data = []
#         for result in results:
#             for task_score in result.task_scores:
#                 results_data.append({
#                     "Question": result.question_text[:60] + "..." if len(result.question_text) > 60 else result.question_text,
#                     "Task": task_score.task_type.replace("_", " ").title(),
#                     "Score": f"{task_score.score:.2f}",
#                     "Environment": result.environment_name,
#                 })
        
#         if results_data:
#             df_results = pd.DataFrame(results_data)
#             st.dataframe(df_results, use_container_width=True, hide_index=True)


# def run_evaluation(persona: Persona, environments: List[Environment], num_envs: int, num_questions: int):
#     """Run the evaluation pipeline."""
#     st.session_state.evaluation_running = True
#     st.session_state.evaluation_results = []
#     st.session_state.persona_score = None
#     st.session_state.detailed_results = []
    
#     # Reset progress
#     st.session_state.progress_data = {
#         "current_phase": "Initializing",
#         "environments_selected": [],
#         "questions_generated": 0,
#         "responses_generated": 0,
#         "evaluations_completed": 0,
#         "total_steps": 0,
#         "completed_steps": 0,
#     }
    
#     try:
#         settings = get_settings()
        
#         # Initialize clients
#         with st.spinner("Initializing LLM clients..."):
#             generator_client = create_llm_client(
#                 provider="openai",
#                 model=settings.generator_model,
#                 temperature=0.9,
#             )
#             agent_client = create_llm_client(
#                 provider="openai",
#                 model=settings.generator_model,
#                 temperature=settings.generator_temperature,
#             )
        
#         # Create weather agent
#         with st.spinner("Creating weather agent..."):
#             weather_agent = WeatherAgent(llm_client=agent_client)
        
#         # Create evaluators
#         with st.spinner("Creating evaluators..."):
#             evaluator1 = LLMEvaluator(
#                 llm_client=create_llm_client(
#                     provider="openai",
#                     model=settings.evaluator_model_1,
#                     temperature=0.0,
#                 )
#             )
#             evaluator2 = LLMEvaluator(
#                 llm_client=create_llm_client(
#                     provider="openai",
#                     model=settings.evaluator_model_1,
#                     temperature=0.0,
#                 )
#             )
#             ensemble = EnsembleEvaluator(evaluators=[evaluator1, evaluator2])
        
#         # Create orchestrator
#         with st.spinner("Setting up evaluation pipeline..."):
#             orchestrator = EvaluationOrchestrator(
#                 generator_client=generator_client,
#                 agent_client=agent_client,
#                 evaluator=ensemble,
#                 environment_pool=environments,
#                 agent=weather_agent,
#             )
        
#         # Run evaluation
#         st.session_state.progress_data["current_phase"] = "Running Evaluation"
        
#         progress_bar = st.progress(0)
#         status_text = st.empty()
        
#         # Calculate total steps (approximate)
#         total_steps = num_envs * 5 * num_questions  # environments * tasks * questions
#         st.session_state.progress_data["total_steps"] = total_steps
        
#         results, persona_score = orchestrator.evaluate_persona_with_score(
#             persona=persona,
#             num_environments=num_envs,
#             num_questions_per_task=num_questions,
#         )
        
#         # Store results
#         st.session_state.evaluation_results = results
#         st.session_state.persona_score = persona_score
#         st.session_state.progress_data["completed_steps"] = total_steps
#         st.session_state.progress_data["current_phase"] = "Completed"
        
#         progress_bar.progress(1.0)
#         status_text.success("✅ Evaluation completed successfully!")
        
#         # Process detailed results for display
#         process_detailed_results(results)
        
#         # Store selected environments for display (extract from results)
#         selected_env_names = list(set(r.environment_name for r in results))
#         st.session_state.progress_data["environments_selected"] = [
#             env for env in environments if env.name in selected_env_names
#         ]
        
#     except Exception as e:
#         st.error(f"❌ Evaluation failed: {str(e)}")
#         logger.error(f"Evaluation error: {e}", exc_info=True)
#     finally:
#         st.session_state.evaluation_running = False


# def process_detailed_results(results: List[EvaluationResult]):
#     """Process results for detailed display."""
#     detailed = []
#     for result in results:
#         for task_score in result.task_scores:
#             detailed.append({
#                 "task": task_score.task_type,
#                 "question": result.question_text,
#                 "response": result.response_text,
#                 "scores": {
#                     "final": task_score.score,
#                     "evaluator_1": task_score.evaluator_scores[0] if task_score.evaluator_scores and len(task_score.evaluator_scores) > 0 else None,
#                     "evaluator_2": task_score.evaluator_scores[1] if task_score.evaluator_scores and len(task_score.evaluator_scores) > 1 else None,
#                 },
#                 "environment": result.environment_name,
#             })
#     st.session_state.detailed_results = detailed


# def main():
#     """Main dashboard function."""
#     init_session_state()
    
#     # Title and header
#     st.title("🌤️ Kairo Evaluation Platform")
#     st.markdown("**Dynamic Agent Evaluation Dashboard**")
#     st.divider()
    
#     # Sidebar configuration
#     num_environments, num_questions = render_config_panel()
    
#     # Load data
#     data_dir = project_root / "data"
#     persona_path = data_dir / "personas" / "weather_persona.json"
#     environments_path = data_dir / "environments" / "weather_environments.json"
    
#     try:
#         persona = load_persona(persona_path)
#         environments = load_environments(environments_path)
#     except Exception as e:
#         st.error(f"Failed to load data: {e}")
#         st.stop()
    
#     # Persona information
#     render_persona_info(persona)
#     st.divider()
    
#     # Control buttons
#     col1, col2, col3 = st.columns([1, 1, 2])
    
#     with col1:
#         if st.button("🚀 Start Evaluation", type="primary", disabled=st.session_state.evaluation_running):
#             run_evaluation(persona, environments, num_environments, num_questions)
#             st.rerun()
    
#     with col2:
#         if st.button("🔄 Reset", disabled=st.session_state.evaluation_running):
#             st.session_state.evaluation_results = []
#             st.session_state.persona_score = None
#             st.session_state.detailed_results = []
#             st.rerun()
    
#     # Progress indicator
#     if st.session_state.evaluation_running:
#         st.info("⏳ Evaluation in progress... Please wait.")
    
#     # Display results
#     if st.session_state.evaluation_results:
#         # Environment selection (if available)
#         if st.session_state.progress_data.get("environments_selected"):
#             render_environment_selection(environments, st.session_state.progress_data["environments_selected"])
#             st.divider()
        
#         # Results dashboard
#         render_results_dashboard(st.session_state.persona_score, st.session_state.evaluation_results)
#         st.divider()
        
#         # Detailed evaluation
#         if st.session_state.detailed_results:
#             render_evaluation_details(st.session_state.detailed_results)
#             st.divider()
            
#             # Agent responses
#             responses_data = [
#                 {
#                     "question": d["question"],
#                     "response": d["response"],
#                     "metadata": {"task": d["task"], "environment": d["environment"]}
#                 }
#                 for d in st.session_state.detailed_results
#             ]
#             render_agent_responses(responses_data)
    
#     # Footer
#     st.divider()
#     st.caption(f"Kairo Evaluation Platform | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# if __name__ == "__main__":
#     main()



"""Streamlit dashboard for Kairo Evaluation Platform (Professional Edition)."""

'''
import sys
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import streamlit as st
from datetime import datetime
import logging
from io import StringIO

# -----------------------------------------------
# Add project root to path
# -----------------------------------------------
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models import Persona, Environment, Question, AgentResponse, EvaluationResult, PersonaScore
from src.pipeline import EvaluationOrchestrator
from src.agents import WeatherAgent
from src.evaluators import LLMEvaluator, EnsembleEvaluator
from src.llm import create_llm_client
from src.config import get_settings
from src.utils.logging import setup_logging, get_logger


# -----------------------------------------------
# Streamlit Log Handler
# -----------------------------------------------
class StreamlitLogHandler(logging.Handler):
    """Custom log handler that pushes logs into Streamlit session_state."""
    def emit(self, record):
        msg = self.format(record)
        if "log_buffer" not in st.session_state:
            st.session_state["log_buffer"] = ""
        st.session_state["log_buffer"] += msg + "\n"


def attach_streamlit_log_handler():
    handler = StreamlitLogHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.addHandler(handler)


# Setup logging
setup_logging()
logger = get_logger(__name__)
attach_streamlit_log_handler()


# -----------------------------------------------
# Session State Init
# -----------------------------------------------
def init_session_state():
    defaults = {
        "evaluation_running": False,
        "evaluation_results": [],
        "persona_score": None,
        "detailed_results": [],
        "log_buffer": "",
        "progress_data": {
            "current_phase": "Not started",
            "environments_selected": [],
            "questions_generated": 0,
            "responses_generated": 0,
            "evaluations_completed": 0,
            "total_steps": 0,
            "completed_steps": 0,
        }
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# -----------------------------------------------
# Loaders
# -----------------------------------------------
def load_persona(file_path: Path) -> Persona:
    with open(file_path, "r", encoding="utf-8") as f:
        return Persona(**json.load(f))


def load_environments(file_path: Path) -> List[Environment]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [Environment(**env) for env in json.load(f)]


# -----------------------------------------------
# Sidebar Config Section
# -----------------------------------------------
def render_config_panel():
    st.sidebar.header("Configuration")

    settings = get_settings()

    st.sidebar.subheader("Model Settings")
    st.sidebar.text(f"Generator: {settings.generator_model}")
    st.sidebar.text(f"Evaluator 1: {settings.evaluator_model_1}")
    st.sidebar.text(f"Evaluator 2: {settings.evaluator_model_2}")
    st.sidebar.text(f"Generator Temp: {settings.generator_temperature}")
    st.sidebar.text(f"Evaluator Temp: {settings.evaluator_temperature}")

    st.sidebar.divider()

    num_environments = st.sidebar.slider(
        "Environments per Persona",
        min_value=1, max_value=10, value=2
    )
    num_questions = st.sidebar.slider(
        "Questions per Task",
        min_value=1, max_value=10, value=2
    )

    st.sidebar.divider()

    # API Keys
    import os
    st.sidebar.subheader("API Keys Status")

    def status(val): return "Available" if val else "Missing"

    st.sidebar.text(f"OpenAI: {status(settings.openai_api_key)}")
    st.sidebar.text(f"Anthropic: {status(settings.anthropic_api_key)}")
    st.sidebar.text(f"AccuWeather: {status(os.getenv('ACCUWEATHER_API_KEY'))}")

    return num_environments, num_questions


# -----------------------------------------------
# Rendering Components
# -----------------------------------------------
def render_persona_info(persona: Persona):
    st.header("Persona Overview")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(persona.name)
        st.write(persona.description)

        if persona.attributes:
            st.write("Attributes:")
            st.write(", ".join([f"{k}: {v}" for k, v in persona.attributes.items()]))

    with col2:
        status = "Evaluating..." if st.session_state.evaluation_running else "Ready"
        st.metric("Status", status)


def render_environment_selection(environments, selected):
    st.header("Selected Environments")
    st.write(f"Total Available: {len(environments)}")
    st.write(f"Selected: {len(selected)}")

    if selected:
        cols = st.columns(min(3, len(selected)))
        for idx, env in enumerate(selected):
            with cols[idx % 3]:
                st.markdown(f"**{env.name}**")
                st.caption(env.description[:120] + "...")


def render_results_dashboard(persona_score: Optional[PersonaScore], results):
    st.header("Evaluation Results")

    if not persona_score:
        st.info("Run an evaluation to see results.")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Score", f"{persona_score.overall_score:.2f}")
    with col2:
        st.metric("Total Questions", persona_score.total_questions)
    with col3:
        st.metric("Total Evaluations", persona_score.total_evaluations)
    with col4:
        avg_per_task = persona_score.overall_score / len(persona_score.task_averages)
        st.metric("Average per Task", f"{avg_per_task:.2f}")

    st.divider()

    # Task-wise bar chart
    import pandas as pd
    import plotly.express as px

    df = pd.DataFrame({
        "Task": [t.replace("_", " ").title() for t in persona_score.task_averages],
        "Score": list(persona_score.task_averages.values())
    })

    fig = px.bar(df, x="Task", y="Score", range_y=[0, 5], title="")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detailed Results")

    results_data = []
    for result in results:
        for t in result.task_scores:
            results_data.append({
                "Question": result.question_text[:60] + "...",
                "Task": t.task_type,
                "Score": t.score,
                "Environment": result.environment_name
            })

    st.dataframe(pd.DataFrame(results_data), use_container_width=True)


def render_evaluation_details(evaluation_data):
    st.header("Full Evaluation Details")

    for idx, e in enumerate(evaluation_data, 1):
        with st.expander(f"Evaluation {idx}"):
            st.markdown("**Question:**")
            st.write(e["question"])

            st.markdown("**Agent Response:**")
            st.text_area("", e["response"], height=120, disabled=True)

            st.markdown("**Scores:**")
            st.write(e["scores"])


def render_live_logs():
    st.header("Live Evaluation Logs")
    st.text_area(
        label="Logs",
        value=st.session_state.get("log_buffer", ""),
        height=250,
        disabled=True
    )


# -----------------------------------------------
# Evaluation Pipeline Execution
# -----------------------------------------------
def run_evaluation(persona, environments, num_envs, num_questions):
    st.session_state.evaluation_running = True
    st.session_state.evaluation_results = []
    st.session_state.persona_score = None
    st.session_state.log_buffer = ""

    logger.info("Starting evaluation...")

    try:
        settings = get_settings()

        generator_client = create_llm_client(
            provider="openai",
            model=settings.generator_model,
            temperature=0.9,
        )
        agent_client = create_llm_client(
            provider="openai",
            model=settings.generator_model,
            temperature=settings.generator_temperature,
        )

        weather_agent = WeatherAgent(llm_client=agent_client)

        evaluator1 = LLMEvaluator(
            llm_client=create_llm_client(
                provider="openai",
                model=settings.evaluator_model_1,
                temperature=0.0
            )
        )
        evaluator2 = LLMEvaluator(
            llm_client=create_llm_client(
                provider="openai",
                model=settings.evaluator_model_2,
                temperature=0.0
            )
        )

        ensemble = EnsembleEvaluator([evaluator1, evaluator2])

        orchestrator = EvaluationOrchestrator(
            generator_client=generator_client,
            agent_client=agent_client,
            evaluator=ensemble,
            environment_pool=environments,
            agent=weather_agent,
        )

        logger.info("Running orchestrator.evaluate_persona_with_score()")

        results, persona_score = orchestrator.evaluate_persona_with_score(
            persona=persona,
            num_environments=num_envs,
            num_questions_per_task=num_questions
        )

        st.session_state.evaluation_results = results
        st.session_state.persona_score = persona_score

        logger.info("Evaluation completed successfully.")

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        st.error(f"Evaluation failed: {str(e)}")

    finally:
        st.session_state.evaluation_running = False


# -----------------------------------------------
# Main
# -----------------------------------------------
def main():
    init_session_state()

    st.title("Kairo Evaluation Platform")
    st.caption("Dynamic LLM Agent Evaluation Dashboard")
    st.divider()

    num_envs, num_questions = render_config_panel()

    # Load data
    data_dir = project_root / "data"
    persona = load_persona(data_dir / "personas" / "weather_persona.json")
    environments = load_environments(data_dir / "environments" / "weather_environments.json")

    # Persona
    render_persona_info(persona)
    st.divider()

    # Controls
    col1, col2, _ = st.columns([1, 1, 3])
    with col1:
        if st.button("Start Evaluation", disabled=st.session_state.evaluation_running):
            run_evaluation(persona, environments, num_envs, num_questions)
            st.rerun()

    with col2:
        if st.button("Reset", disabled=st.session_state.evaluation_running):
            st.session_state.clear()
            st.rerun()

    st.divider()

    # Live logs during evaluation
    render_live_logs()
    st.divider()

    # Results
    if st.session_state.evaluation_results:
        render_environment_selection(
            environments,
            st.session_state.progress_data.get("environments_selected", [])
        )
        st.divider()

        render_results_dashboard(
            st.session_state.persona_score,
            st.session_state.evaluation_results
        )
        st.divider()

        render_evaluation_details(st.session_state.detailed_results)

    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
    
'''
import sys
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import streamlit as st
from datetime import datetime
import logging
import threading

# -----------------------------------------------
# Add project root to path
# -----------------------------------------------
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models import Persona, Environment, Question, AgentResponse, EvaluationResult, PersonaScore
from src.pipeline import EvaluationOrchestrator
from src.agents import WeatherAgent
from src.evaluators import LLMEvaluator, EnsembleEvaluator
from src.llm import create_llm_client
from src.config import get_settings
from src.utils.logging import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)

# -----------------------------------------------
# Simple Log Collector
# -----------------------------------------------
class SimpleLogCollector(logging.Handler):
    def __init__(self):
        super().__init__()
        self.logs = []
    
    def emit(self, record):
        msg = self.format(record)
        self.logs.append(msg)
    
    def get_logs(self):
        return self.logs.copy()
    
    def clear(self):
        self.logs.clear()

# Global collector
LOG_COLLECTOR = SimpleLogCollector()
LOG_COLLECTOR.setLevel(logging.INFO)
LOG_COLLECTOR.setFormatter(logging.Formatter("%(message)s"))

# -----------------------------------------------
# Session State Init
# -----------------------------------------------
def init_session_state():
    defaults = {
        "evaluation_running": False,
        "evaluation_thread": None,
        "evaluation_results": [],
        "persona_score": None,
        "evaluation_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# -----------------------------------------------
# Loaders
# -----------------------------------------------
def load_persona(file_path: Path) -> Persona:
    with open(file_path, "r", encoding="utf-8") as f:
        return Persona(**json.load(f))

def load_environments(file_path: Path) -> List[Environment]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [Environment(**env) for env in json.load(f)]

# -----------------------------------------------
# Sidebar Config
# -----------------------------------------------
def render_config_panel():
    st.sidebar.header("⚙️ Configuration")
    settings = get_settings()

    st.sidebar.subheader("Model Settings")
    st.sidebar.text(f"Generator: {settings.generator_model}")
    st.sidebar.text(f"Evaluator 1: {settings.evaluator_model_1}")
    st.sidebar.text(f"Evaluator 2: {settings.evaluator_model_2}")

    st.sidebar.divider()

    num_environments = st.sidebar.slider("Environments per Persona", 1, 10, 2)
    num_questions = st.sidebar.slider("Questions per Task", 1, 10, 2)

    st.sidebar.divider()

    import os
    st.sidebar.subheader("🔑 API Keys")
    st.sidebar.text(f"OpenAI: {'✅' if settings.openai_api_key else '❌'}")
    st.sidebar.text(f"Anthropic: {'✅' if settings.anthropic_api_key else '❌'}")
    st.sidebar.text(f"AccuWeather: {'✅' if os.getenv('ACCUWEATHER_API_KEY') else '❌'}")

    st.sidebar.divider()
    refresh_rate = st.sidebar.slider("Refresh Rate (sec)", 1, 10, 2)

    return num_environments, num_questions, refresh_rate

# -----------------------------------------------
# Rendering
# -----------------------------------------------
def render_persona_info(persona: Persona):
    st.header("👤 Persona Overview")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(persona.name)
        st.write(persona.description)
        if persona.attributes:
            st.write("**Attributes:**")
            for k, v in persona.attributes.items():
                st.write(f"- {k}: {v}")

    with col2:
        if st.session_state.evaluation_running:
            st.metric("Status", "🔄 Running")
        elif st.session_state.evaluation_error:
            st.metric("Status", "❌ Failed")
        elif st.session_state.evaluation_results:
            st.metric("Status", "✅ Complete")
        else:
            st.metric("Status", "⏸️ Ready")

def render_results_dashboard(persona_score: Optional[PersonaScore], results):
    st.header("📊 Evaluation Results")

    if not persona_score:
        st.info("Run an evaluation to see results.")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Score", f"{persona_score.overall_score:.2f}")
    with col2:
        st.metric("Total Questions", persona_score.total_questions)
    with col3:
        st.metric("Total Evaluations", persona_score.total_evaluations)
    with col4:
        avg_per_task = persona_score.overall_score / len(persona_score.task_averages) if persona_score.task_averages else 0
        st.metric("Average per Task", f"{avg_per_task:.2f}")

    st.divider()

    if persona_score.task_averages:
        import pandas as pd
        import plotly.express as px

        df = pd.DataFrame({
            "Task": [t.replace("_", " ").title() for t in persona_score.task_averages],
            "Score": list(persona_score.task_averages.values())
        })

        fig = px.bar(df, x="Task", y="Score", range_y=[0, 5], 
                     title="Task Performance", color="Score")
        st.plotly_chart(fig, use_container_width=True)

    results_data = []
    for result in results:
        for t in result.task_scores:
            results_data.append({
                "Question": result.question_text[:80] + "..." if len(result.question_text) > 80 else result.question_text,
                "Task": t.task_type.replace("_", " ").title(),
                "Score": t.score,
                "Environment": result.environment_name.replace("_", " ").title()
            })

    if results_data:
        import pandas as pd
        st.dataframe(pd.DataFrame(results_data), use_container_width=True, height=300)

def render_live_logs():
    st.header("📋 Live Evaluation Logs")
    
    all_logs = LOG_COLLECTOR.get_logs()
    log_text = "\n".join(all_logs) if all_logs else "No logs yet. Click 'Start Evaluation' to begin."
    
    st.text_area("Logs", value=log_text, height=400, disabled=True, label_visibility="collapsed")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption(f"📝 {len(all_logs)} lines")
    with col2:
        if all_logs:
            st.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')}")
    with col3:
        if st.button("🗑️ Clear", key="clear_logs"):
            LOG_COLLECTOR.clear()
            st.rerun()

# -----------------------------------------------
# Evaluation Worker
# -----------------------------------------------
def evaluation_worker(persona, environments, num_envs, num_questions):
    try:
        settings = get_settings()

        generator_client = create_llm_client("openai", settings.generator_model, temperature=0.9)
        agent_client = create_llm_client("openai", settings.generator_model, temperature=settings.generator_temperature)

        weather_agent = WeatherAgent(llm_client=agent_client)

        evaluator1 = LLMEvaluator(llm_client=create_llm_client("openai", settings.evaluator_model_1, temperature=0.0))
        evaluator2 = LLMEvaluator(llm_client=create_llm_client("openai", settings.evaluator_model_2, temperature=0.0))
        ensemble = EnsembleEvaluator([evaluator1, evaluator2])

        orchestrator = EvaluationOrchestrator(
            generator_client=generator_client,
            agent_client=agent_client,
            evaluator=ensemble,
            environment_pool=environments,
            agent=weather_agent,
        )

        results, persona_score = orchestrator.evaluate_persona_with_score(
            persona=persona,
            num_environments=num_envs,
            num_questions_per_task=num_questions
        )

        st.session_state.evaluation_results = results
        st.session_state.persona_score = persona_score
        st.session_state.evaluation_error = None

    except Exception as e:
        logger.error(f"EVALUATION FAILED: {str(e)}", exc_info=True)
        st.session_state.evaluation_error = str(e)
    finally:
        st.session_state.evaluation_running = False
        st.session_state.evaluation_thread = None

def start_evaluation(persona, environments, num_envs, num_questions):
    st.session_state.evaluation_results = []
    st.session_state.persona_score = None
    st.session_state.evaluation_error = None
    LOG_COLLECTOR.clear()
    
    st.session_state.evaluation_running = True
    
    thread = threading.Thread(
        target=evaluation_worker,
        args=(persona, environments, num_envs, num_questions),
        daemon=True
    )
    thread.start()
    st.session_state.evaluation_thread = thread

# -----------------------------------------------
# Main
# -----------------------------------------------
def main():
    # Add collector to root logger once
    root = logging.getLogger()
    if LOG_COLLECTOR not in root.handlers:
        root.addHandler(LOG_COLLECTOR)
        root.setLevel(logging.INFO)
        simple_formatter = logging.Formatter("%(message)s")
        for handler in root.handlers:
            if handler is LOG_COLLECTOR:
                continue
            if isinstance(handler, logging.StreamHandler):
                handler.setFormatter(simple_formatter)
    
    init_session_state()

    st.title("🌤️ Kairo Evaluation Platform")
    st.caption("Dynamic LLM Agent Evaluation Dashboard")
    st.divider()

    # Load data
    data_dir = project_root / "data"
    persona = load_persona(data_dir / "personas" / "weather_persona.json")
    environments = load_environments(data_dir / "environments" / "weather_environments.json")

    # Config
    num_envs, num_questions, refresh_rate = render_config_panel()

    # Persona
    render_persona_info(persona)
    st.divider()

    # Controls
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("▶️ Start", disabled=st.session_state.evaluation_running, 
                     use_container_width=True, type="primary"):
            start_evaluation(persona, environments, num_envs, num_questions)
            st.rerun()

    with col2:
        if st.button("🔄 Reset", disabled=st.session_state.evaluation_running, use_container_width=True):
            if not (st.session_state.evaluation_thread and st.session_state.evaluation_thread.is_alive()):
                st.session_state.clear()
                LOG_COLLECTOR.clear()
                st.rerun()
    
    with col3:
        if st.session_state.evaluation_running:
            st.info("🔄 Running...")
        elif st.session_state.evaluation_error:
            st.error(f"❌ {st.session_state.evaluation_error}")

    st.divider()

    # Logs
    render_live_logs()
    
    # Auto-refresh
    if st.session_state.evaluation_running:
        time.sleep(refresh_rate)
        st.rerun()
    
    st.divider()

    # Results
    if st.session_state.evaluation_results:
        render_results_dashboard(st.session_state.persona_score, st.session_state.evaluation_results)

    st.divider()
    st.caption(f"Updated: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
