import streamlit as st
import pandas as pd
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from soap_rag_system import SOAPDatasetProcessor, SOAPGenerator, SOAPEvaluator

# Page configuration
st.set_page_config(
    page_title="SOAP Note Generator",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if 'generator' not in st.session_state:
    st.session_state.generator = None
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = []

def initialize_system():
    """Initialize the SOAP generation system"""
    with st.spinner("Initializing SOAP generation system..."):
        try:
            # Load dataset and create documents
            processor = SOAPDatasetProcessor()
            df = processor.load_dataset()
            documents = processor.prepare_documents(df)
            
            # Initialize generator
            generator = SOAPGenerator(
                llm_model=st.session_state.selected_model,
                vector_store_type=st.session_state.vector_store_type
            )
            generator.setup_rag_chain(documents)
            
            st.session_state.generator = generator
            st.session_state.dataset = df
            st.success("System initialized successfully!")
            return True
        except Exception as e:
            st.error(f"Error initializing system: {str(e)}")
            return False

def main():
    """Main Streamlit application"""
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    
    # Model selection
    st.session_state.selected_model = st.sidebar.selectbox(
        "Select LLM Model",
        ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
        index=0
    )
    
    # Vector store selection
    st.session_state.vector_store_type = st.sidebar.selectbox(
        "Vector Store Type",
        ["faiss", "chroma"],
        index=0
    )
    
    # API Key input
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if api_key:
        import os
        os.environ["OPENAI_API_KEY"] = api_key
    
    # Initialize system button
    if st.sidebar.button("Initialize System"):
        if api_key:
            initialize_system()
        else:
            st.sidebar.error("Please provide OpenAI API Key")
    
    # Main content
    st.title("🏥 RAG-based SOAP Note Generator")
    st.markdown("Generate structured SOAP notes from doctor-patient conversations using RAG and LLMs")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Generate", "📊 Evaluate", "📈 Analytics", "ℹ️ About"])
    
    with tab1:
        st.header("Generate SOAP Note")
        
        if st.session_state.generator is None:
            st.warning("Please initialize the system using the sidebar configuration.")
            return
        
        # Input conversation
        conversation_input = st.text_area(
            "Enter Doctor-Patient Conversation:",
            height=200,
            placeholder="Patient: I've been having chest pain for the past two days...\nDoctor: Can you describe the pain? Is it sharp or dull?...",
            help="Enter the complete conversation between doctor and patient"
        )
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            generate_button = st.button("🔮 Generate SOAP Note", type="primary")
        
        if generate_button and conversation_input:
            with st.spinner("Generating SOAP note..."):
                result = st.session_state.generator.generate_soap_note(conversation_input)
                
                if result["success"]:
                    st.success("SOAP note generated successfully!")
                    
                    # Display generated SOAP note
                    st.subheader("Generated SOAP Note")
                    st.text_area("", value=result["soap_note"], height=300, disabled=True)
                    
                    # Display source documents
                    with st.expander("📚 Source Documents Used"):
                        for i, doc in enumerate(result["source_documents"], 1):
                            st.markdown(f"**Source {i}:**")
                            st.text(doc[:500] + "..." if len(doc) > 500 else doc)
                            st.divider()
                    
                    # Download option
                    st.download_button(
                        label="📥 Download SOAP Note",
                        data=result["soap_note"],
                        file_name=f"soap_note_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                else:
                    st.error(f"Error generating SOAP note: {result.get('error', 'Unknown error')}")
        
        elif generate_button and not conversation_input:
            st.warning("Please enter a conversation to generate SOAP note.")
    
    with tab2:
        st.header("Evaluate Generated SOAP Notes")
        
        if st.session_state.generator is None:
            st.warning("Please initialize the system first.")
            return
        
        # Sample evaluation
        st.subheader("Quick Evaluation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            generated_soap = st.text_area("Generated SOAP Note:", height=200)
        
        with col2:
            reference_soap = st.text_area("Reference SOAP Note:", height=200)
        
        if st.button("📊 Evaluate") and generated_soap and reference_soap:
            evaluator = SOAPEvaluator()
            scores = evaluator.evaluate_single(generated_soap, reference_soap)
            
            # Display scores
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("ROUGE-1 F1", f"{scores['rouge1_f']:.3f}")
            with col2:
                st.metric("ROUGE-2 F1", f"{scores['rouge2_f']:.3f}")
            with col3:
                st.metric("ROUGE-L F1", f"{scores['rougeL_f']:.3f}")
            with col4:
                st.metric("BLEU Score", f"{scores['bleu']:.3f}")
            
            # Store results for analytics
            st.session_state.evaluation_results.append({
                'timestamp': datetime.now(),
                'scores': scores
            })
        
        # Batch evaluation
        st.subheader("Batch Evaluation")
        
        uploaded_file = st.file_uploader(
            "Upload CSV with generated and reference SOAP notes",
            type=['csv'],
            help="CSV should have 'generated' and 'reference' columns"
        )
        
        if uploaded_file is not None:
            df_eval = pd.read_csv(uploaded_file)
            
            if 'generated' in df_eval.columns and 'reference' in df_eval.columns:
                if st.button("📈 Run Batch Evaluation"):
                    evaluator = SOAPEvaluator()
                    
                    with st.spinner("Evaluating batch..."):
                        batch_scores = evaluator.evaluate_batch(
                            df_eval['generated'].tolist(),
                            df_eval['reference'].tolist()
                        )
                    
                    st.subheader("Batch Evaluation Results")
                    
                    # Display average scores
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Avg ROUGE-1", f"{batch_scores['avg_rouge1_f']:.3f}")
                    with col2:
                        st.metric("Avg ROUGE-2", f"{batch_scores['avg_rouge2_f']:.3f}")
                    with col3:
                        st.metric("Avg ROUGE-L", f"{batch_scores['avg_rougeL_f']:.3f}")
                    with col4:
                        st.metric("Avg BLEU", f"{batch_scores['avg_bleu']:.3f}")
            else:
                st.error("CSV must have 'generated' and 'reference' columns")
    
    with tab3:
        st.header("Analytics Dashboard")
        
        if not st.session_state.evaluation_results:
            st.info("No evaluation results available. Run some evaluations first!")
            return
        
        # Prepare data for visualization
        eval_data = []
        for result in st.session_state.evaluation_results:
            eval_data.append({
                'timestamp': result['timestamp'],
                'ROUGE-1': result['scores']['rouge1_f'],
                'ROUGE-2': result['scores']['rouge2_f'],
                'ROUGE-L': result['scores']['rougeL_f'],
                'BLEU': result['scores']['bleu']
            })
        
        df_analytics = pd.DataFrame(eval_data)
        
        # Score trends over time
        st.subheader("📈 Score Trends")
        
        fig = go.Figure()
        
        metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for metric, color in zip(metrics, colors):
            fig.add_trace(go.Scatter(
                x=df_analytics['timestamp'],
                y=df_analytics[metric],
                mode='lines+markers',
                name=metric,
                line=dict(color=color, width=2),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title="Evaluation Metrics Over Time",
            xaxis_title="Time",
            yaxis_title="Score",
            hovermode='x unified',
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Score distribution
        st.subheader("📊 Score Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Box plot
            fig_box = go.Figure()
            
            for metric, color in zip(metrics, colors):
                fig_box.add_trace(go.Box(
                    y=df_analytics[metric],
                    name=metric,
                    marker_color=color
                ))
            
            fig_box.update_layout(
                title="Score Distribution",
                yaxis_title="Score",
                template='plotly_white'
            )
            
            st.plotly_chart(fig_box, use_container_width=True)
        
        with col2:
            # Summary statistics
            st.subheader("📋 Summary Statistics")
            summary_stats = df_analytics[metrics].describe()
            st.dataframe(summary_stats)
    
    with tab4:
        st.header("About SOAP Note Generator")
        
        st.markdown("""
        ### 🎯 Overview
        This RAG-based SOAP Note Generator uses advanced language models and retrieval-augmented generation 
        to create structured medical documentation from doctor-patient conversations.
        
        ### 🔧 Features
        - **RAG Architecture**: Combines retrieval and generation for contextually relevant SOAP notes
        - **Multiple LLM Support**: Compatible with GPT-3.5, GPT-4, and other models
        - **Vector Storage**: Uses FAISS or Chroma for efficient similarity search
        - **Comprehensive Evaluation**: ROUGE and BLEU metrics for quality assessment
        - **Interactive Interface**: Easy-to-use Streamlit interface
        
        ### 📊 SOAP Format
        - **Subjective**: Patient's reported symptoms and history
        - **Objective**: Observable findings and clinical measurements
        - **Assessment**: Clinical impression and diagnosis
        - **Plan**: Treatment recommendations and follow-up
        
        ### 🚀 Getting Started
        1. Configure your API key and model settings in the sidebar
        2. Initialize the system to load the dataset and create vector embeddings
        3. Enter a doctor-patient conversation in the Generate tab
        4. Review and evaluate the generated SOAP note
        
        ### 📚 Dataset
        Uses the `adesouza1/soap_notes` dataset for training examples and retrieval context.
        
        ### ⚠️ Disclaimer
        This tool is for educational and research purposes only. Always verify medical documentation 
        with qualified healthcare professionals.
        """)

if __name__ == "__main__":
    main()
