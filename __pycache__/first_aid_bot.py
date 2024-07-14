import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import os
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

class FirstAidBot:
    def __init__(self):
        # Load first aid data from CSV
        if not os.path.exists('first_aid_data.csv'):
            raise FileNotFoundError("The file 'first_aid_data.csv' was not found. Please make sure it exists in the directory.")
        
        self.first_aid_data = pd.read_csv('first_aid_data.csv')

        # Initialize SentenceTransformer model for encoding advice texts
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.index = self.create_faiss_index()

        # Initialize LangChain LLMChain
        self.llm_chain = self.create_llm_chain()

    def create_faiss_index(self):
        # Convert advice texts to vectors
        advice_vectors = self.model.encode(self.first_aid_data['advice_text'].tolist())
        # Create a FAISS index
        index = faiss.IndexFlatL2(advice_vectors.shape[1])
        index.add(advice_vectors)
        return index  # Return the newly created index

    def create_llm_chain(self):
        # Define a prompt template and initialize LLMChain
        prompt_template = PromptTemplate(input_variables=["user_input"], template="Provide first aid advice for: {user_input}")
        llm = OpenAI(model_name="text-davinci-003")  # Example model, adjust as needed
        return LLMChain(prompt_template=prompt_template, llm=llm)

    def get_first_aid_advice(self, user_input):
        # Convert user input to a vector
        user_vector = self.model.encode([user_input])
        # Search the FAISS index
        _, advice_ids = self.index.search(user_vector, k=1)
        # Retrieve the most relevant advice
        most_relevant_advice = self.first_aid_data.iloc[advice_ids[0][0]]['advice_text']

        # Additional processing to make the response more precise
        precise_advice = self.process_advice(most_relevant_advice)

        # Use LangChain to further refine the advice
        refined_advice = self.llm_chain.run(user_input=user_input)

        return f"{precise_advice}\n\nAdditional Advice: {refined_advice}"

    def process_advice(self, advice):
        # Example of processing to make the advice more precise
        if "Apply pressure to the wound to stop bleeding." in advice:
            advice += "\nClean the wound with mild soap and water, then apply sterile dressing."

        return advice
