---
layout: post
title: "Retriveal Augmented Generation (RAG) Chatbot for 10Q Financial Reports"
date: 2024-04-26
categories: [LLM, Generative AI, Natural Language Processing]
---

While Large Language Models (LLMs) are revolutionary, they sometimes get it wrong—like citing varying figures for something as critical as Tesla's total assets on a given date. In the accompanying figure, you can see ChatGPT4 giving different results when asked the same question multiple times. This problem is called LLM hallucinations. And that's where Retrival Augmented Generation (RAG) comes in. In this blog post, I'll describe how to create a Chabot for 10Q Financial Reports that leverages RAG.

<div style="text-align: center">
<figure>
<img src="/img/llm_hallucination.png" style="display: block; margin: auto;  max-width: 100%;">
<figcaption>LLM hallucination</figcaption>
</figure>
</div>

## What is Retrival Augmented Generation (RAG)? 

It's a framework that combines the strengths of information retrieval and generative language modeling to enhance the capabilities of machine learning systems, particularly in tasks that involve natural language understanding and generation. It involves two main components.

1. **Retrieval Component:** responsible for accessing an external knowledge source, such as a database or a document collection, to retrieve relevant information based on the input query.
2. **Generation Component:** leverages LLMs to generate response based on the context provided by the retrieval component.

## Building RAG Chatbot

### Dataset

The dataset primarily consists of financial documents, specifically 10-Q and 10-K filings from major publicly traded companies, such as Tesla, NVIDIA, and Apple. These documents are obtained from the U.S. Securities and Exchange Commission's (SEC) [EDGAR database](https://www.sec.gov/edgar/searchedgar/companysearch), which is a reliable source for such financial reports. Each 10-Q and 10-K filing within the dataset contains a comprehensive overview of a company's financial performance.

<div style="text-align: center">
<figure>
<img src="/img/tsla_10q.png" style="display: block; margin: auto;  max-width: 80%;">
<figcaption>Tesla 10Q</figcaption>
</figure>
</div>

### Steps

We need to following the following steps to build a RAG Chatbot. 

* **Problem statement:** Given a PDF document and a query, retrieve the relevant details and information from the document as per the query, and synthesize this information to generate accurate answers.
* **Data Ingestion and Processing:** Reading PDFs of financial reports and split the documents for efficient text chunking of long documents.
* **Retrieval-Augmented Generation (RAG):** Combination of document retrieval with the generative capabilities of the chosen language models.
* **Large Language Models:** Evaluation of various models, including GPT-3.5-turbo, LLama 2, Gemma 1.1, etc.
* **Conversation Chain and Prompt Design:** Crafting of a prompt template designed for concise two-sentence financial summaries.
* **User interface:** Designing Chatbot like user interface.

First, we load the 10-Q PDF using PyPDFLoader.

{% highlight python %}
from langchain.document_loaders import PyPDFLoader
# create a loader
loader = PyPDFLoader(r"data/tsla-20230930.pdf")
{% endhighlight %}

We then split data in chunks using a recursive character text splitter to handle large documents.

{% highlight python %}
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=300)
all_splits = text_splitter.split_documents(data)
{% endhighlight %}

We now create the embeddings using Sentence Transformer and HuggingFace embeddings. In order to create vector embeddings, we use the open-source Chroma vector database.

{% highlight python %}
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

model_name = "sentence-transformers/all-mpnet-base-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cuda"})

vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="chroma_db")
{% endhighlight %}

We use HuggingFace to load LLama 2 model and create a HuggingFace pipeline. Since, we're going to use LangChain, we use `HugggingFacePipeline` wrapper from LangChain to create LangChain llm object, which we're going to use to do further processing. 

{% highlight python %}
import transformers
from transformers import LlamaForCausalLM, AutoTokenizer, AutoConfig
from langchain.llms import HuggingFacePipeline

model_config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                            trust_remote_code = True, config = model_config, device_map = 'auto')
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Creating Pipeline
query_pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",)
llm = HuggingFacePipeline(pipeline=query_pipeline)
{% endhighlight %}

If we want to use GPT models from OpenAI, we can diretly use `openai` API.

{% highlight python %}
import os
import openai
from langchain.chat_models import ChatOpenAI

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
{% endhighlight %}

Finally, we create a LangChain chain for our RAG system. We also pass a task-specific prompt to guide LLM for question answering wrt RAG for financial reports.

{% highlight python %}
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Define prompt template
template = """You are an assistant for question-answering tasks for Retrieval Augmented Generation system for the financial reports such as 10Q and 10K.
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use two sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""

prompt = ChatPromptTemplate.from_template(template)
retriever = vectordb.as_retriever()

# Setup RAG pipeline
conversation_chain = (
    {"context": retriever,  "question": RunnablePassthrough()} 
    | prompt 
    | llm
    | StrOutputParser() 
)
{% endhighlight %}

Finally, we invoke our conversation chain on user input.

{% highlight python %}
user_input = "What's the total assets of Tesla?"
output = conversation_chain.invoke(user_input)
{% endhighlight %}

We can integrate our code with some frontend e.g. with Dash to have chatbot like interface.

<div style="text-align: center">
<figure>
<img src="/img/rag_chatbot_10q.png" style="display: block; margin: auto;  max-width: 80%;">
<figcaption>RAG Chatbot</figcaption>
</figure>
</div>

The full code is availble at [https://github.com/kHarshit/Financial_Document_Summarization_through_RAG](https://github.com/kHarshit/Financial_Document_Summarization_through_RAG)
