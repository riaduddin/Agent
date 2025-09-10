import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()

llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

prompt_extract=ChatPromptTemplate.from_template(
    "Extract the technical specifications from the following text. text:\n{text}"
    )

prompt_transform=ChatPromptTemplate.from_template(
    "Transform the following technical specifications into a JSON object with keys: 'CPU', 'RAM', 'Storage' as keys: \n{specifications}")

extraction_chain=llm | prompt_extract | llm | prompt_transform | StrOutputParser()

full_chain= ({
    "specifications": extraction_chain
}
| prompt_transform
| llm
| StrOutputParser()
)