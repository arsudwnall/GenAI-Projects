Objective of the project 

To develop a QnA LLM application using LangChain which can answer the queries based on a user guide. 

Steps followed 

a. Sample user gudie was cleaned 
b. Chunks were created using Semantic Chunking (Topic-based) as that was most suitable for the nature of user guide
c. Future attempts to try hybrid = Semantic Chunking + Fixed length so long user guides can be handled 
d. Then the chunks were vecttorized and tested using LangChain
e. post the validation of LangChain the code was moved to Flask application to support a UI for QnA