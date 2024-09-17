import streamlit as st
from llama_index import SimpleDirectoryReader, ServiceContext, VectorStoreIndex
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from langchain.schema import (SystemMessage, HumanMessage, AiMessage)

def init_page() -> None:
  st.set_page_config(
      page_title="Library Chatbot",
      page_icon="🦙",
  )
  st.header("Library Chatbot 🦙")
  st.sidebar.title("Options")


def select_llm() -> LlamaCPP:
  return LlamaCPP(
      model_path="./llama-2-7b-chat.Q4_K_M.gguf",
      temperature=0.1,
      max_new_tokens=500,
      context_window=3900,
      generate_kwargs={},
      model_kwargs={"n_gpu_layers": 1},
      messages_to_prompt=messages_to_prompt,
      completion_to_prompt=completion_to_prompt,
      verbose=True,
  )

def init_messages() ->None:
  clear_button = st.sidebar.button("Clear conversation", key = "clear")
  if clear_button or "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(
          content="you are a helpful assistant. Reply your answer in markdown format"
        )
    ]
def get_answer(llm, messages) -> str:
  response = llm.complete(messages)
  return response.message.content

def main() -> None:
  init_page()
  llm = select_llm()
  init_messages()

  if user_input :=st.chat_input("Input your question"):
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.spinner("Wait for it..."):
      answer = get_answer(llm, st.session_state.messages)
      print(answer)
    st.session_state.messages.append(AiMessage(content=answer))

messages = st.session_state.get('messages', [])
for message in messages:
  if isinstance(message, AiMessage):
   with st.chat_message('assistant'):
    st.markdown(message.content)
  elif isinstance(message, HumanMessage):
    with st.chat_message('user'):
      st.markdown(message.content)

  if __name__ == "__main__":
    main()
