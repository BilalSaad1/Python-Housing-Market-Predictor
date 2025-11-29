Open terminal and run the .venv
follow these steps:
pip install -r requirements.txt
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
streamlit run app.py
