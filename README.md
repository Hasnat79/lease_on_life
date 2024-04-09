# lease_on_life

## File Structure
``` 
project_root/
│
├── app.py                      # Main script for running the application
├── data/                       # Directory for storing processed data
│   └── Lease.pdf               # Example of processed lease agreement
├── db/                         # Directory for storing local database files
│   ├── 45ed88cd-4827-4ea6-a86a-026a0789a9bd/
│   │   ├── data_level0.bin
│   │   ├── header.bin
│   │   ├── length.bin
│   │   └── link_lists.bin
│   └── chroma.sqlite3          # Example of local database
├── ingest.py                   # Script for processing lease agreement PDF files
├── LaMini-T5-738M/             # Directory for LaMini model files
│   ├── config.json
│   ├── generation_config.json
│   ├── pytorch_model.bin
│   ├── README.md
│   ├── special_tokens_map.json
│   ├── spiece.model
│   ├── tokenizer_config.json
│   └── tokenizer.json
├── lease_agreements/           # Directory for storing lease agreement PDF files
│   └── Lease.pdf               # Example of lease agreement file
└── prompt_template_for_llm.py  # Template for generating prompts for language model
```
## Setup
- Create a conda environment named lease_env with Python 3.9:
```
conda create -n lease_env python=3.9
conda activate lease_env
```
- Install the required modules listed in requirements.txt:
```
pip install -r requirements.txt
```
- Clone the LaMini-T5-738M from huggingface repository
```
git lfs install
git clone https://huggingface.co/MBZUAI/LaMini-T5-738M
```
## Usage
1. Place lease agreement PDF files in the lease_agreements directory.
2. Run app.py to start the application:
```
python app.py
```
3. Upload lease agreement files via the application interface.
4. Ask questions about the lease agreements, and the application will provide answers based on the processed data

## Misc
1. to run and edit streamlit version uncomment streamlit_main() inside app.py and run the following
```
streamlit run app.py --server.headless true