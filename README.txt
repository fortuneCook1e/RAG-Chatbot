1. Create and activate your virtual environment
	# For Conda
	conda create -n myenv python=3.10
	conda activate myenv

2. Run the following command to install all libraries listed in requirements.txt
	pip install -r requirements.txt

3. Navigate to the app directory and run the following to start the server:
	uvicorn src.main:app

4. The server is now started and user can send API request to it.