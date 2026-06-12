run:
	pip install -r requirements.txt
	python3 src/modelisation.py
	python3 test.py
	python3 return_predictor/app.py