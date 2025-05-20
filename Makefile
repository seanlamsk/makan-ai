.PHONY: venv run gen data scrape install clean

run:
	source venv/bin/activate && python3 qa.py

embed:
	source venv/bin/activate && python3 gen_embeddings.py

process:
	source venv/bin/activate && python3 process_data.py

scrape:
	source venv/bin/activate && python3 scraper.py

venv:
	python3 -m venv venv && echo "Virtual environment created. Run 'make install' to install dependencies."

install:
	source venv/bin/activate && pip install -r requirements.txt && echo "Dependencies installed. See readme.md for make commands to start the application."

app:
	source venv/bin/activate && streamlit run app.py

clean:
	rm -rf venv __pycache__ *.pyc

# To deactivate the virtual environment, simply run:
# deactivate
