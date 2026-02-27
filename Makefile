.PHONY: test webui run

test:
	pytest -q

run:
	python main.py

webui:
	streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501
