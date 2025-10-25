.PHONY: install train test lint clean

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	# instalar el paquete en modo editable para que `import src...` funcione en tests
	pip install -e .

train:
	python src/train.py --config config.yaml

test:
	pytest -q

lint:
	flake8 src || true

clean:
	rm -rf mlruns models __pycache__ .pytest_cache *.egg-info build dist