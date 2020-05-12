.PHONY: build install test clean deploy

install:
	pip install -e .

uninstall:
	pip uninstall pareidolia

clean:
	rm -rf build/ dist/

build: clean
	python setup.py sdist bdist_wheel

deploy: build
	twine upload dist/*

test:
	tox

