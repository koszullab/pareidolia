.PHONY: build install test clean deploy demo

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

demo:
	pareidolia data_test/A_1.cool,data_test/A_2.cool,data_test/A_3.cool,data_test/A_4.cool,data_test/smooth_1.cool,data_test/smooth_2.cool,data_test/smooth_3.cool,data_test/smooth_4.cool A,A,A,A,s,s,s,s out_demo
test:
	tox

