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
	pareidolia data_test/B_1.cool,data_test/B_2.cool,data_test/B_3.cool,data_test/B_4.cool,data_test/smooth_1.cool,data_test/smooth_2.cool,data_test/smooth_3.cool,data_test/smooth_4.cool B,B,B,B,s,s,s,s out_demo
test:
	tox

