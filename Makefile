ENV=.venv

install-linux:
	sudo apt install python 3.11 python3-pip python3.11-venv
	sudo apt install graphviz

install-mac:
	brew install python@3.11
	brew link python@3.11
	brew install graphviz

build:
	test -d $(ENV) || python3.11 -m venv $(ENV)
	$(ENV)/bin/pip3 install -U pip
	$(ENV)/bin/pip3 install --trusted-host pypi.python.org -r requirements.txt

clean:
	rm -rf $(ENV)