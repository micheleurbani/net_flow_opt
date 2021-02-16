# NetFlowOpt: resolve maintenance scheduling on networked systems

The repository contains support software to run the model presented in the
paper **Maintenance optimization on a network flow problem: a bi-objective
approach**.
To run the model, a GUI was built using Plotly Dash and it is thus available
as a web-app.

## Get started

Right now, the only way to access the GUI is through `virtualenv` and running
the Flask development server.
In the future, support for deplooyment in Docker will be available.

Clone the project in the desired folder:

```
$ git clone https://github.com/mikiurbi/net_flow_opt.git
$ cd net_flow_opt
```

It is necessary to install PostgreSQL database, and to create a databese named `flask_app`.

### Pip

In the folder where you want to store the project, run the following code from
the command line.

```
python -m venv venv
source venv/bin/activate
(venv) pip install -r requirements.txt
(venv) python wsgy.py
# Load the site at http://127.0.0.1:5010
```

### Documentation

Check the documentation at https://mikiurbi.github.io/net_flow_opt/.
