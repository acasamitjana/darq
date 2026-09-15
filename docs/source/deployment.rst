Deployment and Remote Access
============================

Overview
--------

DARQ can be deployed using Docker Compose.

The application is divided into three main services:

``frontend``
   Gradio web interface. It is exposed on port 7860.

``api``
   FastAPI backend. It is exposed on port 8000.

``worker``
   DARQ processing worker. It does not require a public HTTP port.

Docker Compose allows the complete system to be started and stopped using a
small set of commands.


Requirements
------------

The deployment machine should provide:

* Git.
* Docker.
* Docker Compose.
* Access to the DARQ source repository.

For GPU execution, the machine must additionally provide:

* A compatible NVIDIA GPU.
* NVIDIA drivers.
* NVIDIA Container Toolkit / GPU support for Docker.


Obtaining the project
---------------------

Clone the repository on the machine where DARQ will run:

.. code-block:: bash

   git clone <repository-url>
   cd darq

If a specific branch must be used:

.. code-block:: bash

   git checkout <branch-name>


Validating Docker Compose
-------------------------

Before starting the application, the Docker Compose configuration can be
validated with:

.. code-block:: bash

   docker compose config

This command checks that the Compose configuration can be interpreted
correctly without starting the services.


Starting DARQ
-------------

To build the images and start all services:

.. code-block:: bash

   docker compose up --build

This command runs Docker Compose in the foreground and displays the service
logs directly in the terminal.

For normal server operation, DARQ can be started in the background:

.. code-block:: bash

   docker compose up -d --build

The ``-d`` option starts the containers in detached mode, allowing them to
continue running after returning to the terminal.


Checking the services
---------------------

The current state of the containers can be checked with:

.. code-block:: bash

   docker compose ps

The expected services are:

* ``api``
* ``worker``
* ``frontend``


Viewing logs
------------

Logs from all services can be displayed with:

.. code-block:: bash

   docker compose logs -f

Logs from an individual service can also be inspected.

FastAPI:

.. code-block:: bash

   docker compose logs -f api

DARQ worker:

.. code-block:: bash

   docker compose logs -f worker

Gradio frontend:

.. code-block:: bash

   docker compose logs -f frontend

The worker logs are particularly useful for monitoring the execution of the
DARQ pipeline and diagnosing processing errors.


Accessing DARQ locally
----------------------

If the browser is running on the same machine as Docker, open:

::

   http://localhost:7860

to access the Gradio interface.

FastAPI documentation is available at:

::

   http://localhost:8000/docs

The address ``0.0.0.0`` can appear in application logs because the service
uses it as a listening address. It is not the address that should normally
be entered in the browser.


Docker internal network
-----------------------

Docker Compose provides an internal network between the services.

Inside Docker, the frontend communicates with FastAPI using the service name:

::

   http://api:8000

Inside a container, ``localhost`` refers to that same container. Therefore,
the frontend must use the Compose service name ``api`` instead of
``localhost`` to communicate with the FastAPI container.

From the host computer, however, the published ports are accessed using
``localhost``.


Stopping DARQ
-------------

To stop and remove the containers created by Docker Compose:

.. code-block:: bash

   docker compose down

To restart the services:

.. code-block:: bash

   docker compose restart


Rebuilding after code changes
-----------------------------

If dependencies, the Dockerfile or relevant application code have changed,
the Docker images should be rebuilt.

A normal rebuild can be performed with:

.. code-block:: bash

   docker compose up -d --build

If a completely clean image build is required:

.. code-block:: bash

   docker compose build --no-cache
   docker compose up -d


GPU verification
----------------

On systems configured for NVIDIA GPU execution, first check that the host can
detect the GPU:

.. code-block:: bash

   nvidia-smi

Docker GPU access can be tested with:

.. code-block:: bash

   docker run --rm --gpus all nvidia/cuda:12.9.0-base-ubuntu22.04 nvidia-smi

PyTorch GPU availability inside the DARQ worker can be checked with:

.. code-block:: bash

   docker compose exec worker python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('PyTorch CUDA:', torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

If CUDA is unavailable, inspect the Docker GPU configuration, NVIDIA drivers
and the PyTorch installation used by the worker.


Remote server deployment
------------------------

DARQ can also run permanently on a remote server.

First connect to the server:

.. code-block:: bash

   ssh -p <SSH_PORT> <USER>@<SERVER>

Move to the DARQ deployment directory:

.. code-block:: bash

   cd <DARQ_DEPLOYMENT_DIRECTORY>

Then start the services:

.. code-block:: bash

   docker compose up -d --build

Check that the containers are running:

.. code-block:: bash

   docker compose ps


Remote access using an SSH tunnel
---------------------------------

The Gradio and FastAPI ports do not need to be accessed directly from the
remote network.

An SSH tunnel can forward the remote ports to the user's computer.

Run the following command from the user's local computer:

.. code-block:: bash

   ssh -L 7860:localhost:7860 -L 8000:localhost:8000 -p <SSH_PORT> <USER>@<SERVER>

Keep this SSH connection open while using DARQ.

The first forwarding rule:

::

   -L 7860:localhost:7860

connects port 7860 on the user's computer to port 7860 on the remote server.

The second rule:

::

   -L 8000:localhost:8000

does the same for FastAPI.

After the SSH tunnel has been established, open:

::

   http://localhost:7860

on the local computer to use DARQ.

FastAPI can be accessed locally through the same tunnel at:

::

   http://localhost:8000/docs

Although ``localhost`` is used in the browser, the actual DARQ services are
running on the remote server.


SSH tunnel overview
-------------------

The remote access flow is:

::

   Local computer                         Remote server

   Browser
      |
      | http://localhost:7860
      v
   localhost:7860
      |
      |       encrypted SSH tunnel
      +---------------------------------> localhost:7860
                                               |
                                               v
                                            Gradio
                                               |
                                               v
                                            FastAPI
                                               |
                                               v
                                          DARQ worker
                                               |
                                               v
                                         DARQ pipeline


Updating a server deployment
----------------------------

To update an existing server installation, enter the DARQ directory and
retrieve the latest version of the selected branch:

.. code-block:: bash

   git pull

Then rebuild and restart the containers:

.. code-block:: bash

   docker compose down
   docker compose up -d --build


Inspecting a processing job
---------------------------

Each job stores metadata and logs that can be used to diagnose execution
problems.

The main files to inspect are:

::

   webapp/jobs/<job_id>/meta.json
   webapp/jobs/<job_id>/logs/

The ``meta.json`` file contains the current job state and progress
information.

The worker and pipeline logs provide detailed information if a processing job
fails.


Troubleshooting
---------------

Application does not open
^^^^^^^^^^^^^^^^^^^^^^^^^

Check whether all containers are running:

.. code-block:: bash

   docker compose ps

Then inspect the logs:

.. code-block:: bash

   docker compose logs -f


Worker does not process jobs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Inspect the worker logs:

.. code-block:: bash

   docker compose logs -f worker

Verify that the worker container is running and that the job metadata
contains a valid processing state.


FastAPI does not respond
^^^^^^^^^^^^^^^^^^^^^^^^

Inspect the API logs:

.. code-block:: bash

   docker compose logs -f api

When running locally, verify:

::

   http://localhost:8000/docs


GPU is not detected
^^^^^^^^^^^^^^^^^^^

Check the host first:

.. code-block:: bash

   nvidia-smi

Then verify CUDA inside the worker container using the GPU verification
commands described above.


Remote interface cannot be reached
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Verify that the Docker services are running on the server and that the SSH
tunnel remains open.

Then access:

::

   http://localhost:7860

from the local computer.