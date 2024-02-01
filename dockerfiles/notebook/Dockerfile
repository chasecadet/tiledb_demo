FROM kubeflownotebookswg/jupyter-pytorch
COPY --chown=${NB_USER}:users requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -r /tmp/requirements.txt  --no-cache-dir \
 && rm -f /tmp/requirements.txt 
