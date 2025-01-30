import json
import logging
import argparse
import requests

import httpx

from kserve import Model, ModelServer, model_server


logger = logging.getLogger(__name__)

PREDICTOR_URL_FORMAT = "http://{0}/v1/models/{1}:predict"


class Transformer(Model):
    def __init__(self, name: str, predictor_host: str, protocol: str,
                 use_ssl: bool, vectorstore_name: str = "vectorstore"):
        super().__init__(name)
        # KServe specific arguments
        self.name = name
        self.predictor_host = predictor_host
        self.protocol = protocol
        self.use_ssl = use_ssl
        self.ready = True

        # Transformer specific arguments
        self.vectorstore_name = vectorstore_name
        self.vectorstore_url = self._build_vectorstore_url()

    def _get_namespace(self):
        return (open(
            "/var/run/secrets/kubernetes.io/serviceaccount/namespace", "r")
            .read())

    def _build_vectorstore_url(self):
        domain_name = "svc.cluster.local"
        namespace = self._get_namespace()
        deployment_name = self.vectorstore_name
        model_name = deployment_name
        # Build the vectorstore URL
        svc = f'{deployment_name}-predictor.{namespace}.{domain_name}'
        url = f"http://{svc}/v1/models/{model_name}:predict"
        return url

    @property
    def _http_client(self):
        if self._http_client_instance is None:
            # No Authorization header needed
            self._http_client_instance = httpx.AsyncClient(verify=False)  # Removed headers argument
        return self._http_client_instance

    def preprocess(self, request: dict, headers: dict) -> dict:
        data = request["instances"][0]
        query = data["input"]
        logger.info(f"Received question: {query}")
        num_docs = data.get("num_docs", 4)
        context = data.get("context", None)
        if context:
            logger.info(f"Received context: {context}")
            logger.info(f"Skipping retrieval step...")
            return {"instances": [data]}
        else:
            payload = {"instances":[{"input": query, "num_docs": num_docs}]}
            logger.info(
                f"Receiving relevant docs from: {self.vectorstore_url}")

            response = requests.post(
                self.vectorstore_url, json=payload,
                verify=False)
            response = json.loads(response.text)        
            context = "\n".join(response["predictions"])
            logger.info(f"Received documents: {context}")
            return {"instances": [{**data, **{"context": context}}]}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[model_server.parser])
    parser.add_argument(
        "--predictor_host", help="The URL for the model predict function",
        required=True)
    parser.add_argument(
        "--protocol", help="The protocol for the predictor", default="v1")
    parser.add_argument(
        "--model_name", help="The name that the model is served under.")
    parser.add_argument(
        "--use_ssl", help="Use ssl for connecting to the predictor",
        action='store_true')
    parser.add_argument("--vectorstore_name", default="vectorstore",
                        required=False,
                        help="The name of the Vector Store Inference Service")
    args, _ = parser.parse_known_args()

    logger.info(args)

    model = Transformer(
        args.model_name, args.predictor_host, args.protocol, args.use_ssl,
        args.vectorstore_name)
    ModelServer().start([model])


import json
import logging
import argparse
import requests
import httpx
from kserve import Model, ModelServer, model_server

logger = logging.getLogger(__name__)
http://{0}/v1/chat/completions
LLM_PREDICTOR_URL = "http://{0}/v1/chat/completions"

class Transformer(Model):
    def __init__(self, name: str, predictor_host: str, protocol: str,
                 use_ssl: bool, vectorstore_name: str = "vectorstore"):
        super().__init__(name)
        # KServe specific arguments
        self.name = name
        self.predictor_host = predictor_host
        self.protocol = protocol
        self.use_ssl = use_ssl
        self.ready = True

        # Transformer specific arguments
        self.vectorstore_name = vectorstore_name
        self.vectorstore_url = self._build_vectorstore_url()

    def _get_namespace(self):
        return (open(
            "/var/run/secrets/kubernetes.io/serviceaccount/namespace", "r")
            .read().strip())

    def _build_vectorstore_url(self):
        domain_name = "svc.cluster.local"
        namespace = self._get_namespace()
        deployment_name = self.vectorstore_name
        model_name = deployment_name
        # Build the vectorstore URL
        svc = f'{deployment_name}-predictor.{namespace}.{domain_name}'
        url = f"http://{svc}/v1/models/{model_name}:predict"
        return url

    def preprocess(self, request: dict, headers: dict) -> dict:
        data = request["instances"][0]
        query = data["input"]
        num_docs = data.get("num_docs", 4)
        system_message = data.get("system", "You are an AI assistant.")
        instruction = data.get("instruction", "Answer the question using the context below.")

        logger.info(f"Received question: {query}")
        context = data.get("context", None)

        # If no context is provided, retrieve documents from the vector store
        if not context:
            payload = {"instances": [{"input": query, "num_docs": num_docs}]}
            logger.info(f"Retrieving relevant documents from: {self.vectorstore_url}")

            response = requests.post(self.vectorstore_url, json=payload, verify=False)
            response_data = response.json()

            if response.status_code == 200 and "predictions" in response_data:
                context = "\n".join(response_data["predictions"])
            else:
                context = "No relevant documents found."

        logger.info(f"Retrieved Context:\n{context}")

        # Call the LLM predictor
        llm_payload = {
            "model": "meta/llama-2-7b-chat",
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"{instruction}\n\nContext: {context}\n\nQuestion: {query}"}
            ],
            "temperature": data.get("temperature", 0.5),
            "top_p": data.get("top_p", 1),
            "max_tokens": int(data.get("max_tokens", 256)),
            "stream": False
        }

        logger.info(f"Sending request to LLM predictor at {LLM_PREDICTOR_URL}")
        llm_response = requests.post(LLM_PREDICTOR_URL, json=llm_payload, verify=False)

        if llm_response.status_code == 200:
            result = llm_response.json()["choices"][0]["message"]["content"]
            logger.info(f"LLM Response: {result}")
            return {"predictions": [result]}
        else:
            error_message = f"Error calling LLM predictor: {llm_response.status_code} - {llm_response.text}"
            logger.error(error_message)
            return {"predictions": [error_message]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[model_server.parser])
    parser.add_argument(
        "--predictor_host", help="The URL for the model predict function",
        required=True)
    parser.add_argument(
        "--protocol", help="The protocol for the predictor", default="v1")
    parser.add_argument(
        "--model_name", help="The name that the model is served under.")
    parser.add_argument(
        "--use_ssl", help="Use SSL for connecting to the predictor",
        action='store_true')
    parser.add_argument("--vectorstore_name", default="vectorstore",
                        required=False,
                        help="The name of the Vector Store Inference Service")
    args, _ = parser.parse_known_args()

    logger.info(args)

    model = Transformer(
        args.model_name, args.predictor_host, args.protocol, args.use_ssl,
        args.vectorstore_name)
    ModelServer().start([model])
