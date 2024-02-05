from fastapi import FastAPI
import gradio as gr
import os

app = FastAPI()
CUSTOM_PATH = os.environ.get("CUSTOM_PATH", "/tiledb")

# Define the Gradio interface function
def greet(name, intensity):
    return "Hello " * intensity + name + "!"

# Create the Gradio interface
io = gr.Interface(fn=greet, inputs=["text", "slider"], outputs="text")

# Create a Gradio app from the interface
gradio_app = gr.routes.App.create_app(io)

# Mount the Gradio app on the custom path within the FastAPI application
app.mount(CUSTOM_PATH, gradio_app)

@app.get("/")
def read_main():
    return {"message": "This is the main app. Access the Gradio interface at /tiledb"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

