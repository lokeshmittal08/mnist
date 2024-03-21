# mnist

MLWorkflow Assignment- Easy

 This assignment focuses more on the post training stuff, primarily
 pipeline management and deployment.
 In this assignment, you will design and build a small ML pipeline based on the MNIST example
 from a public repository. You will implement the following components of this pipeline: model
 store, testing suite and an API.
 Please follow these steps:
 1. Please choose your favourite deep learning framework to implement the model training
 portion of the ML pipeline. Feel free to leverage existing tutorial examples and public
 template repositories (pytorch, pytorch-lightning, pytorch-lightning-hydra, tensorflow2
 etc).
 2. Please design your training code to take as input a configuration file
 (initial_experiment.yaml) defining the following parameters and values.
 ● max_epochs: 1
 ● lr: 0.01
 ● batch_size: 64
 3. Please train an initial MNIST model with the configuration defined above. An example
 run command should be similar to:
 $ python train.py--config initial_experiment.yaml
 4. Implement a model checkpoint management system (ie. model-registry). Some good
 ways to implement this would be leveraging packages such as MLFlow, Neptune, etc.
 5. Please provide a set of tests for the model checkpoint management system above.
6. Make an API to interact with the model-store implemented above. Concretely, the API
 must support the following two use cases:
 ● Inference on a given image.
 @app.post(...)
 def run_inference(...):
 ● Train a model given a new configuration file. You can be creative with the
 parameters and values of the new configuration.
 @app.post(...)
 def run_train(...):
 7. The service should be deployed and tested locally. Feel free to utilize packages such as
 FastAPI and Flask to make your life easier.


Commands to be executed :

Package installation : pip install -r requirements.txt
For executing the train : > python train.py --config initial_experiment.yaml
For mlflow server : mlflow ui
for Unit testing : python -m unittest test_model_checkpoint
For making FASTAPI :  uvicorn model_api:app --reload
for API Calling : python .\api_calling.py


